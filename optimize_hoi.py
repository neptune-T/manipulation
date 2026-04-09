import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

def quat_to_matrix_xyzw(quat):
    q = quat / torch.norm(quat)
    x, y, z, w = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
    ]).reshape(3, 3)


def _load_binary_stl_vertices(stl_path):
    if not os.path.exists(stl_path):
        return np.zeros((0, 3), dtype=np.float32)
    with open(stl_path, "rb") as handle:
        handle.read(80)
        tri_count_bytes = handle.read(4)
        if len(tri_count_bytes) != 4:
            return np.zeros((0, 3), dtype=np.float32)
        tri_count = struct.unpack("<I", tri_count_bytes)[0]
        data = handle.read()

    verts = []
    for tri_idx in range(int(tri_count)):
        offset = tri_idx * 50
        if offset + 50 > len(data):
            break
        vals = struct.unpack("<12fH", data[offset : offset + 50])
        verts.extend(
            [
                (vals[3], vals[4], vals[5]),
                (vals[6], vals[7], vals[8]),
                (vals[9], vals[10], vals[11]),
            ]
        )
    if len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32)


def _estimate_palm_reference_local(mano_urdf_path):
    palm_mesh_path = os.path.join(os.path.dirname(mano_urdf_path), "meshes", "palm.stl")
    palm_vertices = _load_binary_stl_vertices(palm_mesh_path)
    if palm_vertices.shape[0] == 0:
        fallback = np.array([-0.04, 0.0, 0.0], dtype=np.float32)
        return fallback.copy(), fallback.copy()

    palm_center_local = palm_vertices.mean(axis=0).astype(np.float32)
    x_thresh = float(np.quantile(palm_vertices[:, 0], 0.30))
    palm_patch = palm_vertices[palm_vertices[:, 0] <= x_thresh]
    if palm_patch.shape[0] < 32:
        palm_patch = palm_vertices
    palm_patch_local = palm_patch.mean(axis=0).astype(np.float32)
    return palm_center_local, palm_patch_local

class ManoChamferSDFOptimizer(nn.Module):
    def __init__(self, mano_urdf_path, init_palm_pos, init_palm_rot, device="cuda"):
        super().__init__()
        self.device = device
        urdf_bytes = open(mano_urdf_path, "rb").read()
        self.chain = pk.build_chain_from_urdf(urdf_bytes).to(device=device)
        
        self.palm_pos = nn.Parameter(torch.tensor(init_palm_pos, dtype=torch.float32, device=device))
        self.palm_rot = nn.Parameter(torch.tensor(init_palm_rot, dtype=torch.float32, device=device))
        self.qpos = nn.Parameter(torch.ones(20, dtype=torch.float32, device=device) * 0.1)

        # 保存初始旋转作为基准，防止优化时手腕乱扭
        self.init_palm_rot = torch.tensor(init_palm_rot, dtype=torch.float32, device=device)
        palm_center_local, palm_patch_local = _estimate_palm_reference_local(mano_urdf_path)
        self.palm_center_local = torch.tensor(palm_center_local, dtype=torch.float32, device=device)
        self.palm_patch_local = torch.tensor(palm_patch_local, dtype=torch.float32, device=device)
        
        self.joint_names = [
            'j_index1y', 'j_index1x', 'j_index2', 'j_index3',
            'j_middle1y', 'j_middle1x', 'j_middle2', 'j_middle3',
            'j_pinky1y', 'j_pinky1x', 'j_pinky2', 'j_pinky3',
            'j_ring1y', 'j_ring1x', 'j_ring2', 'j_ring3',
            'j_thumb1y', 'j_thumb1z', 'j_thumb2', 'j_thumb3'
        ]

        self.q_limits = torch.tensor([
            [-0.349, 0.349], [-0.174, 1.570], [0.0, 1.745], [0.0, 1.745],
            [-0.523, 0.349], [-0.174, 1.570], [0.0, 1.745], [0.0, 1.745],
            [-0.698, 0.349], [-0.174, 1.570], [0.0, 1.745], [0.0, 1.745],
            [-0.523, 0.349], [-0.174, 1.570], [0.0, 1.745], [0.0, 1.745],
            [-0.174, 2.618], [-0.698, 0.698], [0.0, 1.745], [0.0, 1.745]
        ], dtype=torch.float32, device=device)

        self.flexion_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19]
        # 四指（不含拇指）用于一致性约束的 flexion 关节索引
        # joint_names 顺序见上面 self.joint_names
        self.four_finger_flex2 = torch.tensor([2, 6, 14, 10], dtype=torch.long, device=device)  # index2, middle2, ring2, pinky2
        self.four_finger_flex3 = torch.tensor([3, 7, 15, 11], dtype=torch.long, device=device)  # index3, middle3, ring3, pinky3

    def forward(self, handle_pc, handle_center, handle_out, handle_long, step):
        # 1. 前向运动学获取关节点
        joint_dict = {name: self.qpos[i].unsqueeze(0) for i, name in enumerate(self.joint_names)}
        ret = self.chain.forward_kinematics(joint_dict)
        rot_mat = quat_to_matrix_xyzw(self.palm_rot)
        palm_center_world = rot_mat @ self.palm_center_local + self.palm_pos
        palm_patch_world = rot_mat @ self.palm_patch_local + self.palm_pos

        tips = []
        for finger in ['index3', 'middle3', 'ring3', 'pinky3', 'thumb3']:
            tip_pos = rot_mat @ ret[finger].get_matrix()[0, :3, 3] + self.palm_pos
            tips.append(tip_pos)
        tips_tensor = torch.stack(tips)
        
        mids = []
        for finger in ['index2', 'middle2', 'ring2', 'pinky2', 'thumb2']:
            mid_pos = rot_mat @ ret[finger].get_matrix()[0, :3, 3] + self.palm_pos
            mids.append(mid_pos)
        mids_tensor = torch.stack(mids)
        
        pc_tensor = handle_pc.clone().detach().to(self.device)

        # 把手方向信息（确保归一化，防止出现负数爆炸）
        h_center = torch.tensor(handle_center, device=self.device, dtype=torch.float32)
        h_out = torch.tensor(handle_out, device=self.device, dtype=torch.float32)
        h_out = h_out / (torch.norm(h_out) + 1e-6)
        h_long = torch.tensor(handle_long, device=self.device, dtype=torch.float32)
        h_long = h_long / (torch.norm(h_long) + 1e-6)

        # ----------------------------------------------------
        # 动态权重：早期重点规范方向，后期重点强化包裹
        # ----------------------------------------------------
        alpha = min(1.0, step / 2000.0)

        # ----------------------------------------------------
        # 🌟 核心修复 1: 基于解剖学的垂直抓取约束
        # ----------------------------------------------------
        # A. 手掌宽度向量 (从食指中段到小拇指中段) -> 必须与把手平行
        vec_width = mids_tensor[0] - mids_tensor[3] 
        vec_width = vec_width / (torch.norm(vec_width) + 1e-6)
        # 点积的绝对值越接近1越平行，所以用 1 - abs(dot)
        loss_align_width = 1.0 - torch.clamp(torch.abs(torch.sum(vec_width * h_long)), max=1.0)

        # B. 手指出手方向 (从手掌根部到中指中段) -> 必须与把手垂直
        vec_forward = mids_tensor[1] - palm_patch_world
        vec_forward = vec_forward / (torch.norm(vec_forward) + 1e-6)
        # 垂直意味着点积为 0
        loss_align_forward = torch.abs(torch.sum(vec_forward * h_long))
        
        loss_align = loss_align_width + loss_align_forward

        # ----------------------------------------------------
        # 🌟 夹爪式抓握：四指与拇指分居把手两侧，避免深钩取
        # ----------------------------------------------------
        tips_4 = tips_tensor[:4]
        mids_4 = mids_tensor[:4]
        proj_tips = torch.sum((tips_4 - h_center) * h_out, dim=-1)
        proj_mids = torch.sum((mids_4 - h_center) * h_out, dim=-1)

        # ----------------------------------------------------
        # 🌟 新增：四指-拇指对向夹持（把手两侧）
        # ----------------------------------------------------
        # 用 handle_pc 在 h_out 方向上的投影范围估计把手厚度，并定义中分平面：
        # - 四指应在中平面“背后”(负向一侧)
        # - 拇指应在中平面“前面”(正向一侧)
        proj_pc = torch.sum((pc_tensor - h_center) * h_out, dim=-1)
        min_proj = torch.min(proj_pc)
        max_proj = torch.max(proj_pc)
        mid_proj = 0.5 * (min_proj + max_proj)

        thumb_tip = tips_tensor[4]
        proj_thumb = torch.sum((thumb_tip - h_center) * h_out, dim=-1)

        side_margin = 0.003  # 更贴近把手，两侧包络而不是悬停
        # 四指如果跑到中平面前侧 -> 惩罚
        loss_fingers_side = torch.sum(torch.relu(proj_tips - (mid_proj - side_margin)))
        # 拇指如果没到中平面前侧 -> 惩罚
        loss_thumb_side = torch.relu((mid_proj + side_margin) - proj_thumb)
        loss_opposition = loss_fingers_side + loss_thumb_side

        desired_back = mid_proj - 0.55 * (max_proj - min_proj + 1e-6)
        loss_back_surface = torch.mean(torch.abs(proj_tips - desired_back)) + 0.5 * torch.mean(
            torch.abs(proj_mids - desired_back)
        )

        deep_hook_margin = -0.001
        loss_anti_hook = torch.sum(torch.relu((min_proj - deep_hook_margin) - proj_tips))


        # A. 四指 flexion（第 2/3 关节）尽量一致
        flex2 = self.qpos[self.four_finger_flex2]
        flex3 = self.qpos[self.four_finger_flex3]
        loss_flex_consistency = torch.var(flex2) + torch.var(flex3)

        # B. 四指末端方向尽量一致（tip-mid 方向向量）
        finger_dirs = tips_tensor[:4] - mids_tensor[:4]
        finger_dirs = finger_dirs / (torch.norm(finger_dirs, dim=-1, keepdim=True) + 1e-6)
        mean_dir = torch.sum(finger_dirs, dim=0)
        mean_dir = mean_dir / (torch.norm(mean_dir) + 1e-6)
        loss_dir_consistency = torch.mean(1.0 - torch.sum(finger_dirs * mean_dir.unsqueeze(0), dim=-1))
        loss_consistency = loss_flex_consistency + loss_dir_consistency

        # ----------------------------------------------------
        # 🌟 新增：拇指指向修正（让 thumb3 指向四指聚集方向）
        # ----------------------------------------------------
        thumb_dir = tips_tensor[4] - mids_tensor[4]
        thumb_dir = thumb_dir / (torch.norm(thumb_dir) + 1e-6)
        target_thumb_dir = torch.mean(tips_tensor[:4], dim=0) - mids_tensor[4]
        target_thumb_dir = target_thumb_dir / (torch.norm(target_thumb_dir) + 1e-6)
        dot_thumb = torch.sum(thumb_dir * target_thumb_dir)
        dot_thumb = torch.clamp(dot_thumb, -1.0, 1.0)
        loss_thumb_dir = 1.0 - dot_thumb

        proj_long_4 = torch.sum((tips_4 - h_center) * h_long, dim=-1)
        proj_long_thumb = torch.sum((thumb_tip - h_center) * h_long, dim=-1)
        # Centre the four-finger GROUP on the handle, not each finger
        # individually.  Fingers are side-by-side along h_long, so
        # penalising each one's absolute offset from h_center biases
        # the optimizer to extend only the index finger to h_center
        # while the others (anatomically offset) curl into thin air.
        group_center_long = torch.mean(proj_long_4)
        loss_centerline = torch.abs(group_center_long) + 0.5 * torch.abs(proj_long_thumb)

        # ----------------------------------------------------
        # 🌟 新增：力闭合几何先验
        # ----------------------------------------------------
        # 1) 四指平均接触点与拇指接触点应跨过把手法向形成夹持宽度
        finger_pad_center = torch.mean(tips_4, dim=0)
        closure_vec = thumb_tip - finger_pad_center
        closure_dir = closure_vec / (torch.norm(closure_vec) + 1e-6)
        loss_force_axis = 1.0 - torch.clamp(torch.abs(torch.sum(closure_dir * h_out)), max=1.0)

        # 2) 夹持轴不应平行于把手长度；否则只是沿杆滑动，不利于传力
        loss_tangent_leverage = torch.abs(torch.sum(closure_dir * h_long))

        # 3) 拇指与四指应夹住同一段把手，而不是前后错位
        thumb_long = torch.sum((thumb_tip - h_center) * h_long)
        finger_long = torch.mean(torch.sum((tips_4 - h_center) * h_long, dim=-1))
        loss_same_section = torch.abs(thumb_long - finger_long)

        # 4) 夹持宽度要接近把手厚度，不要过大或过小
        closure_width = torch.abs(torch.sum(closure_vec * h_out))
        target_width = 0.90 * (max_proj - min_proj + 1e-6)
        loss_closure_width = torch.abs(closure_width - target_width)

        loss_force_closure = (
            1.2 * loss_force_axis
            + 0.8 * loss_tangent_leverage
            + 0.6 * loss_same_section
            + 0.8 * loss_closure_width
        )

        # ----------------------------------------------------
        # Normal-Assisted Projection Alignment: force middle/ring/pinky
        # to match the index finger's depth along h_out so they wrap
        # on the same plane instead of hovering rigidly.
        # ----------------------------------------------------
        proj_out_tips_4 = torch.sum((tips_4 - h_center) * h_out, dim=-1)    # (4,)
        proj_out_mids_4 = torch.sum((mids_4 - h_center) * h_out, dim=-1)    # (4,)
        ref_tip_depth = proj_out_tips_4[0]   # index finger tip as reference
        ref_mid_depth = proj_out_mids_4[0]   # index finger mid as reference
        loss_normal_align = (
            torch.sum((proj_out_tips_4[1:] - ref_tip_depth) ** 2)
            + 0.5 * torch.sum((proj_out_mids_4[1:] - ref_mid_depth) ** 2)
        )

        # 3. 贴合与防穿透损失
        dists_tips = torch.cdist(tips_tensor, pc_tensor)
        dists_mids = torch.cdist(mids_tensor, pc_tensor)
        loss_chamfer = torch.mean(torch.min(dists_tips, dim=1)[0]) + 0.3 * torch.mean(torch.min(dists_mids, dim=1)[0])

        contact_threshold = 0.015
        per_tip_min_dist = torch.min(dists_tips, dim=1)[0]          # (5,)
        per_mid_min_dist = torch.min(dists_mids, dim=1)[0]          # (5,)
        loss_tip_contact = torch.sum(torch.relu(per_tip_min_dist - contact_threshold))
        loss_mid_contact = torch.sum(torch.relu(per_mid_min_dist - contact_threshold * 2.0))
        loss_contact = loss_tip_contact + 0.5 * loss_mid_contact

        # Split penetration into palm (strict) vs fingers (relaxed).
        # Fingertips and mid-phalanges must be allowed to curl past the
        # handle back surface — a single flat plane blocks mid-transit
        # phalanges and causes the "OK gesture" local optimum where only
        # the index finger wraps.
        door_plane_strict = h_center + h_out * (min_proj - 0.005)   # palm: tight
        door_plane_finger = h_center + h_out * (min_proj - 0.025)   # fingers: generous

        palm_joints = torch.stack([palm_center_world, palm_patch_world])  # (2, 3)
        palm_signed = torch.sum((palm_joints - door_plane_strict) * h_out, dim=-1)
        loss_pen_palm = torch.sum(torch.relu(-palm_signed))

        finger_joints = torch.cat([mids_tensor, tips_tensor], dim=0)      # (10, 3)
        finger_signed = torch.sum((finger_joints - door_plane_finger) * h_out, dim=-1)
        loss_pen_finger = torch.sum(torch.relu(-finger_signed))

        loss_penetration = loss_pen_palm + 0.3 * loss_pen_finger

        # 4. 关节极限与闭合
        loss_limits = torch.sum(torch.relu(self.qpos - self.q_limits[:, 1])) + \
                      torch.sum(torch.relu(self.q_limits[:, 0] - self.qpos))
        q_flexion = self.qpos[self.flexion_indices]
        # 夹持时保持中等闭合，避免过度蜷成 hook
        closure_target = 0.65 + 0.10 * alpha
        loss_closure = torch.sum(torch.relu(closure_target - q_flexion))

        # 5. 锚定与旋转死锁
        # Anchor the palm patch just outside the front handle surface.
        # `h_center` is already the front-face center, so large offsets keep
        # the hand hovering in front of the handle and reproduce the ~3cm gap
        # seen in rollout logs.
        handle_thickness = max_proj - min_proj + 1e-6
        anchor_offset = torch.clamp(handle_thickness * 0.06, min=0.0015, max=0.0060)
        target_palm_pos = h_center + h_out * anchor_offset
        loss_anchor = torch.norm(palm_patch_world - target_palm_pos)
        norm_rot = self.palm_rot / (torch.norm(self.palm_rot) + 1e-6)
        init_rot = self.init_palm_rot / (torch.norm(self.init_palm_rot) + 1e-6)
        loss_rot = 1.0 - torch.clamp(torch.sum(norm_rot * init_rot)**2, max=1.0)

        weight_rot = 30.0 * (1.0 - alpha) + 2.0 * alpha       # 旋转死锁逐渐放开
        weight_anchor = 10.0 * (1.0 - alpha) + 2.0 * alpha    # 锚定限制逐渐变弱
        weight_back_surface = 120.0 * (1.0 - alpha) + 220.0 * alpha
        weight_anti_hook = 30.0 * (1.0 - alpha) + 60.0 * alpha
        weight_closure = 8.0 * (1.0 - alpha) + 40.0 * alpha
        weight_align = 150.0 * (1.0 - alpha) + 50.0 * alpha
        weight_consistency = 20.0 * (1.0 - alpha) + 60.0 * alpha
        weight_opposition = 80.0 * (1.0 - alpha) + 240.0 * alpha
        weight_thumb_dir = 20.0 * (1.0 - alpha) + 120.0 * alpha
        weight_centerline = 20.0 * (1.0 - alpha) + 70.0 * alpha
        weight_force_closure = 60.0 * (1.0 - alpha) + 180.0 * alpha
        # Contact: ramp up aggressively — early stages need fingers close,
        # late stages enforce tight surface contact
        weight_chamfer = 40.0 * (1.0 - alpha) + 120.0 * alpha
        weight_contact = 80.0 * (1.0 - alpha) + 300.0 * alpha
        weight_normal_align = 80.0 * (1.0 - alpha) + 350.0 * alpha

        total_loss = weight_chamfer * loss_chamfer + \
                     weight_contact * loss_contact + \
                     weight_normal_align * loss_normal_align + \
                     weight_back_surface * loss_back_surface + \
                     weight_anti_hook * loss_anti_hook + \
                     weight_opposition * loss_opposition + \
                     weight_force_closure * loss_force_closure + \
                     180.0 * loss_penetration + \
                     weight_closure * loss_closure + \
                     100.0 * loss_limits + \
                     weight_anchor * loss_anchor + \
                     weight_rot * loss_rot + \
                     weight_align * loss_align + \
                     weight_consistency * loss_consistency + \
                     weight_thumb_dir * loss_thumb_dir + \
                     weight_centerline * loss_centerline
        
        valid_quat = self.palm_rot / (torch.norm(self.palm_rot) + 1e-6)
        return total_loss, self.palm_pos, valid_quat, self.qpos
    
def run_optimization(mano_urdf_path, init_p, init_q, handle_pc, handle_center, handle_out, handle_long):
    print(" 开始基于 pinch + force-closure 几何先验的抓取优化...")
    model = ManoChamferSDFOptimizer(mano_urdf_path, init_p, init_q).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.005) 
    
    for step in range(3000): 
        optimizer.zero_grad()
        # 将 step 和 handle_long 传进模型
        loss, opt_pos, opt_rot, opt_qpos = model(handle_pc, handle_center, handle_out, handle_long, step)
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  Step {step:04d}, Loss: {loss.item():.4f}")
            
    print(" 优化完成！已生成更接近力闭合的初始抓取。")
    return opt_pos.detach().cpu().numpy(), opt_rot.detach().cpu().numpy(), opt_qpos.detach().cpu().numpy()

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
        vec_forward = mids_tensor[1] - self.palm_pos
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

        side_margin = 0.003  # 3mm：留出稳定接触的“夹持间隙”
        # 四指如果跑到中平面前侧 -> 惩罚
        loss_fingers_side = torch.sum(torch.relu(proj_tips - (mid_proj - side_margin)))
        # 拇指如果没到中平面前侧 -> 惩罚
        loss_thumb_side = torch.relu((mid_proj + side_margin) - proj_thumb)
        loss_opposition = loss_fingers_side + loss_thumb_side

        desired_back = mid_proj - 0.35 * (max_proj - min_proj + 1e-6)
        loss_back_surface = torch.mean(torch.abs(proj_tips - desired_back)) + 0.5 * torch.mean(
            torch.abs(proj_mids - desired_back)
        )

        deep_hook_margin = 0.006
        loss_anti_hook = torch.sum(torch.relu((min_proj - deep_hook_margin) - proj_tips))

        # ----------------------------------------------------
        # 🌟 新增：四指“同向/同形态”一致性（避免某根手指偏航去蹭点）
        # ----------------------------------------------------
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
        loss_centerline = torch.mean(torch.abs(proj_long_4)) + 0.5 * torch.abs(proj_long_thumb)

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
        target_width = 0.80 * (max_proj - min_proj + 1e-6)
        loss_closure_width = torch.abs(closure_width - target_width)

        loss_force_closure = (
            1.2 * loss_force_axis
            + 0.8 * loss_tangent_leverage
            + 0.6 * loss_same_section
            + 0.8 * loss_closure_width
        )

        # 3. 贴合与防穿透损失
        dists_tips = torch.cdist(tips_tensor, pc_tensor)
        dists_mids = torch.cdist(mids_tensor, pc_tensor)
        loss_chamfer = torch.mean(torch.min(dists_tips, dim=1)[0]) + 0.3 * torch.mean(torch.min(dists_mids, dim=1)[0])

        door_plane_point = h_center + h_out * (min_proj - 0.0015)
        all_joints = torch.cat([self.palm_pos.unsqueeze(0), mids_tensor, tips_tensor], dim=0)
        signed_dists = torch.sum((all_joints - door_plane_point) * h_out, dim=-1)
        loss_penetration = torch.sum(torch.relu(-signed_dists))

        # 4. 关节极限与闭合
        loss_limits = torch.sum(torch.relu(self.qpos - self.q_limits[:, 1])) + \
                      torch.sum(torch.relu(self.q_limits[:, 0] - self.qpos))
        q_flexion = self.qpos[self.flexion_indices]
        # 夹持时保持中等闭合，避免过度蜷成 hook
        closure_target = 0.65 + 0.10 * alpha
        loss_closure = torch.sum(torch.relu(closure_target - q_flexion))

        # 5. 锚定与旋转死锁
        target_palm_pos = h_center + h_out * 0.06
        loss_anchor = torch.norm(self.palm_pos - target_palm_pos)
        norm_rot = self.palm_rot / (torch.norm(self.palm_rot) + 1e-6)
        init_rot = self.init_palm_rot / (torch.norm(self.init_palm_rot) + 1e-6)
        loss_rot = 1.0 - torch.clamp(torch.sum(norm_rot * init_rot)**2, max=1.0)

        weight_rot = 30.0 * (1.0 - alpha) + 2.0 * alpha       # 旋转死锁逐渐放开
        weight_anchor = 10.0 * (1.0 - alpha) + 2.0 * alpha    # 锚定限制逐渐变弱
        weight_back_surface = 80.0 * (1.0 - alpha) + 160.0 * alpha
        weight_anti_hook = 150.0 * (1.0 - alpha) + 260.0 * alpha
        weight_closure = 8.0 * (1.0 - alpha) + 40.0 * alpha
        weight_align = 150.0 * (1.0 - alpha) + 50.0 * alpha
        weight_consistency = 20.0 * (1.0 - alpha) + 60.0 * alpha
        weight_opposition = 80.0 * (1.0 - alpha) + 240.0 * alpha
        weight_thumb_dir = 20.0 * (1.0 - alpha) + 120.0 * alpha
        weight_centerline = 20.0 * (1.0 - alpha) + 70.0 * alpha
        weight_force_closure = 60.0 * (1.0 - alpha) + 180.0 * alpha

        total_loss = 40.0 * loss_chamfer + \
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
    print("🚀 开始基于 pinch + force-closure 几何先验的抓取优化...")
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

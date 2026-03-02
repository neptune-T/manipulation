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
        # 🌟 核心修复 2: 四指包裹损失强化
        # ----------------------------------------------------
        # 逼迫四根手指 (食、中、无名、小指) 指尖深深探入把手后方至少 1cm 处
        tips_4 = tips_tensor[:4] 
        proj_tips = torch.sum((tips_4 - h_center) * h_out, dim=-1)
        # 要求 proj_tips 必须小于 -0.01 (深入内部)
        loss_wrap = torch.sum(torch.relu(proj_tips + 0.01)) 

        # 3. 贴合与防穿透损失
        dists_tips = torch.cdist(tips_tensor, pc_tensor)
        dists_mids = torch.cdist(mids_tensor, pc_tensor)
        loss_chamfer = torch.mean(torch.min(dists_tips, dim=1)[0]) + 0.3 * torch.mean(torch.min(dists_mids, dim=1)[0])

        proj_pc = torch.sum((pc_tensor - h_center) * h_out, dim=-1)
        min_proj = torch.min(proj_pc)
        door_plane_point = h_center + h_out * (min_proj - 0.005) 
        all_joints = torch.cat([self.palm_pos.unsqueeze(0), mids_tensor, tips_tensor], dim=0)
        signed_dists = torch.sum((all_joints - door_plane_point) * h_out, dim=-1)
        loss_penetration = torch.sum(torch.relu(-signed_dists))

        # 4. 关节极限与闭合
        loss_limits = torch.sum(torch.relu(self.qpos - self.q_limits[:, 1])) + \
                      torch.sum(torch.relu(self.q_limits[:, 0] - self.qpos))
        q_flexion = self.qpos[self.flexion_indices]
        loss_closure = torch.sum(torch.relu(0.8 - q_flexion))

        # 5. 锚定与旋转死锁
        target_palm_pos = h_center + h_out * 0.075 
        loss_anchor = torch.norm(self.palm_pos - target_palm_pos)
        norm_rot = self.palm_rot / (torch.norm(self.palm_rot) + 1e-6)
        init_rot = self.init_palm_rot / (torch.norm(self.init_palm_rot) + 1e-6)
        loss_rot = 1.0 - torch.clamp(torch.sum(norm_rot * init_rot)**2, max=1.0)

        # ----------------------------------------------------
        # 动态权重：早期重点规范方向，后期重点强化包裹
        # ----------------------------------------------------
        alpha = min(1.0, step / 2000.0) 

        weight_rot = 30.0 * (1.0 - alpha) + 2.0 * alpha       # 旋转死锁逐渐放开
        weight_anchor = 10.0 * (1.0 - alpha) + 2.0 * alpha    # 锚定限制逐渐变弱
        weight_wrap = 100.0 * (1.0 - alpha) + 500.0 * alpha   # 后期疯狂包裹
        weight_closure = 10.0 * (1.0 - alpha) + 80.0 * alpha  # 后期抓紧
        weight_align = 150.0 * (1.0 - alpha) + 50.0 * alpha   # 强硬的对齐态度

        total_loss = 40.0 * loss_chamfer + \
                     weight_wrap * loss_wrap + \
                     100.0 * loss_penetration + \
                     weight_closure * loss_closure + \
                     100.0 * loss_limits + \
                     weight_anchor * loss_anchor + \
                     weight_rot * loss_rot + \
                     weight_align * loss_align
        
        valid_quat = self.palm_rot / (torch.norm(self.palm_rot) + 1e-6)
        return total_loss, self.palm_pos, valid_quat, self.qpos
    
def run_optimization(mano_urdf_path, init_p, init_q, handle_pc, handle_center, handle_out, handle_long):
    print("🚀 开始基于 Wrap Loss 与动态权重的四指包裹抓取优化...")
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
            
    print(" 优化完成！手部已深度包裹把手。")
    return opt_pos.detach().cpu().numpy(), opt_rot.detach().cpu().numpy(), opt_qpos.detach().cpu().numpy()
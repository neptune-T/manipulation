import os
import torch
import pytorch_kinematics as pk
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R_scipy

def quat_to_matrix_xyzw(quat):
    """批量四元数转旋转矩阵: quat shape (B, 4) -> (B, 3, 3)"""
    q = quat / torch.norm(quat, dim=-1, keepdim=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mat = torch.stack([
        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
    ], dim=-1).reshape(-1, 3, 3)
    return mat

class FastContactCalculator:
    def __init__(self, mano_urdf_path, obj_urdf_path, device="cuda", points_per_link=500):
        """
        利用 GPU 和 PyTorch Kinematics 加速接触标签的计算。
        """
        self.device = device
        print(" 正在初始化 GPU 运动学链与表面点云库...")
        
        tree = ET.parse(obj_urdf_path)
        root = tree.getroot()
        
        # 1. 动态修复 GAPartNet URDF 缺失的 limit 属性
        for limit in root.iter('limit'):
            if 'effort' not in limit.attrib:
                limit.set('effort', '100.0')
            if 'velocity' not in limit.attrib:
                limit.set('velocity', '100.0')
                
        fixed_obj_urdf_bytes = ET.tostring(root)
        
        # 2. 构建双侧运动学链
        self.hand_chain = pk.build_chain_from_urdf(open(mano_urdf_path, "rb").read()).to(device=device)
        self.obj_chain = pk.build_chain_from_urdf(fixed_obj_urdf_bytes).to(device=device)
        
        # 3.  原生解析 XML 加载碰撞 Mesh，彻底解决路径丢失和版本兼容问题
        self.link_pcs = {}
        urdf_dir = os.path.dirname(obj_urdf_path)
        
        for link in root.iter('link'):
            link_name = link.get('name')
            link_points = []
            
            for collision in link.iter('collision'):
                geom = collision.find('geometry')
                if geom is not None:
                    mesh_tag = geom.find('mesh')
                    if mesh_tag is not None:
                        # 拼接出 mesh 的绝对物理路径
                        filename = mesh_tag.get('filename')
                        abs_path = os.path.join(urdf_dir, filename)
                        
                        if os.path.exists(abs_path):
                            # 加载 mesh
                            mesh = trimesh.load(abs_path, force='mesh')
                            # 处理多材质组成的 Scene 格式
                            if isinstance(mesh, trimesh.Scene):
                                geom_list = list(mesh.geometry.values())
                                if len(geom_list) > 0:
                                    mesh = trimesh.util.concatenate(geom_list)
                            
                            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                                # 采样点云
                                pts, _ = trimesh.sample.sample_surface(mesh, points_per_link)
                                
                                # 处理 collision 标签自带的 origin 偏移
                                origin = collision.find('origin')
                                if origin is not None:
                                    xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
                                    rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
                                    rot_mat = R_scipy.from_euler('xyz', rpy).as_matrix()
                                    pts = pts @ rot_mat.T + xyz
                                    
                                link_points.append(pts)
                                
            # 如果这个 link 存在任何碰撞模型，就合并并送入 GPU
            if len(link_points) > 0:
                combined_pts = np.concatenate(link_points, axis=0)
                self.link_pcs[link_name] = torch.tensor(combined_pts, dtype=torch.float32, device=device).unsqueeze(0)
                
        self.hand_joint_names = self.hand_chain.get_joint_parameter_names()
        self.obj_joint_names = self.obj_chain.get_joint_parameter_names()
        self.finger_joints = ['index3', 'middle3', 'ring3', 'pinky3', 'thumb3',
                              'index2', 'middle2', 'ring2', 'pinky2', 'thumb2']
        print(f" 初始化完成！提取了 {len(self.link_pcs)} 个活动部件。")

    def compute_batch_contact(self, hand_root_pos, hand_root_rot, hand_qpos, 
                                    obj_root_pos, obj_root_rot, obj_qpos, thresh=0.015):
        """
        批量计算手与物体的接触标签 (所有输入均为 Tensor，支持 batch_size = B)
        """
        B = hand_qpos.shape[0]
        
        hand_dict = {name: hand_qpos[:, i] for i, name in enumerate(self.hand_joint_names)}
        hand_ret = self.hand_chain.forward_kinematics(hand_dict)
        hand_rot_mat = quat_to_matrix_xyzw(hand_root_rot) 
        
        hand_joints_world = []
        for jname in self.finger_joints:
            local_pos = hand_ret[jname].get_matrix()[:, :3, 3]
            world_pos = torch.bmm(hand_rot_mat, local_pos.unsqueeze(-1)).squeeze(-1) + hand_root_pos
            hand_joints_world.append(world_pos)
        hand_joints_world = torch.stack(hand_joints_world, dim=1) 
        
        obj_dict = {name: obj_qpos[:, i] for i, name in enumerate(self.obj_joint_names)}
        obj_ret = self.obj_chain.forward_kinematics(obj_dict)
        obj_rot_mat = quat_to_matrix_xyzw(obj_root_rot)
        
        all_obj_points_world = []
        for link_name, local_points in self.link_pcs.items():
            if link_name in obj_ret:
                link_tf = obj_ret[link_name].get_matrix()
                link_rot = link_tf[:, :3, :3]
                link_trans = link_tf[:, :3, 3]
                points_obj = torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)
                points_world = torch.matmul(points_obj, obj_rot_mat.transpose(1, 2)) + obj_root_pos.unsqueeze(1)
                all_obj_points_world.append(points_world)
            
        merged_obj_points = torch.cat(all_obj_points_world, dim=1)
        
        dists = torch.cdist(hand_joints_world, merged_obj_points)
        min_dists, _ = torch.min(dists, dim=2)
        contact_labels = min_dists < thresh
        
        return contact_labels, min_dists
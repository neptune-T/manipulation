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
    def __init__(
        self,
        mano_urdf_path,
        obj_urdf_path,
        device="cuda",
        obj_scale=1.0,
        points_per_link=500,
        hand_points_per_link=200,
        hand_contact_links=None,
    ):
        """
        利用 GPU 和 PyTorch Kinematics 加速接触标签的计算。
        """
        self.device = device
        self.obj_scale = torch.tensor(float(obj_scale), dtype=torch.float32, device=device)
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
        self.link_normals = {}
        urdf_dir = os.path.dirname(obj_urdf_path)
        
        for link in root.iter('link'):
            link_name = link.get('name')
            link_points = []
            link_normals = []
            
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
                                # 修复法线方向: GAPartNet mesh 通常是非水密的,
                                # 法线朝向不一致会导致 SDF 符号翻转
                                try:
                                    trimesh.repair.fix_normals(mesh)
                                except Exception:
                                    pass
                                # 采样点云
                                pts, face_idx = trimesh.sample.sample_surface(mesh, points_per_link)
                                face_normals = mesh.face_normals[face_idx]

                                # 质心启发式: 确保法线朝外 (远离质心方向)
                                centroid = mesh.vertices.mean(axis=0)
                                to_centroid = centroid - pts
                                dots = np.sum(face_normals * to_centroid, axis=1)
                                # 如果大多数法线朝内 (>60%), 整体翻转
                                if np.mean(dots > 0) > 0.6:
                                    face_normals = -face_normals

                                # 处理 collision 标签自带的 origin 偏移
                                origin = collision.find('origin')
                                if origin is not None:
                                    xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
                                    rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
                                    rot_mat = R_scipy.from_euler('xyz', rpy).as_matrix()
                                    pts = pts @ rot_mat.T + xyz
                                    face_normals = face_normals @ rot_mat.T
                                    
                                link_points.append(pts)
                                link_normals.append(face_normals)

            # 如果这个 link 存在任何碰撞模型，就合并并送入 GPU
            if len(link_points) > 0:
                combined_pts = np.concatenate(link_points, axis=0)
                self.link_pcs[link_name] = torch.tensor(combined_pts, dtype=torch.float32, device=device).unsqueeze(0)
                combined_normals = np.concatenate(link_normals, axis=0)
                self.link_normals[link_name] = torch.tensor(combined_normals, dtype=torch.float32, device=device).unsqueeze(0)

        # 4. 同样为 MANO 手加载碰撞 mesh 点云（用于"手部顶点/表面->物体表面"的距离接触判定）
        self.hand_link_pcs = {}
        self.hand_link_order = []
        if hand_contact_links is None:
            hand_contact_links = [
                "palm",
                "index1x", "index2", "index3",
                "middle1x", "middle2", "middle3",
                "ring1x", "ring2", "ring3",
                "pinky1x", "pinky2", "pinky3",
                "thumb1z", "thumb2", "thumb3",
            ]
        self.hand_contact_links = set(hand_contact_links)

        try:
            hand_tree = ET.parse(mano_urdf_path)
            hand_root = hand_tree.getroot()
            hand_urdf_dir = os.path.dirname(mano_urdf_path)

            for link in hand_root.iter("link"):
                link_name = link.get("name")
                if link_name not in self.hand_contact_links:
                    continue

                link_points = []
                for collision in link.iter("collision"):
                    geom = collision.find("geometry")
                    if geom is None:
                        continue

                    mesh_tag = geom.find("mesh")
                    if mesh_tag is None:
                        continue

                    filename = mesh_tag.get("filename")
                    if filename is None:
                        continue

                    abs_path = os.path.join(hand_urdf_dir, filename)
                    if not os.path.exists(abs_path):
                        continue

                    mesh = trimesh.load(abs_path, force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        geom_list = list(mesh.geometry.values())
                        if len(geom_list) > 0:
                            mesh = trimesh.util.concatenate(geom_list)

                    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                        continue

                    pts, _ = trimesh.sample.sample_surface(mesh, hand_points_per_link)

                    origin = collision.find("origin")
                    if origin is not None:
                        xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
                        rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
                        rot_mat = R_scipy.from_euler("xyz", rpy).as_matrix()
                        pts = pts @ rot_mat.T + xyz

                    link_points.append(pts)

                if len(link_points) > 0:
                    combined_pts = np.concatenate(link_points, axis=0)
                    self.hand_link_pcs[link_name] = torch.tensor(combined_pts, dtype=torch.float32, device=device).unsqueeze(0)
                    self.hand_link_order.append(link_name)
        except Exception as e:
            print(f"[Warning] Failed to load MANO collision meshes for surface contact: {e}")

        self.hand_joint_names = self.hand_chain.get_joint_parameter_names()
        self.obj_joint_names = self.obj_chain.get_joint_parameter_names()
        self.finger_joints = ['index3', 'middle3', 'ring3', 'pinky3', 'thumb3',
                              'index2', 'middle2', 'ring2', 'pinky2', 'thumb2']
        print(f" 初始化完成！提取了 {len(self.link_pcs)} 个活动部件。")
        if len(self.hand_link_pcs) > 0:
            print(f" 手部接触点云已加载: {self.hand_link_order}")

    def _compute_hand_surface_points_world(
        self,
        hand_root_pos,
        hand_root_rot,
        hand_qpos,
        link_filter=None,
    ):
        """
        将 MANO 手碰撞点云变换到世界坐标系。
        """
        if len(self.hand_link_pcs) == 0:
            raise RuntimeError("No hand surface point clouds loaded.")

        B = hand_qpos.shape[0]
        if link_filter is None:
            link_filter = set(self.hand_link_order)
        else:
            link_filter = set(link_filter)

        hand_dict = {name: hand_qpos[:, i] for i, name in enumerate(self.hand_joint_names)}
        hand_ret = self.hand_chain.forward_kinematics(hand_dict)
        hand_rot_mat = quat_to_matrix_xyzw(hand_root_rot)

        hand_points_world = []
        link_slices = []
        cursor = 0
        for link_name in self.hand_link_order:
            if link_name not in link_filter:
                continue
            if link_name not in self.hand_link_pcs or link_name not in hand_ret:
                continue

            local_points = self.hand_link_pcs[link_name]
            link_tf = hand_ret[link_name].get_matrix()
            link_rot = link_tf[:, :3, :3]
            link_trans = link_tf[:, :3, 3]

            points_palm = torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)
            points_world = torch.matmul(points_palm, hand_rot_mat.transpose(1, 2)) + hand_root_pos.unsqueeze(1)
            hand_points_world.append(points_world)

            n = int(points_world.shape[1])
            link_slices.append((link_name, cursor, cursor + n))
            cursor += n

        if len(hand_points_world) == 0:
            raise RuntimeError("No hand points selected after filtering.")

        merged_hand_points = torch.cat(hand_points_world, dim=1)
        if merged_hand_points.shape[0] != B:
            raise RuntimeError(f"Unexpected batch size for hand points: {merged_hand_points.shape[0]} vs {B}")
        return merged_hand_points, link_slices

    def compute_hand_joint_positions_world(
        self,
        hand_root_pos,
        hand_root_rot,
        hand_qpos,
        joint_names=None,
    ):
        """
        Compute MANO joint/link reference positions in world coordinates.
        """
        if joint_names is None:
            joint_names = list(self.finger_joints)
        else:
            joint_names = list(joint_names)

        hand_dict = {name: hand_qpos[:, i] for i, name in enumerate(self.hand_joint_names)}
        hand_ret = self.hand_chain.forward_kinematics(hand_dict)
        hand_rot_mat = quat_to_matrix_xyzw(hand_root_rot)

        hand_joints_world = []
        valid_joint_names = []
        for jname in joint_names:
            if jname not in hand_ret:
                continue
            local_pos = hand_ret[jname].get_matrix()[:, :3, 3]
            world_pos = torch.bmm(hand_rot_mat, local_pos.unsqueeze(-1)).squeeze(-1) + hand_root_pos
            hand_joints_world.append(world_pos)
            valid_joint_names.append(jname)

        if len(hand_joints_world) == 0:
            raise RuntimeError("No valid MANO joints found for world-position query.")

        return torch.stack(hand_joints_world, dim=1), valid_joint_names

    def compute_object_surface_points_world(
        self,
        obj_root_pos,
        obj_root_rot,
        obj_qpos,
        obj_link_filter=None,
    ):
        """
        Compute object collision surface points in world coordinates.
        """
        obj_dict = {name: obj_qpos[:, i] for i, name in enumerate(self.obj_joint_names)}
        obj_ret = self.obj_chain.forward_kinematics(obj_dict)
        obj_rot_mat = quat_to_matrix_xyzw(obj_root_rot)

        if obj_link_filter is None:
            obj_link_filter = set(self.link_pcs.keys())
        else:
            obj_link_filter = set(obj_link_filter)

        all_obj_points_world = []
        link_slices = []
        cursor = 0
        for link_name, local_points in self.link_pcs.items():
            if link_name not in obj_link_filter:
                continue
            if link_name not in obj_ret:
                continue

            link_tf = obj_ret[link_name].get_matrix()
            link_rot = link_tf[:, :3, :3]
            link_trans = link_tf[:, :3, 3]
            points_obj = (torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)) * self.obj_scale
            points_world = torch.matmul(points_obj, obj_rot_mat.transpose(1, 2)) + obj_root_pos.unsqueeze(1)
            all_obj_points_world.append(points_world)

            n = int(points_world.shape[1])
            link_slices.append((link_name, cursor, cursor + n))
            cursor += n

        if len(all_obj_points_world) == 0:
            raise RuntimeError("No object surface points selected after filtering.")

        return torch.cat(all_obj_points_world, dim=1), link_slices

    def compute_batch_contact(self, hand_root_pos, hand_root_rot, hand_qpos, 
                                    obj_root_pos, obj_root_rot, obj_qpos, thresh=0.015):
        """
        批量计算手与物体的接触标签
        """
        B = hand_qpos.shape[0]

        hand_joints_world, _ = self.compute_hand_joint_positions_world(
            hand_root_pos, hand_root_rot, hand_qpos, joint_names=self.finger_joints
        )
        merged_obj_points, _ = self.compute_object_surface_points_world(
            obj_root_pos, obj_root_rot, obj_qpos
        )
        
        dists = torch.cdist(hand_joints_world, merged_obj_points)
        min_dists, _ = torch.min(dists, dim=2)
        contact_labels = min_dists < thresh
        
        return contact_labels, min_dists

    def compute_batch_surface_contact(
        self,
        hand_root_pos,
        hand_root_rot,
        hand_qpos,
        obj_root_pos,
        obj_root_rot,
        obj_qpos,
        thresh=0.015,
    ):
        """
        计算"手部表面点(近似顶点) -> 物体表面点"的距离接触。
        """
        B = hand_qpos.shape[0]

        if len(self.hand_link_pcs) == 0:
            contact_labels, min_dists = self.compute_batch_contact(
                hand_root_pos, hand_root_rot, hand_qpos, obj_root_pos, obj_root_rot, obj_qpos, thresh=thresh
            )
            link_counts = {"fingertips": contact_labels.to(torch.int32).sum(dim=1)}
            return contact_labels, min_dists, link_counts

        merged_hand_points, link_slices = self._compute_hand_surface_points_world(
            hand_root_pos, hand_root_rot, hand_qpos
        )

        obj_dict = {name: obj_qpos[:, i] for i, name in enumerate(self.obj_joint_names)}
        obj_ret = self.obj_chain.forward_kinematics(obj_dict)
        obj_rot_mat = quat_to_matrix_xyzw(obj_root_rot)

        all_obj_points_world = []
        for link_name, local_points in self.link_pcs.items():
            if link_name in obj_ret:
                link_tf = obj_ret[link_name].get_matrix()
                link_rot = link_tf[:, :3, :3]
                link_trans = link_tf[:, :3, 3]
                points_obj = (torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)) * self.obj_scale
                points_world = torch.matmul(points_obj, obj_rot_mat.transpose(1, 2)) + obj_root_pos.unsqueeze(1)
                all_obj_points_world.append(points_world)

        merged_obj_points = torch.cat(all_obj_points_world, dim=1)

        dists = torch.cdist(merged_hand_points, merged_obj_points)
        min_dists, _ = torch.min(dists, dim=2)
        contact_mask = min_dists < thresh

        link_contact_counts = {}
        for link_name, start, end in link_slices:
            link_contact_counts[link_name] = contact_mask[:, start:end].to(torch.int32).sum(dim=1)

        return contact_mask, min_dists, link_contact_counts

    def compute_batch_signed_distance(
        self,
        query_points_world,
        obj_root_pos,
        obj_root_rot,
        obj_qpos,
        obj_link_filter=None,
        max_penetration_depth=0.06,  # <--- 💡 修改点：放宽到了 0.06 (6cm)，绝不让 Agent 物理上突破免罚区
    ):
        """
        Approximate signed distance from query_points to object surface (B, N).
        """
        if query_points_world.dim() != 3 or query_points_world.shape[-1] != 3:
            raise ValueError(f"query_points_world must be (B, N, 3), got {tuple(query_points_world.shape)}")

        obj_dict = {name: obj_qpos[:, i] for i, name in enumerate(self.obj_joint_names)}
        obj_ret = self.obj_chain.forward_kinematics(obj_dict)
        obj_rot_mat = quat_to_matrix_xyzw(obj_root_rot)

        if obj_link_filter is None:
            obj_link_filter = set(self.link_pcs.keys())
        else:
            obj_link_filter = set(obj_link_filter)

        all_obj_points_world = []
        all_obj_normals_world = []
        for link_name, local_points in self.link_pcs.items():
            if link_name not in obj_link_filter:
                continue
            if link_name not in obj_ret or link_name not in self.link_normals:
                continue

            link_tf = obj_ret[link_name].get_matrix()
            link_rot = link_tf[:, :3, :3]
            link_trans = link_tf[:, :3, 3]

            local_normals = self.link_normals[link_name]
            points_obj = (torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)) * self.obj_scale
            normals_obj = torch.matmul(local_normals, link_rot.transpose(1, 2))

            points_world = torch.matmul(points_obj, obj_rot_mat.transpose(1, 2)) + obj_root_pos.unsqueeze(1)
            normals_world = torch.matmul(normals_obj, obj_rot_mat.transpose(1, 2))

            all_obj_points_world.append(points_world)
            all_obj_normals_world.append(normals_world)

        if len(all_obj_points_world) == 0:
            raise RuntimeError("No object surface points available for signed distance.")

        merged_obj_points = torch.cat(all_obj_points_world, dim=1)  # (B, M, 3)
        merged_obj_normals = torch.cat(all_obj_normals_world, dim=1)  # (B, M, 3)
        merged_obj_normals = merged_obj_normals / (torch.norm(merged_obj_normals, dim=-1, keepdim=True) + 1e-6)

        dists = torch.cdist(query_points_world, merged_obj_points)  # (B, N, M)
        min_dists, min_idx = torch.min(dists, dim=2)  # (B, N)

        idx_expand = min_idx.unsqueeze(-1).expand(-1, -1, 3)
        closest_points = torch.gather(merged_obj_points, 1, idx_expand)
        closest_normals = torch.gather(merged_obj_normals, 1, idx_expand)

        dot = torch.sum((query_points_world - closest_points) * closest_normals, dim=-1)
        sign = torch.where(dot >= 0.0, torch.ones_like(dot), -torch.ones_like(dot))

        # 安全阀: 无符号距离 > max_penetration_depth 的点不可能真的在物体内部
        clearly_outside = min_dists > float(max_penetration_depth)
        sign = torch.where(clearly_outside, torch.ones_like(sign), sign)

        signed_dists = sign * min_dists

        return signed_dists, min_dists, min_idx

    @staticmethod
    def contact_loss_signed_distance(
        signed_dists,
        interact_mask=None,
        lambda_r_interact=0.3,
        near_thresh=0.02,
        contact_thresh=0.0,
        eps=1e-6,
    ):
        """
        contact loss 形式： L = λ_R * L_R + (1 - λ_R) * L_A
        """
        if signed_dists.dim() != 2:
            raise ValueError(f"signed_dists must be (B, N), got {tuple(signed_dists.shape)}")

        B = signed_dists.shape[0]
        device = signed_dists.device
        dtype = signed_dists.dtype

        if interact_mask is None:
            interact_mask = torch.ones((B,), device=device, dtype=dtype)
        else:
            interact_mask = interact_mask.to(device=device, dtype=dtype).view(B)

        lambda_r_interact_t = torch.tensor(float(lambda_r_interact), device=device, dtype=dtype)
        lambda_r = torch.where(interact_mask > 0.5, lambda_r_interact_t, torch.ones_like(interact_mask))

        L_R = torch.mean(torch.relu(-signed_dists) ** 2, dim=1)  # (B,)

        near_mask = (signed_dists > float(contact_thresh)) & (signed_dists < float(near_thresh))
        near_count = near_mask.to(dtype=dtype).sum(dim=1)  # (B,)
        L_A_sum = torch.sum((near_mask.to(dtype=dtype) * signed_dists) ** 2, dim=1)
        L_A = L_A_sum / (near_count + eps)

        L = lambda_r * L_R + (1.0 - lambda_r) * L_A
        return torch.mean(L)

    def compute_batch_contact_loss(
        self,
        hand_root_pos,
        hand_root_rot,
        hand_qpos,
        obj_root_pos,
        obj_root_rot,
        obj_qpos,
        interact_mask=None,
        lambda_r_interact=0.3,
        near_thresh=0.02,
        contact_thresh=0.002,
        hand_link_filter=None,
        obj_link_filter=None,
    ):
        hand_pts, _ = self._compute_hand_surface_points_world(
            hand_root_pos, hand_root_rot, hand_qpos, link_filter=hand_link_filter
        )
        signed_dists, _, _ = self.compute_batch_signed_distance(
            hand_pts, obj_root_pos, obj_root_rot, obj_qpos, obj_link_filter=obj_link_filter
        )
        return self.contact_loss_signed_distance(
            signed_dists,
            interact_mask=interact_mask,
            lambda_r_interact=lambda_r_interact,
            near_thresh=near_thresh,
            contact_thresh=contact_thresh,
        )
from object_gym import ObjectGym
import numpy as np
from utils import read_yaml_config, prepare_gsam_model
import torch
import glob
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import sys
import tqdm
import os
from isaacgym import gymutil
from optimize_hoi import run_optimization
# from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert
import xml.etree.ElementTree as ET
import torch

def parse_joint_info(urdf_path, target_link_name):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    child_to_joint = {}
    for joint in root.findall('joint'):
        child = joint.find('child')
        if child is not None:
            child_to_joint[child.get('link')] = joint
            
    # 1. 找到负责驱动把手的真实活动关节
    active_joint = None
    search_link = target_link_name
    while search_link in child_to_joint:
        joint = child_to_joint[search_link]
        if joint.get('type') in ['revolute', 'prismatic', 'continuous']:
            active_joint = joint
            break
        search_link = joint.find('parent').get('link')
        
    if active_joint is None:
        return None, None, None
        
    # 2. 收集从 base 到 active_joint 的完整路径 (包含 fixed 关节)
    path = []
    curr = active_joint
    while curr is not None:
        path.append(curr)
        parent_link = curr.find('parent').get('link')
        curr = child_to_joint.get(parent_link)
    path.reverse() # 从 root 到 leaf
    
    # 3. 累加所有的 origin 变换，计算基坐标系下的真实 Axis 和 Origin
    current_rot = R.identity()
    current_pos = np.zeros(3)
    
    for joint in path:
        origin = joint.find('origin')
        xyz = np.array([float(x) for x in origin.get('xyz').split()]) if origin is not None and origin.get('xyz') else np.zeros(3)
        rpy = np.array([float(x) for x in origin.get('rpy').split()]) if origin is not None and origin.get('rpy') else np.zeros(3)
        
        # 当前层级的局部轴向 (只有到达 active_joint 时才提取 axis并返回)
        if joint == active_joint:
            axis = joint.find('axis')
            local_axis = np.array([float(x) for x in axis.get('xyz').split()]) if axis is not None and axis.get('xyz') else np.array([0, 0, 1])
            true_axis = current_rot.apply(local_axis)
            true_origin = current_pos + current_rot.apply(xyz)
            return joint.get('type'), true_origin, true_axis
            
        # 累加 fixed 关节的变换
        current_pos = current_pos + current_rot.apply(xyz)
        current_rot = current_rot * R.from_euler('xyz', rpy)
        
    return None, None, None

def generate_kinematic_trajectory(start_pose_6d, joint_type, world_origin, world_axis, open_amount=0.15, steps=100, **kwargs):
    """
    根据物理关节的类型和参数，生成手部的随动轨迹。
    修复：将位移和旋转合并成标准的 [N, 7] 轨迹数组，适配 Isaac Gym 接口。
    """
    combined_traj = []
    
    # 【自适应解包逻辑】
    start_arr = np.array(start_pose_6d)
    if start_arr.ndim == 1 and len(start_arr) == 7:
        hand_pos = start_arr[:3]
        hand_rot_quat = start_arr[3:7]
    else:
        hand_pos = np.array(start_pose_6d[0]).flatten()
        hand_rot_quat = np.array(start_pose_6d[1]).flatten()
    
    hand_rot = R.from_quat(hand_rot_quat)
    
    for i in range(steps + 1):
        fraction = i / float(steps)
        current_amount = fraction * open_amount
        
        if joint_type == 'prismatic':
            # 【抽屉】：手部位置沿着运动轴平移，手部姿态保持不变
            delta_pos = world_axis * current_amount
            new_pos = hand_pos + delta_pos
            new_rot = hand_rot
            
        elif joint_type in ['revolute', 'continuous']:
            # 【柜门】：手部绕着转轴中心 (origin) 旋转
            rot_vec = world_axis * current_amount
            delta_R = R.from_rotvec(rot_vec)
            
            vec_to_hand = hand_pos - world_origin
            new_pos = world_origin + delta_R.apply(vec_to_hand)
            new_rot = delta_R * hand_rot
        else:
            new_pos = hand_pos
            new_rot = hand_rot
        
        full_pose = np.concatenate([new_pos, new_rot.as_quat()])
        combined_traj.append(full_pose)
        
    return np.array(combined_traj)


def matrix_to_quaternion(rot_mats: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) -> quaternions (..., 4) in xyzw."""
    mats = rot_mats.detach().cpu().numpy()
    quat = R.from_matrix(mats).as_quat()  # xyzw
    return torch.tensor(quat, dtype=rot_mats.dtype, device=rot_mats.device)

def quaternion_invert(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Invert unit quaternions in xyzw."""
    out = quat_xyzw.clone()
    out[..., 0:3] *= -1.0
    return out

sys.path.append(sys.path[-1]+"/gym")
torch.set_printoptions(precision=4, sci_mode=False)

# load arguments
args = gymutil.parse_arguments(description="Placement",
    custom_parameters=[
        {"name": "--mode", "type": str, "default": ""},
        {"name": "--task_root", "type": str, "default": "output"},
        {"name": "--config", "type": str, "default": "config"},
        {"name": "--device", "type": str, "default": "cuda"},
        # headless
        {"name": "--headless", "action": 'store_true', "default": False},
        ])

def init_gym(cfgs, task_cfg=None):
    '''
    function: init gym
    input: cfgs, task_cfg
    '''
    # init gsam
    if cfgs["INFERENCE_GSAM"]:
        grounded_dino_model, sam_predictor = prepare_gsam_model(device=args.device)
    else:
        grounded_dino_model, sam_predictor = None, None
        
    # load selected object information (not important for articulated object manipulation)
    selected_obj_names = task_cfg["selected_obj_names"]
    selected_obj_urdfs=task_cfg["selected_urdfs"]
    selected_obj_num = len(selected_obj_names)
    selected_ob_poses = task_cfg["init_obj_pos"]
    selected_ob_pose_rs = [pose[3:] for pose in selected_ob_poses]
    save_root = task_cfg["save_root"]
    cfgs["asset"]["position_noise"] = [0,0,0]
    cfgs["asset"]["rotation_noise"] = 0
    cfgs["asset"]["asset_files"] = selected_obj_urdfs
    cfgs["asset"]["asset_seg_ids"] = [2 + i for i in range(selected_obj_num)]
    cfgs["asset"]["obj_pose_ps"] = selected_ob_poses
    cfgs["asset"]["obj_pose_rs"] = selected_ob_pose_rs

    # init gym
    gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
    
    # refresh observation and run steps to initialize the scene
    gym.refresh_observation(get_visual_obs=False)
    gym.run_steps(pre_steps = 10, refresh_obs=False, print_step=False)
    gym.refresh_observation(get_visual_obs=False)
    gym.save_root = save_root
    
    return gym, cfgs

if args.mode == "run_arti_free_control":
    '''
    function: init gym and run free control
    '''
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    
    # we choose one example object to show the demo, change the path 
    # to the object you want to show!
    paths = ["../partnet_mobility_part/45661/mobility_annotation_gapartnet.urdf"]
    for path in tqdm.tqdm(paths, total=len(paths)):
        gapart_id = path.split("/")[-2]
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        # gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        # task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"[Warning] ID {gapart_id} not found in gapartnet_obj_min_z.json. Using default Z-offset of -1.5.")
            gapartnet_obj_min_z_ = -1.5 
            
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])

        cfgs["HEADLESS"] = args.headless
        cfgs["USE_CUROBO"] = True
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [0.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        print(gym.save_root)
        gym.run_steps(pre_steps = 100, refresh_obs=False, print_step=False)
        
        ############################ change to desired pose ############################
        rotation = np.array([0, 1, 0, 0])
        position = np.array([0.2502,     -0.2000,     0.8517])
        move_pose = np.concatenate([position, rotation])
        ################################################################################
        
        step_num, traj = gym.control_to_pose(move_pose, close_gripper = True, save_video = False, save_root = None, step_num = 0)
        
        gym.clean_up()
        del gym     
        
elif args.mode == "run_arti_open":
    '''
    function: init gym and run open demo
    '''
    
    ROOT = "gapartnet_example"
    # read all paths
    # we choose one example object to show the demo, change the path
    paths = glob.glob(f"assets/{ROOT}/*/mobility_annotation_gapartnet.urdf")
    for path in tqdm.tqdm(paths, total=len(paths)):
        # get gapart id and anno
        gapart_id = path.split("/")[-2]
        gapart_anno_path = "/".join(path.split("/")[:-1]) + "/link_annotation_gapartnet.json"
        gapart_anno = json.load(open(gapart_anno_path, "r"))
        for link_anno in gapart_anno:
            if link_anno["is_gapart"] and link_anno["category"] == "slider_drawer":
                pass
        
        # cfg loading and init gym
        cfgs = read_yaml_config(f"{args.config}.yaml")
        task_root = args.task_root
        task_cfgs_path = "task_config.json"
        with open(task_cfgs_path, "r") as f: task_cfg = json.load(f)
        
        # load articualted object with the bottom at z = 0
        with open("gapartnet_obj_min_z.json", "r") as f: gapartnet_obj_min_z = json.load(f)
        if gapart_id in gapartnet_obj_min_z.keys():
            gapartnet_obj_min_z_ = gapartnet_obj_min_z[gapart_id]
        else:
            print(f"{gapart_id} not in gapartnet_obj_min_z")
            gapartnet_obj_min_z_ = -1.5
            
        # set the save root and other configurations
        task_cfg["save_root"] = "/".join(task_cfgs_path.split("/")[:-1])
        cfgs["HEADLESS"] = args.headless
        cfgs["USE_CUROBO"] = False
        cfgs["asset"]["arti_obj_root"] = ROOT
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        cfgs["asset"]["arti_obj_scale"] = 0.4
        cfgs["asset"]["arti_rotation"] = 0
        cfgs["asset"]["arti_gapartnet_ids"] = [
            gapart_id
        ]
        cfgs["asset"]["arti_obj_pose_ps"] = [
            [.8, 0, -0.4*gapartnet_obj_min_z_]
        ]
        
        # init gym
        gym, cfgs = init_gym(cfgs, task_cfg=task_cfg)

        # get the gapartnet annotation
        gym.get_gapartnet_anno()
        
        # render bbox for visualization and debug
        if not cfgs["HEADLESS"] and True:
            gym.gym.clear_lines(gym.viewer)
        for env_i in range(gym.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(gym.gapart_raw_valid_annos):
                
                all_bbox_now = gym.gapart_init_bboxes[gapart_obj_i]*cfgs["asset"]["arti_obj_scale"]
                
                rotation = R.from_quat(gym.arti_init_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(all_bbox_now, rotation_matrix.T)
                
               
                all_bbox_now = rotated_bbox_now + gym.arti_init_obj_pos_list[env_i]
                
                if not cfgs["HEADLESS"] and True:
                    idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                    for part_i in range(len(gapart_raw_valid_anno)):
                        bbox_now_i = all_bbox_now[part_i]
                        for i in range(len(idx_set)):
                            gym.gym.add_lines(gym.viewer, gym.envs[env_i], 1, 
                                np.concatenate((bbox_now_i[idx_set[i][0]], 
                                                bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                                np.array([1, 0 ,0], dtype=np.float32))
        
        
        # manipulate the object with the last part, change it for other objects
        # get the part bbox and calculate the handle direction
        bbox_id = -1
        
        # 将所有的 bbox 转换为 Tensor
        all_bbox_now_tensor = torch.tensor(all_bbox_now, dtype=torch.float32).to(gym.device).reshape(-1, 8, 3)
        all_bbox_center_front_face = torch.mean(all_bbox_now_tensor[:,0:4,:], dim = 1) 
        
        # 计算方向向量 (向外、长边、短边)
        handle_out = all_bbox_now_tensor[:,0,:] - all_bbox_now_tensor[:,4,:]
        handle_out /= torch.norm(handle_out, dim = 1, keepdim=True)
        handle_long = all_bbox_now_tensor[:,0,:] - all_bbox_now_tensor[:,1,:]
        handle_long /= torch.norm(handle_long, dim = 1, keepdim=True)
        handle_short = all_bbox_now_tensor[:,0,:] - all_bbox_now_tensor[:,3,:]
        handle_short /= torch.norm(handle_short, dim = 1, keepdim=True)
        
        # 计算初始的理想旋转 (使用原来夹爪的旋转作为先验)
        rotations = quaternion_invert(matrix_to_quaternion(torch.cat((handle_long.reshape((-1,1,3)), 
                        handle_short.reshape((-1,1,3)), -handle_out.reshape((-1,1,3))), dim = 1)))
        
        # 提取目标把手的核心数据
        init_position = all_bbox_center_front_face[bbox_id].cpu().numpy()
        handle_out_ = handle_out[bbox_id].cpu().numpy()
        handle_short_ = handle_short[bbox_id].cpu().numpy()
        init_rotation = rotations[bbox_id].cpu().numpy()
        
        bbox_min = all_bbox_now[bbox_id].min(axis=0)
        bbox_max = all_bbox_now[bbox_id].max(axis=0)
        dense_pc = np.random.uniform(bbox_min, bbox_max, size=(1000, 3))
        handle_point_cloud = torch.tensor(dense_pc, dtype=torch.float32, device=gym.device)
        
        print(f"🎯 锁定目标把手，中心位置: {init_position}")

        # ==========================================================
        # 🌟 核心修改区：调用 PyTorch 几何优化，替代 IK
        # ==========================================================
        
        # 1. 设定手部的初始姿态 (在把手正前方 10 厘米处，准备抓取)
        # pre_grasp_position = init_position + 0.10 * handle_out_
        pre_grasp_position = init_position + 0.06 * handle_out_
        mano_urdf_path = "../urdf/mano.urdf"
        
        opt_pos, opt_rot, opt_qpos = run_optimization(
            mano_urdf_path=mano_urdf_path,
            init_p=pre_grasp_position,
            init_q=init_rotation,
            handle_pc=handle_point_cloud,  
            handle_center=init_position,
            handle_out=handle_out_,
            handle_long=handle_long
        )
        
        # 3. 将优化得到的位姿打包
        # 优化得到的完美结果
        opt_pose_6d = np.concatenate([opt_pos, opt_rot])
        
        # 设定手部初始的“完全张开”状态
        open_qpos = np.zeros_like(opt_qpos)
        
        # ==========================================
        # 🎬 动作 3: 提取 URDF 运动学信息，生成完美圆弧轨迹并执行
        # ==========================================
        print("🎬 动作 3: 结合 URDF 运动学，沿物理轨迹拉开部件！")
        
        # 假设当前操作的是 link_0 的门 (你可以根据 bbox_id 动态映射对应的 link_name)
        # ==========================================
        # 自动匹配: 从 bbox_id 获取真实的 link_name 和运动类型
        # ==========================================
        # 假设当前场景只有 1 个可动部件物体 (索引为 0)
        target_link = gym.gapart_link_names[0][bbox_id] 
        urdf_file = path 
        
        print(f"🔍 正在解析部件: {target_link} ...")
        
        # 自动去 URDF 查这个部件是 "旋转" 还是 "平移"，以及它的轴(axis)和中心(origin)
        j_type, local_origin, local_axis = parse_joint_info(urdf_file, target_link)
        
        print(f"⚙️ 识别到关节类型: {j_type}, 运动轴: {local_axis}")
        
        j_type, local_origin, local_axis = parse_joint_info(urdf_file, target_link)
        
        if j_type is not None:
            # 获取物体在 Gym 中的缩放和平移信息
            obj_scale = cfgs["asset"]["arti_obj_scale"]
            obj_world_pos = gym.arti_init_obj_pos_list[0]
            obj_world_rot_quat = gym.arti_init_obj_rot_list[0]
            obj_rot_mat = R.from_quat(obj_world_rot_quat)
            
            # 将局部的 Origin 和 Axis 转换到 Isaac Gym 世界坐标系
            # World_Origin = Obj_Pos + Obj_Rot * (Local_Origin * Scale)
            world_origin = np.array(obj_world_pos) + obj_rot_mat.apply(local_origin * obj_scale)
            # 轴向量只受旋转影响
            world_axis = obj_rot_mat.apply(local_axis)
            
            # 设定要打开的角度或距离 (例如：开门 1.0 弧度，约 57 度)
            open_amount = 1.0 if j_type in ["revolute", "continuous"] else 0.15 
            
            # 生成随动的轨迹
            hand_traj = generate_kinematic_trajectory(
                joint_type=j_type,
                world_axis=world_axis,
                world_origin=world_origin,
                start_pose_6d=opt_pose_6d,
                amount=open_amount,
                steps=100
            )
            
            # 执行跟随轨迹控制
            # gym.follow_trajectory(traj_poses=hand_traj, target_qpos=opt_qpos)
            # 1. 执行轨迹并记录 (获得 60 帧的原始运动状态)
            records = gym.follow_trajectory_and_record(traj_poses=hand_traj, target_qpos=opt_qpos)
            
            # 2. 获取当前环境物体的 URDF 路径和绝对位置

            # 🌟 加上 gym.asset_root，拼接成完整的真实路径！
            obj_urdf_path = os.path.join(
                gym.asset_root, 
                gym.gapartnet_root, 
                str(gym.gapartnet_ids[0]), 
                "mobility_annotation_gapartnet.urdf"
            )
            
            obj_world_pos = gym.arti_init_obj_pos_list[0]
            obj_world_rot = gym.arti_init_obj_rot_list[0]
            
            # 3. 送入 GPU 结算接触，并保存为 Dataset
            save_file = f"output/dataset/grasp_record_{gym.gapartnet_ids[0]}.json"
            gym.process_and_save_dataset(
                records_list=records,
                obj_world_pos=obj_world_pos,
                obj_world_rot=obj_world_rot,
                obj_urdf_path=obj_urdf_path,
                save_path=save_file
            )
        else:
            print("⚠️ 未找到对应的关节信息，跳过动作 3。")
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
import torch
from fast_contact_calc import quat_to_matrix_xyzw
from single_door_rl_task import annotate_single_door_records, parse_joint_info, select_single_door_task


def has_revolute_gapart(urdf_path):
    """Check whether this object has at least one gapart link driven by revolute/continuous joint."""
    anno_path = os.path.join(os.path.dirname(urdf_path), "link_annotation_gapartnet.json")
    if not os.path.exists(anno_path):
        return False
    with open(anno_path, "r") as f:
        annos = json.load(f)

    for anno in annos:
        if not anno.get("is_gapart", False):
            continue
        link_name = anno.get("link_name")
        if link_name is None:
            continue
        j_type, _, _, _, _, _ = parse_joint_info(urdf_path, link_name)
        if j_type in ["revolute", "continuous"]:
            return True
    return False


def _build_handle_point_cloud_from_collision_mesh(
    gym,
    obj_urdf_path,
    link_name,
    handle_center,
    handle_out,
    num_points=1500,
    points_per_link=2500,
    backside_margin=None,
):
    """
    尝试从目标 link 的 collision mesh 采样表面点云（世界坐标）。
    并可选过滤出把手“背后”区域（沿 handle_out 的负方向）。
    """
    try:
        gym._ensure_contact_calc(
            obj_urdf_path=obj_urdf_path,
            points_per_link=max(int(points_per_link), int(num_points)),
            hand_points_per_link=200,
        )
        if not hasattr(gym, "contact_calc"):
            return None

        cc = gym.contact_calc
        if link_name not in cc.link_pcs:
            return None

        # object dof (B=1)
        obj_qpos = gym.dof_pos[
            0,
            gym.mano_num_dofs : gym.mano_num_dofs + gym.arti_obj_num_dofs,
            0,
        ].unsqueeze(0)
        obj_dict = {name: obj_qpos[:, i] for i, name in enumerate(cc.obj_joint_names)}
        obj_ret = cc.obj_chain.forward_kinematics(obj_dict)
        if link_name not in obj_ret:
            return None

        link_tf = obj_ret[link_name].get_matrix()
        link_rot = link_tf[:, :3, :3]
        link_trans = link_tf[:, :3, 3]
        local_points = cc.link_pcs[link_name]
        points_obj = (torch.matmul(local_points, link_rot.transpose(1, 2)) + link_trans.unsqueeze(1)) * cc.obj_scale

        o_pos = torch.tensor(gym.arti_init_obj_pos_list[0], dtype=torch.float32, device=gym.device).unsqueeze(0)
        o_rot = torch.tensor(gym.arti_init_obj_rot_list[0], dtype=torch.float32, device=gym.device).unsqueeze(0)
        obj_rot_mat = quat_to_matrix_xyzw(o_rot)
        points_world = torch.matmul(points_obj, obj_rot_mat.transpose(1, 2)) + o_pos.unsqueeze(1)
        points_world = points_world.squeeze(0)

        # 可选过滤：取把手背后的 patch（四指插入区域）
        if backside_margin is not None:
            center_t = torch.tensor(handle_center, dtype=torch.float32, device=points_world.device)
            out_t = torch.tensor(handle_out, dtype=torch.float32, device=points_world.device)
            out_t = out_t / (torch.norm(out_t) + 1e-6)
            proj = torch.sum((points_world - center_t) * out_t, dim=-1)
            mask_back = proj < -float(backside_margin)
            if mask_back.any():
                back_points = points_world[mask_back]
                # 只有当背后点数足够时才替换，避免空洞/误判导致点云过稀
                if back_points.shape[0] >= max(50, num_points // 4):
                    points_world = back_points

        # 采样到固定数量（可重复采样）
        if points_world.shape[0] <= 0:
            return None
        if points_world.shape[0] >= num_points:
            idx = torch.randperm(points_world.shape[0], device=points_world.device)[:num_points]
            points_world = points_world[idx]
        else:
            idx = torch.randint(0, points_world.shape[0], (num_points,), device=points_world.device)
            points_world = points_world[idx]

        return points_world
    except Exception as e:
        print(f"⚠️ collision handle_pc 构建失败: {e}")
        return None

def generate_kinematic_trajectory(start_pose_6d, joint_type, world_origin, world_axis, open_amount=0.15, steps=100, **kwargs):
    """
    根据物理关节的类型和参数，生成手部的随动轨迹。
    修复：将位移和旋转合并成标准的 [N, 7] 轨迹数组，适配 Isaac Gym 接口。
    """
    combined_traj = []

    # 兼容旧调用：amount -> open_amount
    if "amount" in kwargs and kwargs["amount"] is not None:
        open_amount = kwargs["amount"]

    # 保证旋转/平移轴为单位向量，避免幅度被 axis 长度污染
    world_axis = np.asarray(world_axis, dtype=np.float64)
    axis_norm = np.linalg.norm(world_axis)
    if axis_norm < 1e-8:
        raise ValueError(f"Invalid world_axis: {world_axis}")
    world_axis = world_axis / axis_norm
    
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
    paths = ["../partnet_mobility_part/9117/mobility_annotation_gapartnet.urdf"]
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
    preferred_gapart_id = "11304"
    preferred_path = f"assets/{ROOT}/{preferred_gapart_id}/mobility_annotation_gapartnet.urdf"
    if os.path.exists(preferred_path):
        paths = [preferred_path]
        print(f"🎯 优先测试指定对象: {preferred_gapart_id}")
    else:
        revolute_paths = [p for p in paths if has_revolute_gapart(p)]
        if len(revolute_paths) > 0:
            paths = revolute_paths
            print(f"🔁 已切换到旋转任务集合: {len(paths)} / {len(glob.glob(f'assets/{ROOT}/*/mobility_annotation_gapartnet.urdf'))}")
        else:
            print("⚠️ 未筛到旋转对象，回退到原始对象集合。")
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
        # 旋转把手抓取需要更小的 contact_offset / thickness，避免“空气层”太厚导致无法插入把手背后
        if has_revolute_gapart(path):
            cfgs["asset"]["arti_shape_contact_offset"] = 0.005
            cfgs["asset"]["arti_shape_thickness"] = 0.02
            # 旋转任务需要更强的摩擦与手指夹持力，才能把门“拖动”起来
            cfgs["asset"]["mano_shape_contact_offset"] = 0.005
            cfgs["asset"]["mano_shape_thickness"] = 0.02
            cfgs["asset"]["mano_shape_friction"] = 5.0
            cfgs["asset"]["mano_dof_stiffness"] = 400.0
            cfgs["asset"]["mano_dof_damping"] = 30.0
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
        
        
        task_spec = select_single_door_task(
            asset_dir=os.path.dirname(path),
            door_index=0,
        )
        bbox_id = int(task_spec.handle_bbox_index)
        print(
            f"✅ 单门任务已锁定: door={task_spec.door_link_name}, "
            f"handle={task_spec.handle_link_name}, joint={task_spec.joint_name}, bbox_id={bbox_id}"
        )
        
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
        handle_long_ = handle_long[bbox_id].cpu().numpy()
        init_rotation = rotations[bbox_id].cpu().numpy()

        # ------------------------------------------
        # handle_pc: 旋转任务优先用 collision mesh 表面点云
        # ------------------------------------------
        obj_urdf_path = os.path.join(
            gym.asset_root,
            gym.gapartnet_root,
            str(gym.gapartnet_ids[0]),
            "mobility_annotation_gapartnet.urdf",
        )
        handle_link_name = task_spec.handle_link_name
        handle_joint_type = task_spec.joint_type
        print(f"🧩 用 collision mesh 采样 handle_pc, link={handle_link_name}, joint_type={handle_joint_type}")
        handle_point_cloud = _build_handle_point_cloud_from_collision_mesh(
            gym=gym,
            obj_urdf_path=obj_urdf_path,
            link_name=handle_link_name,
            handle_center=init_position,
            handle_out=handle_out_,
            num_points=1500,
            points_per_link=2500,
            backside_margin=None,
        )
        if handle_point_cloud is None:
            print(f"🧩 handle_pc 回退到 bbox 采样, link={handle_link_name}, joint_type={handle_joint_type}")
            # fallback：平移任务保持原逻辑；旋转任务尽量取“背后”区域
            bbox_min = all_bbox_now[bbox_id].min(axis=0)
            bbox_max = all_bbox_now[bbox_id].max(axis=0)
            dense_pc = np.random.uniform(bbox_min, bbox_max, size=(1000, 3))
            handle_point_cloud = torch.tensor(dense_pc, dtype=torch.float32, device=gym.device)
        print(f"🧩 handle_pc 点数: {int(handle_point_cloud.shape[0])}")
        
        print(f"🎯 锁定目标把手，中心位置: {init_position}")

        # ==========================================================
        # 🌟 核心修改区：调用 PyTorch 几何优化，替代 IK
        # ==========================================================
        
        # 1. 设定手部的初始姿态 (在把手正前方 10 厘米处，准备抓取)
        # pre_grasp_position = init_position + 0.10 * handle_out_
        pre_grasp_position = init_position + 0.06 * handle_out_
        mano_urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "urdf", "mano.urdf"))
        
        opt_pos, opt_rot, opt_qpos = run_optimization(
            mano_urdf_path=mano_urdf_path,
            init_p=pre_grasp_position,
            init_q=init_rotation,
            handle_pc=handle_point_cloud,  
            handle_center=init_position,
            handle_out=handle_out_,
            handle_long=handle_long_
        )
        
        # 3. 将优化得到的位姿打包
        # 优化得到的完美结果
        opt_pose_6d = np.concatenate([opt_pos, opt_rot])

        # ==========================================
        # 🤝 动作 2.5: 基于“手部表面点->物体表面点”距离的接触稳定
        # ==========================================
        # 在拉动前先让手指闭合并结算若干帧；若接触不足则沿 -handle_out_ 方向微推入把手
        # obj_urdf_path 已在 handle_pc 构建时生成，这里复用
        opt_pose_6d, grasp_stable, grasp_info = gym.stabilize_grasp_by_surface_contact(
            start_pose_6d=opt_pose_6d,
            target_qpos=opt_qpos,
            approach_dir=-handle_out_,
            obj_urdf_path=obj_urdf_path,
            surface_contact_thresh=0.015,
            min_contact_points=60,
            required_contact_links=["index3", "middle3", "ring3", "pinky3", "thumb3"],
            min_points_per_link=3,
            settle_steps=8,
            max_iters=12,
            push_step=0.002,
        )
        print(f"🤝 稳定接触: {grasp_stable}, info={grasp_info}")

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
        target_link = task_spec.handle_link_name
        target_cate = task_spec.handle_category

        print(f"🔍 正在解析部件: {target_link} ({target_cate}) ...")

        j_type = task_spec.joint_type
        local_origin = task_spec.joint_origin_local
        local_axis = task_spec.joint_axis_local
        joint_name = task_spec.joint_name
        joint_lower = task_spec.joint_lower
        joint_upper = task_spec.joint_upper
        print(f"⚙️ 识别到关节类型: {j_type}, 关节名: {joint_name}, 运动轴: {local_axis}, limit=({joint_lower}, {joint_upper})")
        
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
            
            obj_dof_index = gym.arti_obj_dof_dict.get(joint_name, None) if joint_name is not None else None

            # 设定要打开的角度或距离
            # 平移分支保持你原来的 0.15，不动；
            # 旋转分支改为基于关节当前值到极限的增量。
            open_amount = 1.0 if j_type in ["revolute", "continuous"] else 0.15
            if j_type in ["revolute", "continuous"] and obj_dof_index is not None:
                if joint_lower is not None and joint_upper is not None:
                    current_obj_dof = gym.dof_pos[0, gym.mano_num_dofs + obj_dof_index, 0].item()
                    delta_upper = float(joint_upper - current_obj_dof)
                    delta_lower = float(joint_lower - current_obj_dof)
                    open_amount = delta_upper if abs(delta_upper) >= abs(delta_lower) else delta_lower
                elif joint_upper is not None:
                    current_obj_dof = gym.dof_pos[0, gym.mano_num_dofs + obj_dof_index, 0].item()
                    open_amount = float(joint_upper - current_obj_dof)
                open_amount = float(np.clip(open_amount, -1.2, 1.2))

            # 按你的要求：旋转也采用和平移一致的开环手部轨迹规划，
            # 不依赖 obj_dof 反馈，避免“物体不先动 -> 手不继续动”的停滞。
            hand_traj = generate_kinematic_trajectory(
                joint_type=j_type,
                world_axis=world_axis,
                world_origin=world_origin,
                start_pose_6d=opt_pose_6d,
                open_amount=open_amount,
                steps=100
            )

            drive_dof_delta_thresh = 0.01 if j_type == "prismatic" else 0.05
            records = gym.follow_trajectory_and_record(
                traj_poses=hand_traj,
                target_qpos=opt_qpos,
                record_surface_contact=True,
                surface_contact_thresh=0.015,
                min_contact_points=60,
                required_contact_links=["index3", "middle3", "ring3", "pinky3", "thumb3"],
                min_points_per_link=3,
                drive_dof_index=obj_dof_index,
                drive_dof_delta_thresh=drive_dof_delta_thresh,
                set_root_velocities=j_type in ["revolute", "continuous"],
            )
            if len(records) > 0:
                records[0]["grasp_stable_init"] = bool(grasp_stable)
                records[0]["grasp_info_init"] = grasp_info

            if obj_dof_index is not None and len(records) > 1:
                try:
                    init_dof = float(records[0]["obj_dof"][obj_dof_index])
                    final_dof = float(records[-1]["obj_dof"][obj_dof_index])
                    delta_dof = final_dof - init_dof
                    stable_frames = [r.get("surface_contact_stable", False) for r in records]
                    stable_ratio = float(sum(bool(x) for x in stable_frames)) / float(len(stable_frames))
                    print(
                        f"📈 运动结果: dof[{obj_dof_index}] {init_dof:.4f} -> {final_dof:.4f} (Δ={delta_dof:.4f}), "
                        f"stable_ratio={stable_ratio:.2f}"
                    )
                except Exception:
                    pass
            
            # 2. 获取当前环境物体的 URDF 路径和绝对位置

            # 🌟 加上 gym.asset_root，拼接成完整的真实路径！
            # obj_urdf_path 已在稳定接触阶段构建，这里复用
            
            obj_world_pos = gym.arti_init_obj_pos_list[0]
            obj_world_rot = gym.arti_init_obj_rot_list[0]
            
            # 3. 送入 GPU 结算接触，并保存为 Dataset
            phase_summary = annotate_single_door_records(
                records_list=records,
                task_spec=task_spec,
                obj_world_pos=obj_world_pos,
                obj_world_rot=obj_world_rot,
                obj_scale=obj_scale,
                contact_target_points=6,
                min_contact_points=30,
            )
            print(f"🧠 单门 phase 标注: {phase_summary}")

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

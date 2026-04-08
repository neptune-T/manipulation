from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
import torch

try:
    from object_gym import ObjectGym
    from optimize_hoi import run_optimization
    from utils import prepare_gsam_model, read_yaml_config
    from fast_contact_calc import quat_to_matrix_xyzw
    from single_door_rl_task import (
        CONTACT_LINK_ORDER,
        SingleDoorRewardConfig,
        SingleDoorTaskSpec,
        build_single_door_observation,
        build_contact_feature_vector,
        compute_single_door_reward,
        extract_single_door_runtime_state,
        get_phase_contact_target,
        select_articulation_task,
        select_single_door_task,
    )
except ImportError:
    from manipulation.object_gym import ObjectGym
    from manipulation.optimize_hoi import run_optimization
    from manipulation.utils import prepare_gsam_model, read_yaml_config
    from manipulation.fast_contact_calc import quat_to_matrix_xyzw
    from manipulation.single_door_rl_task import (
        CONTACT_LINK_ORDER,
        SingleDoorRewardConfig,
        SingleDoorTaskSpec,
        build_single_door_observation,
        build_contact_feature_vector,
        compute_single_door_reward,
        extract_single_door_runtime_state,
        get_phase_contact_target,
        select_articulation_task,
        select_single_door_task,
    )


MANO_Q_LIMITS = np.asarray(
    [
        [-0.349, 0.349],
        [-0.174, 1.570],
        [0.0, 1.745],
        [0.0, 1.745],
        [-0.523, 0.349],
        [-0.174, 1.570],
        [0.0, 1.745],
        [0.0, 1.745],
        [-0.698, 0.349],
        [-0.174, 1.570],
        [0.0, 1.745],
        [0.0, 1.745],
        [-0.523, 0.349],
        [-0.174, 1.570],
        [0.0, 1.745],
        [0.0, 1.745],
        [-0.174, 2.618],
        [-0.698, 0.698],
        [0.0, 1.745],
        [0.0, 1.745],
    ],
    dtype=np.float32,
)

# Power grasp presets: C-shaped hand with high MCP/PIP flexion for full wrap.
# Joint order per finger: [abduction, MCP_flexion, PIP_flexion, DIP_flexion]
# Thumb order: [j_thumb1y, j_thumb1z, j_thumb2, j_thumb3]
PINCH_PRESET_QPOS = np.asarray(
    [
        # index:  abd,   MCP,  PIP,  DIP  — deep wrap
        0.00,  0.95, 1.10, 0.50,
        # middle
        0.00,  1.05, 1.20, 0.50,
        # pinky
        0.00,  0.90, 1.00, 0.40,
        # ring
        0.00,  1.00, 1.10, 0.50,
        # thumb — opposed, wrapping inward
        1.30,  0.30, 0.70, 0.40,
    ],
    dtype=np.float32,
)

PINCH_ACTUATE_QPOS = np.asarray(
    [
        0.00,  1.05, 1.20, 0.60,
        0.00,  1.15, 1.30, 0.60,
        0.00,  1.00, 1.10, 0.50,
        0.00,  1.10, 1.20, 0.60,
        1.40,  0.35, 0.80, 0.50,
    ],
    dtype=np.float32,
)

# Relaxed C-shape for initial approach — fingers moderately curled, ready to wrap
PINCH_TOUCH_QPOS = np.asarray(
    [
        # index
        0.00,  0.55, 0.65, 0.30,
        # middle
        0.00,  0.60, 0.70, 0.30,
        # pinky
        0.00,  0.50, 0.55, 0.20,
        # ring
        0.00,  0.55, 0.65, 0.28,
        # thumb
        1.00,  0.15, 0.40, 0.20,
    ],
    dtype=np.float32,
)


def _normalize(vec: Sequence[float], fallback: Optional[Sequence[float]] = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        return arr / norm
    if fallback is None:
        return np.zeros_like(arr)
    fb = np.asarray(fallback, dtype=np.float32)
    fb_norm = float(np.linalg.norm(fb))
    if fb_norm > 1e-8:
        return fb / fb_norm
    return np.zeros_like(arr)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def _matrix_to_quaternion_xyzw(rot_mat: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot_mat).as_quat().astype(np.float32)


def _build_handle_point_cloud_from_collision_mesh(
    gym,
    obj_urdf_path: str,
    link_name: str,
    handle_center: Sequence[float],
    handle_out: Sequence[float],
    num_points: int = 1500,
    points_per_link: int = 2500,
    backside_margin: Optional[float] = None,
):
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

        obj_qpos = gym.dof_pos[
            0,
            gym.mano_num_dofs : gym.mano_num_dofs + gym.arti_obj_num_dofs,
            0,
        ].unsqueeze(0)
        obj_dict = {name: obj_qpos[:, index] for index, name in enumerate(cc.obj_joint_names)}
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

        if backside_margin is not None:
            center_t = torch.tensor(handle_center, dtype=torch.float32, device=points_world.device)
            out_t = torch.tensor(handle_out, dtype=torch.float32, device=points_world.device)
            out_t = out_t / (torch.norm(out_t) + 1e-6)
            proj = torch.sum((points_world - center_t) * out_t, dim=-1)
            mask_back = proj < -float(backside_margin)
            if mask_back.any():
                back_points = points_world[mask_back]
                if back_points.shape[0] >= max(50, num_points // 4):
                    points_world = back_points

        if points_world.shape[0] <= 0:
            return None
        if points_world.shape[0] >= num_points:
            idx = torch.randperm(points_world.shape[0], device=points_world.device)[:num_points]
            points_world = points_world[idx]
        else:
            idx = torch.randint(0, points_world.shape[0], (num_points,), device=points_world.device)
            points_world = points_world[idx]
        return points_world
    except Exception as exc:
        print(f"⚠️ collision handle_pc 构建失败: {exc}")
        return None


def generate_kinematic_trajectory(
    start_pose_6d: Sequence[float],
    joint_type: str,
    world_origin: Sequence[float],
    world_axis: Sequence[float],
    open_amount: float = 0.15,
    steps: int = 100,
) -> np.ndarray:
    combined_traj = []
    world_axis_arr = _normalize(world_axis, fallback=[0.0, 0.0, 1.0]).astype(np.float64)
    start_arr = np.asarray(start_pose_6d, dtype=np.float64).reshape(-1)
    hand_pos = start_arr[:3]
    hand_rot = R.from_quat(start_arr[3:7])
    world_origin_arr = np.asarray(world_origin, dtype=np.float64)

    for frame in range(int(steps) + 1):
        fraction = frame / float(max(steps, 1))
        current_amount = fraction * float(open_amount)
        if joint_type == "prismatic":
            new_pos = hand_pos + world_axis_arr * current_amount
            new_rot = hand_rot
        elif joint_type in ["revolute", "continuous"]:
            delta_rot = R.from_rotvec(world_axis_arr * current_amount)
            vec_to_hand = hand_pos - world_origin_arr
            new_pos = world_origin_arr + delta_rot.apply(vec_to_hand)
            new_rot = delta_rot * hand_rot
        else:
            new_pos = hand_pos
            new_rot = hand_rot
        combined_traj.append(np.concatenate([new_pos, new_rot.as_quat()], axis=0).astype(np.float32))
    return np.asarray(combined_traj, dtype=np.float32)


@dataclass
class SingleDoorResidualConfig:
    asset_dir: str
    config_path: str = "config.yaml"
    headless: bool = True
    device: str = "cuda"
    num_envs: int = 1
    task_index: int = 0
    door_index: int = 0
    reset_phase: str = "grasp"
    max_episode_steps: int = 120
    action_repeat: int = 2
    settle_steps: int = 6
    pregrasp_offset: float = 0.01
    trajectory_steps: int = 100
    max_open_amount: float = 1.2
    pos_action_scale: float = 0.015
    rot_action_scale: float = 0.06
    synergy_action_scale: float = 0.35
    use_optimized_grasp: bool = True
    stabilize_grasp: bool = True
    use_demo_base_pose: bool = True
    contact_target_points: int = 6
    min_contact_points: int = 30
    reward_config: SingleDoorRewardConfig = field(default_factory=SingleDoorRewardConfig)
    save_root: str = "output/single_door_rl"
    reset_pose_noise: float = 0.0
    reset_rot_noise: float = 0.0
    stabilize_on_reset: bool = True
    curriculum_enabled: bool = True
    curriculum_levels: Tuple[str, ...] = ("grasp", "actuate", "open")
    curriculum_success_threshold: float = 0.8
    curriculum_window: int = 20
    pinch_search_enabled: bool = True
    pinch_search_lateral_step: float = 0.008
    pinch_search_depth_step: float = 0.003
    pinch_search_lateral_trials: int = 5
    pinch_search_yaw_deg: float = 18.0
    pinch_search_roll_deg: float = 12.0
    wrist_rotation_lock: float = 0.85
    grasp_settle_steps: int = 6
    grasp_settle_synergy_scale: float = 0.65
    max_safe_penetration: float = -0.003
    door_plane_buffer: float = 0.003
    palm_safe_buffer: float = 0.005
    handle_passthrough: bool = False


class SingleDoorResidualEnv:
    action_dim = 11

    def __init__(self, config: SingleDoorResidualConfig):
        self.config = config
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.asset_dir = os.path.abspath(config.asset_dir)
        selected_task_index = int(config.task_index if config.task_index is not None else config.door_index)
        self.task_spec: SingleDoorTaskSpec = select_articulation_task(self.asset_dir, task_index=selected_task_index)
        self.asset_root, self.arti_obj_root, self.gapart_id = self._resolve_asset_paths(self.asset_dir)
        self.gym, self.cfgs = self._init_gym()
        self.reward_config = self.config.reward_config
        self.reward_config.success_progress = float(self.task_spec.success_progress)
        self.obj_scale = float(self.cfgs["asset"]["arti_obj_scale"])
        self.drive_dof_index = self.gym.arti_obj_dof_dict.get(self.task_spec.joint_name, None)
        self.object_lower = np.asarray(self.gym.arti_obj_dof_props["lower"], dtype=np.float32).copy()
        self.object_upper = np.asarray(self.gym.arti_obj_dof_props["upper"], dtype=np.float32).copy()
        self.synergy_matrix = self._build_synergy_matrix()
        self.open_qpos = np.zeros(20, dtype=np.float32)
        self.closed_qpos = self._clip_hand_qpos(self.synergy_to_qpos(np.array([0.6, 0.9, 0.8, 0.6, 0.0], dtype=np.float32)))
        self.pregrasp_pose = np.zeros(7, dtype=np.float32)
        self.anchor_pose = np.zeros(7, dtype=np.float32)
        self.anchor_qpos = self.closed_qpos.copy()
        self.demo_traj = np.zeros((1, 7), dtype=np.float32)
        self.demo_info: Dict[str, Any] = {}
        self.last_pose_target = np.zeros(7, dtype=np.float32)
        self.last_qpos_target = self.closed_qpos.copy()
        self.prev_state = None
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self.observation_dim = None
        self.curriculum_level = 0
        self.curriculum_history = []
        self.grasp_settle_counter = 0
        self._prepare_teacher()

    def _resolve_asset_paths(self, asset_dir: str) -> Tuple[str, str, str]:
        asset_dir_abs = os.path.abspath(asset_dir)
        manipulation_assets_root = os.path.join(self.repo_root, "manipulation", "assets")
        if asset_dir_abs.startswith(manipulation_assets_root + os.sep):
            asset_root = manipulation_assets_root
            arti_obj_root = os.path.relpath(os.path.dirname(asset_dir_abs), asset_root)
        elif asset_dir_abs.startswith(self.repo_root + os.sep):
            asset_root = self.repo_root
            arti_obj_root = os.path.relpath(os.path.dirname(asset_dir_abs), asset_root)
        else:
            asset_root = os.path.dirname(os.path.dirname(asset_dir_abs))
            arti_obj_root = os.path.basename(os.path.dirname(asset_dir_abs))
        gapart_id = os.path.basename(asset_dir_abs)
        return asset_root, arti_obj_root, gapart_id

    def _build_task_cfg(self) -> Dict[str, Any]:
        return {
            "selected_obj_names": [],
            "selected_urdfs": [],
            "init_obj_pos": [],
            "save_root": self.config.save_root,
        }

    def _compute_object_spawn_height(self, obj_scale: float) -> float:
        bbox_path = os.path.join(self.asset_dir, "bounding_box.json")
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, "r") as handle:
                    bbox_data = json.load(handle)
                min_corner = bbox_data.get("min", None)
                if min_corner is not None and len(min_corner) >= 3:
                    min_z = float(min_corner[2])
                    return float(-float(obj_scale) * min_z)
            except Exception as exc:
                print(f"⚠️ 读取 bounding_box 失败，回退默认高度: {exc}")
        return 0.0

    def _init_gym(self):
        cfg_path = self.config.config_path
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
        cfgs = read_yaml_config(cfg_path)
        task_cfg = self._build_task_cfg()

        if bool(cfgs.get("INFERENCE_GSAM", False)):
            grounded_dino_model, sam_predictor = prepare_gsam_model(device=self.config.device)
        else:
            grounded_dino_model, sam_predictor = None, None

        cfgs["HEADLESS"] = bool(self.config.headless)
        cfgs["USE_CUROBO"] = False
        cfgs["USE_ARTI"] = True
        cfgs["num_envs"] = int(self.config.num_envs)
        cfgs["asset"]["asset_root"] = self.asset_root
        cfgs["asset"]["arti_obj_root"] = self.arti_obj_root
        cfgs["asset"]["arti_gapartnet_ids"] = [self.gapart_id]
        cfgs["asset"]["asset_files"] = []
        cfgs["asset"]["asset_seg_ids"] = []
        cfgs["asset"]["obj_pose_ps"] = []
        cfgs["asset"]["obj_pose_rs"] = []
        cfgs["asset"]["position_noise"] = [0, 0, 0]
        cfgs["asset"]["rotation_noise"] = 0
        cfgs["asset"]["arti_position_noise"] = 0.0
        cfgs["asset"]["arti_rotation_noise"] = 0.0
        spawn_height = self._compute_object_spawn_height(cfgs["asset"]["arti_obj_scale"])
        cfgs["asset"]["arti_obj_pose_ps"] = [[0.8, 0.0, spawn_height]]
        cfgs["SAVE_ROOT"] = self.config.save_root
        cfgs["asset"]["handle_passthrough"] = bool(self.config.handle_passthrough)

        if self.task_spec.joint_type in ["revolute", "continuous"]:
            cfgs["asset"]["arti_shape_contact_offset"] = 0.005
            cfgs["asset"]["arti_shape_thickness"] = 0.02
            # Slim down MANO collision bodies to prevent ghost-overlap with the
            # door panel.  Smaller contact_offset + thickness gives the fingers
            # enough physical clearance to wrap around the handle without the
            # physics engine treating a near-miss as a solid contact.
            cfgs["asset"]["mano_shape_contact_offset"] = 0.002
            cfgs["asset"]["mano_shape_thickness"] = 0.005
            cfgs["asset"]["mano_shape_friction"] = 5.0
            cfgs["asset"]["mano_dof_stiffness"] = 400.0
            cfgs["asset"]["mano_dof_damping"] = 30.0

        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            gym = ObjectGym(cfgs, grounded_dino_model, sam_predictor)
        finally:
            sys.argv = argv_backup
        gym.refresh_observation(get_visual_obs=False)
        gym.run_steps(pre_steps=10, refresh_obs=False, print_step=False)
        gym.refresh_observation(get_visual_obs=False)
        gym.save_root = task_cfg["save_root"]
        gym.get_gapartnet_anno()
        return gym, cfgs

    def _build_synergy_matrix(self) -> np.ndarray:
        matrix = np.zeros((20, 5), dtype=np.float32)
        # 0: 四指整体朝向把手闭合（近端+中段）
        matrix[[1, 5, 9, 13], 0] = 0.30
        matrix[[2, 6, 10, 14], 0] = 0.60
        matrix[[3, 7, 11, 15], 0] = 0.20

        # 1: 四指远端进一步夹紧，形成类似平行夹爪内侧面
        matrix[[2, 6, 10, 14], 1] = 0.25
        matrix[[3, 7, 11, 15], 1] = 0.75

        # 2: 拇指对向闭合
        matrix[16, 2] = 0.65
        matrix[17, 2] = 0.35
        matrix[18, 2] = 0.60
        matrix[19, 2] = 0.75

        # 3: 拇指根部摆位，帮助从另一侧卡住把手
        matrix[16, 3] = 0.55
        matrix[17, 3] = -0.50
        matrix[18, 3] = 0.15

        # 4: 四指 MCP 横向并拢，形成统一“夹爪面”
        matrix[0, 4] = 0.12
        matrix[4, 4] = 0.04
        matrix[12, 4] = -0.04
        matrix[8, 4] = -0.12
        return matrix

    def zero_action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def random_action(self, scale: float = 1.0) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32) * float(scale)

    def synergy_to_qpos(self, synergy: Sequence[float], base_qpos: Optional[Sequence[float]] = None) -> np.ndarray:
        synergy_arr = np.asarray(synergy, dtype=np.float32).reshape(5)
        base = self.open_qpos if base_qpos is None else np.asarray(base_qpos, dtype=np.float32).reshape(20)
        qpos = base + self.synergy_matrix @ synergy_arr
        return self._clip_hand_qpos(qpos)

    def _clip_hand_qpos(self, qpos: Sequence[float]) -> np.ndarray:
        qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(20)
        return np.clip(qpos_arr, MANO_Q_LIMITS[:, 0], MANO_Q_LIMITS[:, 1]).astype(np.float32)

    def _build_pinch_qpos_targets(self, anchor_qpos: Sequence[float]) -> Dict[str, np.ndarray]:
        anchor = self._clip_hand_qpos(anchor_qpos)
        parallel_touch = self.synergy_to_qpos([0.45, 0.15, 0.42, 0.30, 0.10], base_qpos=self.open_qpos)
        parallel_grasp = self.synergy_to_qpos([0.72, 0.42, 0.72, 0.52, 0.18], base_qpos=self.open_qpos)
        parallel_actuate = self.synergy_to_qpos([0.88, 0.66, 0.92, 0.64, 0.22], base_qpos=self.open_qpos)
        pinch_touch = self._clip_hand_qpos(0.20 * anchor + 0.50 * PINCH_TOUCH_QPOS + 0.30 * parallel_touch)
        pinch_grasp = self._clip_hand_qpos(0.20 * anchor + 0.35 * PINCH_PRESET_QPOS + 0.45 * parallel_grasp)
        pinch_actuate = self._clip_hand_qpos(0.15 * anchor + 0.30 * PINCH_ACTUATE_QPOS + 0.55 * parallel_actuate)
        pinch_open = self._clip_hand_qpos(0.55 * pinch_grasp + 0.45 * pinch_actuate)
        return {
            "approach": self.open_qpos.copy(),
            "touch": pinch_touch.astype(np.float32),
            "grasp": pinch_grasp.astype(np.float32),
            "actuate": pinch_actuate.astype(np.float32),
            "open": pinch_open.astype(np.float32),
        }

    def _score_pinch_contact(self, link_counts: Dict[str, int], contact_count: int) -> float:
        palm = float(link_counts.get("palm", 0))
        index2 = float(link_counts.get("index2", 0))
        middle2 = float(link_counts.get("middle2", 0))
        ring2 = float(link_counts.get("ring2", 0))
        index1x = float(link_counts.get("index1x", 0))
        middle1x = float(link_counts.get("middle1x", 0))
        thumb2 = float(link_counts.get("thumb2", 0))
        # Power grasp: palm and mid-phalanges wrapping the handle are most important
        return float(
            0.35 * contact_count
            + 8.0 * palm
            + 5.0 * (index2 + middle2)
            + 3.0 * (ring2 + thumb2)
            + 2.0 * (index1x + middle1x)
        )

    def _search_pinch_grasp(
        self,
        start_pose: Sequence[float],
        target_qpos: Sequence[float],
        world_geom: Dict[str, np.ndarray],
        obj_urdf_path: str,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        base_pose = np.asarray(start_pose, dtype=np.float32).copy()
        lateral_step = float(self.config.pinch_search_lateral_step)
        depth_step = float(self.config.pinch_search_depth_step)
        trial_count = int(max(1, self.config.pinch_search_lateral_trials))
        offsets = np.linspace(-(trial_count // 2), trial_count // 2, trial_count, dtype=np.float32)
        lateral_offsets = offsets * lateral_step
        depth_offsets = np.asarray([0.0, depth_step], dtype=np.float32)
        yaw_deg = float(self.config.pinch_search_yaw_deg)
        roll_deg = float(self.config.pinch_search_roll_deg)
        orientation_candidates = [(0.0, 0.0), (-yaw_deg, 0.0), (yaw_deg, 0.0), (0.0, -roll_deg), (0.0, roll_deg)]

        best_pose = base_pose.copy()
        best_info: Dict[str, Any] = {"stage": "pinch_search"}
        best_score = -1e9
        best_stable = False

        base_rot = R.from_quat(base_pose[3:7])
        handle_frame = np.stack(
            [
                world_geom["handle_long_world"],
                world_geom["handle_short_world"],
                -world_geom["handle_out_world"],
            ],
            axis=1,
        )

        for yaw_offset_deg, roll_offset_deg in orientation_candidates:
            local_delta = R.from_euler("xyz", [0.0, roll_offset_deg, yaw_offset_deg], degrees=True)
            world_delta = R.from_matrix(handle_frame) * local_delta * R.from_matrix(handle_frame).inv()
            rotated_quat = (world_delta * base_rot).as_quat().astype(np.float32)
            for lateral in lateral_offsets:
                for depth in depth_offsets:
                    candidate_pose = base_pose.copy()
                    candidate_pose[:3] += lateral * world_geom["handle_short_world"]
                    candidate_pose[:3] -= depth * world_geom["handle_out_world"]
                    candidate_pose[3:7] = rotated_quat
                    pose, stable, info = self.gym.stabilize_grasp_by_surface_contact(
                        start_pose_6d=candidate_pose,
                        target_qpos=target_qpos,
                        approach_dir=-world_geom["handle_out_world"],
                        obj_urdf_path=obj_urdf_path,
                        surface_contact_thresh=0.020,
                        min_contact_points=20,
                        required_contact_links=["palm", "index2"],
                        min_points_per_link=2,
                        settle_steps=10,
                        max_iters=10,
                        push_step=depth_step,
                    )
                    score = self._score_pinch_contact(info.get("link_counts", {}), int(info.get("contact_count", 0)))
                    min_dist = float(info.get("min_dist", 0.0))
                    if min_dist < float(self.config.max_safe_penetration):
                        score -= 200.0 * abs(min_dist - float(self.config.max_safe_penetration))
                    if stable:
                        score += 1000.0
                    if score > best_score:
                        best_score = score
                        best_pose = np.asarray(pose, dtype=np.float32).copy()
                        best_stable = bool(stable)
                        best_info = dict(info)
                        best_info["lateral_offset"] = float(lateral)
                        best_info["depth_offset"] = float(depth)
                        best_info["yaw_offset_deg"] = float(yaw_offset_deg)
                        best_info["roll_offset_deg"] = float(roll_offset_deg)
                        best_info["pinch_score"] = float(score)
                    if stable:
                        return best_pose, True, best_info
        return best_pose, best_stable, best_info

    def _sample_bbox_handle_pc(self, world_geom: Dict[str, np.ndarray], num_points: int = 1500) -> torch.Tensor:
        bbox_local = np.asarray(self.task_spec.handle_bbox_local, dtype=np.float32)
        bbox_min = bbox_local.min(axis=0)
        bbox_max = bbox_local.max(axis=0)
        samples_local = np.random.uniform(bbox_min, bbox_max, size=(num_points, 3)).astype(np.float32)
        obj_world_pos = np.asarray(self.gym.arti_init_obj_pos_list[0], dtype=np.float32)
        obj_world_rot = R.from_quat(np.asarray(self.gym.arti_init_obj_rot_list[0], dtype=np.float32))
        samples_world = obj_world_pos + obj_world_rot.apply(samples_local * self.obj_scale)
        return torch.tensor(samples_world, dtype=torch.float32, device=self.gym.device)

    def _estimate_initial_rotation(self, world_geom: Dict[str, np.ndarray]) -> np.ndarray:
        rot_mat = np.stack(
            [
                world_geom["handle_long_world"],
                -world_geom["handle_short_world"],
                world_geom["handle_out_world"],
            ],
            axis=1,
        )
        return R.from_matrix(rot_mat).inv().as_quat().astype(np.float32)

    def _current_world_geom(self) -> Dict[str, np.ndarray]:
        return self.task_spec.world_geometry(
            obj_world_pos=self.gym.arti_init_obj_pos_list[0],
            obj_world_rot=self.gym.arti_init_obj_rot_list[0],
            obj_scale=self.obj_scale,
        )

    def get_handle_geometry_diagnostics(self) -> Dict[str, Any]:
        local_extent = np.asarray(self.task_spec.handle_extent_local, dtype=np.float32) * float(self.obj_scale)
        world_geom = self._current_world_geom()
        diag = {
            "handle_extent_world": _to_jsonable(local_extent),
            "handle_depth_world": float(local_extent[0]),
            "handle_length_world": float(local_extent[1]),
            "handle_width_world": float(local_extent[2]),
            "handle_center_world": _to_jsonable(world_geom["handle_center_world"]),
            "handle_front_center_world": _to_jsonable(world_geom["handle_front_center_world"]),
            "handle_out_world": _to_jsonable(world_geom["handle_out_world"]),
            "handle_long_world": _to_jsonable(world_geom["handle_long_world"]),
            "hinge_origin_world": _to_jsonable(world_geom["hinge_origin_world"]),
            "hinge_axis_world": _to_jsonable(world_geom["hinge_axis_world"]),
        }
        if self.npcs_handle_loc is not None:
            diag["npcs_handle_link"] = self.npcs_handle_loc.handle_link_name
            diag["npcs_handle_category"] = self.npcs_handle_loc.handle_category
            diag["npcs_handle_num_points"] = int(self.npcs_handle_loc.handle_points_world.shape[0])
            diag["npcs_handle_center"] = _to_jsonable(self.npcs_handle_loc.handle_center.cpu().numpy())
            diag["npcs_handle_long_axis"] = _to_jsonable(self.npcs_handle_loc.handle_long_axis.cpu().numpy())
            diag["npcs_handle_normal"] = _to_jsonable(self.npcs_handle_loc.handle_normal.cpu().numpy())
        return diag

    def _compute_open_amount(self) -> float:
        open_amount = 1.0 if self.task_spec.joint_type in ["revolute", "continuous"] else 0.15
        if self.drive_dof_index is not None:
            current_obj_dof = self.gym.dof_pos[0, self.gym.mano_num_dofs + self.drive_dof_index, 0].item()
            joint_lower = self.task_spec.joint_lower
            joint_upper = self.task_spec.joint_upper
            if joint_lower is not None and joint_upper is not None:
                delta_upper = float(joint_upper - current_obj_dof)
                delta_lower = float(joint_lower - current_obj_dof)
                open_amount = delta_upper if abs(delta_upper) >= abs(delta_lower) else delta_lower
            elif joint_upper is not None:
                open_amount = float(joint_upper - current_obj_dof)
        return float(np.clip(open_amount, -self.config.max_open_amount, self.config.max_open_amount))

    def _prepare_teacher(self) -> None:
        world_geom = self._current_world_geom()
        init_rotation = self._estimate_initial_rotation(world_geom)
        pregrasp_position = world_geom["handle_front_center_world"] + self.config.pregrasp_offset * world_geom["handle_out_world"]
        self.pregrasp_pose = np.concatenate([pregrasp_position, init_rotation], axis=0).astype(np.float32)

        mano_urdf_path = os.path.join(self.repo_root, "urdf", "mano.urdf")
        obj_urdf_path = self.task_spec.urdf_path

        # --- Collision-mesh handle PC (primary — actual mesh surface) ---
        handle_point_cloud = None
        self.npcs_handle_loc = None
        handle_point_cloud = _build_handle_point_cloud_from_collision_mesh(
            gym=self.gym,
            obj_urdf_path=obj_urdf_path,
            link_name=self.task_spec.handle_link_name,
            handle_center=world_geom["handle_front_center_world"],
            handle_out=world_geom["handle_out_world"],
            num_points=1500,
            points_per_link=2500,
            backside_margin=None,
        )
        if handle_point_cloud is not None:
            print(f"[HandlePC] Collision mesh: {handle_point_cloud.shape[0]} pts")

        # --- NPCS-based fallback (bbox surface approximation) ---
        if handle_point_cloud is None:
            try:
                from npcs_handle_localization import localize_handle_from_annotations
                npcs_loc = localize_handle_from_annotations(
                    asset_dir=self.asset_dir,
                    target_handle_link=self.task_spec.handle_link_name,
                    num_points=1500,
                    device=self.config.device,
                )
                if npcs_loc is not None and npcs_loc.handle_points_world.shape[0] > 0:
                    obj_pos_t = torch.tensor(
                        self.gym.arti_init_obj_pos_list[0], dtype=torch.float32,
                        device=self.gym.device,
                    )
                    obj_rot_t = torch.tensor(
                        self.gym.arti_init_obj_rot_list[0], dtype=torch.float32,
                        device=self.gym.device,
                    )
                    from fast_contact_calc import quat_to_matrix_xyzw
                    obj_rot_mat = quat_to_matrix_xyzw(obj_rot_t.unsqueeze(0))
                    local_pts = npcs_loc.handle_points_world.to(self.gym.device) * self.obj_scale
                    handle_point_cloud = (local_pts @ obj_rot_mat.squeeze(0).T + obj_pos_t).detach()
                    self.npcs_handle_loc = npcs_loc
                    print(f"[HandlePC] NPCS fallback: {handle_point_cloud.shape[0]} pts, "
                          f"link={npcs_loc.handle_link_name}, cat={npcs_loc.handle_category}")
            except Exception as exc:
                print(f"[HandlePC] NPCS localization unavailable: {exc}")

        # --- Bbox sampling last resort ---
        if handle_point_cloud is None:
            handle_point_cloud = self._sample_bbox_handle_pc(world_geom)
            print(f"[HandlePC] Bbox fallback: {handle_point_cloud.shape[0]} pts")

        if self.config.use_optimized_grasp:
            opt_pos, opt_rot, opt_qpos = run_optimization(
                mano_urdf_path=mano_urdf_path,
                init_p=pregrasp_position,
                init_q=init_rotation,
                handle_pc=handle_point_cloud,
                handle_center=world_geom["handle_front_center_world"],
                handle_out=world_geom["handle_out_world"],
                handle_long=world_geom["handle_long_world"],
            )
            anchor_qpos = self.closed_qpos.copy()  # Discard pinch opt_qpos; keep power grasp preset
            anchor_pose = np.concatenate([opt_pos, opt_rot], axis=0).astype(np.float32)
        else:
            anchor_qpos = self.closed_qpos.copy()
            anchor_pose = self.pregrasp_pose.copy()

        stable = False
        grasp_info: Dict[str, Any] = {}
        if self.config.stabilize_grasp and self.config.pinch_search_enabled:
            anchor_pose, stable, grasp_info = self._search_pinch_grasp(
                start_pose=anchor_pose,
                target_qpos=anchor_qpos,
                world_geom=world_geom,
                obj_urdf_path=obj_urdf_path,
            )
        elif self.config.stabilize_grasp:
            anchor_pose, stable, grasp_info = self.gym.stabilize_grasp_by_surface_contact(
                start_pose_6d=anchor_pose,
                target_qpos=anchor_qpos,
                approach_dir=-world_geom["handle_out_world"],
                obj_urdf_path=obj_urdf_path,
                surface_contact_thresh=0.015,
                min_contact_points=60,
                required_contact_links=["index3", "middle3", "ring3", "pinky3", "thumb3"],
                min_points_per_link=3,
                settle_steps=8,
                max_iters=12,
                push_step=0.002,
            )

        self.anchor_pose = np.asarray(anchor_pose, dtype=np.float32)
        self.anchor_qpos = np.asarray(anchor_qpos, dtype=np.float32)
        self.phase_qpos_targets = self._build_pinch_qpos_targets(self.anchor_qpos)
        self.demo_traj = generate_kinematic_trajectory(
            start_pose_6d=self.anchor_pose,
            joint_type=self.task_spec.joint_type,
            world_origin=world_geom["hinge_origin_world"],
            world_axis=world_geom["hinge_axis_world"],
            open_amount=self._compute_open_amount(),
            steps=self.config.trajectory_steps,
        )
        self.demo_info = {
            "grasp_stable": bool(stable),
            "grasp_info": _to_jsonable(grasp_info),
            "spawn_height": float(self.gym.arti_init_obj_pos_list[0][2]),
            "pregrasp_pose": _to_jsonable(self.pregrasp_pose),
            "anchor_pose": _to_jsonable(self.anchor_pose),
            "anchor_qpos": _to_jsonable(self.anchor_qpos),
            "demo_traj_len": int(len(self.demo_traj)),
        }

    def _stabilize_reset_grasp(self, hand_pose: np.ndarray, hand_qpos: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        world_geom = self._current_world_geom()
        if self.config.pinch_search_enabled:
            stabilized_pose, stable, grasp_info = self._search_pinch_grasp(
                start_pose=hand_pose,
                target_qpos=hand_qpos,
                world_geom=world_geom,
                obj_urdf_path=self.task_spec.urdf_path,
            )
        else:
            stabilized_pose, stable, grasp_info = self.gym.stabilize_grasp_by_surface_contact(
                start_pose_6d=hand_pose,
                target_qpos=hand_qpos,
                approach_dir=-world_geom["handle_out_world"],
                obj_urdf_path=self.task_spec.urdf_path,
                surface_contact_thresh=0.015,
                min_contact_points=40,
                required_contact_links=["palm", "index2", "middle2"],
                min_points_per_link=2,
                settle_steps=8,
                max_iters=12,
                push_step=0.002,
            )
        return np.asarray(stabilized_pose, dtype=np.float32), bool(stable), grasp_info

    def _set_state(
        self,
        hand_pose: Sequence[float],
        hand_qpos: Sequence[float],
        obj_qpos: Optional[Sequence[float]] = None,
        settle_steps: Optional[int] = None,
    ) -> None:
        hand_pose_arr = np.asarray(hand_pose, dtype=np.float32).reshape(7)
        hand_qpos_arr = self._clip_hand_qpos(hand_qpos)
        if obj_qpos is None:
            object_qpos = (
                self.gym.dof_pos[
                    0,
                    self.gym.mano_num_dofs : self.gym.mano_num_dofs + self.gym.arti_obj_num_dofs,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            object_qpos = np.asarray(obj_qpos, dtype=np.float32).reshape(self.gym.arti_obj_num_dofs)

        root_states = self.gym.root_states.clone()
        for env_i in range(self.gym.num_envs):
            mano_idx = self.gym.mano_actor_idxs[env_i]
            root_states[mano_idx, :3] = torch.tensor(hand_pose_arr[:3], dtype=torch.float32, device=self.gym.device)
            root_states[mano_idx, 3:7] = torch.tensor(hand_pose_arr[3:7], dtype=torch.float32, device=self.gym.device)
            root_states[mano_idx, 7:13] = 0.0
        self.gym._set_mano_root_state_tensor(root_states)

        dof_states = self.gym.dof_states.clone()
        num_dof = self.gym.dof_pos.shape[1]
        dof_states_view = dof_states.view(self.gym.num_envs, num_dof, 2)
        dof_states_view[:, : self.gym.mano_num_dofs, 0] = torch.tensor(hand_qpos_arr, dtype=torch.float32, device=self.gym.device)
        dof_states_view[:, : self.gym.mano_num_dofs, 1] = 0.0
        dof_states_view[:, self.gym.mano_num_dofs : self.gym.mano_num_dofs + self.gym.arti_obj_num_dofs, 0] = torch.tensor(
            object_qpos, dtype=torch.float32, device=self.gym.device
        )
        dof_states_view[:, self.gym.mano_num_dofs : self.gym.mano_num_dofs + self.gym.arti_obj_num_dofs, 1] = 0.0
        self.gym.gym.set_dof_state_tensor(self.gym.sim, gymtorch.unwrap_tensor(dof_states))

        pos_action = self.gym.dof_pos.squeeze(-1).clone()
        pos_action[:, : self.gym.mano_num_dofs] = torch.tensor(hand_qpos_arr, dtype=torch.float32, device=self.gym.device)
        self.gym.gym.set_dof_position_target_tensor(self.gym.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.run_steps(
            pre_steps=int(self.config.settle_steps if settle_steps is None else settle_steps),
            refresh_obs=True,
            print_step=False,
        )

    def _apply_hand_targets(
        self,
        hand_pose: Sequence[float],
        hand_qpos: Sequence[float],
        settle_steps: Optional[int] = None,
    ) -> None:
        hand_pose_arr = np.asarray(hand_pose, dtype=np.float32).reshape(7)
        hand_qpos_arr = self._clip_hand_qpos(hand_qpos)

        root_states = self.gym.root_states.clone()
        for env_i in range(self.gym.num_envs):
            mano_idx = self.gym.mano_actor_idxs[env_i]
            root_states[mano_idx, :3] = torch.tensor(hand_pose_arr[:3], dtype=torch.float32, device=self.gym.device)
            root_states[mano_idx, 3:7] = torch.tensor(hand_pose_arr[3:7], dtype=torch.float32, device=self.gym.device)
            root_states[mano_idx, 7:13] = 0.0
        self.gym._set_mano_root_state_tensor(root_states)

        pos_action = self.gym.dof_pos.squeeze(-1).clone()
        pos_action[:, : self.gym.mano_num_dofs] = torch.tensor(hand_qpos_arr, dtype=torch.float32, device=self.gym.device)
        self.gym.gym.set_dof_position_target_tensor(self.gym.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.run_steps(
            pre_steps=int(self.config.action_repeat if settle_steps is None else settle_steps),
            refresh_obs=True,
            print_step=False,
        )

    def _base_pose_for_step(self) -> np.ndarray:
        if self.config.use_demo_base_pose:
            index = int(np.clip(self.step_count, 0, len(self.demo_traj) - 1))
            return self.demo_traj[index].copy()
        return self.last_pose_target.copy()

    def _base_qpos_for_step(self) -> np.ndarray:
        if self.config.use_demo_base_pose:
            return self.phase_qpos_targets.get(self.get_curriculum_phase(), self.anchor_qpos).copy()
        return self.last_qpos_target.copy()

    def get_demo_base(self, step_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if step_index is None:
            step_index = self.step_count
        pose = self.demo_traj[int(np.clip(step_index, 0, len(self.demo_traj) - 1))].copy()
        qpos = self.phase_qpos_targets.get(self.get_curriculum_phase(), self.anchor_qpos).copy()
        return pose.astype(np.float32), qpos.astype(np.float32)

    def get_curriculum_phase(self) -> str:
        levels = list(self.config.curriculum_levels)
        if len(levels) == 0:
            return "grasp"
        return levels[int(np.clip(self.curriculum_level, 0, len(levels) - 1))]

    def update_curriculum(self, episode_success: bool) -> Dict[str, Any]:
        if not self.config.curriculum_enabled:
            return {
                "curriculum_enabled": False,
                "curriculum_phase": self.get_curriculum_phase(),
                "curriculum_level": int(self.curriculum_level),
            }
        self.curriculum_history.append(float(episode_success))
        window = int(max(1, self.config.curriculum_window))
        if len(self.curriculum_history) > window:
            self.curriculum_history = self.curriculum_history[-window:]
        recent_success = float(np.mean(self.curriculum_history))
        if (
            len(self.curriculum_history) >= window
            and recent_success >= float(self.config.curriculum_success_threshold)
            and self.curriculum_level < len(self.config.curriculum_levels) - 1
        ):
            self.curriculum_level += 1
            self.curriculum_history = []
        return {
            "curriculum_enabled": True,
            "curriculum_phase": self.get_curriculum_phase(),
            "curriculum_level": int(self.curriculum_level),
            "recent_success": float(recent_success),
        }

    def _phase_demo_index(self, phase_name: str) -> int:
        if phase_name == "grasp":
            return 0
        if phase_name == "actuate":
            return int(max(0, 0.15 * (len(self.demo_traj) - 1)))
        if phase_name == "open":
            return 0
        return 0

    def _curriculum_reset_pose(self, phase_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if phase_name == "approach":
            return self.pregrasp_pose.copy(), self.open_qpos.copy()
        if phase_name == "grasp":
            return self.anchor_pose.copy(), self.phase_qpos_targets["grasp"].copy()
        if phase_name == "actuate":
            index = self._phase_demo_index("actuate")
            return self.demo_traj[index].copy(), self.phase_qpos_targets["actuate"].copy()
        return self.anchor_pose.copy(), self.phase_qpos_targets["open"].copy()

    def action_to_target(self, action: Sequence[float], step_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        if self.config.use_demo_base_pose:
            base_pose, base_qpos = self.get_demo_base(step_index=step_index)
        else:
            base_pose = self.last_pose_target.copy()
            base_qpos = self.last_qpos_target.copy()
        target_pose = self._apply_pose_action(base_pose, action_arr[:6])
        target_qpos = self._apply_synergy_action(base_qpos, action_arr[6:])
        return target_pose.astype(np.float32), target_qpos.astype(np.float32)

    def get_teacher_action(self, step_index: Optional[int] = None) -> np.ndarray:
        if step_index is None:
            step_index = self.step_count
        if not self.config.use_demo_base_pose:
            return np.zeros(self.action_dim, dtype=np.float32)
        base_pose, base_qpos = self.get_demo_base(step_index=step_index)
        next_index = int(np.clip(step_index + 1, 0, len(self.demo_traj) - 1))
        next_pose = self.demo_traj[next_index].copy()
        teacher_phase = self.get_curriculum_phase()
        if teacher_phase == "grasp":
            pull_pose = next_pose.copy()
            pull_pose[:3] += 0.35 * float(self.config.pos_action_scale) * self.prev_state.open_tangent_world
            pull_pose[:3] -= 0.20 * float(self.config.pos_action_scale) * self.prev_state.handle_out_world
            next_pose = pull_pose
        pos_residual = (next_pose[:3] - base_pose[:3]) / max(self.config.pos_action_scale, 1e-6)
        delta_rot = R.from_quat(next_pose[3:7]) * R.from_quat(base_pose[3:7]).inv()
        rotvec_residual = delta_rot.as_rotvec() / max(self.config.rot_action_scale, 1e-6)
        desired_qpos = self.phase_qpos_targets.get(teacher_phase, self.anchor_qpos).copy()
        if teacher_phase == "grasp":
            desired_qpos = self.phase_qpos_targets.get("actuate", desired_qpos).copy()
        elif teacher_phase == "touch":
            desired_qpos = self.phase_qpos_targets.get("grasp", desired_qpos).copy()
        qpos_delta = desired_qpos - base_qpos
        synergy_residual = np.linalg.pinv(self.synergy_matrix) @ qpos_delta
        synergy_residual = synergy_residual / max(self.config.synergy_action_scale, 1e-6)
        teacher_action = np.concatenate(
            [
                np.asarray(pos_residual, dtype=np.float32),
                np.asarray(rotvec_residual, dtype=np.float32),
                np.asarray(synergy_residual, dtype=np.float32),
            ],
            axis=0,
        )
        return np.clip(teacher_action, -1.0, 1.0).astype(np.float32)

    def get_teacher_targets(self, step_index: Optional[int] = None) -> Dict[str, Any]:
        teacher_action = self.get_teacher_action(step_index=step_index)
        teacher_target_pose, teacher_target_qpos = self.action_to_target(teacher_action, step_index=step_index)
        teacher_phase = self.get_curriculum_phase()
        _, teacher_base_qpos = self.get_demo_base(step_index=step_index)
        teacher_qpos_residual = (
            (teacher_target_qpos - teacher_base_qpos) / max(self.config.synergy_action_scale, 1e-6)
        ).astype(np.float32)
        return {
            "teacher_action": teacher_action.astype(np.float32),
            "teacher_target_pose": teacher_target_pose.astype(np.float32),
            "teacher_target_qpos": teacher_target_qpos.astype(np.float32),
            "teacher_qpos_residual": teacher_qpos_residual.astype(np.float32),
            "teacher_contact_target": get_phase_contact_target(teacher_phase).astype(np.float32),
            "teacher_phase": teacher_phase,
            "teacher_pull_hint": bool(teacher_phase == "grasp"),
        }

    def _apply_pose_action(self, base_pose: Sequence[float], action: Sequence[float]) -> np.ndarray:
        base_pose_arr = np.asarray(base_pose, dtype=np.float32).reshape(7)
        action_arr = np.asarray(action, dtype=np.float32).reshape(6)
        new_pos = base_pose_arr[:3] + action_arr[:3] * float(self.config.pos_action_scale)
        base_rot = R.from_quat(base_pose_arr[3:7])
        delta_rot = R.from_rotvec(action_arr[3:6] * float(self.config.rot_action_scale))
        target_rot = delta_rot * base_rot
        blend = float(np.clip(self.config.wrist_rotation_lock, 0.0, 1.0))
        if blend > 0.0:
            target_rotvec = target_rot.as_rotvec()
            base_rotvec = base_rot.as_rotvec()
            blended_rot = R.from_rotvec((1.0 - blend) * target_rotvec + blend * base_rotvec)
            new_rot = blended_rot.as_quat().astype(np.float32)
        else:
            new_rot = target_rot.as_quat().astype(np.float32)
        return np.concatenate([new_pos.astype(np.float32), new_rot], axis=0)

    def _apply_synergy_action(self, base_qpos: Sequence[float], action: Sequence[float]) -> np.ndarray:
        base_arr = np.asarray(base_qpos, dtype=np.float32).reshape(20)
        synergy_action = np.asarray(action, dtype=np.float32).reshape(5) * float(self.config.synergy_action_scale)
        qpos = base_arr + self.synergy_matrix @ synergy_action
        return self._clip_hand_qpos(qpos)

    def _apply_grasp_settle(self, base_pose: np.ndarray, base_qpos: np.ndarray, action_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        settle_scale = float(self.config.grasp_settle_synergy_scale)
        settle_action = action_arr.copy()
        settle_action[:6] = 0.0
        settle_action[6:] *= settle_scale
        target_pose = base_pose.copy()
        target_qpos = self._apply_synergy_action(base_qpos, settle_action[6:])
        return target_pose.astype(np.float32), target_qpos.astype(np.float32)

    def _apply_door_plane_buffer(self, hand_pose: Sequence[float], runtime_state) -> np.ndarray:
        pose = np.asarray(hand_pose, dtype=np.float32).copy()
        plane_normal = runtime_state.handle_out_world
        plane_point = runtime_state.handle_front_center_world - float(self.config.palm_safe_buffer) * plane_normal
        signed_dist = float(np.dot(pose[:3] - plane_point, plane_normal))
        if signed_dist < 0.0:
            pose[:3] = pose[:3] - signed_dist * plane_normal
        return pose.astype(np.float32)

    def _door_plane_violation(self, runtime_state) -> float:
        plane_normal = runtime_state.handle_out_world
        plane_point = runtime_state.handle_front_center_world - float(self.config.palm_safe_buffer) * plane_normal
        signed_dist = float(np.dot(runtime_state.hand_pos - plane_point, plane_normal))
        return float(max(0.0, -signed_dist))

    def reset(self, phase: Optional[str] = None, pose_noise: Optional[float] = None, rot_noise: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        reset_phase = self.config.reset_phase if phase is None else phase
        if phase is None and self.config.curriculum_enabled:
            reset_phase = self.get_curriculum_phase()
        pose_noise_value = self.config.reset_pose_noise if pose_noise is None else float(pose_noise)
        rot_noise_value = self.config.reset_rot_noise if rot_noise is None else float(rot_noise)
        hand_pose, hand_qpos = self._curriculum_reset_pose(reset_phase)

        if pose_noise_value > 0.0:
            hand_pose[:3] += np.random.uniform(-pose_noise_value, pose_noise_value, size=(3,)).astype(np.float32)
        if rot_noise_value > 0.0:
            delta_rot = R.from_rotvec(np.random.uniform(-rot_noise_value, rot_noise_value, size=(3,)).astype(np.float32))
            hand_pose[3:7] = (delta_rot * R.from_quat(hand_pose[3:7])).as_quat().astype(np.float32)

        self._set_state(hand_pose=hand_pose, hand_qpos=hand_qpos, obj_qpos=self.object_lower, settle_steps=max(self.config.settle_steps, 2))
        reset_stable = False
        reset_grasp_info: Dict[str, Any] = {}
        if reset_phase != "approach" and self.config.stabilize_on_reset:
            hand_pose, reset_stable, reset_grasp_info = self._stabilize_reset_grasp(hand_pose=hand_pose, hand_qpos=hand_qpos)
            self._set_state(hand_pose=hand_pose, hand_qpos=hand_qpos, obj_qpos=None, settle_steps=max(self.config.settle_steps, 2))
        self.last_pose_target = hand_pose.copy()
        self.last_qpos_target = hand_qpos.copy()
        self.prev_action = self.zero_action()
        self.step_count = 0
        self.grasp_settle_counter = int(max(0, self.config.grasp_settle_steps if reset_phase != "approach" else 0))
        self.prev_state = extract_single_door_runtime_state(
            self.gym,
            self.task_spec,
            env_i=0,
            surface_contact_thresh=0.015,
            min_contact_points=self.config.min_contact_points,
            contact_target_points=self.config.contact_target_points,
        )
        obs = build_single_door_observation(
            self.prev_state,
            prev_action=self.prev_action,
            contact_target_points=self.config.contact_target_points,
        )
        self.observation_dim = int(obs.shape[0])
        reset_teacher_targets = self.get_teacher_targets(step_index=0)
        info = {
            "reset_phase": reset_phase,
            "curriculum_phase": self.get_curriculum_phase(),
            "curriculum_level": int(self.curriculum_level),
            "task_spec": self.task_spec.to_dict(),
            "demo_info": self.demo_info,
            "progress": float(self.prev_state.progress),
            "surface_contact_count": int(self.prev_state.surface_contact_count),
            "surface_contact_stable": bool(self.prev_state.surface_contact_stable),
            "surface_contact_min_dist": float(self.prev_state.surface_contact_min_dist),
            "handle_min_dist": float(self.prev_state.handle_min_dist),
            "non_interact_min_dist": float(self.prev_state.non_interact_min_dist),
            "non_interact_penetration_depth": float(self.prev_state.non_interact_penetration_depth),
            "handle_contact_ratio": float(self.prev_state.handle_contact_ratio),
            "non_interact_near_ratio": float(self.prev_state.non_interact_near_ratio),
            "door_plane_violation": float(self._door_plane_violation(self.prev_state)),
            "contact_features": _to_jsonable(build_contact_feature_vector(self.prev_state.surface_contact_link_counts)),
            "contact_target": _to_jsonable(get_phase_contact_target(self.get_curriculum_phase())),
            "contact_link_order": list(CONTACT_LINK_ORDER),
            "reset_grasp_stable": bool(reset_stable),
            "reset_grasp_info": _to_jsonable(reset_grasp_info),
            "grasp_settle_counter": int(self.grasp_settle_counter),
            "teacher_qpos_residual": _to_jsonable(reset_teacher_targets["teacher_qpos_residual"]),
            "pinch_debug": {
                "thumb3": int(self.prev_state.surface_contact_link_counts.get("thumb3", 0)),
                "index3": int(self.prev_state.surface_contact_link_counts.get("index3", 0)),
                "middle3": int(self.prev_state.surface_contact_link_counts.get("middle3", 0)),
                "ring3": int(self.prev_state.surface_contact_link_counts.get("ring3", 0)),
                "pinky3": int(self.prev_state.surface_contact_link_counts.get("pinky3", 0)),
            },
            "observation_dim": int(self.observation_dim),
        }
        return obs.astype(np.float32), info

    def step(self, action: Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        teacher_targets = self.get_teacher_targets(step_index=self.step_count)
        base_pose = self._base_pose_for_step()
        base_qpos = self._base_qpos_for_step()
        in_grasp_settle = bool(self.grasp_settle_counter > 0)
        if in_grasp_settle:
            target_pose, target_qpos = self._apply_grasp_settle(base_pose, base_qpos, action_arr)
            self.grasp_settle_counter = max(0, self.grasp_settle_counter - 1)
        else:
            target_pose = self._apply_pose_action(base_pose, action_arr[:6])
            target_qpos = self._apply_synergy_action(base_qpos, action_arr[6:])

        if self.prev_state is not None:
            target_pose = self._apply_door_plane_buffer(target_pose, self.prev_state)

        self._apply_hand_targets(hand_pose=target_pose, hand_qpos=target_qpos, settle_steps=max(1, self.config.action_repeat))

        state = extract_single_door_runtime_state(
            self.gym,
            self.task_spec,
            env_i=0,
            surface_contact_thresh=0.015,
            min_contact_points=self.config.min_contact_points,
            contact_target_points=self.config.contact_target_points,
        )
        reward_terms = compute_single_door_reward(
            state=state,
            prev_state=self.prev_state,
            action=action_arr,
            prev_action=self.prev_action,
            config=self.reward_config,
        )

        if state.surface_contact_min_dist < float(self.config.max_safe_penetration):
            safe_pose = target_pose.copy()
            safe_pose[:3] += 0.75 * float(self.config.pinch_search_depth_step) * state.handle_out_world
            self._apply_hand_targets(hand_pose=safe_pose, hand_qpos=target_qpos, settle_steps=1)
            state = extract_single_door_runtime_state(
                self.gym,
                self.task_spec,
                env_i=0,
                surface_contact_thresh=0.015,
                min_contact_points=self.config.min_contact_points,
                contact_target_points=self.config.contact_target_points,
            )
            reward_terms = compute_single_door_reward(
                state=state,
                prev_state=self.prev_state,
                action=action_arr,
                prev_action=self.prev_action,
                config=self.reward_config,
            )
            target_pose = safe_pose

        self.prev_state = state
        self.prev_action = action_arr.copy()
        self.last_pose_target = target_pose.copy()
        self.last_qpos_target = target_qpos.copy()
        self.step_count += 1

        done = bool(state.progress >= self.reward_config.success_progress or self.step_count >= self.config.max_episode_steps)
        obs = build_single_door_observation(
            state,
            prev_action=self.prev_action,
            contact_target_points=self.config.contact_target_points,
        )
        info = {
            "step_count": int(self.step_count),
            "curriculum_phase": self.get_curriculum_phase(),
            "teacher_action": _to_jsonable(teacher_targets["teacher_action"]),
            "teacher_target_pose": _to_jsonable(teacher_targets["teacher_target_pose"]),
            "teacher_target_qpos": _to_jsonable(teacher_targets["teacher_target_qpos"]),
            "teacher_qpos_residual": _to_jsonable(teacher_targets["teacher_qpos_residual"]),
            "teacher_pull_hint": bool(teacher_targets["teacher_pull_hint"]),
            "grasp_settle_active": bool(in_grasp_settle),
            "grasp_settle_counter": int(self.grasp_settle_counter),
            "contact_features": _to_jsonable(build_contact_feature_vector(state.surface_contact_link_counts)),
            "contact_target": _to_jsonable(teacher_targets["teacher_contact_target"]),
            "contact_link_order": list(CONTACT_LINK_ORDER),
            "progress": float(state.progress),
            "drive_dof_val": float(state.drive_dof_val),
            "drive_dof_vel": float(state.drive_dof_vel),
            "surface_contact_count": int(state.surface_contact_count),
            "surface_contact_stable": bool(state.surface_contact_stable),
            "surface_contact_min_dist": float(state.surface_contact_min_dist),
            "handle_min_dist": float(state.handle_min_dist),
            "non_interact_min_dist": float(state.non_interact_min_dist),
            "non_interact_penetration_depth": float(state.non_interact_penetration_depth),
            "handle_contact_ratio": float(state.handle_contact_ratio),
            "non_interact_near_ratio": float(state.non_interact_near_ratio),
            "door_plane_violation": float(self._door_plane_violation(state)),
            "surface_contact_link_counts": dict(state.surface_contact_link_counts),
            "pinch_debug": {
                "thumb3": int(state.surface_contact_link_counts.get("thumb3", 0)),
                "index3": int(state.surface_contact_link_counts.get("index3", 0)),
                "middle3": int(state.surface_contact_link_counts.get("middle3", 0)),
                "ring3": int(state.surface_contact_link_counts.get("ring3", 0)),
                "pinky3": int(state.surface_contact_link_counts.get("pinky3", 0)),
            },
            "target_pose": _to_jsonable(target_pose),
            "target_qpos": _to_jsonable(target_qpos),
            "reward_terms": reward_terms,
            "success": bool(state.progress >= self.reward_config.success_progress),
            "task_spec": self.task_spec.to_dict(),
        }
        return obs.astype(np.float32), float(reward_terms["total"]), done, info

    def close(self) -> None:
        if hasattr(self, "gym") and self.gym is not None:
            try:
                self.gym.clean_up()
            except Exception:
                pass

    def export_rollout_summary(self, metrics: Sequence[Dict[str, Any]], save_path: str) -> None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        payload = {
            "config": _to_jsonable(self.config.__dict__),
            "task_spec": self.task_spec.to_dict(),
            "demo_info": self.demo_info,
            "handle_geometry": self.get_handle_geometry_diagnostics(),
            "metrics": _to_jsonable(list(metrics)),
        }
        with open(save_path, "w") as handle:
            json.dump(payload, handle)

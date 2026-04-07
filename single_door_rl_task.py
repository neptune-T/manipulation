from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


PHASE_TO_ID = {
    "approach": 0,
    "touch": 1,
    "grasp": 2,
    "actuate": 3,
    "success": 4,
}

CONTACT_LINK_ORDER = ["thumb3", "index3", "middle3", "ring3", "pinky3", "palm"]


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


def _clip_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def prepare_handle_passthrough_urdf(
    asset_dir: str,
    handle_categories: Sequence[str] = (
        "line_fixed_handle",
        "round_fixed_handle",
        "hinge_knob",
        "revolute_handle",
    ),
    src_urdf: str = "mobility_annotation_gapartnet.urdf",
    dst_urdf: str = "mobility_annotation_gapartnet_passthrough.urdf",
) -> str:
    """Strip collision elements from handle links so MANO fingers can pass through.

    Reads ``link_annotation_gapartnet.json`` to identify which links are handles,
    then writes a modified copy of the URDF with those links' ``<collision>``
    elements removed.  Visual geometry is kept so the handle remains visible.

    Returns the absolute path of the generated URDF.  If the output already
    exists and is newer than both the source URDF and the annotation file,
    it is returned as-is (no rewrite).
    """
    urdf_path = os.path.join(asset_dir, src_urdf)
    anno_path = os.path.join(asset_dir, "link_annotation_gapartnet.json")
    dst_path = os.path.join(asset_dir, dst_urdf)

    # Skip regeneration when output is up-to-date.
    if os.path.exists(dst_path):
        dst_mtime = os.path.getmtime(dst_path)
        if (
            os.path.getmtime(urdf_path) <= dst_mtime
            and os.path.getmtime(anno_path) <= dst_mtime
        ):
            return dst_path

    annos = _read_json(anno_path, [])
    handle_link_names: set = set()
    for anno in annos:
        if not anno.get("is_gapart", False):
            continue
        cat = str(anno.get("category", "")).lower()
        if cat in {c.lower() for c in handle_categories} or "handle" in cat or "knob" in cat:
            link_name = anno.get("link_name")
            if link_name:
                handle_link_names.add(link_name)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    stripped_count = 0
    for link_elem in root.findall("link"):
        if link_elem.get("name") not in handle_link_names:
            continue
        collisions = link_elem.findall("collision")
        for col in collisions:
            link_elem.remove(col)
            stripped_count += 1

    if stripped_count > 0:
        tree.write(dst_path, xml_declaration=True)
        print(
            f"[handle-passthrough] Stripped {stripped_count} collision element(s) "
            f"from handle links {handle_link_names} → {dst_path}"
        )
    else:
        # No handle collision to strip — just copy the original.
        import shutil
        shutil.copy2(urdf_path, dst_path)
        print(f"[handle-passthrough] No handle collisions found; copied URDF → {dst_path}")

    return dst_path


def parse_joint_info(urdf_path: str, target_link_name: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[float], Optional[float]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    child_to_joint: Dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") is not None:
            child_to_joint[child.get("link")] = joint

    active_joint = None
    search_link = target_link_name
    while search_link in child_to_joint:
        joint = child_to_joint[search_link]
        if joint.get("type") in ["revolute", "prismatic", "continuous"]:
            active_joint = joint
            break
        parent = joint.find("parent")
        if parent is None or parent.get("link") is None:
            break
        search_link = parent.get("link")

    if active_joint is None:
        return None, None, None, None, None, None

    path: List[ET.Element] = []
    current_joint: Optional[ET.Element] = active_joint
    while current_joint is not None:
        path.append(current_joint)
        parent_link = current_joint.find("parent")
        if parent_link is None or parent_link.get("link") is None:
            current_joint = None
        else:
            current_joint = child_to_joint.get(parent_link.get("link"))
    path.reverse()

    current_rot = R.identity()
    current_pos = np.zeros(3, dtype=np.float32)

    for joint in path:
        origin = joint.find("origin")
        xyz = np.array([float(x) for x in origin.get("xyz").split()]) if origin is not None and origin.get("xyz") else np.zeros(3, dtype=np.float32)
        rpy = np.array([float(x) for x in origin.get("rpy").split()]) if origin is not None and origin.get("rpy") else np.zeros(3, dtype=np.float32)

        if joint is active_joint:
            axis = joint.find("axis")
            local_axis = np.array([float(x) for x in axis.get("xyz").split()]) if axis is not None and axis.get("xyz") else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            joint_rot = R.from_euler("xyz", rpy)
            true_axis = (current_rot * joint_rot).apply(local_axis)
            true_axis = _normalize(true_axis, fallback=[0.0, 0.0, 1.0])
            true_origin = current_pos + current_rot.apply(xyz)
            joint_name = joint.get("name")

            joint_limit = joint.find("limit")
            lower = float(joint_limit.get("lower")) if joint_limit is not None and joint_limit.get("lower") is not None else None
            upper = float(joint_limit.get("upper")) if joint_limit is not None and joint_limit.get("upper") is not None else None
            return joint.get("type"), true_origin.astype(np.float32), true_axis.astype(np.float32), joint_name, lower, upper

        current_pos = current_pos + current_rot.apply(xyz)
        current_rot = current_rot * R.from_euler("xyz", rpy)

    return None, None, None, None, None, None


def compute_handle_bbox_geometry(bbox_points: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
    bbox = np.asarray(bbox_points, dtype=np.float32).reshape(8, 3)
    center = np.mean(bbox, axis=0)
    center_front_face = np.mean(bbox[:4], axis=0)
    handle_out = _normalize(bbox[0] - bbox[4], fallback=[1.0, 0.0, 0.0])
    handle_long = _normalize(bbox[0] - bbox[1], fallback=[0.0, 0.0, 1.0])
    handle_short = _normalize(bbox[0] - bbox[3], fallback=[0.0, 1.0, 0.0])
    extents = np.array(
        [
            np.linalg.norm(bbox[0] - bbox[4]),
            np.linalg.norm(bbox[0] - bbox[1]),
            np.linalg.norm(bbox[0] - bbox[3]),
        ],
        dtype=np.float32,
    )
    return {
        "bbox": bbox,
        "center": center.astype(np.float32),
        "center_front_face": center_front_face.astype(np.float32),
        "out": handle_out.astype(np.float32),
        "long": handle_long.astype(np.float32),
        "short": handle_short.astype(np.float32),
        "extents": extents,
    }


def _build_parent_map(urdf_path: str) -> Dict[str, Tuple[str, str]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    parent_map: Dict[str, Tuple[str, str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        child_link = child.get("link")
        if parent_link is None or child_link is None:
            continue
        parent_map[child_link] = (parent_link, joint.get("type") or "")
    return parent_map


def _find_handle_for_door(
    door_link_name: str,
    handle_annos: List[Tuple[int, int, Dict[str, Any]]],
    parent_map: Dict[str, Tuple[str, str]],
    urdf_path: str,
    door_joint_name: Optional[str],
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    direct_matches: List[Tuple[int, int, Dict[str, Any]]] = []
    fallback_matches: List[Tuple[int, int, Dict[str, Any]]] = []
    for handle_gapart_idx, handle_full_idx, handle_anno in handle_annos:
        handle_link_name = handle_anno.get("link_name")
        if handle_link_name is None:
            continue
        relation = parent_map.get(handle_link_name)
        if relation is not None and relation[0] == door_link_name:
            direct_matches.append((handle_gapart_idx, handle_full_idx, handle_anno))
            continue
        _, _, _, handle_joint_name, _, _ = parse_joint_info(urdf_path, handle_link_name)
        if door_joint_name is not None and handle_joint_name == door_joint_name:
            fallback_matches.append((handle_gapart_idx, handle_full_idx, handle_anno))
    if direct_matches:
        return sorted(direct_matches, key=lambda item: item[0])[0]
    if fallback_matches:
        return sorted(fallback_matches, key=lambda item: item[0])[0]
    return None


def _read_joints_from_urdf(urdf_path: str) -> Dict[str, Dict[str, Any]]:
    """Read the full kinematic chain from URDF (mirrors the official GAPartNet read_joints_from_urdf_file)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_dict: Dict[str, Dict[str, Any]] = {}
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name is None or joint_type is None:
            continue
        child_elem = joint.find("child")
        parent_elem = joint.find("parent")
        if child_elem is None or parent_elem is None:
            continue
        origin_elem = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin_elem is not None:
            if origin_elem.get("xyz"):
                xyz = [float(x) for x in origin_elem.get("xyz").split()]
            if origin_elem.get("rpy"):
                rpy = [float(x) for x in origin_elem.get("rpy").split()]
        axis_val = None
        if joint_type in ("prismatic", "revolute", "continuous"):
            axis_elem = joint.find("axis")
            if axis_elem is not None and axis_elem.get("xyz"):
                axis_val = [float(x) for x in axis_elem.get("xyz").split()]
            else:
                axis_val = [1.0, 0.0, 0.0]
        joint_dict[joint_name] = {
            "type": joint_type,
            "parent": parent_elem.get("link"),
            "child": child_elem.get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis_val,
        }
    return joint_dict


def _walk_ancestors(link_name: str, joints_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return list of ancestor link names from link_name up to the root."""
    child_to_joint: Dict[str, str] = {}
    for jname, jinfo in joints_dict.items():
        child_to_joint[jinfo["child"]] = jname
    ancestors: List[str] = []
    cur = link_name
    while cur in child_to_joint:
        jname = child_to_joint[cur]
        cur = joints_dict[jname]["parent"]
        ancestors.append(cur)
    return ancestors


def _axangle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis-angle to 3x3 rotation matrix (Rodrigues). Pure numpy, no scipy."""
    ax = np.asarray(axis, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    ax = ax / norm
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = ax[0], ax[1], ax[2]
    return np.array([
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ],
    ], dtype=np.float64)


def _transform_bbox_through_chain(
    bbox: np.ndarray,
    link_name: str,
    joints_dict: Dict[str, Dict[str, Any]],
    joint_qpos: Optional[Dict[str, float]] = None,
    base_link_name: str = "base",
) -> np.ndarray:
    """Transform a bbox from link-local space to world space via forward kinematics.

    Mirrors the official GAPartNet ``query_part_pose_from_joint_qpos`` but uses
    pure numpy instead of SAPIEN so we can run inside Isaac Gym.
    """
    child_to_joint: Dict[str, str] = {}
    for jname, jinfo in joints_dict.items():
        child_to_joint[jinfo["child"]] = jname

    # Build chain from link up to (but not including) the base joint
    chain: List[str] = []
    cur = link_name
    while cur in child_to_joint:
        jname = child_to_joint[cur]
        chain.append(jname)
        cur = joints_dict[jname]["parent"]

    # Compute cumulative transforms for each joint origin+axis in world frame
    # Walk the chain root→leaf, accumulating the FK transform
    chain_reversed = list(reversed(chain))
    cum_pos = np.zeros(3, dtype=np.float64)
    cum_rot = np.eye(3, dtype=np.float64)
    joint_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for jname in chain_reversed:
        jinfo = joints_dict[jname]
        xyz = np.asarray(jinfo["xyz"], dtype=np.float64)
        rpy = np.asarray(jinfo["rpy"], dtype=np.float64)
        local_rot = R.from_euler("xyz", rpy).as_matrix()
        cum_pos = cum_pos + cum_rot @ xyz
        cum_rot = cum_rot @ local_rot
        if jinfo["axis"] is not None:
            world_axis = cum_rot @ np.asarray(jinfo["axis"], dtype=np.float64)
            world_axis = world_axis / (np.linalg.norm(world_axis) + 1e-12)
        else:
            world_axis = np.array([0.0, 0.0, 1.0])
        joint_world[jname] = (cum_pos.copy(), world_axis.copy())

    # Apply joint transformations to bbox (walk from root to leaf, like the official algo)
    pts = np.asarray(bbox, dtype=np.float64).reshape(-1, 3).copy()
    if joint_qpos is None:
        joint_qpos = {}
    for jname in chain_reversed:
        jinfo = joints_dict[jname]
        jtype = jinfo["type"]
        qval = float(joint_qpos.get(jname, 0.0))
        if jtype == "fixed" or abs(qval) < 1e-12:
            continue
        origin, axis = joint_world[jname]
        if jtype == "prismatic":
            pts = pts + axis * qval
        elif jtype in ("revolute", "continuous"):
            rot_mat = _axangle_to_matrix(axis, qval)
            pts = (pts - origin) @ rot_mat.T + origin

    return pts.astype(np.float32)


def _find_handle_for_door_chain(
    door_link_name: str,
    handle_annos: List[Tuple[int, int, Dict[str, Any]]],
    urdf_path: str,
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    """Fallback: walk the full kinematic chain to match handles to doors.

    Uses the official GAPartNet approach — a handle belongs to a door if the
    door link appears anywhere in the handle's ancestor chain (not just as a
    direct parent).
    """
    joints_dict = _read_joints_from_urdf(urdf_path)
    chain_matches: List[Tuple[int, int, Dict[str, Any]]] = []
    for handle_gapart_idx, handle_full_idx, handle_anno in handle_annos:
        handle_link_name = handle_anno.get("link_name")
        if handle_link_name is None:
            continue
        ancestors = _walk_ancestors(handle_link_name, joints_dict)
        if door_link_name in ancestors:
            chain_matches.append((handle_gapart_idx, handle_full_idx, handle_anno))
    if chain_matches:
        return sorted(chain_matches, key=lambda item: item[0])[0]
    return None


def _find_any_handle_by_proximity(
    annos: List[Dict[str, Any]],
    door_link_name: str,
    urdf_path: str,
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    """Last-resort fallback: find the nearest handle annotation by bbox proximity to the door.

    Uses GAPartNet's TARGET_GAPARTS categories to identify valid handles.
    """
    HANDLE_CATEGORIES = {"line_fixed_handle", "round_fixed_handle", "hinge_knob", "revolute_handle", "hinge_handle"}
    joints_dict = _read_joints_from_urdf(urdf_path)

    # Get the door bbox center as reference
    door_center = None
    for anno in annos:
        if anno.get("link_name") == door_link_name and anno.get("bbox"):
            door_bbox = np.asarray(anno["bbox"], dtype=np.float32).reshape(-1, 3)
            door_center = np.mean(door_bbox, axis=0)
            break
    if door_center is None:
        return None

    best_dist = float("inf")
    best_match = None
    gapart_idx = -1
    for anno_idx, anno in enumerate(annos):
        if not anno.get("is_gapart", False):
            continue
        gapart_idx += 1
        cat = str(anno.get("category", "")).lower()
        if cat not in HANDLE_CATEGORIES and "handle" not in cat and "knob" not in cat:
            continue
        bbox = anno.get("bbox", [])
        if len(bbox) != 8:
            continue
        handle_bbox = np.asarray(bbox, dtype=np.float32).reshape(8, 3)
        handle_center = np.mean(handle_bbox, axis=0)
        dist = float(np.linalg.norm(handle_center - door_center))
        if dist < best_dist:
            best_dist = dist
            best_match = (gapart_idx, anno_idx, anno)
    return best_match


def _match_handle_to_door(
    door_link_name: str,
    handle_annos: List[Tuple[int, int, Dict[str, Any]]],
    annos: List[Dict[str, Any]],
    parent_map: Dict[str, Tuple[str, str]],
    urdf_path: str,
    door_joint_name: Optional[str],
) -> Tuple[Optional[Tuple[int, int, Dict[str, Any]]], str]:
    """Associate a handle with a door using progressively weaker GAPartNet-style heuristics."""
    match = _find_handle_for_door(door_link_name, handle_annos, parent_map, urdf_path, door_joint_name)
    if match is not None:
        return match, "direct_parent_or_joint"

    match = _find_handle_for_door_chain(door_link_name, handle_annos, urdf_path)
    if match is not None:
        return match, "ancestor_chain"

    match = _find_any_handle_by_proximity(annos, door_link_name, urdf_path)
    if match is not None:
        return match, "bbox_proximity"

    return None, "unmatched"


def _determine_open_sign(joint_lower: Optional[float], joint_upper: Optional[float]) -> float:
    candidates: List[float] = []
    if joint_upper is not None:
        candidates.append(float(joint_upper))
    if joint_lower is not None:
        candidates.append(float(joint_lower))
    if not candidates:
        return 1.0
    chosen = max(candidates, key=lambda value: abs(value))
    return 1.0 if chosen >= 0.0 else -1.0


@dataclass
class SingleDoorTaskSpec:
    asset_dir: str
    urdf_path: str
    model_cat: str
    door_link_name: str
    handle_link_name: str
    door_anno_index: int
    handle_anno_index: int
    door_bbox_index: int
    handle_bbox_index: int
    door_category: str
    handle_category: str
    handle_match_strategy: str
    joint_name: str
    joint_type: str
    joint_origin_local: np.ndarray
    joint_axis_local: np.ndarray
    joint_lower: Optional[float]
    joint_upper: Optional[float]
    open_sign: float
    handle_bbox_local: np.ndarray
    handle_center_local: np.ndarray
    handle_front_center_local: np.ndarray
    handle_out_local: np.ndarray
    handle_long_local: np.ndarray
    handle_short_local: np.ndarray
    handle_extent_local: np.ndarray
    success_progress: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))

    def world_geometry(self, obj_world_pos: Sequence[float], obj_world_rot: Sequence[float], obj_scale: float = 1.0) -> Dict[str, np.ndarray]:
        world_pos = np.asarray(obj_world_pos, dtype=np.float32)
        world_rot = R.from_quat(np.asarray(obj_world_rot, dtype=np.float32))
        scale = float(obj_scale)

        hinge_origin_world = world_pos + world_rot.apply(self.joint_origin_local * scale)
        hinge_axis_world = _normalize(world_rot.apply(self.joint_axis_local), fallback=self.joint_axis_local)
        handle_center_world = world_pos + world_rot.apply(self.handle_center_local * scale)
        handle_front_center_world = world_pos + world_rot.apply(self.handle_front_center_local * scale)
        handle_out_world = _normalize(world_rot.apply(self.handle_out_local), fallback=self.handle_out_local)
        handle_long_world = _normalize(world_rot.apply(self.handle_long_local), fallback=self.handle_long_local)
        handle_short_world = _normalize(world_rot.apply(self.handle_short_local), fallback=self.handle_short_local)

        radial = handle_center_world - hinge_origin_world
        radial = radial - hinge_axis_world * np.dot(radial, hinge_axis_world)
        radial = _normalize(radial, fallback=handle_out_world)
        open_tangent_world = _normalize(self.open_sign * np.cross(hinge_axis_world, radial), fallback=handle_long_world)

        return {
            "hinge_origin_world": hinge_origin_world.astype(np.float32),
            "hinge_axis_world": hinge_axis_world.astype(np.float32),
            "handle_center_world": handle_center_world.astype(np.float32),
            "handle_front_center_world": handle_front_center_world.astype(np.float32),
            "handle_out_world": handle_out_world.astype(np.float32),
            "handle_long_world": handle_long_world.astype(np.float32),
            "handle_short_world": handle_short_world.astype(np.float32),
            "open_tangent_world": open_tangent_world.astype(np.float32),
        }


ArticulationTaskSpec = SingleDoorTaskSpec


def select_articulation_task(
    asset_dir: str,
    task_index: int = 0,
    preferred_link: Optional[str] = None,
    preferred_handle_link: Optional[str] = None,
) -> ArticulationTaskSpec:
    return select_single_door_task(
        asset_dir=asset_dir,
        door_index=task_index,
        preferred_door_link=preferred_link,
        preferred_handle_link=preferred_handle_link,
    )


@dataclass
class SingleDoorRuntimeState:
    hand_pos: np.ndarray
    hand_rot: np.ndarray
    hand_lin_vel: np.ndarray
    hand_ang_vel: np.ndarray
    hand_qpos: np.ndarray
    hand_qvel: np.ndarray
    obj_qpos: np.ndarray
    obj_qvel: np.ndarray
    drive_dof_index: Optional[int]
    drive_dof_val: float
    drive_dof_vel: float
    progress: float
    surface_contact_count: int
    surface_contact_min_dist: float
    surface_contact_link_counts: Dict[str, int]
    surface_contact_stable: bool
    handle_min_dist: float
    non_interact_min_dist: float
    non_interact_signed_min_dist: float
    non_interact_penetration_depth: float
    handle_attraction_score: float
    non_interact_repulsion_penalty: float
    handle_contact_ratio: float
    non_interact_near_ratio: float
    fingertip_handle_dists: np.ndarray
    fingertip_non_interact_dists: np.ndarray
    hinge_origin_world: np.ndarray
    hinge_axis_world: np.ndarray
    handle_center_world: np.ndarray
    handle_front_center_world: np.ndarray
    handle_out_world: np.ndarray
    handle_long_world: np.ndarray
    handle_short_world: np.ndarray
    open_tangent_world: np.ndarray


@dataclass
class SingleDoorRewardConfig:
    reach_weight: float = 0.25
    align_weight: float = 0.10
    contact_weight: float = 0.8
    opposition_weight: float = 1.8
    hold_weight: float = 0.8
    progress_weight: float = 8.0
    tangent_weight: float = 0.75
    detach_penalty_weight: float = 1.5
    action_l2_weight: float = 0.01
    action_smooth_weight: float = 0.02
    success_bonus: float = 10.0
    success_progress: float = 0.35
    contact_target_points: int = 6
    reach_scale: float = 4.0
    tangent_scale: float = 0.5
    pinch_gate_threshold: float = 0.35
    progress_gate_threshold: float = 0.55
    force_closure_weight: float = 2.0
    force_closure_gate_threshold: float = 0.20
    penetration_progress_penalty: float = 8.0
    palm_penetration_penalty: float = 10.0
    fingertip_penetration_allowance: float = 0.005
    sdf_contact_weight: float = 1.2
    sdf_penetration_weight: float = 1.2
    sdf_target_margin: float = 0.003
    sdf_far_margin: float = 0.020
    outside_grasp_bonus_gate: float = 0.45
    outside_grasp_penalty: float = 4.0
    panel_penetration_weight: float = 5.0
    panel_penetration_scale: float = 200.0
    envelopment_weight: float = 1.5
    part_handle_reward_weight: float = 1.2
    part_non_interact_penalty_weight: float = 4.0
    part_handle_margin: float = 0.02
    part_non_interact_margin: float = 0.015


def _compute_part_aware_runtime_metrics(
    gym,
    task_spec: SingleDoorTaskSpec,
    env_i: int,
    handle_margin: float = 0.02,
    non_interact_margin: float = 0.015,
) -> Dict[str, Any]:
    default_fingertip = np.full((10,), 0.10, dtype=np.float32)
    default_metrics = {
        "handle_min_dist": 0.10,
        "non_interact_min_dist": 0.10,
        "non_interact_signed_min_dist": 0.10,
        "non_interact_penetration_depth": 0.0,
        "handle_attraction_score": 0.0,
        "non_interact_repulsion_penalty": 0.0,
        "handle_contact_ratio": 0.0,
        "non_interact_near_ratio": 0.0,
        "fingertip_handle_dists": default_fingertip,
        "fingertip_non_interact_dists": default_fingertip.copy(),
    }

    if not hasattr(gym, "contact_calc"):
        return default_metrics

    cc = gym.contact_calc
    handle_link_name = str(task_spec.handle_link_name)
    handle_links = [handle_link_name] if handle_link_name in cc.link_pcs else []
    non_interact_links = [name for name in cc.link_pcs.keys() if name != handle_link_name]

    if len(handle_links) == 0:
        return default_metrics

    try:
        hand_root_pos = gym.root_states[gym.mano_actor_idxs[env_i], :3].detach().to(dtype=torch.float32).unsqueeze(0)
        hand_root_rot = gym.root_states[gym.mano_actor_idxs[env_i], 3:7].detach().to(dtype=torch.float32).unsqueeze(0)
        hand_qpos = gym.dof_pos[env_i, : gym.mano_num_dofs, 0].detach().to(dtype=torch.float32).unsqueeze(0)
        obj_root_pos = torch.tensor(gym.arti_init_obj_pos_list[env_i], dtype=torch.float32, device=gym.device).unsqueeze(0)
        obj_root_rot = torch.tensor(gym.arti_init_obj_rot_list[env_i], dtype=torch.float32, device=gym.device).unsqueeze(0)
        obj_qpos = gym.dof_pos[
            env_i,
            gym.mano_num_dofs : gym.mano_num_dofs + gym.arti_obj_num_dofs,
            0,
        ].detach().to(dtype=torch.float32).unsqueeze(0)

        fingertip_world, fingertip_names = cc.compute_hand_joint_positions_world(
            hand_root_pos, hand_root_rot, hand_qpos, joint_names=cc.finger_joints
        )
        handle_points_world, _ = cc.compute_object_surface_points_world(
            obj_root_pos, obj_root_rot, obj_qpos, obj_link_filter=handle_links
        )
        handle_dists = torch.cdist(fingertip_world, handle_points_world).min(dim=-1).values.squeeze(0)

        if len(non_interact_links) > 0:
            non_interact_points_world, _ = cc.compute_object_surface_points_world(
                obj_root_pos, obj_root_rot, obj_qpos, obj_link_filter=non_interact_links
            )
            non_interact_dists = torch.cdist(fingertip_world, non_interact_points_world).min(dim=-1).values.squeeze(0)
            hand_surface_world, _ = cc._compute_hand_surface_points_world(hand_root_pos, hand_root_rot, hand_qpos)
            non_interact_signed_dists, _, _ = cc.compute_batch_signed_distance(
                hand_surface_world,
                obj_root_pos,
                obj_root_rot,
                obj_qpos,
                obj_link_filter=non_interact_links,
            )
            non_interact_signed_min = float(non_interact_signed_dists.min().item())
        else:
            non_interact_dists = torch.full_like(handle_dists, 0.10)
            non_interact_signed_min = 0.10

        handle_margin_t = float(handle_margin)
        non_interact_margin_t = float(non_interact_margin)
        handle_attraction_score = torch.exp(-torch.mean(torch.relu(handle_dists - handle_margin_t)))
        non_interact_repulsion_penalty = torch.mean(torch.relu(non_interact_margin_t - non_interact_dists).pow(2))

        handle_contact_ratio = torch.mean((handle_dists <= handle_margin_t).to(torch.float32))
        non_interact_near_ratio = torch.mean((non_interact_dists <= non_interact_margin_t).to(torch.float32))

        fingertip_handle_dists = handle_dists.detach().cpu().numpy().astype(np.float32)
        fingertip_non_interact_dists = non_interact_dists.detach().cpu().numpy().astype(np.float32)
        if fingertip_handle_dists.shape[0] < 10:
            pad = 10 - fingertip_handle_dists.shape[0]
            fingertip_handle_dists = np.pad(fingertip_handle_dists, (0, pad), constant_values=0.10)
            fingertip_non_interact_dists = np.pad(fingertip_non_interact_dists, (0, pad), constant_values=0.10)

        return {
            "handle_min_dist": float(handle_dists.min().item()),
            "non_interact_min_dist": float(non_interact_dists.min().item()),
            "non_interact_signed_min_dist": float(non_interact_signed_min),
            "non_interact_penetration_depth": float(max(0.0, -non_interact_signed_min)),
            "handle_attraction_score": float(handle_attraction_score.item()),
            "non_interact_repulsion_penalty": float(non_interact_repulsion_penalty.item()),
            "handle_contact_ratio": float(handle_contact_ratio.item()),
            "non_interact_near_ratio": float(non_interact_near_ratio.item()),
            "fingertip_handle_dists": fingertip_handle_dists[:10],
            "fingertip_non_interact_dists": fingertip_non_interact_dists[:10],
        }
    except Exception:
        return default_metrics


def select_single_door_task(asset_dir: str, door_index: int = 0, preferred_door_link: Optional[str] = None, preferred_handle_link: Optional[str] = None) -> SingleDoorTaskSpec:
    urdf_path = os.path.join(asset_dir, "mobility_annotation_gapartnet.urdf")
    anno_path = os.path.join(asset_dir, "link_annotation_gapartnet.json")
    meta_path = os.path.join(asset_dir, "meta.json")

    annos = _read_json(anno_path, [])
    if not annos:
        raise FileNotFoundError(f"Missing or empty link annotations: {anno_path}")

    meta = _read_json(meta_path, {})
    parent_map = _build_parent_map(urdf_path)

    handle_annos: List[Tuple[int, int, Dict[str, Any]]] = []
    door_annos: List[Tuple[int, int, Dict[str, Any]]] = []
    gapart_bbox_index = -1
    for anno_index, anno in enumerate(annos):
        if not anno.get("is_gapart", False):
            continue
        gapart_bbox_index += 1
        category = str(anno.get("category", "")).lower()
        if "handle" in category or "knob" in category:
            handle_annos.append((gapart_bbox_index, anno_index, anno))
        if "door" in category:
            door_annos.append((gapart_bbox_index, anno_index, anno))

    candidates: List[SingleDoorTaskSpec] = []
    unmatched_doors: List[str] = []
    for door_bbox_index, door_anno_index, door_anno in door_annos:
        door_link_name = door_anno.get("link_name")
        if door_link_name is None:
            continue
        if preferred_door_link is not None and door_link_name != preferred_door_link:
            continue

        joint_type, joint_origin_local, joint_axis_local, joint_name, joint_lower, joint_upper = parse_joint_info(urdf_path, door_link_name)
        if joint_type not in ["revolute", "continuous"]:
            continue

        handle_match, handle_match_strategy = _match_handle_to_door(
            door_link_name=door_link_name,
            handle_annos=handle_annos,
            annos=annos,
            parent_map=parent_map,
            urdf_path=urdf_path,
            door_joint_name=joint_name,
        )
        if handle_match is None:
            unmatched_doors.append(str(door_link_name))
            continue

        handle_bbox_index, handle_anno_index, handle_anno = handle_match
        handle_link_name = handle_anno.get("link_name")
        if handle_link_name is None:
            continue
        if preferred_handle_link is not None and handle_link_name != preferred_handle_link:
            continue

        handle_bbox = handle_anno.get("bbox", [])
        if len(handle_bbox) != 8:
            continue
        handle_geom = compute_handle_bbox_geometry(handle_bbox)

        open_sign = _determine_open_sign(joint_lower, joint_upper)
        open_limit = max(abs(float(joint_upper or 0.0)), abs(float(joint_lower or 0.0)))
        success_progress = float(np.clip(0.25 * max(open_limit, 1.0), 0.20, 0.60))

        candidates.append(
            SingleDoorTaskSpec(
                asset_dir=asset_dir,
                urdf_path=urdf_path,
                model_cat=str(meta.get("model_cat", "")),
                door_link_name=door_link_name,
                handle_link_name=handle_link_name,
                door_anno_index=int(door_anno_index),
                handle_anno_index=int(handle_anno_index),
                door_bbox_index=int(door_bbox_index),
                handle_bbox_index=int(handle_bbox_index),
                door_category=str(door_anno.get("category", "")),
                handle_category=str(handle_anno.get("category", "")),
                handle_match_strategy=str(handle_match_strategy),
                joint_name=str(joint_name),
                joint_type=str(joint_type),
                joint_origin_local=np.asarray(joint_origin_local, dtype=np.float32),
                joint_axis_local=np.asarray(joint_axis_local, dtype=np.float32),
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                open_sign=float(open_sign),
                handle_bbox_local=np.asarray(handle_geom["bbox"], dtype=np.float32),
                handle_center_local=np.asarray(handle_geom["center"], dtype=np.float32),
                handle_front_center_local=np.asarray(handle_geom["center_front_face"], dtype=np.float32),
                handle_out_local=np.asarray(handle_geom["out"], dtype=np.float32),
                handle_long_local=np.asarray(handle_geom["long"], dtype=np.float32),
                handle_short_local=np.asarray(handle_geom["short"], dtype=np.float32),
                handle_extent_local=np.asarray(handle_geom["extents"], dtype=np.float32),
                success_progress=success_progress,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No single-door revolute task candidate found under "
            f"{asset_dir} (doors={len(door_annos)} handles={len(handle_annos)} unmatched_doors={unmatched_doors})"
        )

    candidates = sorted(candidates, key=lambda item: (item.door_bbox_index, item.handle_bbox_index))
    selected_index = int(np.clip(door_index, 0, len(candidates) - 1))
    return candidates[selected_index]


def compute_contact_score(link_counts: Dict[str, int], target_points: int = 6) -> float:
    thumb = _clip_unit(link_counts.get("thumb3", 0) / float(target_points))
    index = _clip_unit(link_counts.get("index3", 0) / float(target_points))
    middle = _clip_unit(link_counts.get("middle3", 0) / float(target_points))
    ring = _clip_unit(link_counts.get("ring3", 0) / float(target_points))
    pinky = _clip_unit(link_counts.get("pinky3", 0) / float(target_points))
    palm = _clip_unit(link_counts.get("palm", 0) / float(2 * target_points))
    return float(0.30 * thumb + 0.22 * index + 0.18 * middle + 0.10 * ring + 0.05 * pinky + 0.15 * palm)


def compute_opposition_score(link_counts: Dict[str, int], target_points: int = 6) -> float:
    thumb = _clip_unit(link_counts.get("thumb3", 0) / float(target_points))
    four_fingers = max(
        _clip_unit(link_counts.get("index3", 0) / float(target_points)),
        _clip_unit(link_counts.get("middle3", 0) / float(target_points)),
        _clip_unit(link_counts.get("ring3", 0) / float(target_points)),
    )
    return float(min(thumb, four_fingers))


def compute_force_closure_score(
    link_counts: Dict[str, int],
    target_points: int = 6,
    handle_out_alignment: float = 1.0,
    tangent_alignment: float = 1.0,
) -> float:
    thumb = _clip_unit(link_counts.get("thumb3", 0) / float(target_points))
    index = _clip_unit(link_counts.get("index3", 0) / float(target_points))
    middle = _clip_unit(link_counts.get("middle3", 0) / float(target_points))
    ring = _clip_unit(link_counts.get("ring3", 0) / float(target_points))
    pinky = _clip_unit(link_counts.get("pinky3", 0) / float(target_points))
    finger_wall = max(index, middle, ring)
    finger_support = _clip_unit((index + middle + ring + pinky) / 2.5)
    closure_contact = min(thumb, finger_wall)
    # Breadth bonus: reward having multiple fingers engaged rather than just one strong contact
    engaged_fingers = sum(1.0 for f in [index, middle, ring, pinky] if f > 0.15)
    breadth_bonus = _clip_unit(engaged_fingers / 3.0)
    alignment_term = _clip_unit(0.65 * handle_out_alignment + 0.35 * tangent_alignment)
    return float(closure_contact * (0.45 + 0.35 * finger_support + 0.20 * breadth_bonus) * alignment_term)


def compute_sdf_contact_score(min_dist: float, target_margin: float = 0.002, far_margin: float = 0.015) -> float:
    if not np.isfinite(min_dist):
        return 0.0
    if min_dist < 0.0:
        return float(max(0.0, 1.0 - abs(min_dist) / max(target_margin, 1e-6)))
    if min_dist <= target_margin:
        return 1.0
    if min_dist >= far_margin:
        return 0.0
    return float(1.0 - (min_dist - target_margin) / max(far_margin - target_margin, 1e-6))


def compute_envelopment_score(link_counts: Dict[str, int], target_points: int = 6) -> float:
    """Reward full hand envelopment: bonus when all five fingers + palm wrap around the handle."""
    thumb = _clip_unit(link_counts.get("thumb3", 0) / float(target_points))
    index = _clip_unit(link_counts.get("index3", 0) / float(target_points))
    middle = _clip_unit(link_counts.get("middle3", 0) / float(target_points))
    ring = _clip_unit(link_counts.get("ring3", 0) / float(target_points))
    pinky = _clip_unit(link_counts.get("pinky3", 0) / float(target_points))
    # Geometric mean rewards having all fingers in contact rather than
    # a few fingers with many points.  A single missing finger zeroes
    # out the score, strongly encouraging full wrap-around.
    contacts = [thumb, index, middle, ring, pinky]
    min_contact = min(contacts)
    mean_contact = float(np.mean(contacts))
    # Blend: mostly driven by the weakest link (min) with a mean bonus
    return float(0.7 * min_contact + 0.3 * mean_contact)


def compute_outside_grasp_score(
    link_counts: Dict[str, int],
    force_closure_reward: float,
    palm_contact_cap: float = 0.25,
    target_points: int = 6,
) -> float:
    thumb = _clip_unit(link_counts.get("thumb3", 0) / float(target_points))
    index = _clip_unit(link_counts.get("index3", 0) / float(target_points))
    middle = _clip_unit(link_counts.get("middle3", 0) / float(target_points))
    palm = _clip_unit(link_counts.get("palm", 0) / float(2 * target_points))
    finger_wall = _clip_unit(0.5 * (index + middle))
    palm_suppression = _clip_unit(1.0 - palm / max(palm_contact_cap, 1e-6))
    return float(min(thumb, finger_wall) * (0.5 + 0.5 * force_closure_reward) * palm_suppression)


def build_contact_feature_vector(link_counts: Dict[str, int], target_points: int = 6) -> np.ndarray:
    feats = []
    for link_name in CONTACT_LINK_ORDER:
        denom = float(2 * target_points) if link_name == "palm" else float(target_points)
        feats.append(_clip_unit(link_counts.get(link_name, 0) / max(denom, 1.0)))
    return np.asarray(feats, dtype=np.float32)


def get_phase_contact_target(phase_name: str) -> np.ndarray:
    if phase_name == "approach":
        return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if phase_name == "touch":
        return np.asarray([0.35, 0.25, 0.20, 0.10, 0.05, 0.05], dtype=np.float32)
    if phase_name == "grasp":
        return np.asarray([1.0, 0.95, 0.90, 0.75, 0.55, 0.05], dtype=np.float32)
    if phase_name == "actuate":
        return np.asarray([1.0, 0.95, 0.90, 0.80, 0.60, 0.05], dtype=np.float32)
    if phase_name == "success":
        return np.asarray([0.85, 0.80, 0.75, 0.70, 0.50, 0.05], dtype=np.float32)
    return np.zeros(6, dtype=np.float32)


def infer_single_door_phase(
    progress: float,
    progress_delta: float,
    contact_score: float,
    opposition_score: float,
    force_closure_score: float,
    stable_contact: bool,
    success_progress: float,
) -> str:
    if progress >= float(success_progress):
        return "success"
    if (
        max(opposition_score, force_closure_score) >= 0.40 and progress_delta > 5e-4
    ) or (stable_contact and progress_delta > 2e-4):
        return "actuate"
    if stable_contact or force_closure_score >= 0.35 or opposition_score >= 0.40 or contact_score >= 0.55:
        return "grasp"
    if contact_score >= 0.20:
        return "touch"
    return "approach"


def annotate_single_door_records(
    records_list: List[Dict[str, Any]],
    task_spec: SingleDoorTaskSpec,
    obj_world_pos: Sequence[float],
    obj_world_rot: Sequence[float],
    obj_scale: float = 1.0,
    contact_target_points: int = 6,
    min_contact_points: int = 30,
    success_progress: Optional[float] = None,
) -> Dict[str, Any]:
    if len(records_list) == 0:
        return {"num_records": 0}

    success_threshold = float(task_spec.success_progress if success_progress is None else success_progress)
    world_geom = task_spec.world_geometry(obj_world_pos=obj_world_pos, obj_world_rot=obj_world_rot, obj_scale=obj_scale)

    first_record = records_list[0]
    first_record["single_door_task_spec"] = {
        **task_spec.to_dict(),
        "world_geometry": _jsonify(world_geom),
    }

    max_progress = -1e9
    max_contact = -1
    phase_histogram = {name: 0 for name in PHASE_TO_ID.keys()}
    prev_progress = 0.0

    for index, record in enumerate(records_list):
        drive_val = record.get("drive_dof_val")
        if drive_val is None:
            drive_idx = record.get("drive_dof_index")
            obj_dof = record.get("obj_dof", [])
            if drive_idx is not None and 0 <= int(drive_idx) < len(obj_dof):
                drive_val = float(obj_dof[int(drive_idx)])
            else:
                drive_val = 0.0
        progress = float(task_spec.open_sign * float(drive_val))
        progress_delta = 0.0 if index == 0 else float(progress - prev_progress)
        prev_progress = progress

        link_counts = {str(k): int(v) for k, v in record.get("surface_contact_link_counts", {}).items()}
        contact_score = compute_contact_score(link_counts, target_points=contact_target_points)
        opposition_score = compute_opposition_score(link_counts, target_points=contact_target_points)
        force_closure_score = compute_force_closure_score(link_counts, target_points=contact_target_points)
        stable_contact = bool(record.get("surface_contact_stable", False))
        if not stable_contact:
            stable_contact = bool(record.get("surface_contact_count", 0) >= min_contact_points and opposition_score >= 0.60)

        phase = infer_single_door_phase(
            progress=progress,
            progress_delta=progress_delta,
            contact_score=contact_score,
            opposition_score=opposition_score,
            force_closure_score=force_closure_score,
            stable_contact=stable_contact,
            success_progress=success_threshold,
        )

        phase_histogram[phase] += 1
        max_progress = max(max_progress, progress)
        max_contact = max(max_contact, int(record.get("surface_contact_count", 0)))

        record["single_door_progress"] = float(progress)
        record["single_door_progress_delta"] = float(progress_delta)
        record["single_door_contact_score"] = float(contact_score)
        record["single_door_opposition_score"] = float(opposition_score)
        record["single_door_force_closure_score"] = float(force_closure_score)
        record["single_door_phase"] = phase
        record["single_door_phase_id"] = int(PHASE_TO_ID[phase])
        record["single_door_success"] = bool(progress >= success_threshold)

    return {
        "num_records": len(records_list),
        "max_progress": float(max_progress),
        "max_contact_count": int(max_contact),
        "success": bool(max_progress >= success_threshold),
        "phase_histogram": phase_histogram,
    }


def extract_single_door_runtime_state(
    gym,
    task_spec: SingleDoorTaskSpec,
    env_i: int = 0,
    surface_contact_thresh: float = 0.015,
    min_contact_points: int = 30,
    contact_target_points: int = 6,
    part_handle_margin: float = 0.02,
    part_non_interact_margin: float = 0.015,
) -> SingleDoorRuntimeState:
    if hasattr(gym, "_ensure_contact_calc"):
        gym._ensure_contact_calc(obj_urdf_path=task_spec.urdf_path)

    mano_idx = gym.mano_actor_idxs[env_i]
    root_state = gym.root_states[mano_idx]

    hand_pos = root_state[:3].detach().cpu().numpy().astype(np.float32)
    hand_rot = root_state[3:7].detach().cpu().numpy().astype(np.float32)
    hand_lin_vel = root_state[7:10].detach().cpu().numpy().astype(np.float32)
    hand_ang_vel = root_state[10:13].detach().cpu().numpy().astype(np.float32)
    hand_qpos = gym.dof_pos[env_i, : gym.mano_num_dofs, 0].detach().cpu().numpy().astype(np.float32)
    hand_qvel = gym.dof_vel[env_i, : gym.mano_num_dofs, 0].detach().cpu().numpy().astype(np.float32)
    obj_qpos = gym.dof_pos[env_i, gym.mano_num_dofs : gym.mano_num_dofs + gym.arti_obj_num_dofs, 0].detach().cpu().numpy().astype(np.float32)
    obj_qvel = gym.dof_vel[env_i, gym.mano_num_dofs : gym.mano_num_dofs + gym.arti_obj_num_dofs, 0].detach().cpu().numpy().astype(np.float32)

    drive_dof_index = gym.arti_obj_dof_dict.get(task_spec.joint_name, None)
    if drive_dof_index is None:
        drive_dof_val = 0.0
        drive_dof_vel = 0.0
    else:
        drive_dof_val = float(obj_qpos[int(drive_dof_index)])
        drive_dof_vel = float(obj_qvel[int(drive_dof_index)])

    hand_pose_6d = np.concatenate([hand_pos, hand_rot], axis=0)
    contact_count, link_counts, min_dist = gym._compute_surface_contact_summary(
        hand_pose_6d=hand_pose_6d,
        obj_urdf_path=task_spec.urdf_path,
        surface_contact_thresh=surface_contact_thresh,
    )
    opposition_score = compute_opposition_score(link_counts, target_points=contact_target_points)
    force_closure_score = compute_force_closure_score(link_counts, target_points=contact_target_points)
    stable_contact = bool(contact_count >= int(min_contact_points) and opposition_score >= 0.60)

    obj_world_pos = gym.arti_init_obj_pos_list[env_i]
    obj_world_rot = gym.arti_init_obj_rot_list[env_i]
    obj_scale = gym.cfgs.get("asset", {}).get("arti_obj_scale", 1.0)
    world_geom = task_spec.world_geometry(obj_world_pos=obj_world_pos, obj_world_rot=obj_world_rot, obj_scale=obj_scale)
    part_metrics = _compute_part_aware_runtime_metrics(
        gym=gym,
        task_spec=task_spec,
        env_i=env_i,
        handle_margin=part_handle_margin,
        non_interact_margin=part_non_interact_margin,
    )

    return SingleDoorRuntimeState(
        hand_pos=hand_pos,
        hand_rot=hand_rot,
        hand_lin_vel=hand_lin_vel,
        hand_ang_vel=hand_ang_vel,
        hand_qpos=hand_qpos,
        hand_qvel=hand_qvel,
        obj_qpos=obj_qpos,
        obj_qvel=obj_qvel,
        drive_dof_index=None if drive_dof_index is None else int(drive_dof_index),
        drive_dof_val=float(drive_dof_val),
        drive_dof_vel=float(drive_dof_vel),
        progress=float(task_spec.open_sign * drive_dof_val),
        surface_contact_count=int(contact_count),
        surface_contact_min_dist=float(min_dist),
        surface_contact_link_counts={str(k): int(v) for k, v in link_counts.items()},
        surface_contact_stable=stable_contact,
        handle_min_dist=float(part_metrics["handle_min_dist"]),
        non_interact_min_dist=float(part_metrics["non_interact_min_dist"]),
        non_interact_signed_min_dist=float(part_metrics["non_interact_signed_min_dist"]),
        non_interact_penetration_depth=float(part_metrics["non_interact_penetration_depth"]),
        handle_attraction_score=float(part_metrics["handle_attraction_score"]),
        non_interact_repulsion_penalty=float(part_metrics["non_interact_repulsion_penalty"]),
        handle_contact_ratio=float(part_metrics["handle_contact_ratio"]),
        non_interact_near_ratio=float(part_metrics["non_interact_near_ratio"]),
        fingertip_handle_dists=np.asarray(part_metrics["fingertip_handle_dists"], dtype=np.float32),
        fingertip_non_interact_dists=np.asarray(part_metrics["fingertip_non_interact_dists"], dtype=np.float32),
        hinge_origin_world=world_geom["hinge_origin_world"],
        hinge_axis_world=world_geom["hinge_axis_world"],
        handle_center_world=world_geom["handle_center_world"],
        handle_front_center_world=world_geom["handle_front_center_world"],
        handle_out_world=world_geom["handle_out_world"],
        handle_long_world=world_geom["handle_long_world"],
        handle_short_world=world_geom["handle_short_world"],
        open_tangent_world=world_geom["open_tangent_world"],
    )


def build_single_door_observation(state: SingleDoorRuntimeState, prev_action: Optional[Sequence[float]] = None, contact_target_points: int = 6) -> np.ndarray:
    handle_frame = np.stack(
        [
            state.handle_long_world,
            state.handle_short_world,
            -state.handle_out_world,
        ],
        axis=1,
    )
    hand_rot_mat = R.from_quat(state.hand_rot).as_matrix().astype(np.float32)
    rel_rot = handle_frame.T @ hand_rot_mat
    rel_rot_6d = rel_rot[:, :2].reshape(-1)

    pos_rel = handle_frame.T @ (state.hand_pos - state.handle_front_center_world)
    lin_vel_rel = handle_frame.T @ state.hand_lin_vel
    ang_vel_rel = handle_frame.T @ state.hand_ang_vel
    hinge_rel = handle_frame.T @ (state.hinge_origin_world - state.handle_front_center_world)

    contact_feats = np.array(
        [
            _clip_unit(state.surface_contact_link_counts.get("thumb3", 0) / float(contact_target_points)),
            _clip_unit(state.surface_contact_link_counts.get("index3", 0) / float(contact_target_points)),
            _clip_unit(state.surface_contact_link_counts.get("middle3", 0) / float(contact_target_points)),
            _clip_unit(state.surface_contact_link_counts.get("ring3", 0) / float(contact_target_points)),
            _clip_unit(state.surface_contact_link_counts.get("pinky3", 0) / float(contact_target_points)),
            _clip_unit(state.surface_contact_link_counts.get("palm", 0) / float(2 * contact_target_points)),
            compute_contact_score(state.surface_contact_link_counts, target_points=contact_target_points),
            compute_opposition_score(state.surface_contact_link_counts, target_points=contact_target_points),
            float(state.surface_contact_stable),
        ],
        dtype=np.float32,
    )

    obj_feats = np.array(
        [
            float(state.progress),
            float(state.drive_dof_vel),
            float(state.surface_contact_count) / 100.0,
            float(state.surface_contact_min_dist),
            float(state.handle_min_dist),
            float(state.non_interact_min_dist),
            float(state.non_interact_signed_min_dist),
            float(state.non_interact_penetration_depth),
            float(state.handle_attraction_score),
            float(state.non_interact_repulsion_penalty),
            float(state.handle_contact_ratio),
            float(state.non_interact_near_ratio),
        ],
        dtype=np.float32,
    )
    part_feats = np.concatenate(
        [
            np.asarray(state.fingertip_handle_dists, dtype=np.float32).reshape(-1),
            np.asarray(state.fingertip_non_interact_dists, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )

    prev_action_arr = np.zeros(0, dtype=np.float32) if prev_action is None else np.asarray(prev_action, dtype=np.float32).reshape(-1)

    return np.concatenate(
        [
            pos_rel.astype(np.float32),
            lin_vel_rel.astype(np.float32),
            ang_vel_rel.astype(np.float32),
            hinge_rel.astype(np.float32),
            rel_rot_6d.astype(np.float32),
            obj_feats,
            contact_feats,
            part_feats,
            prev_action_arr,
        ],
        axis=0,
    )


def compute_single_door_reward(
    state: SingleDoorRuntimeState,
    prev_state: Optional[SingleDoorRuntimeState] = None,
    action: Optional[Sequence[float]] = None,
    prev_action: Optional[Sequence[float]] = None,
    config: Optional[SingleDoorRewardConfig] = None,
) -> Dict[str, float]:
    cfg = SingleDoorRewardConfig() if config is None else config

    thumb_contact = float(state.surface_contact_link_counts.get("thumb3", 0))
    finger_contacts = np.asarray(
        [
            float(state.surface_contact_link_counts.get("index3", 0)),
            float(state.surface_contact_link_counts.get("middle3", 0)),
            float(state.surface_contact_link_counts.get("ring3", 0)),
            float(state.surface_contact_link_counts.get("pinky3", 0)),
        ],
        dtype=np.float32,
    )
    strong_finger_count = float(np.sum(finger_contacts >= 2.0))
    top2_fingers = float(np.mean(np.sort(finger_contacts)[-2:])) if finger_contacts.size > 0 else 0.0
    pinch_balance = _clip_unit(min(thumb_contact, top2_fingers) / max(1.0, float(cfg.contact_target_points)))
    pinch_reward = float(0.5 * _clip_unit(strong_finger_count / 2.0) + 0.5 * pinch_balance)
    pinch_gate = _clip_unit((pinch_reward - cfg.pinch_gate_threshold) / max(1e-6, 1.0 - cfg.pinch_gate_threshold))

    reach_dist = float(np.linalg.norm(state.hand_pos - state.handle_front_center_world))
    raw_reach_reward = float(np.exp(-cfg.reach_scale * reach_dist))
    reach_reward = float(raw_reach_reward * (0.2 + 0.8 * pinch_gate))

    hand_rot_mat = R.from_quat(state.hand_rot).as_matrix().astype(np.float32)
    align_long = 0.5 * (1.0 + float(np.dot(hand_rot_mat[:, 0], state.handle_long_world)))
    align_out = 0.5 * (1.0 + float(np.dot(hand_rot_mat[:, 2], -state.handle_out_world)))
    raw_align_reward = float(0.5 * (align_long + align_out))
    align_reward = float(raw_align_reward * (0.25 + 0.75 * pinch_gate))

    contact_reward = compute_contact_score(state.surface_contact_link_counts, target_points=cfg.contact_target_points)
    opposition_reward = compute_opposition_score(state.surface_contact_link_counts, target_points=cfg.contact_target_points)
    tangent_axis_alignment = 0.5 * (1.0 + float(np.dot(hand_rot_mat[:, 1], state.open_tangent_world)))
    force_closure_reward = compute_force_closure_score(
        state.surface_contact_link_counts,
        target_points=cfg.contact_target_points,
        handle_out_alignment=align_out,
        tangent_alignment=tangent_axis_alignment,
    )
    sdf_contact_reward = compute_sdf_contact_score(
        state.surface_contact_min_dist,
        target_margin=cfg.sdf_target_margin,
        far_margin=cfg.sdf_far_margin,
    )
    envelopment_reward = compute_envelopment_score(
        state.surface_contact_link_counts,
        target_points=cfg.contact_target_points,
    )
    outside_grasp_score = compute_outside_grasp_score(
        state.surface_contact_link_counts,
        force_closure_reward=force_closure_reward,
        target_points=cfg.contact_target_points,
    )
    force_closure_gate = _clip_unit(
        (force_closure_reward - cfg.force_closure_gate_threshold) / max(1e-6, 1.0 - cfg.force_closure_gate_threshold)
    )
    hold_reward = float((0.5 * opposition_reward + 0.5 * pinch_reward) if state.surface_contact_stable else 0.0)
    part_handle_reward = float(state.handle_attraction_score * (0.25 + 0.75 * pinch_gate))
    part_non_interact_penalty = float(
        state.non_interact_repulsion_penalty + 2.0 * state.non_interact_penetration_depth
    )

    progress_delta = 0.0 if prev_state is None else float(state.progress - prev_state.progress)
    progress_gate = _clip_unit(
        (
            max(opposition_reward, pinch_reward, force_closure_reward) - cfg.progress_gate_threshold
        ) / max(1e-6, 1.0 - cfg.progress_gate_threshold)
    )
    progress_reward = float(max(0.0, progress_delta) * progress_gate)

    tangent_speed = float(np.dot(state.hand_lin_vel, state.open_tangent_world))
    tangent_reward = float((max(0.0, tangent_speed) / max(cfg.tangent_scale, 1e-6)) * progress_gate)

    detach_penalty = 0.0
    if prev_state is not None and prev_state.surface_contact_stable and not state.surface_contact_stable:
        detach_penalty = 1.0
    if pinch_reward < 0.15 and reach_dist < 0.05:
        detach_penalty += 0.5
    if state.handle_min_dist > float(cfg.part_handle_margin) and state.non_interact_near_ratio > 0.25:
        detach_penalty += 0.5
    # --- Penetration penalties (surface_contact_min_dist < 0 means inside object) ---
    penetration_depth = float(max(0.0, -state.surface_contact_min_dist))
    # Mild penalty that ramps linearly with penetration depth (original behaviour,
    # but now actually fires because min_dist is signed).
    penetration_penalty = 0.0
    if state.surface_contact_min_dist < -0.001:
        penetration_penalty = float(min(0.05, penetration_depth))
        # Nullify any progress/tangent reward obtained via cheating through the panel
        if progress_delta > 0.0:
            progress_reward = 0.0
            tangent_reward = 0.0
    # Heavy flat penalty: a large negative hit whenever any hand point is inside
    # the door panel.  This teaches the agent that "phasing through" is never
    # worth the progress it might gain.  The penalty is proportional to depth
    # but with a steep scale and a guaranteed floor of 1.0 so even a tiny
    # penetration is strongly punished.
    panel_penetration_penalty = 0.0
    if penetration_depth > 0.0:
        panel_penetration_penalty = float(1.0 + cfg.panel_penetration_scale * penetration_depth)
    # Continuous SDF-based repulsion (softer gradient for the optimiser)
    sdf_penetration_penalty = penetration_depth
    palm_contact = float(state.surface_contact_link_counts.get("palm", 0))
    fingertip_contact = float(
        state.surface_contact_link_counts.get("thumb3", 0)
        + state.surface_contact_link_counts.get("index3", 0)
        + state.surface_contact_link_counts.get("middle3", 0)
    )
    palm_penalty = 0.0
    if palm_contact > 0 and state.surface_contact_min_dist < -float(cfg.fingertip_penetration_allowance):
        palm_ratio = palm_contact / max(1.0, fingertip_contact + palm_contact)
        palm_penalty = float(min(0.05, palm_ratio * penetration_depth))

    action_l2 = 0.0 if action is None else float(np.mean(np.square(np.asarray(action, dtype=np.float32))))
    action_smooth = 0.0
    if action is not None and prev_action is not None:
        action_arr = np.asarray(action, dtype=np.float32)
        prev_action_arr = np.asarray(prev_action, dtype=np.float32)
        if action_arr.shape == prev_action_arr.shape:
            action_smooth = float(np.mean(np.square(action_arr - prev_action_arr)))

    success = bool(state.progress >= cfg.success_progress)
    outside_grasp_ok = bool(outside_grasp_score >= cfg.outside_grasp_bonus_gate)
    total = (
        cfg.reach_weight * reach_reward
        + cfg.align_weight * align_reward
        + cfg.contact_weight * contact_reward
        + cfg.sdf_contact_weight * sdf_contact_reward
        + cfg.opposition_weight * opposition_reward
        + cfg.force_closure_weight * force_closure_reward
        + cfg.envelopment_weight * envelopment_reward
        + cfg.hold_weight * hold_reward
        + cfg.part_handle_reward_weight * part_handle_reward
        + 0.75 * pinch_reward
        + cfg.progress_weight * progress_reward
        + cfg.tangent_weight * tangent_reward
        - cfg.detach_penalty_weight * detach_penalty
        - cfg.penetration_progress_penalty * penetration_penalty
        - cfg.sdf_penetration_weight * sdf_penetration_penalty
        - cfg.panel_penetration_weight * panel_penetration_penalty
        - cfg.palm_penetration_penalty * palm_penalty
        - cfg.part_non_interact_penalty_weight * part_non_interact_penalty
        - (0.0 if outside_grasp_ok else cfg.outside_grasp_penalty * max(0.0, state.progress))
        - cfg.action_l2_weight * action_l2
        - cfg.action_smooth_weight * action_smooth
        + (cfg.success_bonus if (success and outside_grasp_ok) else 0.0)
    )

    return {
        "total": float(total),
        "reach": float(reach_reward),
        "raw_reach": float(raw_reach_reward),
        "align": float(align_reward),
        "raw_align": float(raw_align_reward),
        "contact": float(contact_reward),
        "sdf_contact": float(sdf_contact_reward),
        "opposition": float(opposition_reward),
        "force_closure": float(force_closure_reward),
        "force_closure_gate": float(force_closure_gate),
        "envelopment": float(envelopment_reward),
        "outside_grasp": float(outside_grasp_score),
        "outside_grasp_ok": float(outside_grasp_ok),
        "hold": float(hold_reward),
        "part_handle": float(part_handle_reward),
        "part_non_interact_penalty": float(part_non_interact_penalty),
        "handle_min_dist": float(state.handle_min_dist),
        "non_interact_min_dist": float(state.non_interact_min_dist),
        "non_interact_penetration_depth": float(state.non_interact_penetration_depth),
        "handle_contact_ratio": float(state.handle_contact_ratio),
        "non_interact_near_ratio": float(state.non_interact_near_ratio),
        "pinch": float(pinch_reward),
        "pinch_gate": float(pinch_gate),
        "progress_gate": float(progress_gate),
        "progress": float(progress_reward),
        "tangent": float(tangent_reward),
        "detach_penalty": float(detach_penalty),
        "penetration_penalty": float(penetration_penalty),
        "sdf_penetration_penalty": float(sdf_penetration_penalty),
        "panel_penetration_penalty": float(panel_penetration_penalty),
        "palm_penalty": float(palm_penalty),
        "action_l2_penalty": float(action_l2),
        "action_smooth_penalty": float(action_smooth),
        "success": float(success),
        "success_bonus_active": float(success and outside_grasp_ok),
    }

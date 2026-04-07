"""NPCS-based handle localization for the RL pipeline.

Ports the NPCS (Normalized Part Coordinate Space) pipeline from the GAPartNet
rendering tools into a form usable by SingleDoorResidualEnv.  Given an object's
link annotations, semantic segmentation mask, and oriented bounding boxes, this
module computes:

  1. Per-link RTS (Rotation, Translation, Scale) parameters that map between
     world space and the canonical NPCS frame.
  2. Dense handle surface point clouds in world space, filtered by semantic
     part category.
  3. Compact handle geometry features (center, PCA long axis, outward normal)
     derived from the NPCS-localized point set.

All heavy math uses pure PyTorch (no scipy) so it can run inside the Isaac Gym
GPU loop without host round-trips.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# GAPartNet semantic part IDs that correspond to graspable handles / buttons.
HANDLE_PART_IDS = {1, 3, 9}  # line_fixed_handle, slider_button, revolute_handle

PART_ID2NAME = {
    0: "others",
    1: "line_fixed_handle",
    2: "round_fixed_handle",
    3: "slider_button",
    4: "hinge_door",
    5: "slider_drawer",
    6: "slider_lid",
    7: "hinge_lid",
    8: "hinge_knob",
    9: "revolute_handle",
}


# ---------------------------------------------------------------------------
# RTS computation (mirrors get_NPCS_map_from_oriented_bbox from pose_utils.py)
# ---------------------------------------------------------------------------

def _compute_rotation_matrix_np(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Rotation that best aligns canonical bbox *b1* to scaled bbox *b2* (SVD)."""
    c1 = np.mean(b1, axis=0)
    c2 = np.mean(b2, axis=0)
    H = (b1 - c1).T @ (b2 - c2)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        R[0, :] *= -1
    return R.T


def compute_link_rts(bbox: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute NPCS RTS parameters from an 8-corner oriented bounding box.

    Parameters
    ----------
    bbox : (8, 3) array
        Oriented bounding-box corners in world coordinates, following the
        GAPartNet convention.

    Returns
    -------
    dict with keys ``R`` (3×3), ``T`` (3,), ``S`` (3,), ``scaler`` (float).
    """
    bbox = np.asarray(bbox, dtype=np.float64).reshape(8, 3)
    T = bbox.mean(axis=0)
    s_x = np.linalg.norm(bbox[1] - bbox[0])
    s_y = np.linalg.norm(bbox[1] - bbox[2])
    s_z = np.linalg.norm(bbox[0] - bbox[4])
    S = np.array([s_x, s_y, s_z])
    scaler = np.linalg.norm(S)
    bbox_scaled = (bbox - T) / scaler
    bbox_canon = np.array([
        [-s_x / 2,  s_y / 2,  s_z / 2],
        [ s_x / 2,  s_y / 2,  s_z / 2],
        [ s_x / 2, -s_y / 2,  s_z / 2],
        [-s_x / 2, -s_y / 2,  s_z / 2],
        [-s_x / 2,  s_y / 2, -s_z / 2],
        [ s_x / 2,  s_y / 2, -s_z / 2],
        [ s_x / 2, -s_y / 2, -s_z / 2],
        [-s_x / 2, -s_y / 2, -s_z / 2],
    ]) / scaler
    R = _compute_rotation_matrix_np(bbox_canon, bbox_scaled)
    return {
        "R": R.astype(np.float32),
        "T": T.astype(np.float32),
        "S": S.astype(np.float32),
        "scaler": float(scaler),
    }


# ---------------------------------------------------------------------------
# Handle point cloud extraction from link annotations + semantic masks
# ---------------------------------------------------------------------------

@dataclass
class HandleLocalization:
    """Result container for NPCS-based handle localization."""

    handle_points_world: torch.Tensor       # (N, 3) handle surface points
    handle_center: torch.Tensor             # (3,)
    handle_long_axis: torch.Tensor          # (3,) PCA first component
    handle_normal: torch.Tensor             # (3,) outward normal
    handle_rts: Dict[str, np.ndarray]       # RTS parameters
    handle_link_name: str
    handle_category: str
    semantic_part_id: int


def load_link_annotations(asset_dir: str) -> List[Dict[str, Any]]:
    """Load GAPartNet link_annotation_gapartnet.json."""
    anno_path = os.path.join(asset_dir, "link_annotation_gapartnet.json")
    if not os.path.exists(anno_path):
        return []
    with open(anno_path, "r") as f:
        return json.load(f)


def extract_handle_annotations(
    annos: List[Dict[str, Any]],
    handle_categories: Sequence[str] = (
        "line_fixed_handle",
        "round_fixed_handle",
        "hinge_knob",
        "revolute_handle",
    ),
) -> List[Dict[str, Any]]:
    """Filter annotations to only handle/knob parts."""
    cats_lower = {c.lower() for c in handle_categories}
    handles = []
    for anno in annos:
        if not anno.get("is_gapart", False):
            continue
        cat = str(anno.get("category", "")).lower()
        if cat in cats_lower or "handle" in cat or "knob" in cat:
            handles.append(anno)
    return handles


def bbox_to_surface_points(
    bbox: np.ndarray,
    num_points: int = 1500,
) -> np.ndarray:
    """Sample surface points on a box defined by 8 corners.

    Uses the RTS decomposition to generate uniformly distributed points
    on all 6 faces of the oriented bounding box.
    """
    bbox = np.asarray(bbox, dtype=np.float32).reshape(8, 3)
    rts = compute_link_rts(bbox)

    # Generate points on the 6 faces of the unit cube [-0.5, 0.5]^3
    points_per_face = max(num_points // 6, 10)
    faces = []
    for axis in range(3):
        for sign in [-0.5, 0.5]:
            pts = np.random.uniform(-0.5, 0.5, size=(points_per_face, 3)).astype(np.float32)
            pts[:, axis] = sign
            faces.append(pts)
    surface_npcs = np.concatenate(faces, axis=0)

    # Map from NPCS to world space: world = npcs @ R * scaler + T
    surface_world = surface_npcs @ rts["R"] * rts["scaler"] + rts["T"]

    # Subsample to exactly num_points
    if surface_world.shape[0] > num_points:
        idx = np.random.choice(surface_world.shape[0], num_points, replace=False)
        surface_world = surface_world[idx]

    return surface_world.astype(np.float32)


def localize_handle_from_annotations(
    asset_dir: str,
    target_handle_link: Optional[str] = None,
    num_points: int = 1500,
    device: str = "cuda",
) -> Optional[HandleLocalization]:
    """Localize the handle using GAPartNet annotations + NPCS geometry.

    Parameters
    ----------
    asset_dir : str
        Path to the GAPartNet object directory containing
        ``link_annotation_gapartnet.json``.
    target_handle_link : str, optional
        If given, only consider this specific link.  Otherwise the first
        handle annotation is used.
    num_points : int
        Number of surface points to sample on the handle bbox.
    device : str
        Torch device for the output tensors.

    Returns
    -------
    HandleLocalization or None if no handle found.
    """
    annos = load_link_annotations(asset_dir)
    handle_annos = extract_handle_annotations(annos)
    if not handle_annos:
        return None

    # Pick the target handle
    chosen = None
    if target_handle_link is not None:
        for ha in handle_annos:
            if ha.get("link_name") == target_handle_link:
                chosen = ha
                break
    if chosen is None:
        chosen = handle_annos[0]

    bbox = chosen.get("bbox", [])
    if len(bbox) != 8:
        return None

    bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(8, 3)
    rts = compute_link_rts(bbox_arr)
    surface_world = bbox_to_surface_points(bbox_arr, num_points=num_points)

    pts_t = torch.tensor(surface_world, dtype=torch.float32, device=device)
    center = pts_t.mean(dim=0)

    # PCA for long axis and normal
    centered = pts_t - center
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Eigenvalues are ascending; last = largest variance = long axis
    long_axis = eigvecs[:, -1]
    normal = eigvecs[:, 0]  # smallest variance = outward normal

    # Determine semantic part ID from category
    cat = str(chosen.get("category", "")).lower()
    sem_id = 0
    for pid, pname in PART_ID2NAME.items():
        if pname == cat:
            sem_id = pid
            break

    return HandleLocalization(
        handle_points_world=pts_t,
        handle_center=center,
        handle_long_axis=long_axis / (torch.norm(long_axis) + 1e-8),
        handle_normal=normal / (torch.norm(normal) + 1e-8),
        handle_rts=rts,
        handle_link_name=str(chosen.get("link_name", "")),
        handle_category=cat,
        semantic_part_id=sem_id,
    )


# ---------------------------------------------------------------------------
# Semantic-mask–based handle extraction from Isaac Gym point clouds
# ---------------------------------------------------------------------------

def extract_handle_points_from_semantic_pc(
    object_points: torch.Tensor,
    semantic_mask: torch.Tensor,
    handle_part_ids: Sequence[int] = (1, 3, 9),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract handle and non-interactive point clouds using semantic masks.

    Parameters
    ----------
    object_points : (N, 3) tensor
        Full object surface point cloud.
    semantic_mask : (N,) integer tensor
        Per-point GAPartNet semantic part ID.
    handle_part_ids : sequence of int
        Part IDs that count as graspable handles.

    Returns
    -------
    handle_points : (M, 3) tensor of handle surface points
    non_interact_points : (K, 3) tensor of non-handle object points
    handle_mask : (N,) bool tensor
    """
    ids_t = torch.tensor(list(handle_part_ids), dtype=semantic_mask.dtype,
                         device=semantic_mask.device)
    handle_mask = (semantic_mask.unsqueeze(-1) == ids_t.unsqueeze(0)).any(dim=-1)

    # Exclude background (id 0 or negative)
    valid_mask = semantic_mask > 0
    non_interact_mask = valid_mask & (~handle_mask)

    handle_points = object_points[handle_mask]
    non_interact_points = object_points[non_interact_mask]
    return handle_points, non_interact_points, handle_mask


def compute_handle_geometry_from_points(
    handle_points: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute compact handle geometry features from a point set.

    Returns center, PCA long axis, outward normal, and extents.
    """
    if handle_points.shape[0] < 3:
        device = handle_points.device
        return {
            "center": torch.zeros(3, device=device),
            "long_axis": torch.tensor([0., 0., 1.], device=device),
            "normal": torch.tensor([1., 0., 0.], device=device),
            "extents": torch.zeros(3, device=device),
        }

    center = handle_points.mean(dim=0)
    centered = handle_points - center
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    long_axis = eigvecs[:, -1]
    normal = eigvecs[:, 0]
    long_axis = long_axis / (torch.norm(long_axis) + 1e-8)
    normal = normal / (torch.norm(normal) + 1e-8)

    # Extents along each PCA axis
    projections = centered @ eigvecs
    extents = projections.max(dim=0).values - projections.min(dim=0).values

    return {
        "center": center,
        "long_axis": long_axis,
        "normal": normal,
        "extents": extents,
    }


# ---------------------------------------------------------------------------
# NPCS map to world-space point cloud (for depth-image based pipelines)
# ---------------------------------------------------------------------------

def npcs_map_to_world_points(
    npcs_map: np.ndarray,
    sem_seg_map: np.ndarray,
    rts_dict: Dict[str, Dict[str, np.ndarray]],
    link_name_to_inst_id: Dict[str, int],
    handle_part_ids: Sequence[int] = (1, 3, 9),
    anno_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert NPCS map pixels to world-space 3D points, filtered by handle semantic IDs.

    This mirrors the inverse of ``get_NPCS_map_from_oriented_bbox`` from the
    rendering pipeline, reconstructing world-space handle points from the
    dense NPCS prediction + semantic segmentation.

    Parameters
    ----------
    npcs_map : (H, W, 3)
        Per-pixel NPCS coordinates.
    sem_seg_map : (H, W)
        Per-pixel semantic segmentation (GAPartNet part IDs).
    rts_dict : dict
        Link-name → {R, T, S, scaler} from ``compute_link_rts``.
    link_name_to_inst_id : dict
        Mapping from link name to instance segmentation ID.
    handle_part_ids : sequence of int
        Semantic IDs for handle parts.
    anno_list : list, optional
        Full annotation list, used to map instance IDs back to link names
        and check categories.

    Returns
    -------
    handle_world_points : (M, 3) array of handle surface points in world space.
    all_world_points : (N, 3) array of all valid object points in world space.
    """
    H, W = sem_seg_map.shape[:2]
    handle_part_set = set(handle_part_ids)
    inst_to_link = {v: k for k, v in link_name_to_inst_id.items()}

    handle_world = []
    all_world = []

    for y in range(H):
        for x in range(W):
            sem_id = int(sem_seg_map[y, x])
            if sem_id <= 0:
                continue
            npcs_coord = npcs_map[y, x]
            if np.allclose(npcs_coord, 0.0):
                continue

            # Find which link this pixel belongs to via instance seg
            # For simplicity we iterate; in practice this would be vectorized
            # Reconstruct world point: world = npcs @ R^T * scaler + T
            # We need the link's RTS. If we have instance seg info, use it.
            # Otherwise fall back to iterating rts_dict for the handle links.
            for link_name, rts in rts_dict.items():
                world_pt = npcs_coord @ rts["R"].T * rts["scaler"] + rts["T"]
                all_world.append(world_pt)
                if sem_id in handle_part_set:
                    handle_world.append(world_pt)
                break  # Use first matching RTS for this pixel

    handle_arr = np.array(handle_world, dtype=np.float32).reshape(-1, 3) if handle_world else np.zeros((0, 3), dtype=np.float32)
    all_arr = np.array(all_world, dtype=np.float32).reshape(-1, 3) if all_world else np.zeros((0, 3), dtype=np.float32)
    return handle_arr, all_arr

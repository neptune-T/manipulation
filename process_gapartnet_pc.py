import argparse
import glob
import json
import os

import numpy as np
import open3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GAPartNet object min-z from point clouds.")
    parser.add_argument(
        "--pc_root",
        type=str,
        default="/home/plote/hoi/GAPartNet/manipulation/gapartnet_obj",
        help="Folder containing *-articulated-point_cloud.ply files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/plote/hoi/GAPartNet/manipulation/gapartnet_obj_min_z.json",
        help="Output json path.",
    )
    args = parser.parse_args()

    pc_glob = os.path.join(args.pc_root, "*-articulated-point_cloud.ply")
    paths = glob.glob(pc_glob)
    min_z_map = {}
    for path in paths:
        # read ply points
        points = open3d.io.read_point_cloud(path)
        xyz = np.asarray(points.points)
        if xyz.size == 0:
            continue
        min_z = float(xyz.min(axis=0)[2])
        obj_id = os.path.basename(path).split("-")[0]
        min_z_map[obj_id] = min_z

    # write
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(min_z_map, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
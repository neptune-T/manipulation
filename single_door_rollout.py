import argparse
import os
import isaacgym

import numpy as np

try:
    from single_door_residual_env import SingleDoorResidualConfig, SingleDoorResidualEnv
except ImportError:
    from single_door_residual_env import SingleDoorResidualConfig, SingleDoorResidualEnv


def main():
    default_asset_dir = os.path.join(os.path.dirname(__file__), "assets", "gapartnet_example", "11712")
    parser = argparse.ArgumentParser(description="Run a minimal single-door residual rollout.")
    parser.add_argument("--asset-dir", type=str, default=default_asset_dir)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--door-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--policy", type=str, default="zero", choices=["zero", "random"])
    parser.add_argument("--random-scale", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no-optimize", action="store_true", default=False)
    parser.add_argument("--no-demo-base", action="store_true", default=False)
    parser.add_argument("--reset-phase", type=str, default="grasp", choices=["approach", "grasp"])
    parser.add_argument("--reset-pose-noise", type=float, default=0.0)
    parser.add_argument("--reset-rot-noise", type=float, default=0.0)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    config = SingleDoorResidualConfig(
        asset_dir=args.asset_dir,
        config_path=args.config,
        door_index=args.door_index,
        headless=bool(args.headless),
        use_optimized_grasp=not bool(args.no_optimize),
        use_demo_base_pose=not bool(args.no_demo_base),
        reset_phase=args.reset_phase,
        reset_pose_noise=float(args.reset_pose_noise),
        reset_rot_noise=float(args.reset_rot_noise),
    )
    env = SingleDoorResidualEnv(config)

    try:
        obs, info = env.reset()
        geom = env.get_handle_geometry_diagnostics()
        print(
            f"reset phase={info['reset_phase']} "
            f"contact={info['surface_contact_count']} stable={int(info['surface_contact_stable'])} "
            f"progress={info['progress']:.4f}"
        )
        print(
            f"reset_grasp_stable={int(info['reset_grasp_stable'])} "
            f"obs_dim={info['observation_dim']} curriculum={info['curriculum_phase']}"
        )
        print(
            f"task door={info['task_spec']['door_link_name']} "
            f"handle={info['task_spec']['handle_link_name']} joint={info['task_spec']['joint_name']}"
        )
        print(
            f"handle length={geom['handle_length_world']:.4f} "
            f"depth={geom['handle_depth_world']:.4f} width={geom['handle_width_world']:.4f}"
        )
        print(
            f"teacher stable={int(info['demo_info']['grasp_stable'])} "
            f"demo_traj_len={info['demo_info']['demo_traj_len']} "
            f"demo_base={int(config.use_demo_base_pose)} pinch_mode=1"
        )
        print(f"reset pinch={info['pinch_debug']}")
        print(f"reset teacher_qpos_residual_l1={np.mean(np.abs(np.asarray(info.get('teacher_qpos_residual', np.zeros(20))))):.4f}")
        print(f"reset target_qpos_head={np.asarray(info['demo_info']['anchor_qpos'])[:8]}")
        print(f"reset grasp_info={info['demo_info']['grasp_info']}")
        print(f"reset settle_counter={info['grasp_settle_counter']}")

        metrics = []
        total_reward = 0.0
        for step in range(int(args.steps)):
            if args.policy == "random":
                action = env.random_action(scale=args.random_scale)
            else:
                action = env.zero_action()

            obs, reward, done, step_info = env.step(action)
            total_reward += float(reward)
            metrics.append(
                {
                    "step": step,
                    "reward": float(reward),
                    "progress": float(step_info["progress"]),
                    "contact": int(step_info["surface_contact_count"]),
                    "stable": bool(step_info["surface_contact_stable"]),
                    "success": bool(step_info["success"]),
                    "drive_dof_val": float(step_info["drive_dof_val"]),
                    "drive_dof_vel": float(step_info["drive_dof_vel"]),
                    "target_pose": step_info["target_pose"],
                    "target_qpos": step_info["target_qpos"],
                    "reward_terms": step_info["reward_terms"],
                }
            )

            if step % 10 == 0 or done:
                print(
                    f"step={step:03d} reward={reward:.3f} progress={step_info['progress']:.4f} "
                    f"dof={step_info['drive_dof_val']:.4f} vel={step_info['drive_dof_vel']:.4f} "
                    f"contact={step_info['surface_contact_count']:03d} stable={int(step_info['surface_contact_stable'])} "
                    f"success={int(step_info['success'])} "
                    f"contact_r={step_info['reward_terms']['contact']:.3f} prog_r={step_info['reward_terms']['progress']:.3f} "
                    f"sdf_r={step_info['reward_terms']['sdf_contact']:.3f} pinch_r={step_info['reward_terms']['pinch']:.3f} gate={step_info['reward_terms']['pinch_gate']:.3f} "
                    f"fc_r={step_info['reward_terms']['force_closure']:.3f} fc_g={step_info['reward_terms']['force_closure_gate']:.3f} "
                    f"min_dist={step_info['surface_contact_min_dist']:.4f} plane_v={step_info['door_plane_violation']:.4f} pen={step_info['reward_terms']['penetration_penalty']:.4f} sdf_pen={step_info['reward_terms']['sdf_penetration_penalty']:.4f} palm_pen={step_info['reward_terms']['palm_penalty']:.4f} "
                    f"phase={step_info['curriculum_phase']} "
                    f"settle={int(step_info['grasp_settle_active'])}:{step_info['grasp_settle_counter']} "
                    f"pull={int(step_info['teacher_pull_hint'])} "
                    f"teacher_max={np.max(np.abs(np.asarray(step_info['teacher_action']))):.3f} "
                    f"teacher_qpos_l1={np.mean(np.abs(np.asarray(step_info['teacher_qpos_residual']))):.3f} "
                    f"pinch={step_info['pinch_debug']}"
                )

            if done:
                break

        print(f"done total_reward={total_reward:.3f} final_progress={metrics[-1]['progress']:.4f}")

        if args.save_json:
            save_path = args.save_json
            if not os.path.isabs(save_path):
                save_path = os.path.join(os.getcwd(), save_path)
            env.export_rollout_summary(metrics, save_path)
            print(f"saved rollout summary to {save_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

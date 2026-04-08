import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

try:
    from single_door_residual_env import SingleDoorResidualConfig, SingleDoorResidualEnv
except ImportError:
    from single_door_residual_env import SingleDoorResidualConfig, SingleDoorResidualEnv

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class PPOConfig:
    total_updates: int = 200
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.008
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 128
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    action_std_init: float = 0.40
    save_every: int = 20
    bc_coef: float = 0.1
    bc_coef_min: float = 0.01
    bc_coef_decay: float = 0.90
    contact_aux_coef: float = 0.35
    value_clip: float = 0.2
    teacher_forcing_coef: float = 0.25
    teacher_forcing_min: float = 0.05
    teacher_forcing_decay: float = 0.90
    teacher_forcing_success_threshold: float = 0.80
    teacher_forcing_window: int = 20
    target_kl: float = 0.015


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, action_std_init: float = 0.35):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.contact_head = nn.Linear(hidden_dim, 16)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(action_std_init))

    def forward(self, obs: torch.Tensor):
        feat = self.backbone(obs)
        mean = self.actor_mean(feat)
        value = self.critic(feat).squeeze(-1)
        contact_logit = self.contact_head(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value, contact_logit

    def act(self, obs: torch.Tensor):
        mean, std, value, contact_logit = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value, contact_logit

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value, contact_logit = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value, contact_logit, mean


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = np.zeros_like(rewards[0], dtype=np.float32)
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - np.asarray(dones[step], dtype=np.float32)
        next_val = next_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[step] = last_adv
    returns = advantages + values
    return advantages, returns


def save_checkpoint(model, optimizer, update, path, metadata):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "metadata": metadata,
        },
        path,
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the single-door residual task.")
    parser.add_argument("--asset-dir", type=str, default=os.path.join(os.path.dirname(__file__), "assets", "gapartnet_example", "45936"))
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--door-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="output/single_door_ppo")
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--reset-phase", type=str, default="grasp", choices=["approach", "grasp"])
    parser.add_argument("--reset-pose-noise", type=float, default=0.002)
    parser.add_argument("--reset-rot-noise", type=float, default=0.02)
    parser.add_argument("--policy-init-std", type=float, default=0.40)
    parser.add_argument("--bc-coef", type=float, default=0.1)
    parser.add_argument("--bc-coef-min", type=float, default=0.01)
    parser.add_argument("--bc-coef-decay", type=float, default=0.90)
    parser.add_argument("--teacher-forcing-coef", type=float, default=0.25)
    parser.add_argument("--teacher-forcing-min", type=float, default=0.05)
    parser.add_argument("--teacher-forcing-decay", type=float, default=0.90)
    parser.add_argument("--teacher-forcing-success-threshold", type=float, default=0.80)
    parser.add_argument("--teacher-forcing-window", type=int, default=20)
    parser.add_argument("--entropy-coef", type=float, default=0.008)
    parser.add_argument("--contact-aux-coef", type=float, default=0.35)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--curriculum-enabled", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="gapartnet-hoi")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    args = parser.parse_args()

    env_cfg = SingleDoorResidualConfig(
        asset_dir=args.asset_dir,
        config_path=args.config,
        door_index=args.door_index,
        num_envs=int(args.num_envs),
        headless=bool(args.headless),
        device=args.device,
        reset_phase=args.reset_phase,
        reset_pose_noise=float(args.reset_pose_noise),
        reset_rot_noise=float(args.reset_rot_noise),
        curriculum_enabled=bool(args.curriculum_enabled),
    )
    ppo_cfg = PPOConfig(
        total_updates=int(args.total_updates),
        rollout_steps=int(args.rollout_steps),
        action_std_init=float(args.policy_init_std),
        bc_coef=float(args.bc_coef),
        bc_coef_min=float(args.bc_coef_min),
        bc_coef_decay=float(args.bc_coef_decay),
        entropy_coef=float(args.entropy_coef),
        contact_aux_coef=float(args.contact_aux_coef),
        teacher_forcing_coef=float(args.teacher_forcing_coef),
        teacher_forcing_min=float(args.teacher_forcing_min),
        teacher_forcing_decay=float(args.teacher_forcing_decay),
        teacher_forcing_success_threshold=float(args.teacher_forcing_success_threshold),
        teacher_forcing_window=int(args.teacher_forcing_window),
        target_kl=float(args.target_kl),
    )

    env = SingleDoorResidualEnv(env_cfg)
    torch_device = torch.device(args.device)
    wandb_run = None

    try:
        if bool(args.wandb):
            if wandb is None:
                raise ImportError("wandb is not installed. Please `pip install wandb` in your env.")
            wandb_init_kwargs = {
                "project": args.wandb_project,
                "config": _to_jsonable(
                    {
                        "env_cfg": asdict(env_cfg),
                        "ppo_cfg": asdict(ppo_cfg),
                        "asset_dir": args.asset_dir,
                        "device": args.device,
                    }
                ),
            }
            if args.wandb_name:
                wandb_init_kwargs["name"] = args.wandb_name
            if args.wandb_entity:
                wandb_init_kwargs["entity"] = args.wandb_entity
            wandb_run = wandb.init(**wandb_init_kwargs)

        obs, reset_info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        num_envs = int(obs.shape[0])
        obs_dim = int(obs.shape[-1])
        action_dim = int(env.action_dim)
        model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=ppo_cfg.hidden_dim,
            action_std_init=ppo_cfg.action_std_init,
        ).to(torch_device)
        optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

        episode_return = np.zeros((num_envs,), dtype=np.float32)
        episode_length = np.zeros((num_envs,), dtype=np.int32)
        global_step = 0
        history = []
        train_stats_history = []

        for update in range(1, ppo_cfg.total_updates + 1):
            # Linear learning rate annealing
            lr_frac = 1.0 - (update - 1) / ppo_cfg.total_updates
            for param_group in optimizer.param_groups:
                param_group["lr"] = ppo_cfg.lr * lr_frac

            obs_buf = np.zeros((ppo_cfg.rollout_steps, num_envs, obs_dim), dtype=np.float32)
            action_buf = np.zeros((ppo_cfg.rollout_steps, num_envs, action_dim), dtype=np.float32)
            logp_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)
            reward_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)
            done_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)
            value_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)
            contact_vec_target_buf = np.zeros((ppo_cfg.rollout_steps, num_envs, 16), dtype=np.float32)
            teacher_action_buf = np.zeros((ppo_cfg.rollout_steps, num_envs, action_dim), dtype=np.float32)
            progress_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)
            stable_contact_buf = np.zeros((ppo_cfg.rollout_steps, num_envs), dtype=np.float32)

            for step in range(ppo_cfg.rollout_steps):
                global_step += num_envs
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
                with torch.no_grad():
                    action_tensor, logp_tensor, value_tensor, contact_logit_tensor = model.act(obs_tensor)
                sampled_action = action_tensor.cpu().numpy().astype(np.float32)
                teacher_action = env.get_teacher_action(step_index=step)
                action = (
                    (1.0 - ppo_cfg.teacher_forcing_coef) * sampled_action
                    + ppo_cfg.teacher_forcing_coef * teacher_action
                ).astype(np.float32)
                next_obs, reward, done, info = env.step(action)

                # Recompute log_prob for the *blended* action that was actually executed,
                # so that old_logp matches the action stored in action_buf.
                with torch.no_grad():
                    blended_tensor = torch.tensor(action, dtype=torch.float32, device=torch_device)
                    mean_t, std_t, _, _ = model.forward(obs_tensor)
                    blended_logp = torch.distributions.Normal(mean_t, std_t).log_prob(blended_tensor).sum(-1)

                obs_buf[step] = obs
                action_buf[step] = action
                logp_buf[step] = np.atleast_1d(np.asarray(blended_logp.cpu().numpy(), dtype=np.float32))
                reward_buf[step] = np.atleast_1d(np.asarray(reward, dtype=np.float32))
                done_buf[step] = np.atleast_1d(np.asarray(done, dtype=np.float32))
                value_buf[step] = np.atleast_1d(np.asarray(value_tensor.cpu().numpy(), dtype=np.float32))
                contact_vec_target_buf[step] = np.asarray(info["contact_target"], dtype=np.float32).reshape(num_envs, 16)
                teacher_action_buf[step] = np.asarray(teacher_action, dtype=np.float32).reshape(num_envs, action_dim)
                progress_buf[step] = np.atleast_1d(np.asarray(info["progress"], dtype=np.float32))
                stable_contact_buf[step] = np.atleast_1d(np.asarray(info["surface_contact_stable"], dtype=np.float32))

                obs = np.asarray(next_obs, dtype=np.float32)
                if obs.ndim == 1:
                    obs = obs[None, :]
                episode_return += np.atleast_1d(np.asarray(reward, dtype=np.float32))
                episode_length += 1

                done_arr = np.atleast_1d(np.asarray(done, dtype=bool))
                success_arr = np.atleast_1d(np.asarray(info["success"], dtype=bool))
                progress_arr = np.atleast_1d(np.asarray(info["progress"], dtype=np.float32))
                if np.any(done_arr):
                    curriculum_info = env.update_curriculum(bool(np.mean(success_arr.astype(np.float32)) > 0.5))
                    for env_i in np.where(done_arr)[0]:
                        history.append(
                            {
                                "global_step": global_step,
                                "env_i": int(env_i),
                                "episode_return": float(episode_return[env_i]),
                                "episode_length": int(episode_length[env_i]),
                                "progress": float(progress_arr[env_i]),
                                "success": bool(success_arr[env_i]),
                                "curriculum_phase": curriculum_info["curriculum_phase"],
                                "curriculum_level": curriculum_info["curriculum_level"],
                            }
                        )
                    obs, reset_info = env.reset()
                    obs = np.asarray(obs, dtype=np.float32)
                    if obs.ndim == 1:
                        obs = obs[None, :]
                    episode_return[done_arr] = 0.0
                    episode_length[done_arr] = 0

            with torch.no_grad():
                next_value = model.forward(torch.tensor(obs, dtype=torch.float32, device=torch_device))[2].cpu().numpy().astype(np.float32)

            advantages, returns = compute_gae(
                reward_buf,
                done_buf,
                value_buf,
                next_value,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
            )
            # Normalize advantages per-minibatch below instead of globally,
            # to reduce bias with small batch sizes.

            flat_obs = obs_buf.reshape(-1, obs_dim)
            flat_actions = action_buf.reshape(-1, action_dim)
            flat_logp = logp_buf.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_contact = contact_vec_target_buf.reshape(-1, 16)
            flat_teacher_action = teacher_action_buf.reshape(-1, action_dim)
            flat_values = value_buf.reshape(-1)

            obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=torch_device)
            action_tensor = torch.tensor(flat_actions, dtype=torch.float32, device=torch_device)
            old_logp_tensor = torch.tensor(flat_logp, dtype=torch.float32, device=torch_device)
            adv_tensor = torch.tensor(flat_adv, dtype=torch.float32, device=torch_device)
            ret_tensor = torch.tensor(flat_returns, dtype=torch.float32, device=torch_device)
            contact_vec_target_tensor = torch.tensor(flat_contact, dtype=torch.float32, device=torch_device)
            teacher_action_tensor = torch.tensor(flat_teacher_action, dtype=torch.float32, device=torch_device)
            old_value_tensor = torch.tensor(flat_values, dtype=torch.float32, device=torch_device)

            batch_size = ppo_cfg.rollout_steps * num_envs
            inds = np.arange(batch_size)
            last_stats = {}
            kl_early_stop = False
            for _ in range(ppo_cfg.epochs):
                if kl_early_stop:
                    break
                np.random.shuffle(inds)
                for start in range(0, batch_size, ppo_cfg.minibatch_size):
                    end = start + ppo_cfg.minibatch_size
                    mb_inds = inds[start:end]
                    mb_obs = obs_tensor[mb_inds]
                    mb_actions = action_tensor[mb_inds]
                    mb_old_logp = old_logp_tensor[mb_inds]
                    mb_adv = adv_tensor[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    mb_returns = ret_tensor[mb_inds]
                    mb_contact_vec_target = contact_vec_target_tensor[mb_inds]
                    mb_teacher_action = teacher_action_tensor[mb_inds]
                    mb_old_value = old_value_tensor[mb_inds]

                    new_logp, entropy, value, contact_logit, action_mean = model.evaluate_actions(mb_obs, mb_actions)
                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_pred_clipped = mb_old_value + torch.clamp(value - mb_old_value, -ppo_cfg.value_clip, ppo_cfg.value_clip)
                    value_loss_unclipped = (value - mb_returns) ** 2
                    value_loss_clipped = (value_pred_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                    entropy_loss = entropy.mean()
                    bc_loss = torch.mean((action_mean - mb_teacher_action) ** 2)
                    contact_aux_loss = nn.functional.binary_cross_entropy_with_logits(contact_logit, mb_contact_vec_target)
                    loss = (
                        policy_loss
                        + ppo_cfg.value_coef * value_loss
                        - ppo_cfg.entropy_coef * entropy_loss
                        + ppo_cfg.bc_coef * bc_loss
                        + ppo_cfg.contact_aux_coef * contact_aux_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                    optimizer.step()
                    approx_kl = torch.mean((ratio - 1) - torch.log(ratio)).item()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > ppo_cfg.clip_ratio).float()).item()
                    if approx_kl > ppo_cfg.target_kl:
                        kl_early_stop = True
                        break
                    last_stats = {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                        "bc_loss": float(bc_loss.item()),
                        "contact_aux_loss": float(contact_aux_loss.item()),
                        "approx_kl": float(approx_kl),
                        "clipfrac": float(clipfrac),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }

            recent = history[-5:]
            avg_return = float(np.mean([item["episode_return"] for item in recent])) if recent else 0.0
            avg_progress = float(np.mean([item["progress"] for item in recent])) if recent else 0.0
            success_rate = float(np.mean([float(item["success"]) for item in recent])) if recent else 0.0
            rollout_contact_rate = float(np.mean(stable_contact_buf))
            rollout_progress = float(np.mean(progress_buf))
            tf_recent = history[-int(max(1, ppo_cfg.teacher_forcing_window)) :]
            tf_recent_success = float(np.mean([float(item["success"]) for item in tf_recent])) if tf_recent else 0.0
            if tf_recent and tf_recent_success >= float(ppo_cfg.teacher_forcing_success_threshold):
                ppo_cfg.teacher_forcing_coef = max(
                    float(ppo_cfg.teacher_forcing_min),
                    float(ppo_cfg.teacher_forcing_coef) * float(ppo_cfg.teacher_forcing_decay),
                )
                ppo_cfg.bc_coef = max(
                    float(ppo_cfg.bc_coef_min),
                    float(ppo_cfg.bc_coef) * float(ppo_cfg.bc_coef_decay),
                )
            elif update % 50 == 0:
                # Slow time-based fallback decay to prevent permanently stuck
                # imitation when the success threshold is never reached.
                slow_decay = 0.98
                ppo_cfg.teacher_forcing_coef = max(
                    float(ppo_cfg.teacher_forcing_min),
                    float(ppo_cfg.teacher_forcing_coef) * slow_decay,
                )
                ppo_cfg.bc_coef = max(
                    float(ppo_cfg.bc_coef_min),
                    float(ppo_cfg.bc_coef) * slow_decay,
                )
            print(
                f"update={update:04d} avg_return={avg_return:.3f} "
                f"avg_progress={avg_progress:.4f} success_rate={success_rate:.2f} "
                f"rollout_contact={rollout_contact_rate:.2f} rollout_prog={rollout_progress:.4f} "
                f"bc={last_stats.get('bc_loss', 0.0):.4f} contact_aux={last_stats.get('contact_aux_loss', 0.0):.4f} "
                f"tf={ppo_cfg.teacher_forcing_coef:.2f} bc_coef={ppo_cfg.bc_coef:.3f} tf_recent={tf_recent_success:.2f} "
                f"phase={env.get_curriculum_phase()}"
            )
            train_stats_history.append(
                {
                    "update": int(update),
                    "avg_return": float(avg_return),
                    "avg_progress": float(avg_progress),
                    "success_rate": float(success_rate),
                    "rollout_contact_rate": float(rollout_contact_rate),
                    "rollout_progress": float(rollout_progress),
                    "bc_coef": float(ppo_cfg.bc_coef),
                    "bc_recent_success": float(tf_recent_success),
                    "teacher_forcing_coef": float(ppo_cfg.teacher_forcing_coef),
                    "teacher_forcing_recent_success": float(tf_recent_success),
                    "curriculum_phase": env.get_curriculum_phase(),
                    "curriculum_level": int(env.curriculum_level),
                    **last_stats,
                }
            )
            if wandb_run is not None:
                wandb.log(train_stats_history[-1], step=update)

            if update % ppo_cfg.save_every == 0 or update == ppo_cfg.total_updates:
                checkpoint_path = os.path.join(args.save_dir, f"ppo_update_{update:04d}.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    path=checkpoint_path,
                    metadata={
                        "env_cfg": asdict(env_cfg),
                        "ppo_cfg": asdict(ppo_cfg),
                        "task_spec": env.task_spec.to_dict(),
                        "history_tail": history[-20:],
                        "train_stats": last_stats,
                        "curriculum_phase": env.get_curriculum_phase(),
                    },
                )

        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "train_history.json"), "w") as handle:
            json.dump(_to_jsonable(history), handle)
        with open(os.path.join(args.save_dir, "train_stats.json"), "w") as handle:
            json.dump(_to_jsonable(train_stats_history), handle)
        with open(os.path.join(args.save_dir, "train_config.json"), "w") as handle:
            json.dump(
                _to_jsonable({
                    "env_cfg": asdict(env_cfg),
                    "ppo_cfg": asdict(ppo_cfg),
                    "task_spec": env.task_spec.to_dict(),
                    "demo_info": env.demo_info,
                }),
                handle,
            )
    finally:
        if wandb_run is not None:
            wandb.finish()
        env.close()


if __name__ == "__main__":
    main()

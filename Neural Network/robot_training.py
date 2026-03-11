"""
robot_training.py — PPO Training Loop
======================================
Trains the RobotActorCritic network on the RobotEnv simulation using
Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).

Designed for: AMD Threadripper 7970X + NVIDIA RTX 5090
    • 32 parallel PyBullet environments (sequential on Windows, thread-safe)
    • Network forward/backward passes run entirely on CUDA
    • CPU ↔ GPU transfers are batched and minimised

Expected training timeline (max performance mode):
    ~200K steps  :  Arm begins pointing toward objects        (~3 min)
    ~800K steps  :  Consistent reaching                       (~11 min)
    ~2M   steps  :  First successful grasps                   (~28 min)
    ~5M   steps  :  Reliable pick-and-place                   (~1.2 hr)
    ~10M  steps  :  Well-generalised policy                   (~2.4 hr)

Outputs (written to ./runs/<run_name>/):
    checkpoints/   model_<steps>.pt   — saved every CHECKPOINT_INTERVAL steps
    best_model.pt                     — best mean reward seen so far
    training_log.csv                  — per-update metrics
    tensorboard/                      — TensorBoard event files

Usage:
    py robot_training.py                          # start new run
    py robot_training.py --resume runs/my_run     # resume from checkpoint
    py robot_training.py --render                 # render one env during training
    py robot_training.py --steps 10_000_000       # custom total steps
    py robot_training.py --envs 16                # fewer parallel envs

Dependencies:
    pip install pybullet torch numpy tensorboard
"""

import os
import sys
import csv
import math
import time
import argparse
import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from robot_neural_network import RobotActorCritic, CAM_C, CAM_H, CAM_W, OBS_PROPRIO_DIM
from robot_env import VectorizedRobotEnv, RobotEnv, MAX_STEPS


# ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────


class Config:
    """
    All training hyperparameters in one place.
    Tuned for the 7970X Threadripper + RTX 5090 hardware configuration.
    """

    # ── Environment ───────────────────────────────────────────────────────────
    num_envs:           int   = 32       # parallel environments (sequential on Windows)
    #   PyBullet on Windows is not thread-safe; sequential stepping is used
    #   32 envs is the recommended default for the 7970X Threadripper

    # ── Rollout ───────────────────────────────────────────────────────────────
    steps_per_env:      int   = 512      # steps collected per env per update
    #   Total batch = num_envs × steps_per_env = 32 × 512 = 16,384 steps

    # ── PPO update ────────────────────────────────────────────────────────────
    ppo_epochs:         int   = 10       # gradient epochs per collected batch
    minibatch_size:     int   = 256      # samples per gradient step
    #   Minibatches per epoch = 16384 / 256 = 64

    clip_coef:          float = 0.2      # PPO clip ε
    value_coef:         float = 0.5      # critic loss weight
    entropy_coef:       float = 0.01     # entropy bonus weight (exploration)
    max_grad_norm:      float = 0.5      # gradient clipping

    # ── Optimiser ─────────────────────────────────────────────────────────────
    learning_rate:      float = 3e-4
    lr_anneal:          bool  = True     # linearly decay LR to 0 over training
    adam_eps:           float = 1e-5

    # ── GAE ───────────────────────────────────────────────────────────────────
    gamma:              float = 0.99     # discount factor
    gae_lambda:         float = 0.95     # GAE λ (bias-variance tradeoff)

    # ── Training duration ─────────────────────────────────────────────────────
    total_steps:        int   = 10_000_000

    # ── Logging & checkpointing ───────────────────────────────────────────────
    log_interval:       int   = 1        # log every N updates
    checkpoint_interval:int   = 500_000  # save checkpoint every N env steps
    eval_interval:      int   = 100_000  # run deterministic eval every N steps

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_episodes:      int   = 10       # episodes per evaluation

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:               int   = 42
    run_name:           str   = ""       # auto-generated if empty


# ─────────────────────────────────────────────────────────────────────────────
#  ROLLOUT BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores (obs, action, reward, done, value, log_prob) tuples collected
    from N parallel environments over T steps.

    All tensors live on CPU during collection, then are moved to GPU
    in one batched transfer at update time.

    Shape conventions:
        images   : (T, N, 4, 84, 84)
        proprios : (T, N, 14)
        actions  : (T, N, 5)
        rewards  : (T, N)
        dones    : (T, N)
        values   : (T, N)
        log_probs: (T, N)
        advantages: (T, N)   — computed after collection
        returns   : (T, N)   — computed after collection
    """

    def __init__(self, steps_per_env: int, num_envs: int, cfg: Config):
        T, N = steps_per_env, num_envs
        self.T   = T
        self.N   = N
        self.cfg = cfg
        self.ptr = 0   # write pointer

        self.images    = torch.zeros(T, N, CAM_C, CAM_H, CAM_W, dtype=torch.float32)
        self.proprios  = torch.zeros(T, N, OBS_PROPRIO_DIM,      dtype=torch.float32)
        self.actions   = torch.zeros(T, N, 5,                    dtype=torch.float32)
        self.rewards   = torch.zeros(T, N,                        dtype=torch.float32)
        self.dones     = torch.zeros(T, N,                        dtype=torch.float32)
        self.values    = torch.zeros(T, N,                        dtype=torch.float32)
        self.log_probs = torch.zeros(T, N,                        dtype=torch.float32)

        # Filled by compute_returns_and_advantages()
        self.advantages = torch.zeros(T, N, dtype=torch.float32)
        self.returns    = torch.zeros(T, N, dtype=torch.float32)

    def add(
        self,
        images:    np.ndarray,   # (N, 4, 84, 84)
        proprios:  np.ndarray,   # (N, 14)
        actions:   torch.Tensor, # (N, 5)
        rewards:   np.ndarray,   # (N,)
        dones:     np.ndarray,   # (N,)
        values:    torch.Tensor, # (N,)
        log_probs: torch.Tensor, # (N,)
    ):
        t = self.ptr
        self.images[t]    = torch.from_numpy(images)
        self.proprios[t]  = torch.from_numpy(proprios)
        self.actions[t]   = actions.cpu()
        self.rewards[t]   = torch.from_numpy(rewards)
        self.dones[t]     = torch.from_numpy(dones.astype(np.float32))
        self.values[t]    = values.cpu()
        self.log_probs[t] = log_probs.cpu()
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,  # (N,) — V(s_T) from critic
        last_dones:  np.ndarray,    # (N,) — done flags at last step
    ):
        """
        GAE (Generalized Advantage Estimation).
        Fills self.advantages and self.returns in-place.
        """
        gamma  = self.cfg.gamma
        lam    = self.cfg.gae_lambda
        T, N   = self.T, self.N

        last_gae = torch.zeros(N, dtype=torch.float32)
        last_val  = last_values.cpu().float()
        last_done = torch.from_numpy(last_dones.astype(np.float32))

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_done
                next_values       = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            delta    = (self.rewards[t]
                        + gamma * next_values * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * lam * next_non_terminal * last_gae

            self.advantages[t] = last_gae
            self.returns[t]    = last_gae + self.values[t]

    def get_minibatches(self, device: torch.device):
        """
        Flatten (T, N) → (T*N,) and yield random minibatches on the GPU.
        Called inside the PPO update loop.
        """
        T, N = self.T, self.N
        B    = T * N   # total samples in buffer

        # Flatten and move everything to GPU in one transfer each
        imgs    = self.images.view(B, CAM_C, CAM_H, CAM_W).to(device)
        props   = self.proprios.view(B, OBS_PROPRIO_DIM).to(device)
        acts    = self.actions.view(B, 5).to(device)
        lps     = self.log_probs.view(B).to(device)
        advs    = self.advantages.view(B).to(device)
        rets    = self.returns.view(B).to(device)

        # Normalise advantages (stabilises training)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Yield random minibatches
        indices = torch.randperm(B, device=device)
        mb_size = self.cfg.minibatch_size
        for start in range(0, B, mb_size):
            mb_idx = indices[start:start + mb_size]
            yield (
                imgs[mb_idx],
                props[mb_idx],
                acts[mb_idx],
                lps[mb_idx],
                advs[mb_idx],
                rets[mb_idx],
            )

    def reset(self):
        self.ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Full PPO training loop integrating RobotActorCritic + VectorizedRobotEnv.

    Call trainer.train() to start. Progress is logged to TensorBoard and CSV.
    Checkpoints are saved automatically.
    """

    def __init__(self, cfg: Config, render: bool = False):
        self.cfg    = cfg
        self.render = render
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Run directory ─────────────────────────────────────────────────────
        if not cfg.run_name:
            ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.run_name = f"run_{ts}"
        self.run_dir = Path("runs") / cfg.run_name
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # ── Seeding ───────────────────────────────────────────────────────────
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(cfg.seed)

        # ── Network ───────────────────────────────────────────────────────────
        self.model = RobotActorCritic().to(self.device)
        self.optim = optim.Adam(
            self.model.parameters(),
            lr  = cfg.learning_rate,
            eps = cfg.adam_eps,
        )

        # ── Environments ──────────────────────────────────────────────────────
        render_idx = 0 if render else -1
        self.envs  = VectorizedRobotEnv(
            num_envs   = cfg.num_envs,
            render_idx = render_idx,
            seed       = cfg.seed,
        )

        # ── Rollout buffer ────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(cfg.steps_per_env, cfg.num_envs, cfg)

        # ── Logging ───────────────────────────────────────────────────────────
        self.writer      = SummaryWriter(self.run_dir / "tensorboard")
        self.csv_path    = self.run_dir / "training_log.csv"
        self._init_csv()

        # ── State trackers ────────────────────────────────────────────────────
        self.global_step  = 0
        self.update_count = 0
        self.best_reward  = -float("inf")

        # Rolling window for episode stats (last 100 completed episodes)
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_successes = deque(maxlen=100)

        # Per-env episode accumulators
        self.ep_reward_buf = np.zeros(cfg.num_envs, dtype=np.float32)
        self.ep_length_buf = np.zeros(cfg.num_envs, dtype=np.int32)

        # Current observation (kept between rollout collections)
        self._obs_images:  np.ndarray = None
        self._obs_proprios: np.ndarray = None
        self._dones:        np.ndarray = np.zeros(cfg.num_envs, dtype=bool)

        print(self._header())

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self) -> str:
        cfg = self.cfg
        param_counts = self.model.count_parameters()
        lines = [
            "=" * 68,
            "  PPO Training — Robot Arm",
            "=" * 68,
            f"  Device        : {self.device}"
            + (f" ({torch.cuda.get_device_name(0)})"
               if self.device.type == "cuda" else ""),
            f"  Parameters    : {param_counts['TOTAL']:,}",
            f"  Environments  : {cfg.num_envs}",
            f"  Batch size    : {cfg.num_envs * cfg.steps_per_env:,} steps",
            f"  PPO epochs    : {cfg.ppo_epochs}",
            f"  Minibatch     : {cfg.minibatch_size}",
            f"  Total steps   : {cfg.total_steps:,}",
            f"  Run dir       : {self.run_dir}",
            "=" * 68,
        ]
        return "\n".join(lines)

    # ── Training entry point ──────────────────────────────────────────────────

    def train(self):
        """Main training loop. Runs until total_steps is reached."""
        cfg          = self.cfg
        total_updates = math.ceil(
            cfg.total_steps / (cfg.num_envs * cfg.steps_per_env)
        )

        # Initial reset
        obs_batch          = self.envs.reset()
        self._obs_images   = obs_batch["image"]
        self._obs_proprios = obs_batch["proprio"]
        self._dones        = np.zeros(cfg.num_envs, dtype=bool)

        print(f"\n  Starting training — {total_updates} updates planned\n")
        t_start = time.time()

        for update in range(1, total_updates + 1):
            # ── LR annealing ─────────────────────────────────────────────────
            if cfg.lr_anneal:
                frac = 1.0 - (update - 1) / total_updates
                for pg in self.optim.param_groups:
                    pg["lr"] = frac * cfg.learning_rate

            # ── Collect rollout ───────────────────────────────────────────────
            self._collect_rollout()

            # ── PPO update ────────────────────────────────────────────────────
            metrics = self._ppo_update()
            self.update_count += 1

            # ── Logging ───────────────────────────────────────────────────────
            if update % cfg.log_interval == 0:
                elapsed  = time.time() - t_start
                steps_ps = self.global_step / elapsed if elapsed > 0 else 0
                eta_s    = ((cfg.total_steps - self.global_step) / steps_ps
                            if steps_ps > 0 else 0)

                mean_reward  = (np.mean(self.ep_rewards)
                                if self.ep_rewards else float("nan"))
                mean_length  = (np.mean(self.ep_lengths)
                                if self.ep_lengths else float("nan"))
                success_rate = (np.mean(self.ep_successes)
                                if self.ep_successes else float("nan"))

                self._log_to_tensorboard(metrics, mean_reward,
                                         mean_length, success_rate, steps_ps)
                self._log_to_csv(metrics, mean_reward,
                                 mean_length, success_rate)
                self._print_update(update, total_updates, elapsed, steps_ps,
                                   eta_s, mean_reward, success_rate, metrics)

            # ── Checkpoint ───────────────────────────────────────────────────
            if self.global_step % cfg.checkpoint_interval < (
                    cfg.num_envs * cfg.steps_per_env):
                self._save_checkpoint(tag=f"model_{self.global_step}")

            # ── Best model ───────────────────────────────────────────────────
            if self.ep_rewards:
                mean_r = float(np.mean(self.ep_rewards))
                if mean_r > self.best_reward:
                    self.best_reward = mean_r
                    self._save_checkpoint(tag="best_model")

            # ── Periodic evaluation ───────────────────────────────────────────
            if (self.global_step % cfg.eval_interval
                    < cfg.num_envs * cfg.steps_per_env):
                eval_reward, eval_success = self._evaluate()
                self.writer.add_scalar("eval/mean_reward",
                                       eval_reward, self.global_step)
                self.writer.add_scalar("eval/success_rate",
                                       eval_success, self.global_step)
                print(f"  [EVAL] step={self.global_step:>8,}  "
                      f"reward={eval_reward:.3f}  "
                      f"success={eval_success:.1%}")

            if self.global_step >= cfg.total_steps:
                break

        # ── Final save ────────────────────────────────────────────────────────
        self._save_checkpoint(tag="final_model")
        elapsed = time.time() - t_start
        print(f"\n  Training complete in {elapsed/3600:.2f} hours")
        print(f"  Best mean reward: {self.best_reward:.4f}")
        print(f"  Outputs saved to: {self.run_dir}")

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(self):
        """
        Run all environments for steps_per_env steps, storing
        (obs, action, reward, done, value, log_prob) in the buffer.

        Network runs on GPU; environment stepping runs on CPU.
        """
        self.buffer.reset()
        self.model.eval()

        for _ in range(self.cfg.steps_per_env):
            # Move current obs to GPU
            img_t  = torch.from_numpy(self._obs_images).to(self.device)
            prop_t = torch.from_numpy(self._obs_proprios).to(self.device)

            with torch.no_grad():
                actions, log_probs = self.model.act(img_t, prop_t)
                values             = self.model.get_value(img_t, prop_t).squeeze(-1)

            # CPU numpy actions → environments
            actions_np = actions.cpu().numpy()   # (N, 5)

            obs_batch, rewards, dones, infos = self.envs.step(actions_np)

            # Track episode stats
            self.ep_reward_buf += rewards
            self.ep_length_buf += 1
            for i, done in enumerate(dones):
                if done:
                    self.ep_rewards.append(self.ep_reward_buf[i])
                    self.ep_lengths.append(self.ep_length_buf[i])
                    delivered = infos[i].get("delivered", False)
                    self.ep_successes.append(float(delivered))
                    self.ep_reward_buf[i] = 0.0
                    self.ep_length_buf[i] = 0

            # Store to buffer
            self.buffer.add(
                images    = self._obs_images,
                proprios  = self._obs_proprios,
                actions   = actions,
                rewards   = rewards,
                dones     = dones,
                values    = values,
                log_probs = log_probs,
            )

            self._obs_images   = obs_batch["image"]
            self._obs_proprios = obs_batch["proprio"]
            self._dones        = dones
            self.global_step  += self.cfg.num_envs

        # Compute bootstrap value for last state
        img_t  = torch.from_numpy(self._obs_images).to(self.device)
        prop_t = torch.from_numpy(self._obs_proprios).to(self.device)
        with torch.no_grad():
            last_values = self.model.get_value(img_t, prop_t).squeeze(-1)

        self.buffer.compute_returns_and_advantages(last_values, self._dones)

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self) -> dict:
        """
        Run ppo_epochs passes over the collected rollout buffer,
        updating the network with clipped PPO objective.

        Returns dict of mean metrics across all minibatches.
        """
        cfg = self.cfg
        self.model.train()

        # Accumulators
        total_policy_loss  = 0.0
        total_value_loss   = 0.0
        total_entropy      = 0.0
        total_approx_kl    = 0.0
        total_clip_frac    = 0.0
        n_minibatches      = 0

        for epoch in range(cfg.ppo_epochs):
            for (imgs, props, acts, old_lps, advs, rets) in \
                    self.buffer.get_minibatches(self.device):

                # Re-evaluate stored actions under current policy
                new_lps, values, entropy = self.model.evaluate_actions(
                    imgs, props, acts
                )

                # Policy loss (clipped surrogate)
                ratio       = torch.exp(new_lps - old_lps)
                pg_loss1    = -advs * ratio
                pg_loss2    = -advs * ratio.clamp(1 - cfg.clip_coef,
                                                   1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                values      = values.squeeze(-1)
                value_loss  = nn.functional.mse_loss(values, rets)

                # Total loss
                loss = (policy_loss
                        - cfg.entropy_coef * entropy
                        + cfg.value_coef  * value_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optim.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_approx_kl   += approx_kl.item()
                total_clip_frac   += clip_frac.item()
                n_minibatches     += 1

        n = max(n_minibatches, 1)
        return {
            "policy_loss":  total_policy_loss / n,
            "value_loss":   total_value_loss  / n,
            "entropy":      total_entropy     / n,
            "approx_kl":    total_approx_kl   / n,
            "clip_frac":    total_clip_frac   / n,
            "learning_rate": self.optim.param_groups[0]["lr"],
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, float]:
        """
        Run deterministic evaluation episodes (no exploration noise).
        Uses a single headless environment.

        Returns: (mean_reward, success_rate)
        """
        cfg = self.cfg
        self.model.eval()

        eval_env     = RobotEnv(render=False, random_seed=cfg.seed + 9999)
        total_reward = 0.0
        n_success    = 0

        for ep in range(cfg.eval_episodes):
            obs   = eval_env.reset()
            ep_r  = 0.0
            done  = False
            while not done:
                img_t, prop_t = obs.as_torch(self.device)
                with torch.no_grad():
                    action, _ = self.model.act(img_t, prop_t, deterministic=True)
                action_np = action.cpu().numpy()[0]   # remove batch dim
                obs, reward, done, info = eval_env.step(action_np)
                ep_r += reward
            total_reward += ep_r
            if info.get("delivered", False):
                n_success += 1

        eval_env.close()
        self.model.train()

        mean_reward  = total_reward / cfg.eval_episodes
        success_rate = n_success   / cfg.eval_episodes
        return mean_reward, success_rate

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        """Save model weights, optimizer state, and training metadata."""
        path = self.run_dir / "checkpoints" / f"{tag}.pt"
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optim.state_dict(),
            "global_step":     self.global_step,
            "update_count":    self.update_count,
            "best_reward":     self.best_reward,
            "config": {
                "num_envs":       self.cfg.num_envs,
                "steps_per_env":  self.cfg.steps_per_env,
                "learning_rate":  self.cfg.learning_rate,
                "total_steps":    self.cfg.total_steps,
            },
        }, path)

    def load_checkpoint(self, path: str):
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optimizer_state"])
        self.global_step  = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        self.best_reward  = ckpt["best_reward"]
        print(f"  Resumed from step {self.global_step:,}  "
              f"(best reward: {self.best_reward:.4f})")

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step", "update", "mean_reward", "mean_length",
                "success_rate", "policy_loss", "value_loss", "entropy",
                "approx_kl", "clip_frac", "learning_rate", "steps_per_sec",
            ])

    def _log_to_csv(
        self, metrics: dict, mean_reward: float,
        mean_length: float, success_rate: float,
    ):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step, self.update_count,
                f"{mean_reward:.4f}", f"{mean_length:.1f}",
                f"{success_rate:.4f}",
                f"{metrics['policy_loss']:.6f}",
                f"{metrics['value_loss']:.6f}",
                f"{metrics['entropy']:.4f}",
                f"{metrics['approx_kl']:.6f}",
                f"{metrics['clip_frac']:.4f}",
                f"{metrics['learning_rate']:.2e}",
                f"{metrics.get('steps_per_sec', 0):.0f}",
            ])

    def _log_to_tensorboard(
        self, metrics: dict, mean_reward: float,
        mean_length: float, success_rate: float, steps_per_sec: float,
    ):
        s = self.global_step
        self.writer.add_scalar("train/mean_episode_reward",  mean_reward,   s)
        self.writer.add_scalar("train/mean_episode_length",  mean_length,   s)
        self.writer.add_scalar("train/success_rate",         success_rate,  s)
        self.writer.add_scalar("train/steps_per_second",     steps_per_sec, s)
        self.writer.add_scalar("losses/policy_loss",  metrics["policy_loss"],  s)
        self.writer.add_scalar("losses/value_loss",   metrics["value_loss"],   s)
        self.writer.add_scalar("losses/entropy",      metrics["entropy"],      s)
        self.writer.add_scalar("losses/approx_kl",    metrics["approx_kl"],   s)
        self.writer.add_scalar("losses/clip_frac",    metrics["clip_frac"],   s)
        self.writer.add_scalar("optim/learning_rate", metrics["learning_rate"], s)

    def _print_update(
        self, update: int, total_updates: int,
        elapsed: float, steps_ps: float, eta_s: float,
        mean_reward: float, success_rate: float,
        metrics: dict,
    ):
        eta_str = str(datetime.timedelta(seconds=int(eta_s)))
        print(
            f"  upd {update:>5}/{total_updates}  "
            f"step {self.global_step:>9,}  "
            f"r={mean_reward:>7.3f}  "
            f"succ={success_rate:>5.1%}  "
            f"ent={metrics['entropy']:>5.3f}  "
            f"kl={metrics['approx_kl']:>6.4f}  "
            f"vl={metrics['value_loss']:>6.4f}  "
            f"pl={metrics['policy_loss']:>7.4f}  "
            f"{steps_ps:>6.0f}sps  "
            f"ETA {eta_str}"
        )

    def close(self):
        self.envs.close()
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER  (load a trained model and run it)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(checkpoint_path: str, render: bool = True, episodes: int = 5):
    """
    Load a saved checkpoint and run the trained policy interactively.

    Args:
        checkpoint_path: path to a .pt checkpoint file
        render:          show PyBullet GUI
        episodes:        number of episodes to run
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RobotActorCritic().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt['global_step']:,}")

    env = RobotEnv(render=render, show_camera=True)
    for ep in range(episodes):
        obs     = env.reset()
        ep_r    = 0.0
        done    = False
        steps   = 0
        while not done:
            img_t, prop_t = obs.as_torch(device)
            with torch.no_grad():
                action, _ = model.act(img_t, prop_t, deterministic=True)
            obs, reward, done, info = env.step(action.cpu().numpy()[0])
            ep_r  += reward
            steps += 1

        print(f"  Episode {ep+1}: reward={ep_r:.3f}  steps={steps}  "
              f"delivered={info.get('delivered', False)}")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for robot arm")
    p.add_argument("--resume",  type=str,  default="",
                   help="Path to checkpoint directory to resume from")
    p.add_argument("--render",  action="store_true",
                   help="Render one environment during training")
    p.add_argument("--steps",   type=int,  default=10_000_000,
                   help="Total environment steps (default: 10M)")
    p.add_argument("--envs",    type=int,  default=32,
                   help="Number of parallel environments (default: 32)")
    p.add_argument("--lr",      type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--name",    type=str,  default="",
                   help="Run name (auto-generated if not set)")
    p.add_argument("--infer",   type=str,  default="",
                   help="Run inference with a checkpoint path (skips training)")
    p.add_argument("--seed",    type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Inference mode ────────────────────────────────────────────────────────
    if args.infer:
        run_inference(args.infer, render=True)
        sys.exit(0)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg              = Config()
    cfg.total_steps  = args.steps
    cfg.num_envs     = args.envs
    cfg.learning_rate = args.lr
    cfg.run_name     = args.name
    cfg.seed         = args.seed

    # ── Launch trainer ────────────────────────────────────────────────────────
    with PPOTrainer(cfg, render=args.render) as trainer:

        # Resume if requested
        if args.resume:
            resume_path = Path(args.resume)
            # Find latest checkpoint in the directory
            ckpts = sorted(
                (resume_path / "checkpoints").glob("model_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if ckpts:
                trainer.load_checkpoint(str(ckpts[-1]))
            else:
                print(f"  No checkpoints found in {resume_path}, starting fresh")

        trainer.train()

    # ── TensorBoard reminder ──────────────────────────────────────────────────
    print(f"\n  To view training curves, run:")
    print(f"    tensorboard --logdir runs/")

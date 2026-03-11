"""
Robot Arm Neural Network — PPO Actor-Critic
============================================
Built specifically for: my_robot URDF (Onshape export)

Joint Map (extracted directly from robot.urdf):
  [0] shoulder_yaw   | revolute  | [-π,       π      ]
  [1] shoulder_pitch | revolute  | [-2.1306,  1.01099]
  [2] elbow_pitch    | revolute  | [-5.04657, 0.363952]
  [3] wrist_roll     | revolute  | [-π,       π      ]
  [4] rack_left      | prismatic | [-1.0,     1.0    ]  ← gripper drive joint
      rack_right     |           |  (mirrors rack_left via URDF <mimic> tag)
      gripper_pinion |           |  (mirrors rack_left via URDF <mimic> tag)

Camera: Arducam physically present on link_3 in the URDF.
        Virtual camera attached to this link in PyBullet simulation.

Total parameters: ~993,711  (≈ 1 million)

Architecture:
  ┌──────────────────────────────────────────┐
  │         CNN IMAGE TRUNK                  │
  │  (4,84,84) RGBD                         │
  │  Conv(4→32,8×8,s4) → (B,32,20,20)      │
  │  Conv(32→32,4×4,s2)→ (B,32,9,9)        │
  │  Conv(32→64,3×3,s1)→ (B,64,7,7)        │
  │  Flatten(3136) → FC(192)                │
  └──────────────┬───────────────────────────┘
                 │192
  ┌──────────────▼───────────────────────────┐
  │      PROPRIOCEPTION ENCODER              │
  │  Input(14): 5 pos + 5 vel               │
  │             + 3 target xyz + 1 gripper  │
  │  FC(128) → LayerNorm → FC(64)           │
  └──────────────┬───────────────────────────┘
                 │64
  Cat([192,64]) = 256
                 │
  ┌──────────────▼───────────────────────────┐
  │         FUSION MLP                       │
  │  FC(460)→LayerNorm→FC(256)→LN→FC(192)  │
  └───────┬──────────────────────────────────┘
          │192
   ┌──────┴──────┐
   ▼             ▼
[ACTOR]       [CRITIC]
FC(128)       FC(128)
→ μ(5), σ(5)  → V(1)

Usage:
    python robot_neural_network.py           # self-test: shapes + param count
    python robot_neural_network.py --train   # starts training (needs pybullet)
"""

import math
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ─────────────────────────────────────────────────────────────────────────────
#  JOINT CONFIGURATION  (from robot.urdf)
# ─────────────────────────────────────────────────────────────────────────────

JOINT_CONFIG = {
    "shoulder_yaw": {
        "index": 0, "type": "revolute",
        "lower": -math.pi,   "upper":  math.pi,
    },
    "shoulder_pitch": {
        "index": 1, "type": "revolute",
        "lower": -2.1306,    "upper":  1.01099,
    },
    "elbow_pitch": {
        "index": 2, "type": "revolute",
        "lower": -5.04657,   "upper":  0.363952,
    },
    "wrist_roll": {
        "index": 3, "type": "revolute",
        "lower": -math.pi,   "upper":  math.pi,
    },
    # rack_right and gripper_pinion are <mimic> joints: not direct outputs.
    "rack_left": {
        "index": 4, "type": "prismatic",
        "lower": -1.0,       "upper":  1.0,
    },
}

NUM_JOINTS      = 5       # controllable outputs
OBS_PROPRIO_DIM = 14      # 5 pos + 5 vel + 3 target xyz + 1 gripper
CAM_H, CAM_W   = 84, 84
CAM_C           = 4       # RGB(3) + Depth(1)

_s = sorted(JOINT_CONFIG.values(), key=lambda c: c["index"])
_JOINT_LOWER = torch.tensor([c["lower"] for c in _s], dtype=torch.float32)
_JOINT_UPPER = torch.tensor([c["upper"] for c in _s], dtype=torch.float32)
_JOINT_MID   = (_JOINT_LOWER + _JOINT_UPPER) / 2.0
_JOINT_HALF  = (_JOINT_UPPER - _JOINT_LOWER) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalise_joint_positions(pos: torch.Tensor) -> torch.Tensor:
    """Raw URDF joint positions → [-1, 1]."""
    mid  = _JOINT_MID.to(pos.device)
    half = _JOINT_HALF.to(pos.device)
    return (pos - mid) / half


def denormalise_actions(actions: torch.Tensor) -> torch.Tensor:
    """Network outputs in [-1, 1] -> raw joint values (rad / m), clamped to URDF limits.
    Guards against float32 precision at the tanh boundary (e.g. tanh ~= +/-pi)."""
    mid  = _JOINT_MID.to(actions.device)
    half = _JOINT_HALF.to(actions.device)
    lo   = _JOINT_LOWER.to(actions.device)
    hi   = _JOINT_UPPER.to(actions.device)
    return (actions * half + mid).clamp(min=lo, max=hi)


# ─────────────────────────────────────────────────────────────────────────────
#  CNN IMAGE TRUNK  (~645 K params)
# ─────────────────────────────────────────────────────────────────────────────

class CameraTrunk(nn.Module):
    """84×84×4 RGBD → 192-d feature vector."""

    def __init__(self, in_channels: int = CAM_C, out_dim: int = 192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),  # →(32,20,20)
            nn.Conv2d(32,          32, kernel_size=4, stride=2), nn.ReLU(),  # →(32, 9, 9)
            nn.Conv2d(32,          64, kernel_size=3, stride=1), nn.ReLU(),  # →(64, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(img))   # (B,192)


# ─────────────────────────────────────────────────────────────────────────────
#  PROPRIOCEPTION ENCODER  (~10 K params)
# ─────────────────────────────────────────────────────────────────────────────

class ProprioEncoder(nn.Module):
    """
    14-d normalised state vector → 64-d embedding.

    Vector layout:
        [0:5]   normalised joint positions
        [5:10]  joint velocities  (÷ 10, URDF velocity limit)
        [10:13] target xyz        (÷ 0.5 m, workspace radius)
        [13]    gripper state     (0=open, 1=closed)
    """

    def __init__(self, in_dim: int = OBS_PROPRIO_DIM, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)    # (B,64)


# ─────────────────────────────────────────────────────────────────────────────
#  ACTOR-CRITIC NETWORK  (~993,711 params total)
# ─────────────────────────────────────────────────────────────────────────────

class RobotActorCritic(nn.Module):
    """
    PPO Actor-Critic for the 5-DOF arm + rack-and-pinion gripper.

    Inputs:
        img    : (B, 4, 84, 84)  RGBD from wrist Arducam
        proprio: (B, 14)         normalised proprioception

    Outputs:
        Actor  : μ (B,5) + log_σ (B,5)  in [-1,1] (tanh-squashed)
        Critic : V(s) (B,1)
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self):
        super().__init__()

        self.camera_trunk = CameraTrunk()    # → 192-d
        self.proprio_enc  = ProprioEncoder() # →  64-d

        # Fusion: cat([192,64])=256 → 460 → 256 → 192
        self.fusion = nn.Sequential(
            nn.Linear(256, 460), nn.LayerNorm(460), nn.ReLU(),
            nn.Linear(460, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 192), nn.ReLU(),
        )

        self.actor_fc      = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
        self.actor_mu      = nn.Linear(128, NUM_JOINTS)
        self.actor_log_std = nn.Linear(128, NUM_JOINTS)

        self.critic = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_mu.weight,      gain=0.01)
        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight,    gain=1.0)

    def _features(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        cam  = self.camera_trunk(img)
        prop = self.proprio_enc(proprio)
        return self.fusion(torch.cat([cam, prop], dim=-1))   # (B,192)

    def _dist(self, features: torch.Tensor) -> Normal:
        x       = self.actor_fc(features)
        mu      = torch.tanh(self.actor_mu(x))
        log_std = self.actor_log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    # ── Public API ───────────────────────────────────────────────────────────

    def act(
        self,
        img: torch.Tensor,
        proprio: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action in [-1,1] normalised space.

        Returns:
            action_norm : (B, 5)
            log_prob    : (B,)
        """
        dist   = self._dist(self._features(img, proprio))
        action = dist.mean if deterministic else dist.rsample().clamp(-1.0, 1.0)
        return action, dist.log_prob(action).sum(dim=-1)

    def act_in_joint_space(
        self,
        img: torch.Tensor,
        proprio: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Like act(), but returns denormalised joint-space values (rad / m).
        Pass these directly to PyBullet setJointMotorControl2().
        """
        norm, lp = self.act(img, proprio, deterministic)
        return denormalise_actions(norm), lp

    def get_value(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Returns V(s). Shape: (B, 1)"""
        return self.critic(self._features(img, proprio))

    def evaluate_actions(
        self,
        img: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the current policy (PPO update step).

        Returns:
            log_probs : (B,)
            values    : (B, 1)
            entropy   : scalar
        """
        features  = self._features(img, proprio)
        dist      = self._dist(features)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1).mean()
        values    = self.critic(features)
        return log_probs, values, entropy

    def count_parameters(self) -> dict[str, int]:
        counts = {
            "camera_trunk": sum(p.numel() for p in self.camera_trunk.parameters()),
            "proprio_enc":  sum(p.numel() for p in self.proprio_enc.parameters()),
            "fusion":       sum(p.numel() for p in self.fusion.parameters()),
            "actor_head":  (sum(p.numel() for p in self.actor_fc.parameters())
                            + sum(p.numel() for p in self.actor_mu.parameters())
                            + sum(p.numel() for p in self.actor_log_std.parameters())),
            "critic_head":  sum(p.numel() for p in self.critic.parameters()),
        }
        counts["TOTAL"] = sum(counts.values())
        return counts


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_proprio_tensor(
    joint_positions:  list[float],
    joint_velocities: list[float],
    target_xyz:       list[float],
    gripper_state:    float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build and normalise the 14-d proprioception vector from PyBullet data.

    Joint ordering must match JOINT_CONFIG indices (0→4):
        shoulder_yaw, shoulder_pitch, elbow_pitch, wrist_roll, rack_left

    Returns: (1, 14) tensor.
    """
    pos  = torch.tensor(joint_positions,  dtype=torch.float32)
    vel  = torch.tensor(joint_velocities, dtype=torch.float32) / 10.0
    tgt  = torch.tensor(target_xyz,       dtype=torch.float32) / 0.5
    grip = torch.tensor([gripper_state],  dtype=torch.float32)
    return torch.cat([normalise_joint_positions(pos), vel, tgt, grip]) \
                 .unsqueeze(0).to(device)


def build_image_tensor(
    rgb:   np.ndarray,
    depth: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Stack PyBullet getCameraImage() outputs into a (1,4,84,84) tensor.

    rgb  : (84,84,3) uint8
    depth: (84,84)   float32 already in [0,1] from PyBullet
    """
    rgb_f   = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    depth_f = torch.from_numpy(depth).float().unsqueeze(0)
    return torch.cat([rgb_f, depth_f], dim=0).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  REWARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    ee_pos:           np.ndarray,
    target_pos:       np.ndarray,
    goal_pos:         np.ndarray,
    grasped:          bool,
    delivered:        bool,
    joint_positions:  list[float],
    prev_dist_target: float,
    prev_dist_goal:   float,
) -> tuple[float, bool, dict]:
    """
    Per-step reward for the pick-and-place task.

    Components:
        reach_reward    : +1.0 × improvement in dist-to-target  (dense)
        grasp_bonus     : +1.0 on successful grasp
        transport_reward: +1.0 × improvement in dist-to-goal    (post-grasp)
        delivery_bonus  : +5.0 on delivery → episode ends
        time_penalty    : -0.001 per step
        limit_penalty   : -0.1  per joint within 5 % of hard limit

    Returns: (reward, done, info_dict)
    """
    reward = 0.0
    done   = False
    info   = {}

    dist_target = float(np.linalg.norm(ee_pos - target_pos))
    dist_goal   = float(np.linalg.norm(ee_pos - goal_pos))

    reward += (prev_dist_target - dist_target)          # reach shaping
    if grasped:
        reward += 1.0
        reward += (prev_dist_goal - dist_goal)          # transport shaping
    if delivered:
        reward += 5.0
        done    = True

    reward -= 0.001                                     # time penalty

    violations = 0
    for cfg in JOINT_CONFIG.values():
        lo, hi = cfg["lower"], cfg["upper"]
        margin = (hi - lo) * 0.05
        if joint_positions[cfg["index"]] < lo + margin \
        or joint_positions[cfg["index"]] > hi - margin:
            reward -= 0.1
            violations += 1

    info.update(dist_to_target=dist_target, dist_to_goal=dist_goal,
                limit_violations=violations, delivered=delivered)
    return reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA CONFIGURATION  (Arducam on link_3, confirmed in URDF)
# ─────────────────────────────────────────────────────────────────────────────
#  URDF visual origin: xyz="-0.049945 0 -0.0708"  rpy="3.14159 0 -1.5708"

CAMERA_CONFIG = {
    "image_width":  CAM_W,
    "image_height": CAM_H,
    "fov_deg":      60.0,
    "near_plane":   0.01,
    "far_plane":    2.0,
    "local_offset": [-0.049945, 0.0, -0.0708],   # in link_3 frame
}


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    SEP = "=" * 62
    print(SEP)
    print("  Robot Arm Neural Network — Self-Test")
    print(SEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    model = RobotActorCritic().to(device)

    print("\n  Parameter counts:")
    for name, n in model.count_parameters().items():
        if name == "TOTAL":
            print(f"  {'─'*38}")
        print(f"  {name:<20}: {n:>10,}")

    B         = 8
    dummy_img = torch.rand(B, CAM_C, CAM_H, CAM_W, device=device)
    dummy_pro = torch.rand(B, OBS_PROPRIO_DIM, device=device) * 2 - 1

    print(f"\n  Forward pass (B={B}):")
    with torch.no_grad():
        acts, lps              = model.act(dummy_img, dummy_pro)
        vals                   = model.get_value(dummy_img, dummy_pro)
        lp2, v2, ent           = model.evaluate_actions(dummy_img, dummy_pro, acts)

    print(f"  act()              acts={tuple(acts.shape)}  log_probs={tuple(lps.shape)}")
    print(f"  get_value()        vals={tuple(vals.shape)}")
    print(f"  evaluate_actions() entropy={ent.item():.4f}")

    print(f"\n  Sample actions — batch[0] in joint space:")
    jv = denormalise_actions(acts[0].cpu())
    for name, cfg in JOINT_CONFIG.items():
        v    = jv[cfg["index"]].item()
        lo, hi = cfg["lower"], cfg["upper"]
        unit = "m" if cfg["type"] == "prismatic" else "rad"
        eps  = 1e-4   # tolerance for floating-point boundary hits (e.g. tanh ≈ π)
        ok   = "✓" if lo - eps <= v <= hi + eps else "✗ OUT OF RANGE"
        print(f"  {name:<20}: {v:+.4f} {unit}  [{lo:.3f},{hi:.3f}]  {ok}")

    print("\n  build_proprio_tensor:", end=" ")
    p = build_proprio_tensor([0.0]*5, [0.1]*5, [0.3, 0.0, 0.2], 0.5, device)
    print(f"shape={tuple(p.shape)}  ✓")

    print("  build_image_tensor:", end="  ")
    img_t = build_image_tensor(
        (np.random.rand(84,84,3)*255).astype(np.uint8),
        np.random.rand(84,84).astype(np.float32), device)
    print(f"shape={tuple(img_t.shape)}  ✓")

    print("  compute_reward:", end="       ")
    r, done, info = compute_reward(
        np.array([0.1,0,0.2]), np.array([0.3,0,0.1]), np.array([0,0.3,0]),
        False, False, [0.0]*5, 0.25, 0.40)
    print(f"reward={r:.4f}  done={done}  ✓")

    print(f"\n{SEP}")
    print("  All tests passed ✓")
    print(SEP)


if __name__ == "__main__":
    if "--train" in sys.argv:
        try:
            import pybullet  # noqa: F401
        except ImportError:
            print("ERROR: pybullet not installed.  Run:  pip install pybullet")
            sys.exit(1)
        print("Training mode: see robot_training.py for the full PPO loop.")
    else:
        _run_self_test()
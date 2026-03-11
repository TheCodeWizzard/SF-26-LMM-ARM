"""
robot_visualizer.py — Real-Time Training Visualizer
=====================================================
Displays a live dashboard while robot_training.py runs in parallel.
Can also replay a trained policy from a checkpoint.

Dashboard layout (single window, 4 panels):

  ┌─────────────────────┬─────────────────────┐
  │                     │                     │
  │   WRIST CAMERA      │   NEURAL NETWORK    │
  │   (live feed)       │   ARCHITECTURE      │
  │                     │                     │
  ├─────────────────────┼─────────────────────┤
  │                     │                     │
  │   REWARD / SUCCESS  │   LOSS CURVES       │
  │   CURVES            │   (policy, value,   │
  │                     │    entropy, KL)     │
  └─────────────────────┴─────────────────────┘

  Bottom bar: step count, SPS, ETA, joint states, grasp status

Data sources:
  • training_log.csv   — polled every second for new rows
  • robot.urdf + env   — one live RobotEnv instance for camera + joint display
  • checkpoint .pt     — loaded for policy rollout in replay mode

Modes:
  py robot_visualizer.py --watch runs/my_run       # live dashboard during training
  py robot_visualizer.py --replay runs/my_run      # replay best_model.pt
  py robot_visualizer.py --arch                    # network architecture only

Dependencies:
    pip install matplotlib numpy torch pybullet
"""

import sys
import csv
import time
import math
import threading
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # works on Windows; fall back to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from robot_neural_network import (
    RobotActorCritic, JOINT_CONFIG, NUM_JOINTS,
    CAM_C, CAM_H, CAM_W, OBS_PROPRIO_DIM,
)
from robot_env import RobotEnv, CTRL_JOINTS, JOINT_LIMITS, MAX_STEPS

import torch


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

BG        = "#0e1117"   # near-black background
PANEL_BG  = "#161b22"   # slightly lighter panel
GRID_COL  = "#21262d"   # subtle grid lines
TEXT_COL  = "#e6edf3"   # primary text
DIM_TEXT  = "#8b949e"   # secondary text
ACCENT1   = "#58a6ff"   # blue  — reward curve
ACCENT2   = "#3fb950"   # green — success rate
ACCENT3   = "#f78166"   # red/orange — policy loss
ACCENT4   = "#d2a8ff"   # purple — value loss
ACCENT5   = "#ffa657"   # orange — entropy
ACCENT6   = "#79c0ff"   # light blue — KL


# ─────────────────────────────────────────────────────────────────────────────
#  CSV LOG READER  (polls training_log.csv for new rows)
# ─────────────────────────────────────────────────────────────────────────────

class LogReader:
    """
    Reads training_log.csv written by robot_training.py.
    Polls the file on a background thread every `poll_interval` seconds.
    Thread-safe: all data is accessed via the `data` property.
    """

    COLUMNS = [
        "global_step", "update", "mean_reward", "mean_length",
        "success_rate", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", "learning_rate", "steps_per_sec",
    ]

    def __init__(self, csv_path: Path, poll_interval: float = 1.0):
        self._path          = csv_path
        self._poll_interval = poll_interval
        self._lock          = threading.Lock()
        self._rows_read     = 0

        # Rolling history (last 2000 updates)
        maxlen = 2000
        self._data = {col: deque(maxlen=maxlen) for col in self.COLUMNS}
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while self._running:
            self._read_new_rows()
            time.sleep(self._poll_interval)

    def _read_new_rows(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, newline="") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            new_rows = rows[self._rows_read:]
            if not new_rows:
                return
            with self._lock:
                for row in new_rows:
                    for col in self.COLUMNS:
                        try:
                            self._data[col].append(float(row[col]))
                        except (KeyError, ValueError):
                            pass
                self._rows_read += len(new_rows)
        except Exception:
            pass

    @property
    def data(self) -> dict:
        with self._lock:
            return {k: list(v) for k, v in self._data.items()}

    @property
    def num_rows(self) -> int:
        return self._rows_read

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK ARCHITECTURE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_network_architecture(ax: plt.Axes):
    """
    Draw a static diagram of the RobotActorCritic architecture on `ax`.
    Annotated with layer names, shapes, and parameter counts.
    """
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Network Architecture  (~993K params)",
                 color=TEXT_COL, fontsize=9, pad=6, fontweight="bold")

    # ── Layer definitions  [x_centre, y_centre, width, height, label, sublabel, colour]
    layers = [
        # CNN branch
        (1.5, 8.5, 1.8, 0.55, "Camera Input",   "(4, 84, 84)",   "#1f6feb"),
        (1.5, 7.6, 1.8, 0.55, "Conv 8×8 s4",    "→ (32,20,20)",  "#388bfd"),
        (1.5, 6.7, 1.8, 0.55, "Conv 4×4 s2",    "→ (32,9,9)",    "#388bfd"),
        (1.5, 5.8, 1.8, 0.55, "Conv 3×3 s1",    "→ (64,7,7)",    "#388bfd"),
        (1.5, 4.9, 1.8, 0.55, "Flatten + FC",   "→ 192-d",       "#1f6feb"),

        # Proprio branch
        (5.0, 8.5, 1.8, 0.55, "Proprio Input",  "(14,)",         "#1a7f37"),
        (5.0, 7.6, 1.8, 0.55, "FC + LayerNorm", "→ 128-d",       "#2ea043"),
        (5.0, 6.7, 1.8, 0.55, "FC",             "→ 64-d",        "#2ea043"),

        # Fusion
        (3.25, 3.7, 2.2, 0.55, "Concat [256]",  "192 + 64",      "#6e40c9"),
        (3.25, 2.8, 2.2, 0.55, "FC + LN",       "→ 460-d",       "#8957e5"),
        (3.25, 1.9, 2.2, 0.55, "FC + LN",       "→ 256-d",       "#8957e5"),
        (3.25, 1.0, 2.2, 0.55, "FC",            "→ 192-d",       "#6e40c9"),
    ]

    # Actor / Critic heads
    heads = [
        (1.5,  -0.2, 1.8, 0.55, "Actor FC",  "128-d",  "#b08800"),
        (5.0,  -0.2, 1.8, 0.55, "Critic FC", "128-d",  "#b08800"),
        (1.5,  -1.1, 1.8, 0.55, "μ  σ",      "5 + 5",  "#d29922"),
        (5.0,  -1.1, 1.8, 0.55, "V(s)",      "1",      "#d29922"),
    ]

    def draw_box(ax, x, y, w, h, label, sublabel, colour):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.04",
            facecolor=colour + "33",   # translucent fill
            edgecolor=colour,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.06, label, ha="center", va="center",
                color=TEXT_COL, fontsize=7.0, fontweight="bold")
        ax.text(x, y - 0.15, sublabel, ha="center", va="center",
                color=DIM_TEXT, fontsize=6.0)

    def arrow(ax, x1, y1, x2, y2, colour="#555566"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                   lw=1.0, mutation_scale=8))

    # Draw main layers
    for layer in layers:
        draw_box(ax, *layer)

    # Draw head layers (shifted down — ax y goes 0→10 so heads are near bottom)
    for x, y, w, h, label, sublabel, colour in heads:
        draw_box(ax, x, y + 10, w, h, label, sublabel, colour)

    # ── Arrows: CNN branch ─────────────────────────────────────────────────
    for ya, yb in [(8.5, 7.6), (7.6, 6.7), (6.7, 5.8), (5.8, 4.9)]:
        arrow(ax, 1.5, ya - 0.28, 1.5, yb + 0.28, ACCENT1)

    # ── Arrows: Proprio branch ─────────────────────────────────────────────
    for ya, yb in [(8.5, 7.6), (7.6, 6.7)]:
        arrow(ax, 5.0, ya - 0.28, 5.0, yb + 0.28, ACCENT2)

    # ── Converge to fusion ─────────────────────────────────────────────────
    arrow(ax, 1.5,  4.62, 2.8,  3.98, ACCENT1)
    arrow(ax, 5.0,  6.42, 3.70, 3.98, ACCENT2)

    # ── Fusion chain ───────────────────────────────────────────────────────
    for ya, yb in [(3.7, 2.8), (2.8, 1.9), (1.9, 1.0)]:
        arrow(ax, 3.25, ya - 0.28, 3.25, yb + 0.28, "#8957e5")

    # ── Split to heads ─────────────────────────────────────────────────────
    arrow(ax, 2.5,  0.72, 1.8,  9.52, "#d29922")   # → Actor
    arrow(ax, 4.0,  0.72, 4.7,  9.52, "#d29922")   # → Critic

    # ── Actor head chain ───────────────────────────────────────────────────
    arrow(ax, 1.5, 9.52, 1.5, 8.72, "#d29922")
    # ── Critic head chain ─────────────────────────────────────────────────
    arrow(ax, 5.0, 9.52, 5.0, 8.72, "#d29922")

    # ── Parameter count annotations ───────────────────────────────────────
    ax.text(0.1, 9.6, "CNN\n645K", color=ACCENT1, fontsize=6.5,
            va="top", ha="left", style="italic")
    ax.text(6.2, 9.6, "Proprio\n10K", color=ACCENT2, fontsize=6.5,
            va="top", ha="left", style="italic")
    ax.text(4.5, 3.7, "Fusion\n287K", color="#8957e5", fontsize=6.5,
            va="center", ha="left", style="italic")
    ax.text(6.2, 9.3, "Heads\n51K", color="#d29922", fontsize=6.5,
            va="top", ha="left", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
#  JOINT STATE BAR WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def draw_joint_bars(ax: plt.Axes, joint_positions: list[float]):
    """
    Draw horizontal bars showing each joint's current position
    within its URDF limits.
    """
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(CTRL_JOINTS))

    colours = [ACCENT1, ACCENT2, ACCENT5, ACCENT4, ACCENT3]
    units   = ["rad", "rad", "rad", "rad", "m"]

    for i, (name, pos) in enumerate(zip(CTRL_JOINTS, joint_positions)):
        lo, hi = JOINT_LIMITS[name]
        frac   = (pos - lo) / (hi - lo)   # 0 → 1
        y      = len(CTRL_JOINTS) - 1 - i
        colour = colours[i % len(colours)]

        # Track background
        ax.barh(y + 0.5, 1.0, height=0.55, left=0,
                color=GRID_COL, align="center")
        # Fill
        ax.barh(y + 0.5, max(frac, 0.01), height=0.55, left=0,
                color=colour + "99", align="center")
        # Label
        short = name.replace("shoulder_", "sh_").replace("_pitch", "_p") \
                    .replace("_yaw", "_y").replace("_roll", "_r") \
                    .replace("rack_left", "gripper")
        ax.text(0.01, y + 0.5, short, va="center", ha="left",
                color=TEXT_COL, fontsize=7.5, fontweight="bold")
        ax.text(0.99, y + 0.5, f"{pos:+.3f} {units[i]}",
                va="center", ha="right", color=DIM_TEXT, fontsize=7.0)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

class RobotVisualizer:
    """
    Live training dashboard.

    In --watch mode: reads training_log.csv and runs a single RobotEnv
    with the latest checkpoint to show camera + joint states.

    In --replay mode: loads best_model.pt and runs the full policy.
    """

    def __init__(
        self,
        run_dir:      Path,
        mode:         str  = "watch",   # "watch" | "replay" | "arch"
        update_hz:    float = 2.0,      # dashboard refresh rate
    ):
        self.run_dir   = run_dir
        self.mode      = mode
        self.update_hz = update_hz
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Log reader ────────────────────────────────────────────────────────
        csv_path      = run_dir / "training_log.csv"
        self.log      = LogReader(csv_path)

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = RobotActorCritic().to(self.device)
        self._load_latest_checkpoint()

        # ── Live environment (headless — just for obs) ────────────────────────
        self.env:      RobotEnv  = None
        self._obs                = None
        self._joint_positions    = [0.0] * NUM_JOINTS
        self._reward_history     = deque(maxlen=200)
        self._ep_done            = False
        self._ep_reward          = 0.0
        self._info               = {}
        self._env_lock           = threading.Lock()

        if mode in ("watch", "replay"):
            self._start_env_thread()

        # ── Figure setup ──────────────────────────────────────────────────────
        plt.style.use("dark_background")
        self.fig = plt.figure(
            figsize=(14, 8),
            facecolor=BG,
            num="Robot Training Visualizer",
        )
        self.fig.patch.set_facecolor(BG)
        self._build_layout()
        self._ani_running = True

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        """Create the 2×2 grid + bottom status bar."""
        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[10, 1],
            hspace=0.08,
            left=0.04, right=0.97, top=0.95, bottom=0.04,
        )
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[0],
            hspace=0.35, wspace=0.28,
        )

        self.ax_cam    = self.fig.add_subplot(inner[0, 0])  # wrist camera
        self.ax_arch   = self.fig.add_subplot(inner[0, 1])  # network arch
        self.ax_reward = self.fig.add_subplot(inner[1, 0])  # reward curves
        self.ax_loss   = self.fig.add_subplot(inner[1, 1])  # loss curves
        self.ax_status = self.fig.add_subplot(outer[1])     # status bar

        for ax in [self.ax_cam, self.ax_arch,
                   self.ax_reward, self.ax_loss, self.ax_status]:
            ax.set_facecolor(PANEL_BG)

        # Static: draw network architecture once
        draw_network_architecture(self.ax_arch)

        # Camera placeholder
        self.ax_cam.set_facecolor(PANEL_BG)
        self.ax_cam.axis("off")
        self.ax_cam.set_title("Wrist Camera  (RGBD → 84×84)",
                               color=TEXT_COL, fontsize=9,
                               pad=6, fontweight="bold")
        blank = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self._cam_img = self.ax_cam.imshow(blank, interpolation="nearest",
                                            aspect="equal")

        # Overlay text on camera
        self._cam_label = self.ax_cam.text(
            0.02, 0.97, "Waiting for env...",
            transform=self.ax_cam.transAxes,
            color=ACCENT2, fontsize=7.5, va="top",
            bbox=dict(facecolor=BG + "cc", edgecolor="none", pad=2),
        )

        self.fig.canvas.manager.set_window_title("Robot Training Visualizer")

    # ── Environment thread ────────────────────────────────────────────────────

    def _start_env_thread(self):
        """Run a single headless RobotEnv on a background thread."""
        def _run():
            self.env = RobotEnv(render=False, random_seed=0)
            obs      = self.env.reset()
            ep_r     = 0.0

            while self._ani_running:
                # Load latest checkpoint periodically
                self._load_latest_checkpoint()

                img_t, prop_t = obs.as_torch(self.device)
                with torch.no_grad():
                    deterministic = (self.mode == "replay")
                    action, _     = self.model.act(
                        img_t, prop_t, deterministic=deterministic
                    )
                action_np = action.cpu().numpy()[0]

                obs, reward, done, info = self.env.step(action_np)
                ep_r += reward

                with self._env_lock:
                    self._obs             = obs
                    self._joint_positions = (
                        self.env._get_joint_positions().tolist()
                    )
                    self._reward_history.append(ep_r)
                    self._info            = info
                    self._ep_done         = done
                    self._ep_reward       = ep_r

                if done:
                    obs  = self.env.reset()
                    ep_r = 0.0

                time.sleep(0.01)   # ~100 Hz env, dashboard refreshes at update_hz

        self._env_thread = threading.Thread(target=_run, daemon=True)
        self._env_thread.start()

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_latest_checkpoint(self):
        """
        Try to load the best_model.pt checkpoint.
        Falls back to the most recent model_<N>.pt if best not available.
        Silently skips if no checkpoint exists yet.
        """
        ckpt_dir = self.run_dir / "checkpoints"
        candidates = []

        best = ckpt_dir / "best_model.pt"
        if best.exists():
            candidates.append(best)

        numbered = sorted(
            ckpt_dir.glob("model_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
        )
        candidates.extend(numbered)

        if not candidates:
            return

        path = candidates[0]
        try:
            ckpt = torch.load(str(path), map_location=self.device,
                              weights_only=True)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self._ckpt_step = ckpt.get("global_step", 0)
        except Exception:
            pass   # checkpoint may be mid-write; skip silently

    # ── Update callbacks ──────────────────────────────────────────────────────

    def _update_camera_panel(self):
        """Refresh the wrist camera image."""
        with self._env_lock:
            obs  = self._obs
            info = dict(self._info)
            ep_r = self._ep_reward
            done = self._ep_done

        if obs is None:
            return

        # obs.image is (4, 84, 84); take RGB channels → (84, 84, 3)
        rgb = (obs.image[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
        self._cam_img.set_data(rgb)

        grasped   = info.get("grasped", False)
        d_target  = info.get("dist_to_target", 0.0)
        d_goal    = info.get("dist_to_goal", 0.0)
        step      = info.get("step", 0)
        delivered = info.get("delivered", False)

        status = (
            f"step {step}/{MAX_STEPS}  "
            f"reward {ep_r:+.2f}\n"
            f"dist_target {d_target:.3f}m  "
            f"dist_goal {d_goal:.3f}m\n"
            f"{'✓ GRASPED' if grasped else '○ reaching'}  "
            f"{'★ DELIVERED!' if delivered else ''}"
        )
        colour = ACCENT2 if delivered else (ACCENT5 if grasped else ACCENT1)
        self._cam_label.set_text(status)
        self._cam_label.set_color(colour)

    def _update_reward_panel(self, data: dict):
        """Redraw reward + success rate curves."""
        ax = self.ax_reward
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Episode Reward & Success Rate",
                     color=TEXT_COL, fontsize=9, pad=5, fontweight="bold")
        ax.tick_params(colors=DIM_TEXT, labelsize=7)
        ax.spines[:].set_color(GRID_COL)
        ax.yaxis.label.set_color(DIM_TEXT)
        ax.xaxis.label.set_color(DIM_TEXT)

        steps   = data["global_step"]
        rewards = data["mean_reward"]
        success = data["success_rate"]

        if len(steps) < 2:
            ax.text(0.5, 0.5, "Waiting for data…",
                    ha="center", va="center", transform=ax.transAxes,
                    color=DIM_TEXT, fontsize=9)
            return

        steps_m = [s / 1e6 for s in steps]   # convert to millions

        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL_BG)
        ax2.tick_params(colors=DIM_TEXT, labelsize=7)
        ax2.spines[:].set_color(GRID_COL)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Success Rate", color=ACCENT2, fontsize=7.5)

        ax.plot(steps_m, rewards, color=ACCENT1, linewidth=1.5,
                label="Mean Reward", alpha=0.9)
        ax2.plot(steps_m, success, color=ACCENT2, linewidth=1.2,
                 label="Success Rate", alpha=0.85, linestyle="--")

        # Smoothed reward overlay
        if len(rewards) >= 10:
            window = min(50, len(rewards) // 4)
            kernel = np.ones(window) / window
            smooth = np.convolve(rewards, kernel, mode="valid")
            sx     = steps_m[window - 1:][:len(smooth)]
            ax.plot(sx, smooth, color=ACCENT1, linewidth=2.5, alpha=0.6)

        ax.set_xlabel("Steps (millions)", color=DIM_TEXT, fontsize=7.5)
        ax.set_ylabel("Mean Episode Reward", color=ACCENT1, fontsize=7.5)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)

        # Current values annotation
        if rewards:
            ax.annotate(
                f"{rewards[-1]:.2f}",
                xy=(steps_m[-1], rewards[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=ACCENT1, fontsize=7, fontweight="bold",
            )
        if success:
            ax2.annotate(
                f"{success[-1]:.1%}",
                xy=(steps_m[-1], success[-1]),
                xytext=(8, 0), textcoords="offset points",
                color=ACCENT2, fontsize=7, fontweight="bold",
            )

        handles = [
            mpatches.Patch(color=ACCENT1, label="Mean Reward"),
            mpatches.Patch(color=ACCENT2, label="Success Rate"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper left",
                  facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL)

    def _update_loss_panel(self, data: dict):
        """Redraw loss curves."""
        ax = self.ax_loss
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Training Losses",
                     color=TEXT_COL, fontsize=9, pad=5, fontweight="bold")
        ax.tick_params(colors=DIM_TEXT, labelsize=7)
        ax.spines[:].set_color(GRID_COL)

        steps = data["global_step"]
        if len(steps) < 2:
            ax.text(0.5, 0.5, "Waiting for data…",
                    ha="center", va="center", transform=ax.transAxes,
                    color=DIM_TEXT, fontsize=9)
            return

        steps_m = [s / 1e6 for s in steps]

        curves = [
            ("policy_loss", "Policy Loss",  ACCENT3),
            ("value_loss",  "Value Loss",   ACCENT4),
            ("entropy",     "Entropy",      ACCENT5),
            ("approx_kl",   "Approx KL",   ACCENT6),
        ]

        for key, label, colour in curves:
            vals = data.get(key, [])
            if vals and len(vals) == len(steps_m):
                ax.plot(steps_m, vals, color=colour, linewidth=1.2,
                        label=label, alpha=0.85)
                # Smoothed overlay
                if len(vals) >= 10:
                    w  = min(30, len(vals) // 4)
                    sm = np.convolve(vals, np.ones(w)/w, mode="valid")
                    sx = steps_m[w - 1:][:len(sm)]
                    ax.plot(sx, sm, color=colour, linewidth=2.0, alpha=0.5)

        ax.set_xlabel("Steps (millions)", color=DIM_TEXT, fontsize=7.5)
        ax.set_ylabel("Loss", color=DIM_TEXT, fontsize=7.5)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=7, loc="upper right",
                  facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL)

    def _update_status_bar(self, data: dict):
        """Update the bottom status bar with live training metrics."""
        ax = self.ax_status
        ax.cla()
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        steps     = data["global_step"]
        sps_list  = data["steps_per_sec"]
        lr_list   = data["learning_rate"]

        cur_step  = int(steps[-1])      if steps    else 0
        sps       = float(sps_list[-1]) if sps_list else 0.0
        lr        = float(lr_list[-1])  if lr_list  else 0.0
        total     = 10_000_000
        pct       = cur_step / total * 100

        # Progress bar
        bar_w = 0.35
        ax.barh(0.5, bar_w, height=0.5, left=0.01,
                color=GRID_COL, align="center")
        ax.barh(0.5, bar_w * pct / 100, height=0.5, left=0.01,
                color=ACCENT1 + "aa", align="center")
        ax.text(0.01 + bar_w / 2, 0.5,
                f"{cur_step:,} / {total:,}  ({pct:.1f}%)",
                ha="center", va="center",
                color=TEXT_COL, fontsize=8, fontweight="bold")

        # Stats
        with self._env_lock:
            joints = list(self._joint_positions)

        joint_str = "  ".join(
            f"{n.replace('shoulder_','sh_').replace('_pitch','_p').replace('_yaw','_y').replace('_roll','_r').replace('rack_left','grip')}: "
            f"{v:+.2f}"
            for n, v in zip(CTRL_JOINTS, joints)
        )

        ax.text(0.38, 0.75, f"Steps/sec: {sps:,.0f}   LR: {lr:.2e}",
                color=DIM_TEXT, fontsize=7.5, va="center")
        ax.text(0.38, 0.25, joint_str,
                color=DIM_TEXT, fontsize=7.0, va="center")

        # Device
        dev_str = (f"CUDA: {torch.cuda.get_device_name(0)}"
                   if torch.cuda.is_available() else "CPU")
        ax.text(0.98, 0.5, dev_str,
                ha="right", va="center",
                color=ACCENT2, fontsize=7.5)

    # ── Main update ───────────────────────────────────────────────────────────

    def _update(self, _frame=None):
        """Called by matplotlib animation timer to refresh all panels."""
        data = self.log.data

        self._update_camera_panel()
        self._update_reward_panel(data)
        self._update_loss_panel(data)
        self._update_status_bar(data)

        self.fig.canvas.draw_idle()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        """Start the dashboard. Blocks until the window is closed."""
        from matplotlib.animation import FuncAnimation

        interval_ms = int(1000 / self.update_hz)
        self._animation = FuncAnimation(
            self.fig, self._update,
            interval  = interval_ms,
            cache_frame_data = False,
        )

        print(f"  Dashboard running at {self.update_hz} Hz — close window to exit")
        plt.show()
        self._ani_running = False

    def close(self):
        self._ani_running = False
        if self.env:
            self.env.close()
        self.log.stop()
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE-ONLY MODE
# ─────────────────────────────────────────────────────────────────────────────

def show_architecture_only():
    """Display just the network architecture diagram, no training data needed."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    draw_network_architecture(ax)
    fig.suptitle("RobotActorCritic  —  ~993,711 Parameters",
                 color=TEXT_COL, fontsize=11, fontweight="bold", y=0.98)

    # Parameter table
    model  = RobotActorCritic()
    counts = model.count_parameters()
    table_text = "\n".join(
        f"  {k:<22}: {v:>10,}" for k, v in counts.items()
    )
    fig.text(0.02, 0.02, table_text, color=DIM_TEXT,
             fontsize=7.5, family="monospace", va="bottom")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robot training visualizer")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--watch",  type=str, metavar="RUN_DIR",
                   help="Live dashboard — watch a training run in progress")
    g.add_argument("--replay", type=str, metavar="RUN_DIR",
                   help="Replay best_model.pt from a completed run")
    g.add_argument("--arch",   action="store_true",
                   help="Show network architecture diagram only")
    p.add_argument("--hz", type=float, default=2.0,
                   help="Dashboard refresh rate in Hz (default: 2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.arch:
        show_architecture_only()
        sys.exit(0)

    run_dir = Path(args.watch or args.replay)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    mode = "watch" if args.watch else "replay"
    viz  = RobotVisualizer(run_dir=run_dir, mode=mode, update_hz=args.hz)

    try:
        viz.run()
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
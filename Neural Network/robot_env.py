"""
robot_env.py — PyBullet Simulation Environment
================================================
Simulation environment for my_robot URDF (Onshape export).

Robot structure (from URDF):
    base
    └── shoulder_yaw   (revolute, ±π)
        └── shoulder_pitch (revolute, -2.13 → 1.01)
            └── link_1 (~0.234 m)
                └── elbow_pitch (revolute, -5.05 → 0.36)
                    └── link_2 (~0.130 m)
                        └── wrist_roll (revolute, ±π)
                            └── link_3
                                ├── rack_left  (prismatic, ±1.0 m) ← gripper drive
                                ├── rack_right (prismatic, mirrored)
                                └── gripper_pinion (revolute, mimic)

Camera: Arducam on link_3 at local offset [-0.049945, 0, -0.0708]
Workspace: max reach ~0.453 m, practical ~0.340 m radius

Typical usage:
    env = RobotEnv(render=True)
    obs = env.reset()
    while True:
        action = ...                      # numpy array shape (5,) in [-1, 1]
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()

Vectorised usage (for PPO with 16 parallel envs):
    envs = VectorizedRobotEnv(num_envs=16)
    obs_batch = envs.reset()             # dict with 'image':(16,4,84,84) 'proprio':(16,14)
    actions = ...                        # (16, 5)
    obs_batch, rewards, dones, infos = envs.step(actions)

Dependencies:
    pip install pybullet numpy opencv-python
"""

import os
import math
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pybullet as pb
import pybullet_data

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (all sourced from robot.urdf)
# ─────────────────────────────────────────────────────────────────────────────

URDF_PATH = Path(__file__).parent / "robot.urdf"

# Controllable joints — order must match neural_network.py JOINT_CONFIG indices
CTRL_JOINTS = [
    "shoulder_yaw",    # index 0
    "shoulder_pitch",  # index 1
    "elbow_pitch",     # index 2
    "wrist_roll",      # index 3
    "rack_left",       # index 4  (gripper drive; rack_right+pinion are mimic)
]

# Joint limits from URDF <limit> tags
JOINT_LIMITS = {
    "shoulder_yaw":   (-math.pi,      math.pi),
    "shoulder_pitch": (-2.1306,       1.01099),
    "elbow_pitch":    (-5.04657,      0.363952),
    "wrist_roll":     (-math.pi,      math.pi),
    "rack_left":      (-1.0,          1.0),
    # mimic joints — driven automatically by rack_left
    "rack_right":     (-1.0,          1.0),
    "gripper_pinion": (-2.39442,      1.02642),
}

# Effort/velocity limits from URDF
JOINT_MAX_FORCE    = 10.0   # Nm / N
JOINT_MAX_VELOCITY = 10.0   # rad/s or m/s

# Workspace (computed from URDF joint origins)
LINK1_LENGTH  = 0.2341   # m
LINK2_LENGTH  = 0.1302   # m
LINK3_LENGTH  = 0.0889   # m  (to gripper tip)
MAX_REACH     = LINK1_LENGTH + LINK2_LENGTH + LINK3_LENGTH  # 0.453 m
MIN_REACH     = 0.05     # m  (avoid singularities near base)
BASE_HEIGHT   = 0.0619   # m  (shoulder_yaw origin z from URDF)

# Camera config (Arducam on link_3, from URDF visual origin)
CAM_WIDTH     = 84
CAM_HEIGHT    = 84
CAM_FOV       = 60.0     # degrees
CAM_NEAR      = 0.01     # m
CAM_FAR       = 2.0      # m
# Local offset of Arducam in link_3 frame: xyz="-0.049945 0 -0.0708"
CAM_LOCAL_POS = [-0.049945, 0.0, -0.0708]

# Target object
OBJECT_RADIUS  = 0.025   # m  — small sphere as grasp target
OBJECT_MASS    = 0.05    # kg
GOAL_RADIUS    = 0.05    # m  — delivery zone radius (success threshold)
GRASP_DISTANCE = 0.06    # m  — end-effector must be within this to attempt grasp

# Episode settings
MAX_STEPS      = 1000    # steps per episode before timeout
PHYSICS_HZ     = 240     # PyBullet default simulation frequency
CTRL_HZ        = 30      # control frequency (network runs at this rate)
PHYSICS_STEPS_PER_CTRL = PHYSICS_HZ // CTRL_HZ   # = 8 physics steps per action


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """
    Full observation returned by env.step() and env.reset().

    image  : (4, 84, 84) float32 RGBD in [0, 1]  — wrist camera
    proprio: (14,)       float32 normalised proprioception:
                             [0:5]  joint positions normalised to [-1, 1]
                             [5:10] joint velocities / 10
                             [10:13] target xyz / 0.5
                             [13]   gripper state [0=open, 1=closed]
    """
    image:   np.ndarray   # (4, 84, 84) float32
    proprio: np.ndarray   # (14,)       float32

    def as_torch(self, device=None):
        """Convert to torch tensors with batch dim. Requires torch."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch not installed")
        import torch
        dev = device or torch.device("cpu")
        img = torch.from_numpy(self.image).unsqueeze(0).to(dev)      # (1,4,84,84)
        pro = torch.from_numpy(self.proprio).unsqueeze(0).to(dev)    # (1,14)
        return img, pro


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RobotEnv:
    """
    Single-instance PyBullet environment for the 5-DOF arm.

    Args:
        render       : show PyBullet GUI (True) or run headless (False)
        urdf_path    : path to robot.urdf  (defaults to same directory as this file)
        random_seed  : RNG seed for reproducible object placement
        show_camera  : open an OpenCV window showing the wrist camera feed
        goal_pos     : fixed goal position [x,y,z]; if None, randomised each episode
    """

    def __init__(
        self,
        render:      bool            = False,
        urdf_path:   Optional[Path]  = None,
        random_seed: Optional[int]   = None,
        show_camera: bool            = False,
        goal_pos:    Optional[list]  = None,
    ):
        self._render      = render
        self._urdf_path   = Path(urdf_path) if urdf_path else URDF_PATH
        self._rng         = np.random.default_rng(random_seed)
        self._show_camera = show_camera and _CV2_AVAILABLE
        self._fixed_goal  = np.array(goal_pos, dtype=np.float32) if goal_pos else None

        # PyBullet state
        self._client:    int = -1
        self._robot_id:  int = -1
        self._object_id: int = -1
        self._goal_id:   int = -1
        self._plane_id:  int = -1

        # Joint index maps (populated after URDF load)
        self._joint_name_to_idx: dict[str, int] = {}
        self._ctrl_indices:      list[int]       = []
        self._link_name_to_idx:  dict[str, int]  = {}

        # Episode state
        self._step_count:    int   = 0
        self._grasped:       bool  = False
        self._prev_dist_target: float = 0.0
        self._prev_dist_goal:   float = 0.0
        self._target_pos:    np.ndarray = np.zeros(3, dtype=np.float32)
        self._goal_pos:      np.ndarray = np.zeros(3, dtype=np.float32)
        self._gripper_state: float = 0.0   # 0=open, 1=closed

        self._connect()
        self._load_scene()

    # ── Connection & scene setup ──────────────────────────────────────────────

    def _connect(self):
        """Start PyBullet with GUI or DIRECT mode."""
        mode = pb.GUI if self._render else pb.DIRECT
        self._client = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._client)
        pb.setGravity(0, 0, -9.81, physicsClientId=self._client)
        pb.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._client)

        if self._render:
            # Camera looking at the robot from a useful angle
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.2, 0.0, 0.1],
                physicsClientId=self._client,
            )
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_SHADOWS, 1,
                physicsClientId=self._client,
            )

    def _load_scene(self):
        """Load ground plane, robot URDF, target object, and goal marker."""
        cid = self._client

        # Ground plane
        self._plane_id = pb.loadURDF(
            "plane.urdf", physicsClientId=cid
        )

        # Robot
        if not self._urdf_path.exists():
            raise FileNotFoundError(
                f"URDF not found at {self._urdf_path}\n"
                f"Place robot.urdf in the same folder as robot_env.py"
            )
        self._robot_id = pb.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=cid,
        )

        # Build joint/link index maps
        num_joints = pb.getNumJoints(self._robot_id, physicsClientId=cid)
        for i in range(num_joints):
            info = pb.getJointInfo(self._robot_id, i, physicsClientId=cid)
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self._joint_name_to_idx[name] = i
            self._link_name_to_idx[link_name] = i

        # Ordered list of controllable joint indices (matches JOINT_CONFIG order)
        self._ctrl_indices = [
            self._joint_name_to_idx[n] for n in CTRL_JOINTS
        ]

        # Disable default velocity motors (required for position control to work)
        for idx in self._ctrl_indices:
            pb.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=pb.VELOCITY_CONTROL,
                force=0,
                physicsClientId=cid,
            )

        # Target object — small red sphere
        obj_col  = pb.createCollisionShape(pb.GEOM_SPHERE, radius=OBJECT_RADIUS,
                                           physicsClientId=cid)
        obj_vis  = pb.createVisualShape(pb.GEOM_SPHERE, radius=OBJECT_RADIUS,
                                        rgbaColor=[0.9, 0.15, 0.15, 1.0],
                                        physicsClientId=cid)
        self._object_id = pb.createMultiBody(
            baseMass=OBJECT_MASS,
            baseCollisionShapeIndex=obj_col,
            baseVisualShapeIndex=obj_vis,
            basePosition=[0.3, 0.0, OBJECT_RADIUS],
            physicsClientId=cid,
        )

        # Goal marker — translucent green sphere (no collision)
        goal_vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=GOAL_RADIUS,
                                        rgbaColor=[0.15, 0.9, 0.15, 0.35],
                                        physicsClientId=cid)
        self._goal_id = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_vis,
            basePosition=[0.0, 0.3, OBJECT_RADIUS],
            physicsClientId=cid,
        )

    # ── Episode control ───────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment to a new episode.

        - Randomises object position within the reachable workspace
        - Randomises goal position (or uses fixed goal if provided)
        - Resets robot to a safe home configuration
        - Returns the first observation

        Returns:
            obs: Observation (image + proprio)
        """
        cid = self._client

        # Reset episode state
        self._step_count    = 0
        self._grasped       = False
        self._gripper_state = 0.0

        # ── Reset robot to home position ──────────────────────────────────────
        home_positions = {
            "shoulder_yaw":   0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch":    -1.0,
            "wrist_roll":     0.0,
            "rack_left":      -0.5,   # gripper open
        }
        for name, pos in home_positions.items():
            idx = self._joint_name_to_idx[name]
            pb.resetJointState(self._robot_id, idx, pos,
                               physicsClientId=cid)

        # ── Randomise target object position ──────────────────────────────────
        self._target_pos = self._sample_reachable_position()
        pb.resetBasePositionAndOrientation(
            self._object_id,
            self._target_pos.tolist(),
            [0, 0, 0, 1],
            physicsClientId=cid,
        )

        # ── Randomise or fix goal position ────────────────────────────────────
        if self._fixed_goal is not None:
            self._goal_pos = self._fixed_goal.copy()
        else:
            self._goal_pos = self._sample_reachable_position(
                exclude_near=self._target_pos, min_separation=0.15
            )
        pb.resetBasePositionAndOrientation(
            self._goal_id,
            self._goal_pos.tolist(),
            [0, 0, 0, 1],
            physicsClientId=cid,
        )

        # ── Step physics a few times to let things settle ─────────────────────
        for _ in range(10):
            pb.stepSimulation(physicsClientId=cid)

        # ── Initialise distance trackers ──────────────────────────────────────
        ee_pos = self._get_ee_position()
        self._prev_dist_target = float(np.linalg.norm(ee_pos - self._target_pos))
        self._prev_dist_goal   = float(np.linalg.norm(ee_pos - self._goal_pos))

        return self._get_observation()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Observation, float, bool, dict]:
        """
        Apply one control action and advance the simulation.

        Args:
            action: (5,) float array in [-1, 1], one value per controllable joint.
                    Index mapping:
                        [0] shoulder_yaw
                        [1] shoulder_pitch
                        [2] elbow_pitch
                        [3] wrist_roll
                        [4] rack_left  (gripper: -1=open, +1=closed)

        Returns:
            obs    : Observation
            reward : float
            done   : bool  (True on success or timeout)
            info   : dict  with diagnostic fields
        """
        cid = self._client

        # ── Denormalise actions to joint space ────────────────────────────────
        joint_targets = self._denormalise_action(action)

        # ── Apply position control to controllable joints ─────────────────────
        for i, idx in enumerate(self._ctrl_indices):
            pb.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=float(joint_targets[i]),
                force=JOINT_MAX_FORCE,
                maxVelocity=JOINT_MAX_VELOCITY,
                physicsClientId=cid,
            )

        # ── Drive mimic joints to follow rack_left ────────────────────────────
        self._update_mimic_joints(joint_targets[4])

        # ── Step physics at PHYSICS_HZ, control at CTRL_HZ ───────────────────
        for _ in range(PHYSICS_STEPS_PER_CTRL):
            pb.stepSimulation(physicsClientId=cid)
            if self._render:
                time.sleep(1.0 / PHYSICS_HZ)

        self._step_count += 1

        # ── Update gripper state ──────────────────────────────────────────────
        # rack_left position: -1=open, +1=closed → normalise to [0,1]
        rack_pos = self._get_joint_positions()[4]
        self._gripper_state = (rack_pos + 1.0) / 2.0   # [-1,1] → [0,1]

        # ── Check grasp ───────────────────────────────────────────────────────
        ee_pos = self._get_ee_position()
        obj_pos = np.array(
            pb.getBasePositionAndOrientation(self._object_id, physicsClientId=cid)[0],
            dtype=np.float32,
        )
        dist_to_object = float(np.linalg.norm(ee_pos - obj_pos))

        if (not self._grasped
                and dist_to_object < GRASP_DISTANCE
                and self._gripper_state > 0.6):
            self._grasped = True
            # Freeze object to end-effector (simple grasp simulation)
            self._attach_object_to_ee()

        if self._grasped:
            # Keep object at end-effector position
            self._update_object_position()
            obj_pos = self._target_pos = np.array(
                pb.getBasePositionAndOrientation(
                    self._object_id, physicsClientId=cid)[0],
                dtype=np.float32,
            )

        # ── Compute reward ────────────────────────────────────────────────────
        dist_to_target  = float(np.linalg.norm(ee_pos - self._target_pos))
        dist_to_goal    = float(np.linalg.norm(obj_pos - self._goal_pos))
        delivered       = self._grasped and dist_to_goal < GOAL_RADIUS
        joint_positions = self._get_joint_positions()

        reward, done, info = self._compute_reward(
            ee_pos, self._target_pos, self._goal_pos,
            self._grasped, delivered,
            joint_positions,
            self._prev_dist_target, self._prev_dist_goal,
        )

        self._prev_dist_target = dist_to_target
        self._prev_dist_goal   = dist_to_goal

        # Timeout
        if self._step_count >= MAX_STEPS:
            done = True
            info["timeout"] = True

        # ── Get observation ───────────────────────────────────────────────────
        obs = self._get_observation()

        # ── Optional: show camera feed ────────────────────────────────────────
        if self._show_camera:
            self._display_camera_feed(obs.image)

        info.update(
            step          = self._step_count,
            grasped       = self._grasped,
            dist_to_target= dist_to_target,
            dist_to_goal  = dist_to_goal,
            ee_pos        = ee_pos.tolist(),
        )
        return obs, reward, done, info

    def close(self):
        """Disconnect from PyBullet."""
        if self._client >= 0:
            pb.disconnect(physicsClientId=self._client)
            self._client = -1
        if self._show_camera and _CV2_AVAILABLE:
            cv2.destroyAllWindows()

    # ── Observation building ──────────────────────────────────────────────────

    def _get_observation(self) -> Observation:
        image   = self._get_camera_image()    # (4, 84, 84) float32
        proprio = self._get_proprio()         # (14,)       float32
        return Observation(image=image, proprio=proprio)

    def _get_proprio(self) -> np.ndarray:
        """
        Build the 14-d normalised proprioception vector.
        Layout: [pos×5 | vel×5 | target_xyz | gripper_state]
        """
        positions  = self._get_joint_positions()   # (5,) raw
        velocities = self._get_joint_velocities()  # (5,) raw

        # Normalise positions to [-1, 1] using URDF limits
        pos_norm = np.array([
            self._normalise(positions[i],
                            JOINT_LIMITS[CTRL_JOINTS[i]][0],
                            JOINT_LIMITS[CTRL_JOINTS[i]][1])
            for i in range(len(CTRL_JOINTS))
        ], dtype=np.float32)

        vel_norm  = (velocities / JOINT_MAX_VELOCITY).astype(np.float32)
        target_n  = (self._target_pos / 0.5).astype(np.float32)   # workspace ~0.5m
        grip      = np.array([self._gripper_state], dtype=np.float32)

        return np.concatenate([pos_norm, vel_norm, target_n, grip])

    def _get_camera_image(self) -> np.ndarray:
        """
        Render a 84×84 RGBD image from the wrist-mounted Arducam.

        The camera pose is computed by transforming CAM_LOCAL_POS from
        link_3's frame into world coordinates each step.

        Returns:
            (4, 84, 84) float32 array: channels [R, G, B, Depth] all in [0, 1]
        """
        cid = self._client

        # Get link_3 world pose
        link3_idx   = self._link_name_to_idx.get("link_3", -1)
        if link3_idx >= 0:
            link_state  = pb.getLinkState(
                self._robot_id, link3_idx,
                computeForwardKinematics=True,
                physicsClientId=cid,
            )
            link_pos    = np.array(link_state[4])   # world link frame position
            link_orn    = link_state[5]              # world link frame orientation (quat)
        else:
            # Fallback: use base position
            link_pos = np.zeros(3)
            link_orn = pb.getQuaternionFromEuler([0, 0, 0])

        # Transform camera local offset into world frame
        rot_matrix  = np.array(pb.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
        cam_world   = link_pos + rot_matrix @ np.array(CAM_LOCAL_POS)

        # Camera looks along the -Z axis of link_3 (Arducam orientation from URDF)
        # Forward vector in link_3 local frame is [0, 0, -1]
        forward_local = np.array([0.0, 0.0, -1.0])
        up_local      = np.array([0.0, 1.0,  0.0])
        forward_world = rot_matrix @ forward_local
        up_world      = rot_matrix @ up_local

        target_world  = cam_world + forward_world

        view_matrix = pb.computeViewMatrix(
            cameraEyePosition    = cam_world.tolist(),
            cameraTargetPosition = target_world.tolist(),
            cameraUpVector       = up_world.tolist(),
            physicsClientId      = cid,
        )
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov         = CAM_FOV,
            aspect      = CAM_WIDTH / CAM_HEIGHT,
            nearVal     = CAM_NEAR,
            farVal      = CAM_FAR,
            physicsClientId = cid,
        )

        _, _, rgba, depth_raw, _ = pb.getCameraImage(
            width            = CAM_WIDTH,
            height           = CAM_HEIGHT,
            viewMatrix       = view_matrix,
            projectionMatrix = proj_matrix,
            renderer         = pb.ER_TINY_RENDERER,   # fast headless renderer
            physicsClientId  = cid,
        )

        # RGB: uint8 → float32 [0, 1]
        rgb = np.array(rgba, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)
        rgb_f = rgb[:, :, :3].astype(np.float32) / 255.0   # (84,84,3)

        # Depth: linearise from PyBullet's non-linear depth buffer
        depth_buf = np.array(depth_raw, dtype=np.float32).reshape(CAM_HEIGHT, CAM_WIDTH)
        depth_lin = (CAM_FAR * CAM_NEAR
                     / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buf))
        depth_n   = (depth_lin - CAM_NEAR) / (CAM_FAR - CAM_NEAR)   # [0,1]
        depth_n   = depth_n[:, :, np.newaxis]                         # (84,84,1)

        # Stack → (84,84,4), then transpose → (4,84,84)
        rgbd = np.concatenate([rgb_f, depth_n], axis=2)               # (84,84,4)
        return rgbd.transpose(2, 0, 1).astype(np.float32)             # (4,84,84)

    # ── Joint state accessors ─────────────────────────────────────────────────

    def _get_joint_positions(self) -> np.ndarray:
        """Raw joint positions for the 5 controllable joints. Shape: (5,)"""
        states = pb.getJointStates(
            self._robot_id, self._ctrl_indices,
            physicsClientId=self._client,
        )
        return np.array([s[0] for s in states], dtype=np.float32)

    def _get_joint_velocities(self) -> np.ndarray:
        """Raw joint velocities for the 5 controllable joints. Shape: (5,)"""
        states = pb.getJointStates(
            self._robot_id, self._ctrl_indices,
            physicsClientId=self._client,
        )
        return np.array([s[1] for s in states], dtype=np.float32)

    def _get_ee_position(self) -> np.ndarray:
        """
        End-effector world position, estimated as the midpoint between
        rack_left and rack_right links (the gripper jaw centre).
        Falls back to link_3 position if gripper links are unavailable.
        """
        cid = self._client

        # Try to use rack_left link as the EE reference
        rack_idx = self._link_name_to_idx.get("rack_left", -1)
        if rack_idx >= 0:
            state = pb.getLinkState(
                self._robot_id, rack_idx,
                computeForwardKinematics=True,
                physicsClientId=cid,
            )
            return np.array(state[4], dtype=np.float32)

        # Fallback: link_3
        link3_idx = self._link_name_to_idx.get("link_3", -1)
        if link3_idx >= 0:
            state = pb.getLinkState(
                self._robot_id, link3_idx,
                computeForwardKinematics=True,
                physicsClientId=cid,
            )
            return np.array(state[4], dtype=np.float32)

        return np.zeros(3, dtype=np.float32)

    # ── Mimic joint handling ──────────────────────────────────────────────────

    def _update_mimic_joints(self, rack_left_raw: float):
        """
        Drive rack_right and gripper_pinion to mirror rack_left.
        In the real robot these are driven by the URDF <mimic> tag;
        PyBullet ignores <mimic>, so we enforce it manually.
        """
        cid = self._client

        # rack_right: same displacement, opposite direction
        rack_right_idx = self._joint_name_to_idx.get("rack_right", -1)
        if rack_right_idx >= 0:
            pb.setJointMotorControl2(
                self._robot_id, rack_right_idx,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=-rack_left_raw,
                force=JOINT_MAX_FORCE,
                maxVelocity=JOINT_MAX_VELOCITY,
                physicsClientId=cid,
            )

        # gripper_pinion: rack_left position maps to pinion rotation
        # multiplier=1.0 per URDF <mimic multiplier="1.0"/>
        pinion_idx = self._joint_name_to_idx.get("gripper_pinion", -1)
        if pinion_idx >= 0:
            # Clamp to pinion limits from URDF: [-2.39442, 1.02642]
            pinion_target = float(np.clip(
                rack_left_raw,
                JOINT_LIMITS["gripper_pinion"][0],
                JOINT_LIMITS["gripper_pinion"][1],
            ))
            pb.setJointMotorControl2(
                self._robot_id, pinion_idx,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=pinion_target,
                force=JOINT_MAX_FORCE,
                maxVelocity=JOINT_MAX_VELOCITY,
                physicsClientId=cid,
            )

    # ── Grasp simulation ──────────────────────────────────────────────────────

    def _attach_object_to_ee(self):
        """
        Freeze the object at the current end-effector position.
        This simulates a successful grasp without a full contact constraint.
        In future: replace with pb.createConstraint() for physically-based grasping.
        """
        pb.resetBaseVelocity(
            self._object_id,
            linearVelocity  = [0, 0, 0],
            angularVelocity = [0, 0, 0],
            physicsClientId = self._client,
        )

    def _update_object_position(self):
        """Keep grasped object locked to end-effector position each step."""
        ee_pos = self._get_ee_position()
        pb.resetBasePositionAndOrientation(
            self._object_id,
            ee_pos.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self._client,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(
        ee_pos:           np.ndarray,
        target_pos:       np.ndarray,
        goal_pos:         np.ndarray,
        grasped:          bool,
        delivered:        bool,
        joint_positions:  np.ndarray,
        prev_dist_target: float,
        prev_dist_goal:   float,
    ) -> tuple[float, bool, dict]:
        """
        Per-step shaped reward for pick-and-place.

        Components:
            reach shaping   : +1.0 × improvement toward object   (always active)
            grasp bonus     : +1.0 on first successful grasp
            transport shaping: +1.0 × improvement toward goal    (post-grasp)
            delivery bonus  : +5.0 on successful delivery
            time penalty    : -0.001 per step
            limit penalty   : -0.1 per joint within 5% of limit
        """
        reward = 0.0
        done   = False
        info   = {}

        dist_target = float(np.linalg.norm(ee_pos - target_pos))
        dist_goal   = float(np.linalg.norm(
            np.array(pb.getBasePositionAndOrientation.__doc__ or [0])  # dummy
            if False else ee_pos - goal_pos
        ))
        dist_goal = float(np.linalg.norm(ee_pos - goal_pos))

        # Reach shaping
        reward += (prev_dist_target - dist_target)
        info["reach_delta"] = prev_dist_target - dist_target

        # Grasp bonus (one-time)
        if grasped:
            reward += 1.0
            # Transport shaping
            reward += (prev_dist_goal - dist_goal)
            info["transport_delta"] = prev_dist_goal - dist_goal

        # Delivery
        if delivered:
            reward += 5.0
            done    = True
            info["delivered"] = True

        # Time penalty
        reward -= 0.001

        # Joint limit penalty (5% soft margin)
        violations = 0
        for i, name in enumerate(CTRL_JOINTS):
            lo, hi = JOINT_LIMITS[name]
            margin = (hi - lo) * 0.05
            if joint_positions[i] < lo + margin or joint_positions[i] > hi - margin:
                reward -= 0.1
                violations += 1
        info["limit_violations"] = violations

        return reward, done, info

    # ── Workspace sampling ────────────────────────────────────────────────────

    def _sample_reachable_position(
        self,
        exclude_near: Optional[np.ndarray] = None,
        min_separation: float = 0.0,
        max_attempts: int = 50,
    ) -> np.ndarray:
        """
        Sample a random position within the robot's reachable workspace.

        Workspace is approximated as an annular sector on the ground plane:
            radius : [MIN_REACH, MAX_REACH * 0.85]
            height : [OBJECT_RADIUS, MAX_REACH * 0.4]
            angle  : [-π, π]  (full 360° around base)
        """
        for _ in range(max_attempts):
            r     = self._rng.uniform(MIN_REACH, MAX_REACH * 0.85)
            theta = self._rng.uniform(-math.pi, math.pi)
            z     = self._rng.uniform(OBJECT_RADIUS, MAX_REACH * 0.4)

            pos = np.array([
                r * math.cos(theta),
                r * math.sin(theta),
                z,
            ], dtype=np.float32)

            if exclude_near is None:
                return pos
            if np.linalg.norm(pos - exclude_near) >= min_separation:
                return pos

        # Fallback if rejection sampling fails
        return np.array([0.25, 0.0, OBJECT_RADIUS], dtype=np.float32)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        """Map [lo, hi] → [-1, 1]."""
        mid  = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return (value - mid) / half

    @staticmethod
    def _denormalise_action(action: np.ndarray) -> np.ndarray:
        """
        Map network action in [-1, 1] → raw joint targets (rad / m),
        clamped to URDF limits.
        """
        result = np.zeros(len(CTRL_JOINTS), dtype=np.float32)
        for i, name in enumerate(CTRL_JOINTS):
            lo, hi = JOINT_LIMITS[name]
            mid    = (lo + hi) / 2.0
            half   = (hi - lo) / 2.0
            result[i] = float(np.clip(action[i] * half + mid, lo, hi))
        return result

    def _display_camera_feed(self, image: np.ndarray):
        """Show the wrist camera feed in an OpenCV window."""
        if not _CV2_AVAILABLE:
            return
        # image is (4, 84, 84); take RGB channels → (84, 84, 3), scale up for visibility
        rgb = (image[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (336, 336), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Wrist Camera", rgb)
        cv2.waitKey(1)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_actions(self) -> int:
        return len(CTRL_JOINTS)

    @property
    def observation_shapes(self) -> dict:
        return {
            "image":   (4, CAM_HEIGHT, CAM_WIDTH),
            "proprio": (14,),
        }

    @property
    def target_position(self) -> np.ndarray:
        return self._target_pos.copy()

    @property
    def goal_position(self) -> np.ndarray:
        return self._goal_pos.copy()

    @property
    def is_grasped(self) -> bool:
        return self._grasped

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  VECTORISED ENVIRONMENT  (runs N envs in parallel threads for PPO)
# ─────────────────────────────────────────────────────────────────────────────

class VectorizedRobotEnv:
    """
    Runs N independent RobotEnv instances sequentially on the main thread.

    PyBullet on Windows does not support creating or resetting physics clients
    from background threads — doing so causes a deadlock during startup.
    All PyBullet calls therefore happen on the main thread, one env at a time.
    This is slower than true parallelism but completely stable on Windows.

    For the 185H the recommended num_envs is 4-8 for stable performance.

    Args:
        num_envs   : number of environments (recommend 4-8 on Windows)
        render_idx : index of env to show in GUI (-1 = none)
        seed       : base random seed (each env gets seed + i)
    """

    def __init__(
        self,
        num_envs:   int = 4,
        render_idx: int = -1,
        seed:       int = 0,
        urdf_path:  Optional[Path] = None,
    ):
        self.num_envs = num_envs
        self._envs: list[RobotEnv] = []

        # Sequential init — PyBullet is NOT thread-safe on Windows
        print(f"  Initialising {num_envs} environments (sequential, Windows-safe)...")
        for i in range(num_envs):
            print(f"    env {i+1}/{num_envs}...", end=" ", flush=True)
            render = (i == render_idx)
            env = RobotEnv(
                render      = render,
                urdf_path   = urdf_path,
                random_seed = seed + i,
            )
            self._envs.append(env)
            print("ready")

        print(f"  All {num_envs} environments ready.\n")

        # Results storage
        self._obs_images   = np.zeros((num_envs, 4, CAM_HEIGHT, CAM_WIDTH), dtype=np.float32)
        self._obs_proprios = np.zeros((num_envs, 14), dtype=np.float32)
        self._rewards      = np.zeros(num_envs, dtype=np.float32)
        self._dones        = np.zeros(num_envs, dtype=bool)
        self._infos: list[dict] = [{} for _ in range(num_envs)]

    def reset(self) -> dict[str, np.ndarray]:
        """
        Reset all environments sequentially.
        Returns dict with keys 'image' (N,4,84,84) and 'proprio' (N,14).
        """
        for i in range(self.num_envs):
            obs = self._envs[i].reset()
            self._obs_images[i]   = obs.image
            self._obs_proprios[i] = obs.proprio

        return {"image":   self._obs_images.copy(),
                "proprio": self._obs_proprios.copy()}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        """
        Step all environments sequentially with their respective actions.

        Args:
            actions: (N, 5) float32 array in [-1, 1]

        Returns:
            obs    : dict {'image': (N,4,84,84), 'proprio': (N,14)}
            rewards: (N,) float32
            dones  : (N,) bool
            infos  : list of N dicts
        """
        for i in range(self.num_envs):
            obs, reward, done, info = self._envs[i].step(actions[i])
            if done:
                obs = self._envs[i].reset()
            self._obs_images[i]   = obs.image
            self._obs_proprios[i] = obs.proprio
            self._rewards[i]      = reward
            self._dones[i]        = done
            self._infos[i]        = info

        return (
            {"image":   self._obs_images.copy(),
             "proprio": self._obs_proprios.copy()},
            self._rewards.copy(),
            self._dones.copy(),
            list(self._infos),
        )

    def close(self):
        for env in self._envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    SEP = "=" * 62
    print(SEP)
    print("  Robot Environment — Self-Test")
    print(SEP)

    # ── Test 1: Single environment, headless ─────────────────────────────────
    print("\n  [1/4] Single environment (headless) ...")
    with RobotEnv(render=False, random_seed=42) as env:

        print(f"  Observation shapes : {env.observation_shapes}")
        print(f"  Num actions        : {env.num_actions}")

        obs = env.reset()
        print(f"  reset() image      : {obs.image.shape} "
              f"range=[{obs.image.min():.3f}, {obs.image.max():.3f}]")
        print(f"  reset() proprio    : {obs.proprio.shape} "
              f"range=[{obs.proprio.min():.3f}, {obs.proprio.max():.3f}]")
        print(f"  Target pos         : {env.target_position}")
        print(f"  Goal   pos         : {env.goal_position}")

        # ── Test 2: Random rollout ────────────────────────────────────────────
        print("\n  [2/4] Random rollout (50 steps) ...")
        total_reward = 0.0
        for step in range(50):
            action = np.random.uniform(-1, 1, size=5).astype(np.float32)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                obs = env.reset()
                break

        print(f"  Steps completed    : {info['step']}")
        print(f"  Total reward       : {total_reward:.4f}")
        print(f"  Final dist target  : {info['dist_to_target']:.4f} m")
        print(f"  Final dist goal    : {info['dist_to_goal']:.4f} m")
        print(f"  Grasped            : {info['grasped']}")
        print(f"  Limit violations   : {info['limit_violations']}")

    # ── Test 3: Denormalise action ────────────────────────────────────────────
    print("\n  [3/4] Action denormalisation ...")
    env_tmp = RobotEnv.__new__(RobotEnv)   # no __init__, just test static method
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw    = RobotEnv._denormalise_action(action)
    for i, name in enumerate(CTRL_JOINTS):
        lo, hi = JOINT_LIMITS[name]
        unit = "m" if name == "rack_left" else "rad"
        mid  = (lo + hi) / 2.0
        ok   = "✓" if abs(raw[i] - mid) < 1e-4 else "✗"
        print(f"  {name:<20}: action=0.0 → {raw[i]:+.4f} {unit}  "
              f"(expected mid={mid:+.4f})  {ok}")

    # ── Test 4: Vectorised environment ────────────────────────────────────────
    print("\n  [4/4] Vectorised environment (4 envs) ...")
    with VectorizedRobotEnv(num_envs=4, seed=0) as envs:
        obs_batch = envs.reset()
        print(f"  reset() image shape  : {obs_batch['image'].shape}")
        print(f"  reset() proprio shape: {obs_batch['proprio'].shape}")

        actions = np.random.uniform(-1, 1, size=(4, 5)).astype(np.float32)
        obs_batch, rewards, dones, infos = envs.step(actions)
        print(f"  step() rewards       : {rewards}")
        print(f"  step() dones         : {dones}")

    print(f"\n{SEP}")
    print("  All tests passed ✓")
    print(f"  Ready for training with robot_training.py")
    print(SEP)


if __name__ == "__main__":
    import sys
    if "--render" in sys.argv:
        # Interactive demo: render one episode with GUI
        print("Running interactive demo (close window to exit) ...")
        with RobotEnv(render=True, show_camera=True, random_seed=0) as env:
            obs = env.reset()
            for _ in range(MAX_STEPS):
                action = np.random.uniform(-1, 1, size=5).astype(np.float32)
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
    else:
        _run_self_test()
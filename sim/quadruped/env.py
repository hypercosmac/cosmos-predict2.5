# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unitree Go2 Quadruped Environment in Outdoor Scene

A high-resolution MuJoCo environment featuring the Unitree Go2 quadruped
robot navigating a lush outdoor park/orchard setting. Includes:
- Realistic Go2 mesh model with 12 actuated joints
- Outdoor terrain with grass, dirt paths, trees
- Multiple camera angles for cinematic rendering
- Built-in locomotion controllers (trot gait, turn, etc.)

Usage:
    from sim.quadruped.env import Go2Env

    env = Go2Env(render_width=1280, render_height=720)
    obs, info = env.reset()
    rgb = env.render()  # Returns (720, 1280, 3) uint8
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError:
    raise ImportError("MuJoCo required. Install: pip install mujoco")


# ============================================================================
# Gait Generator
# ============================================================================

class TrotGaitGenerator:
    """
    Generates trot gait joint targets for the Go2 quadruped.

    In a trot gait, diagonal leg pairs move together:
    - Phase 0: FL + RR swing, FR + RL stance
    - Phase 1: FR + RL swing, FL + RR stance

    The gait is parameterized by:
    - frequency: steps per second
    - stride_length: how far each step reaches (forward/backward)
    - body_height: target standing height
    - turn_rate: yaw angular velocity for turning
    """

    # Joint order in Go2: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #                      RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf

    # Standing pose (from Go2 keyframe "home")
    STAND_ANGLES = np.array([
        0.0, 0.9, -1.8,   # FL
        0.0, 0.9, -1.8,   # FR
        0.0, 0.9, -1.8,   # RL
        0.0, 0.9, -1.8,   # RR
    ])

    def __init__(self, frequency: float = 2.0, dt: float = 0.002):
        self.frequency = frequency
        self.dt = dt
        self.phase = 0.0

    def get_targets(
        self,
        forward_speed: float = 0.0,
        lateral_speed: float = 0.0,
        turn_rate: float = 0.0,
        body_height: float = 0.27,
    ) -> np.ndarray:
        """
        Compute joint angle targets for current gait phase.

        The trot gait swings diagonal leg pairs (FL+RR, FR+RL) in anti-phase.
        Each leg cycle has a swing phase (foot in the air, moving forward)
        and a stance phase (foot on ground, pushing backward).

        The thigh joint swings forward/back for stride. The calf joint lifts
        during swing to clear the ground, and extends during stance for push-off.

        Args:
            forward_speed: Desired forward velocity (-1 to 1)
            lateral_speed: Desired lateral velocity (-1 to 1)
            turn_rate: Desired yaw rate (-1 to 1)
            body_height: Target body height (meters)

        Returns:
            Joint angle targets, shape (12,)
        """
        # Advance phase
        self.phase += self.frequency * self.dt * 2 * math.pi
        self.phase = self.phase % (2 * math.pi)

        targets = self.STAND_ANGLES.copy()

        # Scale inputs — tuned for the Go2's joint ranges.
        # IMPORTANT: The Go2 thigh joint axis is +y. By the right-hand rule,
        # positive rotation swings the leg BACKWARD. So we negate the stride
        # so that forward_speed > 0 produces forward walking.
        stride = forward_speed * (-0.4)    # negated: positive speed = forward motion
        lateral = lateral_speed * 0.15     # hip abduction/adduction
        yaw = turn_rate * 0.2             # differential hip offset for turning

        # Trot gait: diagonal pairs in anti-phase
        # FL + RR (phase 0), FR + RL (phase + pi)
        phase_fl_rr = self.phase
        phase_fr_rl = self.phase + math.pi

        swing_fl_rr = math.sin(phase_fl_rr)
        swing_fr_rl = math.sin(phase_fr_rl)

        # Leg lift during swing (only when leg is moving forward = positive sin)
        lift_fl_rr = max(0, swing_fl_rr) * 0.5
        lift_fr_rl = max(0, swing_fr_rl) * 0.5

        # Stance push: extend calf slightly during ground contact for propulsion
        push_fl_rr = max(0, -swing_fl_rr) * 0.15
        push_fr_rl = max(0, -swing_fr_rl) * 0.15

        def apply_leg(idx, swing, lift, push, lat_sign, yaw_sign):
            # idx: start index in the 12-dim array (0, 3, 6, 9)
            targets[idx + 0] += lat_sign * lateral + yaw_sign * yaw   # hip abduction
            targets[idx + 1] += stride * swing                        # thigh forward/back
            targets[idx + 2] += -lift + push                          # calf: lift in swing, push in stance

        # FL (left front) - indices 0,1,2
        apply_leg(0, swing_fl_rr, lift_fl_rr, push_fl_rr, lat_sign=1, yaw_sign=1)
        # FR (right front) - indices 3,4,5
        apply_leg(3, swing_fr_rl, lift_fr_rl, push_fr_rl, lat_sign=-1, yaw_sign=1)
        # RL (left rear) - indices 6,7,8
        apply_leg(6, swing_fr_rl, lift_fr_rl, push_fr_rl, lat_sign=1, yaw_sign=-1)
        # RR (right rear) - indices 9,10,11
        apply_leg(9, swing_fl_rr, lift_fl_rr, push_fl_rr, lat_sign=-1, yaw_sign=-1)

        return targets


# ============================================================================
# Main Environment
# ============================================================================

class Go2Env:
    """
    Unitree Go2 quadruped in an outdoor environment.

    Provides:
    - High-resolution RGB rendering (up to 1920x1080)
    - Built-in locomotion with trot gait generator
    - Natural language command interface via command_velocity()
    - Multiple cinematic camera views

    Attributes:
        render_width, render_height: Image resolution
        JOINT_NAMES: List of 12 actuated joint names
    """

    JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    ACTUATOR_NAMES = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    # Named locations in the scene
    LANDMARKS = {
        "waypoint_a": np.array([2.5, 0.5]),
        "waypoint_b": np.array([-2.0, -1.5]),
        "tree_1": np.array([3.0, 1.5]),
        "tree_2": np.array([-2.5, 2.0]),
        "tree_3": np.array([1.0, -3.5]),
        "tree_4": np.array([-3.8, -1.0]),
        "ball": np.array([2.0, -0.5]),
        "center": np.array([0.0, 0.0]),
    }

    def __init__(
        self,
        render_width: int = 1280,
        render_height: int = 720,
        sim_steps_per_action: int = 10,
    ):
        """
        Args:
            render_width: Width of rendered images
            render_height: Height of rendered images
            sim_steps_per_action: Physics steps per control step (dt=0.002 -> 50Hz control at 10)
        """
        self.render_width = render_width
        self.render_height = render_height
        self.sim_steps_per_action = sim_steps_per_action

        # Load model
        scene_path = Path(__file__).parent / "assets" / "outdoor_scene.xml"
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene XML not found at {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self._camera = mujoco.MjvCamera()

        # Gait controller
        self.gait = TrotGaitGenerator(frequency=3.0, dt=self.model.opt.timestep)

        # PD gains for joint position tracking — higher gains for snappy tracking
        # Hip abduction needs less gain; thigh and calf need more for propulsion
        self._kp = np.tile([30.0, 60.0, 60.0], 4)   # [hip, thigh, calf] x 4 legs
        self._kd = np.tile([1.0, 3.0, 3.0], 4)

        # State
        self._step_count = 0
        self._target_pos = np.zeros(2)
        self._current_command = {"forward": 0.0, "lateral": 0.0, "turn": 0.0}

        # Camera presets
        self._camera_presets = {
            "follow_behind": {
                "distance": 2.0,
                "elevation": -20,
                "azimuth_offset": 180,  # Behind the robot
                "height_offset": 0.5,
            },
            "follow_side": {
                "distance": 2.5,
                "elevation": -15,
                "azimuth_offset": 90,
                "height_offset": 0.4,
            },
            "cinematic_high": {
                "distance": 4.0,
                "elevation": -35,
                "azimuth_offset": 210,
                "height_offset": 0.8,
            },
            "close_up": {
                "distance": 1.2,
                "elevation": -10,
                "azimuth_offset": 160,
                "height_offset": 0.3,
            },
            "bird_eye": {
                "distance": 6.0,
                "elevation": -70,
                "azimuth_offset": 0,
                "height_offset": 0.0,
            },
            "dramatic_low": {
                "distance": 1.8,
                "elevation": -5,
                "azimuth_offset": 200,
                "height_offset": 0.15,
            },
        }
        self._active_camera = "cinematic_high"

        # Cache body/joint IDs
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._waypoint_a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "waypoint_A")
        self._waypoint_b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "waypoint_B")
        self._ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_position(self) -> np.ndarray:
        """Robot base (x, y, z)."""
        return self.data.xpos[self._base_body_id].copy()

    @property
    def base_orientation(self) -> np.ndarray:
        """Robot base quaternion (w, x, y, z)."""
        return self.data.xquat[self._base_body_id].copy()

    @property
    def base_yaw(self) -> float:
        """Robot heading angle from quaternion."""
        q = self.base_orientation
        # Extract yaw from quaternion
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
        return math.atan2(siny_cosp, cosy_cosp)

    @property
    def base_velocity(self) -> np.ndarray:
        """Robot base linear velocity."""
        return self.data.cvel[self._base_body_id, 3:6].copy()

    @property
    def joint_positions(self) -> np.ndarray:
        """Current joint positions (12,)."""
        # First 7 qpos are freejoint (pos + quat), then 12 joints
        return self.data.qpos[7:19].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Current joint velocities (12,)."""
        # First 6 qvel are freejoint (lin + ang), then 12 joints
        return self.data.qvel[6:18].copy()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.

        Options:
            robot_pos: (x, y) starting position
            robot_yaw: starting heading
            target: (x, y) target position
        """
        options = options or {}
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Reset gait
        self.gait.phase = 0.0

        # Set initial pose from keyframe "home"
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Override robot position
        if "robot_pos" in options:
            self.data.qpos[0] = options["robot_pos"][0]
            self.data.qpos[1] = options["robot_pos"][1]
        else:
            self.data.qpos[0] = rng.uniform(-1.0, 1.0)
            self.data.qpos[1] = rng.uniform(-1.0, 1.0)

        # Override robot yaw (modify quaternion)
        if "robot_yaw" in options:
            yaw = options["robot_yaw"]
        else:
            yaw = rng.uniform(-math.pi, math.pi)
        self.data.qpos[3] = math.cos(yaw / 2)  # qw
        self.data.qpos[4] = 0.0                 # qx
        self.data.qpos[5] = 0.0                 # qy
        self.data.qpos[6] = math.sin(yaw / 2)  # qz

        # Set target
        if "target" in options:
            self._target_pos = np.array(options["target"][:2])
        else:
            targets = list(self.LANDMARKS.values())
            self._target_pos = targets[rng.integers(len(targets))].copy()

        # Move waypoint A to target
        if self._waypoint_a_id >= 0:
            self.model.body_pos[self._waypoint_a_id, :2] = self._target_pos

        # Forward to settle
        mujoco.mj_forward(self.model, self.data)

        # Zero commands
        self._current_command = {"forward": 0.0, "lateral": 0.0, "turn": 0.0}

        obs = self.render()
        return obs, self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def command_velocity(
        self,
        forward: float = 0.0,
        lateral: float = 0.0,
        turn: float = 0.0,
    ):
        """
        Set velocity command for the robot.

        Args:
            forward: Forward speed, -1 (backward) to 1 (forward)
            lateral: Lateral speed, -1 (right) to 1 (left)
            turn: Turn rate, -1 (right) to 1 (left)
        """
        self._current_command = {
            "forward": np.clip(forward, -1, 1),
            "lateral": np.clip(lateral, -1, 1),
            "turn": np.clip(turn, -1, 1),
        }

    def step(
        self,
        action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.

        If action is None, uses the gait generator with current velocity command.
        If action is provided, it should be joint torques (12,).
        """
        for _ in range(self.sim_steps_per_action):
            if action is None:
                # Use gait generator
                targets = self.gait.get_targets(
                    forward_speed=self._current_command["forward"],
                    lateral_speed=self._current_command["lateral"],
                    turn_rate=self._current_command["turn"],
                )
                # PD control to convert position targets to torques
                pos_err = targets - self.joint_positions
                vel_err = -self.joint_velocities
                torques = self._kp * pos_err + self._kd * vel_err
                self.data.ctrl[:12] = np.clip(torques, -23.7, 23.7)
            else:
                self.data.ctrl[:12] = np.clip(action, -23.7, 23.7)

            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute reward (distance to target)
        pos_2d = self.base_position[:2]
        dist = np.linalg.norm(pos_2d - self._target_pos)
        reward = -dist
        if dist < 0.3:
            reward += 10.0

        # Termination: fell over or reached target
        height = self.base_position[2]
        fell = height < 0.15
        reached = dist < 0.3
        terminated = fell or reached
        truncated = self._step_count >= 2000

        obs = self.render()
        info = self._get_info()
        info["fell"] = fell
        info["reached_target"] = reached

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def set_camera(self, preset: str):
        """Set camera to a named preset."""
        if preset in self._camera_presets:
            self._active_camera = preset

    def render(self, camera: Optional[str] = None) -> np.ndarray:
        """
        Render the scene.

        Args:
            camera: Camera preset name. If None, uses active camera.

        Returns:
            RGB image (H, W, 3) uint8
        """
        preset_name = camera or self._active_camera
        preset = self._camera_presets.get(preset_name, self._camera_presets["cinematic_high"])

        # Get robot position and heading for follow camera
        robot_pos = self.base_position
        robot_yaw_deg = math.degrees(self.base_yaw)

        # Configure camera
        self._camera.lookat[:] = [
            robot_pos[0],
            robot_pos[1],
            robot_pos[2] + preset["height_offset"],
        ]
        self._camera.distance = preset["distance"]
        self._camera.elevation = preset["elevation"]
        self._camera.azimuth = robot_yaw_deg + preset["azimuth_offset"]

        self.renderer.update_scene(self.data, camera=self._camera)
        return self.renderer.render()

    def render_all_views(self) -> Dict[str, np.ndarray]:
        """Render from all camera presets."""
        views = {}
        for name in self._camera_presets:
            views[name] = self.render(camera=name)
        return views

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> Dict[str, Any]:
        pos = self.base_position
        dist = np.linalg.norm(pos[:2] - self._target_pos)
        return {
            "robot_position": pos[:2].copy(),
            "robot_height": pos[2],
            "robot_yaw": self.base_yaw,
            "robot_state": np.array([pos[0], pos[1], self.base_yaw]),
            "target_position": self._target_pos.copy(),
            "distance_to_target": dist,
            "step": self._step_count,
            "success": dist < 0.3,
            "joint_positions": self.joint_positions.copy(),
        }

    def set_target(self, position: np.ndarray):
        """Move the target waypoint to a new position."""
        self._target_pos = np.array(position[:2])
        if self._waypoint_a_id >= 0:
            self.model.body_pos[self._waypoint_a_id, :2] = self._target_pos

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Factory Environment for Cosmos-Predict2.5 World Model Demo

A MuJoCo-based simulated factory with a mobile robot that can navigate
between workstations, pick up objects, and deliver them. Designed to
demonstrate action-conditioned world model capabilities.

Usage:
    from sim.factory.env import FactoryEnv

    env = FactoryEnv(render_size=256)
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space_sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rgb = env.render()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError:
    raise ImportError(
        "MuJoCo is required for the factory simulation. "
        "Install it with: pip install mujoco"
    )


class FactoryEnv:
    """
    Factory environment with mobile robot for world model demos.

    The robot has a mobile base (x, y, yaw) and a simple 2-DOF arm with gripper.
    Tasks include navigation to goals and object manipulation.

    Attributes:
        render_size: Size of rendered RGB images (square)
        camera_name: Name of camera to use for rendering
        max_episode_steps: Maximum steps per episode
    """

    # Action dimensions
    # [base_vx, base_vy, base_vyaw, shoulder, elbow, gripper]
    ACTION_DIM = 6

    # Navigation-only action subset [base_vx, base_vy, base_vyaw]
    NAV_ACTION_DIM = 3

    # Predefined locations in the factory
    LOCATIONS = {
        "center": np.array([0.0, 0.0]),
        "workstation1": np.array([-1.2, 1.2]),
        "workstation2": np.array([1.2, 1.2]),
        "storage": np.array([-1.3, -1.2]),
        "delivery": np.array([1.2, -1.2]),
    }

    def __init__(
        self,
        render_size: int = 256,
        camera_name: str = "overview",
        max_episode_steps: int = 200,
        navigation_only: bool = False,
        headless: bool = True,
    ):
        """
        Initialize the factory environment.

        Args:
            render_size: Width and height of rendered images
            camera_name: Camera to use ('overview', 'robot_view', or 'side')
            max_episode_steps: Maximum steps before truncation
            navigation_only: If True, only use base movement (3D action)
            headless: If True, use offscreen rendering
        """
        self.render_size = render_size
        self.camera_name = camera_name
        self.max_episode_steps = max_episode_steps
        self.navigation_only = navigation_only
        self.headless = headless

        # Load MuJoCo model
        self._model_path = Path(__file__).parent / "assets" / "factory.xml"
        if not self._model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {self._model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self.data = mujoco.MjData(self.model)

        # Add cameras programmatically if not in XML
        self._setup_cameras()

        # Create renderer
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self._camera = mujoco.MjvCamera()

        # State tracking
        self._step_count = 0
        self._goal_position = np.zeros(2)
        self._task_type = "navigation"  # 'navigation' or 'manipulation'

        # Cache actuator and joint indices
        self._cache_indices()

    def _cache_indices(self):
        """Cache frequently used indices for efficiency."""
        # Actuator indices
        self._act_base_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_x")
        self._act_base_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_y")
        self._act_base_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_yaw")
        self._act_shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder")
        self._act_elbow = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow")
        self._act_gripper = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

        # Joint indices
        self._jnt_robot_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_x")
        self._jnt_robot_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_y")
        self._jnt_robot_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_yaw")

        # Body indices
        self._body_robot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
        self._body_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self._body_gripper = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    def _setup_cameras(self):
        """Configure camera views for rendering."""
        # The model spec doesn't easily allow adding cameras after loading,
        # so we'll configure the renderer to use different viewpoints

        # Overview camera settings (will be applied during render)
        self._camera_configs = {
            "overview": {
                "lookat": np.array([0.0, 0.0, 0.3]),
                "distance": 5.0,
                "azimuth": 135,
                "elevation": -45,
            },
            "robot_view": {
                "lookat": np.array([0.0, 0.0, 0.2]),
                "distance": 2.5,
                "azimuth": 180,
                "elevation": -30,
            },
            "side": {
                "lookat": np.array([0.0, 0.0, 0.3]),
                "distance": 4.5,
                "azimuth": 90,
                "elevation": -20,
            },
            "top_down": {
                "lookat": np.array([0.0, 0.0, 0.0]),
                "distance": 5.5,
                "azimuth": 0,
                "elevation": -90,
            },
        }

    @property
    def action_dim(self) -> int:
        """Return action dimension based on mode."""
        return self.NAV_ACTION_DIM if self.navigation_only else self.ACTION_DIM

    def action_space_sample(self) -> np.ndarray:
        """Sample a random action."""
        if self.navigation_only:
            # Base velocity: vx, vy in [-0.5, 0.5], vyaw in [-1, 1]
            return np.concatenate([
                np.random.uniform(-0.5, 0.5, 2),
                np.random.uniform(-1.0, 1.0, 1),
            ])
        else:
            # Full action: base + arm
            return np.concatenate([
                np.random.uniform(-0.5, 0.5, 2),   # base vx, vy
                np.random.uniform(-1.0, 1.0, 1),   # base vyaw
                np.random.uniform(-0.5, 0.5, 2),   # shoulder, elbow
                np.random.uniform(0, 0.04, 1),     # gripper
            ])

    @property
    def robot_position(self) -> np.ndarray:
        """Get robot (x, y) position."""
        qpos_addr_x = self.model.jnt_qposadr[self._jnt_robot_x]
        qpos_addr_y = self.model.jnt_qposadr[self._jnt_robot_y]
        return np.array([self.data.qpos[qpos_addr_x], self.data.qpos[qpos_addr_y]])

    @property
    def robot_yaw(self) -> float:
        """Get robot yaw angle."""
        qpos_addr = self.model.jnt_qposadr[self._jnt_robot_yaw]
        return self.data.qpos[qpos_addr]

    @property
    def robot_state(self) -> np.ndarray:
        """Get full robot state (x, y, yaw)."""
        return np.array([*self.robot_position, self.robot_yaw])

    @property
    def goal_position(self) -> np.ndarray:
        """Get current goal position."""
        return self._goal_position.copy()

    @property
    def end_effector_position(self) -> np.ndarray:
        """Get gripper/end-effector position in world frame."""
        return self.data.xpos[self._body_gripper].copy()

    def _set_goal_position(self, position: np.ndarray):
        """Move the goal marker to a new position."""
        self._goal_position = position.copy()
        # Update goal marker body position
        body_id = self._body_goal
        self.model.body_pos[body_id, :2] = position

    def _randomize_robot_pose(self, rng: np.random.Generator):
        """Randomize robot starting position and orientation."""
        # Random position within bounds (avoiding obstacles)
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            x = rng.uniform(-1.5, 1.5)
            y = rng.uniform(-1.5, 1.5)

            # Check distance from workstations and storage
            pos = np.array([x, y])
            min_dist = min(
                np.linalg.norm(pos - loc)
                for loc in self.LOCATIONS.values()
                if not np.allclose(loc, self.LOCATIONS["center"])
            )
            valid = min_dist > 0.5  # Keep away from obstacles
            attempts += 1

        qpos_addr_x = self.model.jnt_qposadr[self._jnt_robot_x]
        qpos_addr_y = self.model.jnt_qposadr[self._jnt_robot_y]
        qpos_addr_yaw = self.model.jnt_qposadr[self._jnt_robot_yaw]

        self.data.qpos[qpos_addr_x] = x
        self.data.qpos[qpos_addr_y] = y
        self.data.qpos[qpos_addr_yaw] = rng.uniform(-np.pi, np.pi)

    def _randomize_goal(self, rng: np.random.Generator):
        """Randomize goal position."""
        # Choose a named location or random position
        if rng.random() < 0.7:
            # Use a named location
            location_names = list(self.LOCATIONS.keys())
            location_names.remove("center")  # Don't use center as goal
            chosen = rng.choice(location_names)
            goal = self.LOCATIONS[chosen] + rng.uniform(-0.2, 0.2, 2)
        else:
            # Random position
            goal = rng.uniform([-1.5, -1.5], [1.5, 1.5])

        self._set_goal_position(goal)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict with:
                - 'robot_pos': (x, y) starting position
                - 'robot_yaw': starting orientation
                - 'goal_pos': (x, y) goal position
                - 'task': 'navigation' or 'manipulation'

        Returns:
            observation: RGB image of the scene
            info: Dictionary with state information
        """
        options = options or {}
        rng = np.random.default_rng(seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Set task type
        self._task_type = options.get("task", "navigation")

        # Set robot pose
        if "robot_pos" in options:
            qpos_addr_x = self.model.jnt_qposadr[self._jnt_robot_x]
            qpos_addr_y = self.model.jnt_qposadr[self._jnt_robot_y]
            self.data.qpos[qpos_addr_x] = options["robot_pos"][0]
            self.data.qpos[qpos_addr_y] = options["robot_pos"][1]
        else:
            self._randomize_robot_pose(rng)

        if "robot_yaw" in options:
            qpos_addr = self.model.jnt_qposadr[self._jnt_robot_yaw]
            self.data.qpos[qpos_addr] = options["robot_yaw"]

        # Set goal
        if "goal_pos" in options:
            self._set_goal_position(np.array(options["goal_pos"]))
        else:
            self._randomize_goal(rng)

        # Forward dynamics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        obs = self.render()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action vector
                Navigation mode (3D): [vx, vy, vyaw]
                Full mode (6D): [vx, vy, vyaw, shoulder, elbow, gripper]

        Returns:
            observation: RGB image
            reward: Reward signal
            terminated: True if goal reached
            truncated: True if max steps exceeded
            info: State information
        """
        action = np.asarray(action).flatten()

        # Apply action to actuators
        if self.navigation_only:
            assert len(action) >= 3, f"Navigation action must be 3D, got {len(action)}"
            self.data.ctrl[self._act_base_x] = action[0]
            self.data.ctrl[self._act_base_y] = action[1]
            self.data.ctrl[self._act_base_yaw] = action[2]
            # Keep arm in default position
            self.data.ctrl[self._act_shoulder] = 0.0
            self.data.ctrl[self._act_elbow] = 0.0
            self.data.ctrl[self._act_gripper] = 0.02
        else:
            assert len(action) >= 6, f"Full action must be 6D, got {len(action)}"
            self.data.ctrl[self._act_base_x] = action[0]
            self.data.ctrl[self._act_base_y] = action[1]
            self.data.ctrl[self._act_base_yaw] = action[2]
            self.data.ctrl[self._act_shoulder] = action[3]
            self.data.ctrl[self._act_elbow] = action[4]
            self.data.ctrl[self._act_gripper] = action[5]

        # Simulate
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Get observation
        obs = self.render()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_success()
        truncated = self._step_count >= self.max_episode_steps

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        """Compute reward based on task."""
        distance_to_goal = np.linalg.norm(self.robot_position - self._goal_position)

        if self._task_type == "navigation":
            # Dense reward: negative distance
            reward = -distance_to_goal

            # Bonus for reaching goal
            if distance_to_goal < 0.2:
                reward += 10.0

        else:  # manipulation
            reward = -distance_to_goal
            # Could add manipulation-specific rewards here

        return reward

    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        distance_to_goal = np.linalg.norm(self.robot_position - self._goal_position)
        return distance_to_goal < 0.15

    def _get_info(self) -> Dict[str, Any]:
        """Get information about current state."""
        distance_to_goal = np.linalg.norm(self.robot_position - self._goal_position)
        return {
            "robot_position": self.robot_position.copy(),
            "robot_yaw": self.robot_yaw,
            "robot_state": self.robot_state.copy(),
            "goal_position": self._goal_position.copy(),
            "distance_to_goal": distance_to_goal,
            "step": self._step_count,
            "task": self._task_type,
            "success": distance_to_goal < 0.15,
            "end_effector_position": self.end_effector_position.copy(),
        }

    def render(self, camera_name: Optional[str] = None) -> np.ndarray:
        """
        Render the scene as an RGB image.

        Args:
            camera_name: Override default camera ('overview', 'robot_view', 'side', 'top_down')

        Returns:
            RGB image as numpy array (H, W, 3) uint8
        """
        camera = camera_name or self.camera_name
        config = self._camera_configs.get(camera, self._camera_configs["overview"])

        # Configure camera view
        self._camera.lookat[:] = config["lookat"]
        self._camera.distance = config["distance"]
        self._camera.azimuth = config["azimuth"]
        self._camera.elevation = config["elevation"]

        # Update renderer scene with our custom camera
        self.renderer.update_scene(self.data, camera=self._camera)

        # Render
        rgb = self.renderer.render()

        return rgb

    def render_all_views(self) -> Dict[str, np.ndarray]:
        """Render from all camera views."""
        views = {}
        for name in self._camera_configs:
            views[name] = self.render(camera_name=name)
        return views

    def close(self):
        """Clean up resources."""
        if hasattr(self, "renderer"):
            self.renderer.close()


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_random_actions():
    """Run a quick demo with random actions."""
    print("Creating factory environment...")
    env = FactoryEnv(render_size=256, navigation_only=True)

    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial robot position: {info['robot_position']}")
    print(f"Goal position: {info['goal_position']}")

    frames = [obs]
    actions = []

    print("Running 50 random steps...")
    for i in range(50):
        action = env.action_space_sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        actions.append(action)

        if terminated:
            print(f"Goal reached at step {i}!")
            break

    print(f"Final robot position: {info['robot_position']}")
    print(f"Final distance to goal: {info['distance_to_goal']:.3f}")

    env.close()

    return frames, actions, info


if __name__ == "__main__":
    # Run demo when executed directly
    frames, actions, info = demo_random_actions()
    print(f"\nCollected {len(frames)} frames and {len(actions)} actions")

    # Optionally save video
    try:
        import imageio
        video_path = "/tmp/factory_demo.mp4"
        imageio.mimwrite(video_path, frames, fps=20)
        print(f"Saved demo video to {video_path}")
    except ImportError:
        print("Install imageio to save demo video: pip install imageio[ffmpeg]")

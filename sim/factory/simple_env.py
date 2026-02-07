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
Simple 2D Factory Environment (No MuJoCo Required)

A lightweight 2D simulation of a factory floor with a mobile robot.
Uses only NumPy for physics and basic rendering - no external physics
engine required. Good for quick demos and testing.

This is a fallback for when MuJoCo is not available.

Usage:
    from sim.factory.simple_env import SimpleFactoryEnv

    env = SimpleFactoryEnv(render_size=256)
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space_sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]):
    """Draw a filled circle on the image."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = color


def draw_rectangle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
    """Draw a filled rectangle on the image."""
    img[y1:y2, x1:x2] = color


def draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], thickness: int = 2):
    """Draw a line using Bresenham's algorithm with thickness."""
    # Simple implementation - draws pixels along the line
    num_points = max(abs(x2 - x1), abs(y2 - y1), 1) * 2
    xs = np.linspace(x1, x2, num_points).astype(int)
    ys = np.linspace(y1, y2, num_points).astype(int)

    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] = color


class SimpleFactoryEnv:
    """
    Simple 2D factory environment with differential-drive robot.

    No external physics engine required - uses simple kinematic simulation
    with collision detection.

    Attributes:
        render_size: Size of rendered RGB images (square)
        max_episode_steps: Maximum steps per episode
    """

    # World bounds (meters)
    WORLD_SIZE = 4.0  # 4m x 4m factory floor

    # Robot parameters
    ROBOT_RADIUS = 0.15  # meters

    # Predefined locations
    LOCATIONS = {
        "center": np.array([0.0, 0.0]),
        "workstation1": np.array([-1.2, 1.2]),
        "workstation2": np.array([1.2, 1.2]),
        "storage": np.array([-1.3, -1.2]),
        "delivery": np.array([1.2, -1.2]),
    }

    # Obstacles: (x, y, width, height)
    OBSTACLES = [
        (-1.2, 1.2, 0.5, 0.4),   # Workstation 1
        (1.2, 1.2, 0.5, 0.5),    # Workstation 2
        (-1.3, -1.2, 0.7, 0.5),  # Storage shelves
    ]

    def __init__(
        self,
        render_size: int = 256,
        max_episode_steps: int = 200,
        dt: float = 0.05,
    ):
        """
        Initialize the simple factory environment.

        Args:
            render_size: Width and height of rendered images
            max_episode_steps: Maximum steps before truncation
            dt: Simulation timestep (seconds)
        """
        self.render_size = render_size
        self.max_episode_steps = max_episode_steps
        self.dt = dt

        # Robot state: [x, y, yaw]
        self._robot_pos = np.zeros(2)
        self._robot_yaw = 0.0

        # Goal state
        self._goal_pos = np.zeros(2)

        # Step counter
        self._step_count = 0

        # Random state
        self._rng = np.random.default_rng()

    @property
    def action_dim(self) -> int:
        return 3  # [vx, vy, vyaw]

    def action_space_sample(self) -> np.ndarray:
        """Sample a random action."""
        return np.array([
            self._rng.uniform(-0.5, 0.5),   # vx
            self._rng.uniform(-0.3, 0.3),   # vy
            self._rng.uniform(-1.0, 1.0),   # vyaw
        ])

    @property
    def robot_position(self) -> np.ndarray:
        return self._robot_pos.copy()

    @property
    def robot_yaw(self) -> float:
        return self._robot_yaw

    @property
    def robot_state(self) -> np.ndarray:
        return np.array([self._robot_pos[0], self._robot_pos[1], self._robot_yaw])

    @property
    def goal_position(self) -> np.ndarray:
        return self._goal_pos.copy()

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        # World: [-2, 2] x [-2, 2]
        # Pixel: [0, render_size] x [0, render_size]
        # Note: y is flipped (pixel y increases downward)
        px = int((x + self.WORLD_SIZE / 2) / self.WORLD_SIZE * self.render_size)
        py = int((self.WORLD_SIZE / 2 - y) / self.WORLD_SIZE * self.render_size)
        return px, py

    def _meters_to_pixels(self, m: float) -> int:
        """Convert meters to pixels."""
        return int(m / self.WORLD_SIZE * self.render_size)

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check if position collides with obstacles or walls."""
        x, y = pos

        # Wall collision
        half = self.WORLD_SIZE / 2 - self.ROBOT_RADIUS
        if abs(x) > half or abs(y) > half:
            return True

        # Obstacle collision (simple AABB check)
        for ox, oy, ow, oh in self.OBSTACLES:
            # Expand obstacle by robot radius
            if (ox - ow / 2 - self.ROBOT_RADIUS < x < ox + ow / 2 + self.ROBOT_RADIUS and
                oy - oh / 2 - self.ROBOT_RADIUS < y < oy + oh / 2 + self.ROBOT_RADIUS):
                return True

        return False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        options = options or {}

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        # Set robot pose
        if "robot_pos" in options:
            self._robot_pos = np.array(options["robot_pos"])
        else:
            # Random valid position
            for _ in range(100):
                self._robot_pos = self._rng.uniform(-1.5, 1.5, 2)
                if not self._check_collision(self._robot_pos):
                    break

        if "robot_yaw" in options:
            self._robot_yaw = options["robot_yaw"]
        else:
            self._robot_yaw = self._rng.uniform(-np.pi, np.pi)

        # Set goal
        if "goal_pos" in options:
            self._goal_pos = np.array(options["goal_pos"])
        else:
            # Random goal near a location
            if self._rng.random() < 0.7:
                loc_names = [k for k in self.LOCATIONS.keys() if k != "center"]
                chosen = self._rng.choice(loc_names)
                self._goal_pos = self.LOCATIONS[chosen] + self._rng.uniform(-0.3, 0.3, 2)
            else:
                self._goal_pos = self._rng.uniform(-1.5, 1.5, 2)

        obs = self.render()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        action = np.asarray(action).flatten()
        vx, vy, vyaw = action[0], action[1], action[2]

        # Update yaw
        self._robot_yaw += vyaw * self.dt
        self._robot_yaw = np.arctan2(np.sin(self._robot_yaw), np.cos(self._robot_yaw))

        # Compute world-frame velocity
        cos_yaw = np.cos(self._robot_yaw)
        sin_yaw = np.sin(self._robot_yaw)

        # Transform body velocity to world frame
        vx_world = vx * cos_yaw - vy * sin_yaw
        vy_world = vx * sin_yaw + vy * cos_yaw

        # Update position
        new_pos = self._robot_pos + np.array([vx_world, vy_world]) * self.dt

        # Collision check
        if not self._check_collision(new_pos):
            self._robot_pos = new_pos

        self._step_count += 1

        # Get observation
        obs = self.render()

        # Compute reward
        distance = np.linalg.norm(self._robot_pos - self._goal_pos)
        reward = -distance
        if distance < 0.2:
            reward += 10.0

        # Termination
        terminated = distance < 0.15
        truncated = self._step_count >= self.max_episode_steps

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get state information."""
        distance = np.linalg.norm(self._robot_pos - self._goal_pos)
        return {
            "robot_position": self._robot_pos.copy(),
            "robot_yaw": self._robot_yaw,
            "robot_state": self.robot_state,
            "goal_position": self._goal_pos.copy(),
            "distance_to_goal": distance,
            "step": self._step_count,
            "success": distance < 0.15,
        }

    def render(self) -> np.ndarray:
        """Render the scene as RGB image."""
        img = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        # Floor (gray)
        img[:, :] = [100, 100, 100]

        # Grid lines
        for i in range(5):
            coord = int(i / 4 * self.render_size)
            img[coord:coord + 1, :] = [80, 80, 80]
            img[:, coord:coord + 1] = [80, 80, 80]

        # Walls
        wall_color = (60, 60, 60)
        wall_thickness = 4
        draw_rectangle(img, 0, 0, self.render_size, wall_thickness, wall_color)
        draw_rectangle(img, 0, self.render_size - wall_thickness, self.render_size, self.render_size, wall_color)
        draw_rectangle(img, 0, 0, wall_thickness, self.render_size, wall_color)
        draw_rectangle(img, self.render_size - wall_thickness, 0, self.render_size, self.render_size, wall_color)

        # Obstacles (brown/tan)
        for ox, oy, ow, oh in self.OBSTACLES:
            px, py = self._world_to_pixel(ox, oy)
            pw = self._meters_to_pixels(ow)
            ph = self._meters_to_pixels(oh)
            draw_rectangle(img,
                          px - pw // 2, py - ph // 2,
                          px + pw // 2, py + ph // 2,
                          (139, 119, 101))  # Brown

        # Delivery zone (light green)
        dx, dy = self._world_to_pixel(1.2, -1.2)
        dsize = self._meters_to_pixels(0.8)
        draw_rectangle(img,
                      dx - dsize // 2, dy - dsize // 2,
                      dx + dsize // 2, dy + dsize // 2,
                      (144, 238, 144))

        # Goal (green circle)
        gx, gy = self._world_to_pixel(self._goal_pos[0], self._goal_pos[1])
        gr = self._meters_to_pixels(0.15)
        draw_circle(img, gx, gy, gr, (50, 205, 50))

        # Robot (blue circle with direction indicator)
        rx, ry = self._world_to_pixel(self._robot_pos[0], self._robot_pos[1])
        rr = self._meters_to_pixels(self.ROBOT_RADIUS)
        draw_circle(img, rx, ry, rr, (65, 105, 225))  # Royal blue

        # Direction indicator (red line)
        dir_len = self._meters_to_pixels(0.2)
        dx_dir = int(dir_len * np.cos(self._robot_yaw))
        dy_dir = int(-dir_len * np.sin(self._robot_yaw))  # Negative because y is flipped
        draw_line(img, rx, ry, rx + dx_dir, ry + dy_dir, (255, 100, 100), thickness=3)

        return img

    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run a quick demo."""
    print("Creating SimpleFactoryEnv...")
    env = SimpleFactoryEnv(render_size=256)

    print("Resetting...")
    obs, info = env.reset(seed=42)
    print(f"Robot: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Goal: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")

    frames = [obs]

    print("Running 50 steps...")
    for i in range(50):
        action = env.action_space_sample()
        obs, reward, term, trunc, info = env.step(action)
        frames.append(obs)
        if term:
            print(f"Goal reached at step {i}!")
            break

    print(f"Final distance: {info['distance_to_goal']:.2f}")
    env.close()

    # Try to save
    try:
        import imageio
        imageio.mimwrite("/tmp/simple_factory_demo.mp4", frames, fps=20)
        print("Saved video to /tmp/simple_factory_demo.mp4")
    except ImportError:
        print("Install imageio to save video")

    return frames


if __name__ == "__main__":
    demo()

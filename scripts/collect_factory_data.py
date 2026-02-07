#!/usr/bin/env python3
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
Data Collection Script for Factory Environment

Collects training data for Cosmos-Predict2.5 action-conditioned world model.
Generates episodes of a mobile robot navigating in the factory environment.

Usage:
    python scripts/collect_factory_data.py --output_dir datasets/factory --num_episodes 5000

Output format:
    datasets/factory/
    ├── annotation/
    │   ├── train/
    │   │   ├── episode_00000.json
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── videos/
        ├── episode_00000.mp4
        └── ...
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def generate_smooth_actions(
    num_steps: int,
    action_dim: int = 3,
    rng: Optional[np.random.Generator] = None,
    smoothness: float = 0.3,
) -> np.ndarray:
    """
    Generate smooth random action sequences using interpolation.

    Uses cubic spline interpolation between random control points
    to produce physically plausible motion patterns.

    Args:
        num_steps: Number of action steps to generate
        action_dim: Dimension of action space (3 for nav, 6 for full)
        rng: Random number generator
        smoothness: Lower = more control points = less smooth (0.1 to 1.0)

    Returns:
        Actions array of shape (num_steps, action_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Number of control points based on smoothness
    num_ctrl = max(3, int(num_steps * smoothness))

    # Generate control points
    ctrl_times = np.linspace(0, 1, num_ctrl)
    ctrl_values = np.zeros((num_ctrl, action_dim))

    # Action ranges for navigation: [vx, vy, vyaw]
    action_ranges = [
        (-0.4, 0.4),   # vx
        (-0.4, 0.4),   # vy
        (-0.8, 0.8),   # vyaw
        (-0.5, 0.5),   # shoulder (if used)
        (-0.5, 0.5),   # elbow (if used)
        (0.0, 0.04),   # gripper (if used)
    ]

    for d in range(action_dim):
        low, high = action_ranges[d]
        ctrl_values[:, d] = rng.uniform(low, high, num_ctrl)

    # Interpolate to get smooth actions
    from scipy.interpolate import interp1d

    t = np.linspace(0, 1, num_steps)
    actions = np.zeros((num_steps, action_dim))

    for d in range(action_dim):
        f = interp1d(ctrl_times, ctrl_values[:, d], kind="cubic")
        actions[:, d] = f(t)

    return actions


def generate_goal_directed_actions(
    start_pos: np.ndarray,
    start_yaw: float,
    goal_pos: np.ndarray,
    num_steps: int,
    rng: Optional[np.random.Generator] = None,
    noise_scale: float = 0.2,
) -> np.ndarray:
    """
    Generate actions that move toward the goal with some randomness.

    This creates more "purposeful" trajectories that show goal-seeking behavior,
    which is useful for training world models to understand navigation.

    Args:
        start_pos: Starting (x, y) position
        start_yaw: Starting orientation
        goal_pos: Goal (x, y) position
        num_steps: Number of steps
        rng: Random generator
        noise_scale: Amount of random noise to add

    Returns:
        Actions array (num_steps, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    actions = np.zeros((num_steps, 3))
    current_pos = start_pos.copy()
    current_yaw = start_yaw

    dt = 0.05  # Assumed time step

    for i in range(num_steps):
        # Direction to goal
        to_goal = goal_pos - current_pos
        distance = np.linalg.norm(to_goal)

        if distance < 0.1:
            # Near goal, small random motions
            actions[i] = rng.uniform(-0.1, 0.1, 3)
        else:
            # Compute desired heading
            desired_yaw = np.arctan2(to_goal[1], to_goal[0])
            yaw_error = desired_yaw - current_yaw

            # Normalize to [-pi, pi]
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

            # Proportional control with noise
            vyaw = np.clip(yaw_error * 2.0, -1.0, 1.0) + rng.normal(0, noise_scale * 0.5)

            # Forward velocity based on alignment
            alignment = np.cos(yaw_error)
            vx = np.clip(alignment * 0.5 * min(distance, 1.0), -0.5, 0.5)
            vx += rng.normal(0, noise_scale * 0.3)

            # Small lateral velocity
            vy = rng.normal(0, noise_scale * 0.2)

            actions[i] = [vx, vy, vyaw]

            # Simple integration for next iteration
            current_yaw += vyaw * dt
            current_pos[0] += vx * np.cos(current_yaw) * dt - vy * np.sin(current_yaw) * dt
            current_pos[1] += vx * np.sin(current_yaw) * dt + vy * np.cos(current_yaw) * dt

    return actions


def collect_episode(
    episode_id: int,
    output_dir: Path,
    render_size: int = 256,
    min_steps: int = 20,
    max_steps: int = 60,
    seed: Optional[int] = None,
    action_type: str = "mixed",
) -> Dict:
    """
    Collect a single episode of data.

    Args:
        episode_id: Unique episode identifier
        output_dir: Base output directory
        render_size: Size of rendered images
        min_steps: Minimum episode length
        max_steps: Maximum episode length
        seed: Random seed
        action_type: 'random', 'goal_directed', or 'mixed'

    Returns:
        Metadata dictionary for the episode
    """
    from sim.factory.env import FactoryEnv

    rng = np.random.default_rng(seed)
    env = FactoryEnv(render_size=render_size, navigation_only=True)

    # Reset environment
    obs, info = env.reset(seed=seed)

    # Determine episode length
    num_steps = rng.integers(min_steps, max_steps + 1)

    # Generate actions
    if action_type == "random":
        actions = generate_smooth_actions(num_steps, action_dim=3, rng=rng)
    elif action_type == "goal_directed":
        actions = generate_goal_directed_actions(
            info["robot_position"],
            info["robot_yaw"],
            info["goal_position"],
            num_steps,
            rng=rng,
        )
    else:  # mixed
        if rng.random() < 0.5:
            actions = generate_smooth_actions(num_steps, action_dim=3, rng=rng)
        else:
            actions = generate_goal_directed_actions(
                info["robot_position"],
                info["robot_yaw"],
                info["goal_position"],
                num_steps,
                rng=rng,
            )

    # Collect trajectory
    frames = [obs]
    states = [info["robot_state"].tolist()]
    rewards = []
    dones = []

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        states.append(info["robot_state"].tolist())
        rewards.append(reward)
        dones.append(terminated or truncated)

        if terminated or truncated:
            break

    env.close()

    # Truncate actions to match actual steps taken
    actual_steps = len(frames) - 1
    actions = actions[:actual_steps].tolist()

    # Prepare episode ID string
    ep_str = f"episode_{episode_id:05d}"

    # Save video
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{ep_str}.mp4"

    try:
        import imageio
        imageio.mimwrite(str(video_path), frames, fps=20)
    except Exception as e:
        print(f"Warning: Could not save video for {ep_str}: {e}")
        return None

    # Create annotation
    annotation = {
        "episode_id": ep_str,
        "state": states,  # List of [x, y, yaw] for each frame
        "action": actions,  # List of [vx, vy, vyaw] for each action
        "goal": info["goal_position"].tolist(),
        "videos": {
            "0": {"video_path": f"videos/{ep_str}.mp4"}
        },
        # Dummy fields for compatibility with existing Cosmos dataset format
        "continuous_gripper_state": [0.0] * len(states),
        "num_frames": len(frames),
        "success": info["success"],
        "final_distance": info["distance_to_goal"],
    }

    return annotation


def save_annotations(
    annotations: List[Dict],
    output_dir: Path,
    split: str,
):
    """Save annotations to JSON files."""
    ann_dir = output_dir / "annotation" / split
    ann_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        if ann is None:
            continue
        ep_id = ann["episode_id"]
        ann_path = ann_dir / f"{ep_id}.json"
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Collect factory environment data for Cosmos-Predict2.5"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/factory",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Total number of episodes to collect",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes for validation",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=256,
        help="Size of rendered images (square)",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=20,
        help="Minimum steps per episode",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="mixed",
        choices=["random", "goal_directed", "mixed"],
        help="Type of action generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.num_episodes} episodes to {output_dir}")
    print(f"Render size: {args.render_size}x{args.render_size}")
    print(f"Episode length: {args.min_steps}-{args.max_steps} steps")
    print(f"Action type: {args.action_type}")

    # Split episodes
    num_val = int(args.num_episodes * args.val_ratio)
    num_train = args.num_episodes - num_val

    print(f"Train: {num_train}, Val: {num_val}")

    # Collect training episodes
    print("\nCollecting training episodes...")
    train_annotations = []

    # Use sequential collection for simplicity (parallel can cause MuJoCo issues)
    from tqdm import tqdm

    for i in tqdm(range(num_train), desc="Training"):
        ann = collect_episode(
            episode_id=i,
            output_dir=output_dir,
            render_size=args.render_size,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            seed=args.seed + i,
            action_type=args.action_type,
        )
        if ann is not None:
            train_annotations.append(ann)

    print(f"Collected {len(train_annotations)} training episodes")
    save_annotations(train_annotations, output_dir, "train")

    # Collect validation episodes
    print("\nCollecting validation episodes...")
    val_annotations = []

    for i in tqdm(range(num_val), desc="Validation"):
        ann = collect_episode(
            episode_id=num_train + i,
            output_dir=output_dir,
            render_size=args.render_size,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            seed=args.seed + num_train + i,
            action_type=args.action_type,
        )
        if ann is not None:
            val_annotations.append(ann)

    print(f"Collected {len(val_annotations)} validation episodes")
    save_annotations(val_annotations, output_dir, "val")

    # Print summary
    total_frames = sum(ann["num_frames"] for ann in train_annotations + val_annotations if ann)
    print(f"\nCollection complete!")
    print(f"Total episodes: {len(train_annotations) + len(val_annotations)}")
    print(f"Total frames: {total_frames}")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

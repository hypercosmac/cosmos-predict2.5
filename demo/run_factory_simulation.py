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
Factory Simulation Demo - Quick Start

A simple script to run and visualize the factory simulation environment.
This demonstrates the simulation that can be augmented with Cosmos-Predict2.5
world model capabilities.

Usage:
    python demo/run_factory_simulation.py
    python demo/run_factory_simulation.py --save-video
    python demo/run_factory_simulation.py --interactive
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np


def get_factory_env(render_size: int = 256):
    """Get factory environment, preferring MuJoCo but falling back to simple."""
    from sim.factory import HAS_MUJOCO, SimpleFactoryEnv

    if HAS_MUJOCO:
        from sim.factory import FactoryEnv
        print("Using MuJoCo-based FactoryEnv")
        return FactoryEnv(render_size=render_size, navigation_only=True)
    else:
        print("MuJoCo not available, using SimpleFactoryEnv")
        return SimpleFactoryEnv(render_size=render_size)


def run_random_demo(save_video: bool = False, output_path: str = "demo_output"):
    """Run demo with random actions."""
    print("Creating factory environment...")
    env = get_factory_env(render_size=256)

    print("Resetting environment...")
    obs, info = env.reset(seed=42)

    print(f"\nRobot position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Robot yaw: {info['robot_yaw']:.2f} rad")
    print(f"Goal position: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
    print(f"Distance to goal: {info['distance_to_goal']:.2f}")

    frames = [obs]
    total_reward = 0

    print("\nRunning 100 random steps...")
    for step in range(100):
        # Sample random action
        action = env.action_space_sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: pos=({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f}), "
                  f"dist={info['distance_to_goal']:.2f}")

        if terminated:
            print(f"\nGoal reached at step {step}!")
            break

        if truncated:
            print(f"\nEpisode truncated at step {step}")
            break

    print(f"\nFinal position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Final distance to goal: {info['distance_to_goal']:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success: {info['success']}")

    env.close()

    if save_video:
        try:
            import imageio
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "factory_random_demo.mp4"
            imageio.mimwrite(str(video_path), frames, fps=20)
            print(f"\nSaved video to {video_path}")
        except ImportError:
            print("\nInstall imageio to save videos: pip install imageio[ffmpeg]")

    return frames


def run_goal_directed_demo(save_video: bool = False, output_path: str = "demo_output"):
    """Run demo with simple goal-directed controller."""
    print("Creating factory environment...")
    env = get_factory_env(render_size=256)

    print("Resetting environment...")
    obs, info = env.reset(seed=123)

    print(f"\nRobot position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Goal position: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
    print(f"Distance to goal: {info['distance_to_goal']:.2f}")

    frames = [obs]

    print("\nRunning goal-directed controller...")
    for step in range(200):
        # Simple proportional controller
        robot_pos = info["robot_position"]
        robot_yaw = info["robot_yaw"]
        goal_pos = info["goal_position"]

        # Compute direction to goal
        to_goal = goal_pos - robot_pos
        distance = np.linalg.norm(to_goal)
        desired_yaw = np.arctan2(to_goal[1], to_goal[0])

        # Compute yaw error
        yaw_error = desired_yaw - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Normalize

        # P-controller
        vyaw = np.clip(yaw_error * 2.0, -1.0, 1.0)

        # Move forward when aligned
        alignment = np.cos(yaw_error)
        vx = np.clip(alignment * min(distance, 0.5), -0.5, 0.5)

        action = np.array([vx, 0.0, vyaw])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)

        if step % 30 == 0:
            print(f"Step {step}: dist={info['distance_to_goal']:.2f}, yaw_err={yaw_error:.2f}")

        if terminated:
            print(f"\nGoal reached at step {step}!")
            break

        if truncated:
            print(f"\nEpisode truncated at step {step}")
            break

    print(f"\nFinal distance to goal: {info['distance_to_goal']:.2f}")
    print(f"Success: {info['success']}")

    env.close()

    if save_video:
        try:
            import imageio
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "factory_goal_directed_demo.mp4"
            imageio.mimwrite(str(video_path), frames, fps=20)
            print(f"\nSaved video to {video_path}")
        except ImportError:
            print("\nInstall imageio to save videos: pip install imageio[ffmpeg]")

    return frames


def run_multiview_demo(save_video: bool = False, output_path: str = "demo_output"):
    """Run demo showing all camera views (MuJoCo only)."""
    from sim.factory import HAS_MUJOCO

    if not HAS_MUJOCO:
        print("Multiview demo requires MuJoCo. Install with: pip install mujoco")
        print("Running single-view demo instead...")
        return run_random_demo(save_video=save_video, output_path=output_path)

    from sim.factory import FactoryEnv

    print("Creating factory environment...")
    env = FactoryEnv(render_size=256, navigation_only=True)

    print("Resetting environment...")
    obs, info = env.reset(seed=42)

    print("\nRendering all camera views...")
    views = env.render_all_views()

    # Create grid of views
    import numpy as np

    # Stack views in a 2x2 grid
    view_names = ["overview", "robot_view", "side", "top_down"]
    grid_images = []

    for name in view_names:
        if name in views:
            img = views[name].copy()
            # Add label
            try:
                import cv2
                cv2.putText(img, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except ImportError:
                pass
            grid_images.append(img)

    if len(grid_images) == 4:
        top_row = np.hstack([grid_images[0], grid_images[1]])
        bottom_row = np.hstack([grid_images[2], grid_images[3]])
        grid = np.vstack([top_row, bottom_row])

        if save_video:
            try:
                import cv2
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                img_path = output_dir / "factory_multiview.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"\nSaved multiview image to {img_path}")
            except ImportError:
                print("\nInstall opencv-python to save images: pip install opencv-python")

    env.close()
    return views


def show_planning_concept(save_video: bool = False, output_path: str = "demo_output"):
    """
    Demonstrate the planning-via-imagination concept.

    This shows how the robot could use a world model to:
    1. Generate multiple candidate action sequences
    2. Imagine the outcomes (using simulation as ground truth)
    3. Select the best trajectory
    4. Execute it
    """
    print("=" * 60)
    print("Planning-via-Imagination Concept Demo")
    print("=" * 60)

    env = get_factory_env(render_size=256)
    obs, info = env.reset(seed=42)

    print(f"\nRobot at ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Goal at ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")

    # Step 1: Generate candidate action sequences
    print("\n1. Generating 6 candidate action sequences...")
    num_candidates = 6
    horizon = 20
    rng = np.random.default_rng(42)

    def linear_interp(ctrl_times, ctrl_values, t):
        """Simple linear interpolation without scipy."""
        result = np.zeros(len(t))
        for i, ti in enumerate(t):
            # Find surrounding control points
            idx = np.searchsorted(ctrl_times, ti, side='right') - 1
            idx = max(0, min(idx, len(ctrl_times) - 2))
            t0, t1 = ctrl_times[idx], ctrl_times[idx + 1]
            v0, v1 = ctrl_values[idx], ctrl_values[idx + 1]
            # Linear interpolation
            alpha = (ti - t0) / (t1 - t0) if t1 != t0 else 0
            result[i] = v0 + alpha * (v1 - v0)
        return result

    candidates = []
    for i in range(num_candidates):
        # Generate smooth random actions using linear interpolation
        num_ctrl = 5  # More control points for smoother motion
        ctrl_times = np.linspace(0, 1, num_ctrl)
        ctrl_values = rng.uniform(-0.5, 0.5, (num_ctrl, 3))
        ctrl_values[:, 2] *= 2  # Larger angular velocity range

        t = np.linspace(0, 1, horizon)
        actions = np.zeros((horizon, 3))
        for d in range(3):
            actions[:, d] = linear_interp(ctrl_times, ctrl_values[:, d], t)

        candidates.append(actions)

    # Step 2: "Imagine" outcomes (using actual simulation)
    print("2. Imagining outcomes for each candidate...")
    imagined_videos = []
    final_distances = []
    initial_state = {
        "robot_pos": info["robot_position"].copy(),
        "robot_yaw": info["robot_yaw"],
    }

    for i, actions in enumerate(candidates):
        # Reset to initial state
        env.reset(seed=42)

        frames = [env.render()]
        for action in actions:
            obs, _, _, _, info = env.step(action)
            frames.append(obs)

        imagined_videos.append(np.array(frames))
        final_distances.append(info["distance_to_goal"])
        print(f"   Candidate {i + 1}: final distance = {info['distance_to_goal']:.2f}")

    # Step 3: Score and select best
    print("\n3. Selecting best trajectory...")
    scores = [-d for d in final_distances]  # Negative distance = better
    best_idx = np.argmax(scores)
    print(f"   Best: Candidate {best_idx + 1} (distance = {final_distances[best_idx]:.2f})")

    # Step 4: Execute best trajectory
    print("\n4. Executing chosen trajectory...")
    env.reset(seed=42)
    actual_frames = [env.render()]

    for action in candidates[best_idx]:
        obs, _, terminated, _, info = env.step(action)
        actual_frames.append(obs)
        if terminated:
            break

    print(f"   Final position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"   Final distance: {info['distance_to_goal']:.2f}")

    env.close()

    # Create visualization
    if save_video:
        try:
            import cv2
            import imageio

            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create imagination grid (6 trajectories, middle frame)
            h, w = imagined_videos[0][0].shape[:2]
            frame_idx = horizon // 2

            grid_images = []
            for i, (video, dist) in enumerate(zip(imagined_videos, final_distances)):
                frame = video[frame_idx].copy()

                if i == best_idx:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)
                    label = f"Plan {i + 1} (BEST)"
                else:
                    label = f"Plan {i + 1}"

                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Dist: {dist:.2f}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                grid_images.append(frame)

            # 2x3 grid
            rows = [np.hstack(grid_images[i * 3:(i + 1) * 3]) for i in range(2)]
            grid = np.vstack(rows)

            grid_path = output_dir / "planning_imagination_grid.png"
            cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"\nSaved imagination grid to {grid_path}")

            # Side-by-side comparison video
            best_imagined = imagined_videos[best_idx]
            comparison_frames = []

            for i in range(min(len(best_imagined), len(actual_frames))):
                imagined = best_imagined[i].copy()
                actual = actual_frames[i].copy()

                cv2.putText(imagined, "IMAGINED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(actual, "ACTUAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                combined = np.hstack([imagined, actual])
                comparison_frames.append(combined)

            video_path = output_dir / "planning_comparison.mp4"
            imageio.mimwrite(str(video_path), comparison_frames, fps=10)
            print(f"Saved comparison video to {video_path}")

        except ImportError as e:
            print(f"\nInstall dependencies for visualization: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Factory Simulation Demo")
    parser.add_argument("--mode", type=str, default="planning",
                       choices=["random", "goal", "multiview", "planning"],
                       help="Demo mode to run")
    parser.add_argument("--save-video", action="store_true",
                       help="Save output video/images")
    parser.add_argument("--output", type=str, default="demo_output/factory",
                       help="Output directory")
    args = parser.parse_args()

    if args.mode == "random":
        run_random_demo(save_video=args.save_video, output_path=args.output)
    elif args.mode == "goal":
        run_goal_directed_demo(save_video=args.save_video, output_path=args.output)
    elif args.mode == "multiview":
        run_multiview_demo(save_video=args.save_video, output_path=args.output)
    elif args.mode == "planning":
        show_planning_concept(save_video=args.save_video, output_path=args.output)


if __name__ == "__main__":
    main()

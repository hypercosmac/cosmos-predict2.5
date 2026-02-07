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
Factory World Model Demo - Robot Augmentation with Cosmos-Predict2.5

This demo showcases how robots can be augmented with world model capabilities
using Cosmos-Predict2.5 for action-conditioned video prediction.

The demo illustrates:
1. A mobile robot in a simulated factory environment
2. The robot imagining multiple possible futures based on different action sequences
3. Planning by selecting the best imagined trajectory toward a goal
4. Executing the chosen plan and comparing prediction vs reality

This demonstrates the concept of "planning via imagination" where the world model
serves as a learned simulator for evaluating actions before execution.

Usage:
    # With trained checkpoint:
    python demo/factory_world_model_demo.py --checkpoint path/to/checkpoint

    # Simulation-only mode (no world model):
    python demo/factory_world_model_demo.py --simulation-only

    # Generate comparison video:
    python demo/factory_world_model_demo.py --checkpoint path/to/ckpt --output demo_output/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class ActionSampler:
    """
    Generates diverse candidate action sequences for planning.

    Produces action trajectories that explore different movement strategies:
    - Goal-directed: Actions that move toward the goal
    - Exploratory: Random smooth trajectories
    - Turning: Rotational maneuvers
    """

    def __init__(
        self,
        horizon: int = 15,
        action_dim: int = 3,
        seed: Optional[int] = None,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def sample_smooth_trajectory(self) -> np.ndarray:
        """Generate smooth random trajectory using spline interpolation."""
        from scipy.interpolate import interp1d

        # Control points
        num_ctrl = 4
        ctrl_times = np.linspace(0, 1, num_ctrl)
        ctrl_values = self.rng.uniform(-0.5, 0.5, (num_ctrl, self.action_dim))

        # Adjust ranges per dimension
        ctrl_values[:, 2] *= 2  # Larger angular velocities

        # Interpolate
        t = np.linspace(0, 1, self.horizon)
        actions = np.zeros((self.horizon, self.action_dim))
        for d in range(self.action_dim):
            f = interp1d(ctrl_times, ctrl_values[:, d], kind="cubic")
            actions[:, d] = f(t)

        return actions

    def sample_goal_directed(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """Generate actions that move toward the goal."""
        actions = np.zeros((self.horizon, self.action_dim))

        # Direction to goal
        to_goal = goal_pos - robot_pos
        distance = np.linalg.norm(to_goal)
        desired_yaw = np.arctan2(to_goal[1], to_goal[0])
        yaw_error = np.arctan2(np.sin(desired_yaw - robot_yaw), np.cos(desired_yaw - robot_yaw))

        for i in range(self.horizon):
            # Decay the correction over time
            decay = np.exp(-i * 0.1)

            # Turn toward goal
            vyaw = np.clip(yaw_error * 1.5 * decay, -1.0, 1.0)
            vyaw += self.rng.normal(0, 0.1)

            # Move forward when aligned
            alignment = np.cos(yaw_error)
            vx = np.clip(alignment * 0.4, 0, 0.5)
            vx += self.rng.normal(0, 0.05)

            # Small lateral noise
            vy = self.rng.normal(0, 0.05)

            actions[i] = [vx, vy, vyaw]

            # Update estimate for next step
            robot_yaw += vyaw * 0.05
            yaw_error = np.arctan2(np.sin(desired_yaw - robot_yaw), np.cos(desired_yaw - robot_yaw))

        return actions

    def sample_diverse_trajectories(
        self,
        num_samples: int,
        robot_pos: np.ndarray,
        robot_yaw: float,
        goal_pos: np.ndarray,
    ) -> List[np.ndarray]:
        """Sample a diverse set of action trajectories."""
        trajectories = []

        for i in range(num_samples):
            if i == 0:
                # First trajectory: goal-directed
                traj = self.sample_goal_directed(robot_pos, robot_yaw, goal_pos)
            elif i == 1:
                # Second: slightly noisy goal-directed
                traj = self.sample_goal_directed(robot_pos, robot_yaw, goal_pos)
                traj += self.rng.normal(0, 0.15, traj.shape)
            else:
                # Rest: random smooth
                traj = self.sample_smooth_trajectory()

            trajectories.append(np.clip(traj, -1, 1))

        return trajectories


class TrajectoryScorer:
    """
    Scores imagined trajectories based on task objectives.

    Evaluates trajectories by:
    - Distance to goal at end of trajectory
    - Collision indicators (if detectable from images)
    - Smoothness of motion
    """

    def __init__(self, goal_pos: np.ndarray):
        self.goal_pos = goal_pos

    def score_from_states(
        self,
        final_pos: np.ndarray,
        trajectory_states: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Score based on known final position."""
        distance = np.linalg.norm(final_pos - self.goal_pos)

        # Negative distance as score (closer = better)
        score = -distance

        # Bonus for reaching goal
        if distance < 0.2:
            score += 5.0

        return score

    def score_from_video(
        self,
        video: np.ndarray,
        goal_color: Tuple[int, int, int] = (50, 200, 75),  # Green goal marker
    ) -> float:
        """
        Score based on visual analysis of imagined video.

        This is a simple heuristic that looks for the goal marker color
        in the final frames. In practice, you'd use a learned value network.
        """
        # Look at last few frames
        final_frames = video[-3:]

        # Convert goal color to detection range
        goal_rgb = np.array(goal_color)

        total_goal_pixels = 0
        for frame in final_frames:
            # Simple color matching
            diff = np.abs(frame.astype(float) - goal_rgb)
            mask = np.all(diff < 50, axis=-1)
            total_goal_pixels += np.sum(mask)

        # More goal pixels visible = robot is closer to goal
        # This is a very rough heuristic
        score = total_goal_pixels * 0.001

        return score


class WorldModelPlanner:
    """
    Planning with action-conditioned world model.

    Uses Cosmos-Predict2.5 to imagine future outcomes of different action
    sequences, then selects the best trajectory based on task objectives.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_file: Optional[str] = None,
        num_candidates: int = 6,
        horizon: int = 15,
        device: str = "cuda",
    ):
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.action_sampler = ActionSampler(horizon=horizon)

        # Load world model if checkpoint provided
        self.model = None
        if checkpoint_path is not None:
            self._load_model(checkpoint_path, config_file)

    def _load_model(self, checkpoint_path: str, config_file: Optional[str] = None):
        """Load Cosmos-Predict2.5 world model."""
        try:
            import torch
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

            print(f"Loading world model from {checkpoint_path}...")

            self.model = Video2WorldInference(
                experiment_name="factory_action_conditioned",
                ckpt_path=checkpoint_path,
                s3_credential_path="",
                context_parallel_size=1,
                config_file=config_file,
            )
            print("World model loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load world model: {e}")
            print("Running in simulation-only mode.")
            self.model = None

    def imagine_trajectory(
        self,
        initial_frame: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Generate imagined future video for given actions.

        Args:
            initial_frame: Current observation (H, W, 3) uint8
            actions: Action sequence (horizon, 3)

        Returns:
            Imagined video (horizon+1, H, W, 3) uint8
        """
        if self.model is None:
            # No model - return dummy video (copies of initial frame)
            return np.stack([initial_frame] * (len(actions) + 1))

        import torch

        # Prepare input
        # Frame: (H, W, C) -> (1, C, T, H, W)
        frame_tensor = torch.from_numpy(initial_frame).permute(2, 0, 1)  # (C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)

        # Expand to full sequence length with zeros
        num_frames = len(actions) + 1
        video_input = torch.zeros(1, 3, num_frames, initial_frame.shape[0], initial_frame.shape[1], dtype=torch.uint8)
        video_input[:, :, 0] = frame_tensor.squeeze(2)

        # Actions: (horizon, 3) -> tensor
        action_tensor = torch.from_numpy(actions).float()

        # Generate with world model
        try:
            video = self.model.generate_vid2world(
                prompt="",
                input_path=video_input,
                action=action_tensor,
                guidance=0,  # No CFG for action-conditioned
                num_video_frames=num_frames,
                num_latent_conditional_frames=1,
                resolution=f"{initial_frame.shape[0]},{initial_frame.shape[1]}",
                seed=42,
            )

            # Convert from [-1, 1] to [0, 255]
            video_np = ((video[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            return video_np

        except Exception as e:
            print(f"Warning: World model inference failed: {e}")
            return np.stack([initial_frame] * num_frames)

    def plan(
        self,
        observation: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        goal_pos: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Plan best action sequence using world model imagination.

        Args:
            observation: Current RGB observation
            robot_pos: Current (x, y) position
            robot_yaw: Current orientation
            goal_pos: Goal (x, y) position

        Returns:
            best_actions: Selected action sequence
            imagined_videos: List of imagined videos for each candidate
            scores: Score for each candidate
        """
        # Sample diverse action candidates
        action_candidates = self.action_sampler.sample_diverse_trajectories(
            self.num_candidates,
            robot_pos,
            robot_yaw,
            goal_pos,
        )

        # Imagine outcomes for each candidate
        imagined_videos = []
        for actions in action_candidates:
            video = self.imagine_trajectory(observation, actions)
            imagined_videos.append(video)

        # Score trajectories
        scorer = TrajectoryScorer(goal_pos)
        scores = []
        for i, (actions, video) in enumerate(zip(action_candidates, imagined_videos)):
            # Use both state-based and video-based scoring
            # Estimate final position from actions (simple integration)
            est_pos = robot_pos.copy()
            est_yaw = robot_yaw
            dt = 0.05
            for a in actions:
                est_yaw += a[2] * dt
                est_pos[0] += a[0] * np.cos(est_yaw) * dt - a[1] * np.sin(est_yaw) * dt
                est_pos[1] += a[0] * np.sin(est_yaw) * dt + a[1] * np.cos(est_yaw) * dt

            state_score = scorer.score_from_states(est_pos)
            video_score = scorer.score_from_video(video)

            total_score = state_score + video_score * 0.1
            scores.append(total_score)

        # Select best
        best_idx = np.argmax(scores)
        best_actions = action_candidates[best_idx]

        return best_actions, imagined_videos, scores


def create_visualization(
    imagined_videos: List[np.ndarray],
    actual_frames: List[np.ndarray],
    scores: List[float],
    best_idx: int,
    output_path: Path,
):
    """
    Create visualization comparing imagined and actual trajectories.

    Generates:
    1. Side-by-side video: Best imagined vs actual execution
    2. Grid image: All imagined futures with scores
    """
    if not HAS_CV2:
        print("OpenCV not available for visualization")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Get dimensions
    h, w = actual_frames[0].shape[:2]

    # 1. Side-by-side comparison video
    best_imagined = imagined_videos[best_idx]
    comparison_frames = []

    num_frames = min(len(best_imagined), len(actual_frames))
    for i in range(num_frames):
        # Create side-by-side frame
        imagined = best_imagined[i].copy()
        actual = actual_frames[i].copy()

        # Add labels
        cv2.putText(imagined, "IMAGINED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(actual, "ACTUAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add frame number
        cv2.putText(imagined, f"t={i}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(actual, f"t={i}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Combine
        combined = np.hstack([imagined, actual])
        comparison_frames.append(combined)

    # Save video
    if HAS_IMAGEIO:
        video_path = output_path / "comparison.mp4"
        imageio.mimwrite(str(video_path), comparison_frames, fps=10)
        print(f"Saved comparison video to {video_path}")

    # 2. Grid of all imagined futures
    # Show middle frame from each trajectory
    frame_idx = len(imagined_videos[0]) // 2

    grid_images = []
    for i, (video, score) in enumerate(zip(imagined_videos, scores)):
        frame = video[frame_idx].copy()

        # Highlight best
        if i == best_idx:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 3)
            label = f"Plan {i + 1} (BEST)"
        else:
            label = f"Plan {i + 1}"

        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score:.2f}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        grid_images.append(frame)

    # Arrange in 2x3 grid
    rows = []
    for row_idx in range(2):
        row_images = grid_images[row_idx * 3:(row_idx + 1) * 3]
        if len(row_images) < 3:
            # Pad with black
            row_images.extend([np.zeros_like(grid_images[0])] * (3 - len(row_images)))
        rows.append(np.hstack(row_images))

    grid = np.vstack(rows)

    # Save grid image
    grid_path = output_path / "imagination_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved imagination grid to {grid_path}")


def run_demo(
    checkpoint_path: Optional[str] = None,
    output_dir: str = "demo_output",
    num_planning_steps: int = 3,
    simulation_only: bool = False,
    seed: int = 42,
):
    """
    Run the factory world model demo.

    Args:
        checkpoint_path: Path to trained Cosmos-Predict checkpoint
        output_dir: Directory for output videos and images
        num_planning_steps: Number of plan-execute cycles
        simulation_only: If True, skip world model (just show simulation)
        seed: Random seed
    """
    from sim.factory.env import FactoryEnv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Factory World Model Demo")
    print("=" * 60)

    # Initialize environment
    print("\nInitializing factory environment...")
    env = FactoryEnv(render_size=256, navigation_only=True)

    # Initialize planner
    if simulation_only or checkpoint_path is None:
        print("Running in simulation-only mode (no world model)")
        planner = WorldModelPlanner(
            checkpoint_path=None,
            num_candidates=6,
            horizon=15,
        )
    else:
        print(f"Loading world model from {checkpoint_path}...")
        planner = WorldModelPlanner(
            checkpoint_path=checkpoint_path,
            num_candidates=6,
            horizon=15,
        )

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset(seed=seed)

    print(f"Robot start: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Goal: ({info['goal_position'][0]:.2f}, {info['goal_position'][1]:.2f})")
    print(f"Initial distance: {info['distance_to_goal']:.2f}")

    all_frames = [obs]
    all_imagined = []

    # Main planning loop
    for step in range(num_planning_steps):
        print(f"\n--- Planning Step {step + 1}/{num_planning_steps} ---")

        # Plan with world model
        print("Imagining possible futures...")
        best_actions, imagined_videos, scores = planner.plan(
            observation=obs,
            robot_pos=info["robot_position"],
            robot_yaw=info["robot_yaw"],
            goal_pos=info["goal_position"],
        )

        best_idx = np.argmax(scores)
        print(f"Selected plan {best_idx + 1} with score {scores[best_idx]:.2f}")

        # Store imagined videos for visualization
        all_imagined.append((imagined_videos, scores, best_idx))

        # Execute best actions
        print("Executing chosen plan...")
        executed_frames = [obs]

        for i, action in enumerate(best_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            executed_frames.append(obs)
            all_frames.append(obs)

            if terminated:
                print(f"Goal reached at action {i + 1}!")
                break

            if truncated:
                print("Episode truncated")
                break

        if terminated:
            break

        print(f"Position after execution: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
        print(f"Distance to goal: {info['distance_to_goal']:.2f}")

        # Create visualization for this planning step
        create_visualization(
            imagined_videos=imagined_videos,
            actual_frames=executed_frames,
            scores=scores,
            best_idx=best_idx,
            output_path=output_path / f"step_{step + 1}",
        )

    # Final status
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"Final position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Final distance to goal: {info['distance_to_goal']:.2f}")
    print(f"Success: {info['success']}")

    # Save full trajectory video
    if HAS_IMAGEIO:
        full_video_path = output_path / "full_trajectory.mp4"
        imageio.mimwrite(str(full_video_path), all_frames, fps=20)
        print(f"\nSaved full trajectory to {full_video_path}")

    env.close()
    print(f"\nOutputs saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Factory World Model Demo - Robot augmentation with Cosmos-Predict2.5"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained Cosmos-Predict checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_output/factory",
        help="Output directory for videos and images",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of plan-execute cycles",
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Run without world model (simulation only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    run_demo(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_planning_steps=args.num_steps,
        simulation_only=args.simulation_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

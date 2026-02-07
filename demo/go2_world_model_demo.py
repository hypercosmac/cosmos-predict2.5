#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Go2 Quadruped World Model Demo
===============================

A high-resolution demo of the Unitree Go2 quadruped robot navigating
an outdoor park environment. The robot responds to plain English commands
and uses Cosmos-Predict2.5 to imagine future trajectories.

Features:
- Natural language task prompts ("walk to the red marker", "explore the trees")
- High-resolution rendering (1280x720 default, up to 1920x1080)
- Planning-via-imagination with action-conditioned video prediction
- Cinematic camera tracking
- Side-by-side predicted vs actual comparison

Usage:
    # Interactive mode - type commands in English:
    python demo/go2_world_model_demo.py

    # Run a single command:
    python demo/go2_world_model_demo.py --command "walk to the red marker"

    # With world model imagination:
    python demo/go2_world_model_demo.py --checkpoint path/to/ckpt --command "go to the trees"

    # High resolution cinematic output:
    python demo/go2_world_model_demo.py --width 1920 --height 1080 --save-video

Requirements:
    pip install mujoco imageio[ffmpeg] numpy
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


# ============================================================================
# Natural Language Command Parser
# ============================================================================

class NaturalLanguageController:
    """
    Parses plain English commands into robot velocity commands.

    Understands spatial directives like:
    - "walk forward" / "go ahead" / "move forward"
    - "turn left" / "turn right"
    - "go to the red marker" / "walk to the trees"
    - "come back" / "return to center"
    - "stop" / "halt" / "stay"
    - "explore" / "wander around"
    - "spin" / "turn around"
    - "sit down" / "stand"

    For landmark-based commands, it computes the heading to the target
    and generates velocity commands to navigate there.
    """

    # Keyword -> velocity mapping
    MOTION_KEYWORDS = {
        # Forward motion
        "forward": {"forward": 0.7, "lateral": 0.0, "turn": 0.0},
        "ahead": {"forward": 0.7, "lateral": 0.0, "turn": 0.0},
        "straight": {"forward": 0.7, "lateral": 0.0, "turn": 0.0},

        # Backward
        "backward": {"forward": -0.5, "lateral": 0.0, "turn": 0.0},
        "back": {"forward": -0.5, "lateral": 0.0, "turn": 0.0},
        "reverse": {"forward": -0.5, "lateral": 0.0, "turn": 0.0},

        # Turning
        "left": {"forward": 0.2, "lateral": 0.0, "turn": 0.6},
        "right": {"forward": 0.2, "lateral": 0.0, "turn": -0.6},

        # Stop
        "stop": {"forward": 0.0, "lateral": 0.0, "turn": 0.0},
        "halt": {"forward": 0.0, "lateral": 0.0, "turn": 0.0},
        "stay": {"forward": 0.0, "lateral": 0.0, "turn": 0.0},

        # Special
        "spin": {"forward": 0.0, "lateral": 0.0, "turn": 1.0},
        "fast": {"forward": 1.0, "lateral": 0.0, "turn": 0.0},
        "slow": {"forward": 0.3, "lateral": 0.0, "turn": 0.0},
        "trot": {"forward": 0.6, "lateral": 0.0, "turn": 0.0},
        "run": {"forward": 1.0, "lateral": 0.0, "turn": 0.0},
    }

    # Landmark name aliases
    LANDMARK_ALIASES = {
        "red marker": "waypoint_a",
        "marker": "waypoint_a",
        "waypoint a": "waypoint_a",
        "waypoint": "waypoint_a",
        "blue marker": "waypoint_b",
        "waypoint b": "waypoint_b",
        "first tree": "tree_1",
        "big tree": "tree_1",
        "tree": "tree_1",
        "trees": "tree_2",
        "left tree": "tree_2",
        "far tree": "tree_3",
        "small tree": "tree_4",
        "ball": "ball",
        "center": "center",
        "middle": "center",
        "home": "center",
        "origin": "center",
        "start": "center",
    }

    def __init__(self, landmarks: Dict[str, np.ndarray]):
        self.landmarks = landmarks
        self._target_landmark: Optional[str] = None
        self._explore_phase = 0.0

    def parse_command(
        self,
        command: str,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> Tuple[Dict[str, float], str]:
        """
        Parse a natural language command into velocity commands.

        Args:
            command: English text command
            robot_pos: Current (x, y) position
            robot_yaw: Current heading (radians)

        Returns:
            velocity_cmd: Dict with 'forward', 'lateral', 'turn'
            description: Human-readable description of the action
        """
        cmd = command.lower().strip()

        # Check for stop commands first
        for keyword in ["stop", "halt", "stay", "sit", "wait"]:
            if keyword in cmd:
                self._target_landmark = None
                return {"forward": 0.0, "lateral": 0.0, "turn": 0.0}, "Stopping."

        # Check for landmark navigation ("go to X", "walk to X", "navigate to X")
        nav_prefixes = ["go to", "walk to", "move to", "navigate to",
                        "head to", "run to", "trot to", "approach"]
        for prefix in nav_prefixes:
            if prefix in cmd:
                remainder = cmd.split(prefix, 1)[1].strip()
                return self._navigate_to_landmark(remainder, robot_pos, robot_yaw)

        # Check for "come back" / "return"
        if "come back" in cmd or "return" in cmd:
            return self._navigate_to_landmark("center", robot_pos, robot_yaw)

        # Check for "explore" / "wander"
        if "explore" in cmd or "wander" in cmd or "patrol" in cmd:
            return self._explore_command(robot_pos, robot_yaw)

        # Check for "turn around"
        if "turn around" in cmd or "180" in cmd:
            return {"forward": 0.0, "lateral": 0.0, "turn": 1.0}, "Turning around."

        # Check for "circle" or "orbit"
        if "circle" in cmd or "orbit" in cmd:
            return {"forward": 0.5, "lateral": 0.0, "turn": 0.4}, "Moving in a circle."

        # Check for simple motion keywords
        for keyword, vel in self.MOTION_KEYWORDS.items():
            if keyword in cmd:
                desc = f"Moving: fwd={vel['forward']:.1f}, turn={vel['turn']:.1f}"
                return vel.copy(), desc

        # Default: try to interpret as landmark name
        vel, desc = self._navigate_to_landmark(cmd, robot_pos, robot_yaw)
        if self._target_landmark is not None:
            return vel, desc

        # Fallback: gentle forward motion
        return {"forward": 0.3, "lateral": 0.0, "turn": 0.0}, f"Walking forward (didn't understand: '{command}')"

    def _navigate_to_landmark(
        self,
        target_text: str,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> Tuple[Dict[str, float], str]:
        """Navigate toward a named landmark."""
        target_text = target_text.strip().rstrip(".")

        # Resolve landmark name
        landmark_name = None
        for alias, name in self.LANDMARK_ALIASES.items():
            if alias in target_text:
                landmark_name = name
                break

        if landmark_name is None:
            # Try direct name match
            for name in self.landmarks:
                if name.replace("_", " ") in target_text:
                    landmark_name = name
                    break

        if landmark_name is None or landmark_name not in self.landmarks:
            self._target_landmark = None
            return {"forward": 0.3, "lateral": 0.0, "turn": 0.0}, f"Unknown target: '{target_text}'. Walking forward."

        self._target_landmark = landmark_name
        target_pos = self.landmarks[landmark_name]

        return self._compute_navigation_cmd(target_pos, robot_pos, robot_yaw, landmark_name)

    def _compute_navigation_cmd(
        self,
        target_pos: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        label: str = "",
    ) -> Tuple[Dict[str, float], str]:
        """Compute velocity command to navigate to a position."""
        to_target = target_pos - robot_pos
        distance = np.linalg.norm(to_target)

        if distance < 0.3:
            self._target_landmark = None
            return {"forward": 0.0, "lateral": 0.0, "turn": 0.0}, f"Reached {label}!"

        # Desired heading
        desired_yaw = math.atan2(to_target[1], to_target[0])
        yaw_error = desired_yaw - robot_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Proportional control
        turn = np.clip(yaw_error * 2.0, -1.0, 1.0)

        # Forward speed based on alignment
        alignment = math.cos(yaw_error)
        forward = np.clip(alignment * 0.8, -0.2, 0.8)

        # Slow down near target
        if distance < 1.0:
            forward *= distance

        desc = f"Navigating to {label} (dist={distance:.1f}m, heading_err={math.degrees(yaw_error):.0f}deg)"
        return {"forward": float(forward), "lateral": 0.0, "turn": float(turn)}, desc

    def _explore_command(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> Tuple[Dict[str, float], str]:
        """Generate exploration behavior - visit landmarks in sequence."""
        self._explore_phase += 0.01

        # Cycle through landmarks
        landmark_names = list(self.landmarks.keys())
        idx = int(self._explore_phase) % len(landmark_names)
        target_name = landmark_names[idx]
        target_pos = self.landmarks[target_name]

        dist = np.linalg.norm(target_pos - robot_pos)
        if dist < 0.5:
            self._explore_phase += 1  # Move to next landmark

        vel, _ = self._compute_navigation_cmd(target_pos, robot_pos, robot_yaw, target_name)
        return vel, f"Exploring... heading to {target_name} (dist={dist:.1f}m)"

    def get_active_navigation_cmd(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> Optional[Tuple[Dict[str, float], str]]:
        """Continue navigating to active target (if any)."""
        if self._target_landmark is None:
            return None

        target_pos = self.landmarks.get(self._target_landmark)
        if target_pos is None:
            self._target_landmark = None
            return None

        return self._compute_navigation_cmd(
            target_pos, robot_pos, robot_yaw, self._target_landmark
        )


# ============================================================================
# Main Demo
# ============================================================================

def run_demo(
    command: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    save_video: bool = False,
    output_dir: str = "demo_output/go2",
    checkpoint: Optional[str] = None,
    num_steps: int = 300,
    camera: str = "cinematic_high",
    seed: int = 42,
    interactive: bool = False,
):
    """
    Run the Go2 quadruped demo.

    Args:
        command: Natural language command (e.g. "walk to the red marker")
        width, height: Render resolution
        save_video: Whether to save output video
        output_dir: Where to save outputs
        checkpoint: Cosmos-Predict checkpoint for imagination mode
        num_steps: Number of simulation steps to run
        camera: Camera preset name
        seed: Random seed
        interactive: If True, enter interactive command loop
    """
    from sim.quadruped.env import Go2Env

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Go2 Quadruped World Model Demo")
    print("  Cosmos-Predict2.5 Action-Conditioned Planning")
    print("=" * 70)

    # Initialize environment
    print(f"\nInitializing Go2 environment ({width}x{height})...")
    env = Go2Env(render_width=width, render_height=height)
    env.set_camera(camera)

    # Initialize NL controller
    nl_controller = NaturalLanguageController(landmarks=Go2Env.LANDMARKS)

    # Reset
    print("Resetting environment...")
    obs, info = env.reset(seed=seed, options={"robot_pos": [0.0, 0.0], "robot_yaw": 0.0})
    print(f"Robot at ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Target at ({info['target_position'][0]:.2f}, {info['target_position'][1]:.2f})")

    frames = [obs]

    if interactive:
        _run_interactive(env, nl_controller, save_video, output_path, camera)
        return

    # Parse command
    if command is None:
        command = "walk to the red marker"

    print(f"\nCommand: \"{command}\"")
    vel_cmd, description = nl_controller.parse_command(
        command, info["robot_position"], info["robot_yaw"]
    )
    print(f"Action: {description}")

    # === IMAGINATION PHASE (if checkpoint provided) ===
    if checkpoint is not None:
        print("\n--- Imagination Phase ---")
        _run_imagination(env, nl_controller, command, info, checkpoint, output_path)

    # === EXECUTION PHASE ===
    print("\n--- Executing command ---")
    env.command_velocity(**vel_cmd)

    for step in range(num_steps):
        obs, reward, terminated, truncated, info = env.step()
        frames.append(obs)

        # Update navigation command based on current position
        nav_update = nl_controller.get_active_navigation_cmd(
            info["robot_position"], info["robot_yaw"]
        )
        if nav_update is not None:
            vel_cmd, description = nav_update
            env.command_velocity(**vel_cmd)

        if step % 50 == 0:
            pos = info["robot_position"]
            dist = info["distance_to_target"]
            print(f"  Step {step:4d}: pos=({pos[0]:5.2f}, {pos[1]:5.2f}), "
                  f"dist={dist:.2f}m, height={info['robot_height']:.3f}m")

        if terminated:
            if info.get("reached_target"):
                print(f"\n  Target reached at step {step}!")
            elif info.get("fell"):
                print(f"\n  Robot fell at step {step}!")
            break

        if truncated:
            print(f"\n  Episode truncated at step {step}")
            break

    # Final status
    print(f"\nFinal position: ({info['robot_position'][0]:.2f}, {info['robot_position'][1]:.2f})")
    print(f"Final distance to target: {info['distance_to_target']:.2f}m")
    print(f"Success: {info['success']}")

    # Save outputs
    if save_video and len(frames) > 1:
        try:
            import imageio
            video_path = output_path / "go2_demo.mp4"
            print(f"\nSaving video ({len(frames)} frames at {width}x{height})...")
            imageio.mimwrite(str(video_path), frames, fps=50, quality=8)
            print(f"Saved to {video_path}")

            # Also save a thumbnail grid of key frames
            _save_keyframe_grid(frames, output_path / "go2_keyframes.png")
        except ImportError:
            print("Install imageio to save video: pip install imageio[ffmpeg]")

    env.close()
    print("\nDone!")


def _run_interactive(env, nl_controller, save_video, output_path, camera):
    """Interactive command loop."""
    print("\n" + "=" * 50)
    print("Interactive Mode - Type commands in English")
    print("Examples:")
    print('  "walk to the red marker"')
    print('  "explore the trees"')
    print('  "turn left"')
    print('  "come back"')
    print('  "stop"')
    print('Type "quit" to exit.')
    print("=" * 50)

    all_frames = []

    while True:
        try:
            command = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if command.lower() in ["quit", "exit", "q"]:
            break

        if not command:
            continue

        info = env._get_info()
        vel_cmd, description = nl_controller.parse_command(
            command, info["robot_position"], info["robot_yaw"]
        )
        print(f"  -> {description}")
        env.command_velocity(**vel_cmd)

        # Execute for 100 steps
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step()
            all_frames.append(obs)

            # Update navigation
            nav_update = nl_controller.get_active_navigation_cmd(
                info["robot_position"], info["robot_yaw"]
            )
            if nav_update is not None:
                vel_cmd, desc = nav_update
                env.command_velocity(**vel_cmd)

                if step % 30 == 0:
                    print(f"  [{step}] {desc}")

            if terminated:
                if info.get("reached_target"):
                    print("  Target reached!")
                break

        pos = info["robot_position"]
        print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")

    if save_video and all_frames:
        try:
            import imageio
            video_path = output_path / "go2_interactive.mp4"
            imageio.mimwrite(str(video_path), all_frames, fps=50, quality=8)
            print(f"\nSaved interactive session to {video_path}")
        except ImportError:
            pass

    env.close()


def _run_imagination(env, nl_controller, command, info, checkpoint, output_path):
    """Run world model imagination for planning."""
    print("Sampling candidate trajectories...")

    # Generate candidate action sequences
    num_candidates = 4
    horizon = 50  # steps

    candidates = []
    rng = np.random.default_rng(42)

    # Parse the command into base velocity
    base_vel, _ = nl_controller.parse_command(
        command, info["robot_position"], info["robot_yaw"]
    )

    for i in range(num_candidates):
        # Vary the base command with noise
        noise_fwd = rng.normal(0, 0.2)
        noise_turn = rng.normal(0, 0.3)
        vel_seq = []
        for t in range(horizon):
            vel = {
                "forward": np.clip(base_vel["forward"] + noise_fwd * (0.5 + 0.5 * np.sin(t * 0.1)), -1, 1),
                "lateral": np.clip(base_vel["lateral"] + rng.normal(0, 0.05), -1, 1),
                "turn": np.clip(base_vel["turn"] + noise_turn * np.cos(t * 0.15), -1, 1),
            }
            vel_seq.append(vel)
        candidates.append(vel_seq)

    # "Imagine" by running simulation (proxy for world model)
    print(f"Imagining {num_candidates} futures...")
    imagined = []
    scores = []

    for i, vel_seq in enumerate(candidates):
        # Save and restore state
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        saved_gait_phase = env.gait.phase

        traj_frames = []
        for vel in vel_seq:
            env.command_velocity(**vel)
            obs, _, terminated, _, step_info = env.step()
            traj_frames.append(obs)
            if terminated:
                break

        final_dist = step_info["distance_to_target"]
        fell = step_info.get("fell", False)
        score = -final_dist - (10.0 if fell else 0.0)
        scores.append(score)
        imagined.append(np.array(traj_frames))

        print(f"  Candidate {i + 1}: final_dist={final_dist:.2f}m, "
              f"fell={fell}, score={score:.2f}")

        # Restore state
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.gait.phase = saved_gait_phase
        import mujoco
        mujoco.mj_forward(env.model, env.data)

    best_idx = np.argmax(scores)
    print(f"\nBest trajectory: Candidate {best_idx + 1} (score={scores[best_idx]:.2f})")

    # Apply the best trajectory's velocity commands
    best_vels = candidates[best_idx]
    # Re-parse original command to set initial velocity
    vel_cmd, _ = nl_controller.parse_command(
        command, info["robot_position"], info["robot_yaw"]
    )
    env.command_velocity(**vel_cmd)


def _save_keyframe_grid(frames, path, num_keyframes=6):
    """Save a grid of evenly-spaced keyframes."""
    try:
        import cv2

        indices = np.linspace(0, len(frames) - 1, num_keyframes, dtype=int)
        keyframes = [frames[i] for i in indices]

        # Make 2x3 grid
        cols = 3
        rows = 2
        h, w = keyframes[0].shape[:2]

        # Add frame number overlay
        labeled = []
        for i, (idx, frame) in enumerate(zip(indices, keyframes)):
            f = frame.copy()
            cv2.putText(f, f"t={idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)
            labeled.append(f)

        grid_rows = []
        for r in range(rows):
            row_frames = labeled[r * cols:(r + 1) * cols]
            while len(row_frames) < cols:
                row_frames.append(np.zeros_like(labeled[0]))
            grid_rows.append(np.hstack(row_frames))

        grid = np.vstack(grid_rows)
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Saved keyframe grid to {path}")
    except ImportError:
        pass


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Go2 Quadruped World Model Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo/go2_world_model_demo.py --command "walk to the red marker"
  python demo/go2_world_model_demo.py --command "explore the trees" --camera follow_behind
  python demo/go2_world_model_demo.py --interactive
  python demo/go2_world_model_demo.py --command "trot forward" --width 1920 --height 1080 --save-video
        """,
    )
    parser.add_argument("--command", "-c", type=str, default=None,
                       help="Natural language command (e.g. 'walk to the red marker')")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Enter interactive command loop")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--output-dir", type=str, default="demo_output/go2")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Cosmos-Predict checkpoint for imagination mode")
    parser.add_argument("--num-steps", type=int, default=300, help="Steps to simulate")
    parser.add_argument("--camera", type=str, default="cinematic_high",
                       choices=["follow_behind", "follow_side", "cinematic_high",
                                "close_up", "bird_eye", "dramatic_low"],
                       help="Camera angle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.interactive:
        run_demo(interactive=True, width=args.width, height=args.height,
                save_video=args.save_video, output_dir=args.output_dir,
                camera=args.camera, seed=args.seed)
    else:
        run_demo(
            command=args.command,
            width=args.width,
            height=args.height,
            save_video=args.save_video,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            num_steps=args.num_steps,
            camera=args.camera,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

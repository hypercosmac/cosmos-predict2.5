#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Go2 Live Interactive GUI
========================

Opens a real-time MuJoCo viewer window with the Go2 quadruped in the outdoor
scene. Type plain English commands in the terminal and watch the robot
execute them live in the 3D viewport.

Controls:
  - Type commands in the terminal (e.g. "walk to the red marker")
  - Mouse drag to orbit the camera in the viewer window
  - Scroll to zoom
  - Double-click to re-center

Usage:
    python demo/go2_live_gui.py
    python demo/go2_live_gui.py --command "trot forward"
"""

import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import mujoco
import mujoco.viewer


# ── Inline NL controller (keep self-contained) ──────────────────────────

LANDMARK_ALIASES = {
    "red marker": "waypoint_a", "marker": "waypoint_a",
    "waypoint a": "waypoint_a", "waypoint": "waypoint_a",
    "blue marker": "waypoint_b", "waypoint b": "waypoint_b",
    "first tree": "tree_1", "big tree": "tree_1", "tree": "tree_1",
    "trees": "tree_2", "left tree": "tree_2",
    "far tree": "tree_3", "small tree": "tree_4",
    "ball": "ball", "center": "center", "middle": "center",
    "home": "center", "origin": "center", "start": "center",
}

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


def resolve_landmark(text: str) -> Optional[np.ndarray]:
    """Return position array if text contains a known landmark."""
    text = text.lower().strip().rstrip(".")
    for alias, name in LANDMARK_ALIASES.items():
        if alias in text:
            return LANDMARKS[name].copy()
    for name, pos in LANDMARKS.items():
        if name.replace("_", " ") in text:
            return pos.copy()
    return None


def parse_command(cmd: str) -> Dict[str, float]:
    """Parse a plain-English command into {forward, lateral, turn} dict."""
    c = cmd.lower().strip()

    # Stop
    for kw in ("stop", "halt", "stay", "wait", "sit"):
        if kw in c:
            return {"forward": 0.0, "lateral": 0.0, "turn": 0.0}

    # Explore / patrol
    if any(kw in c for kw in ("explore", "wander", "patrol")):
        return {"forward": 0.5, "lateral": 0.0, "turn": 0.25}

    # Turn-only
    if "turn around" in c or "180" in c:
        return {"forward": 0.0, "lateral": 0.0, "turn": 1.0}
    if "spin" in c:
        return {"forward": 0.0, "lateral": 0.0, "turn": 1.0}
    if "circle" in c or "orbit" in c:
        return {"forward": 0.5, "lateral": 0.0, "turn": 0.4}

    # Simple directional
    if "left" in c and "turn" in c:
        return {"forward": 0.2, "lateral": 0.0, "turn": 0.7}
    if "right" in c and "turn" in c:
        return {"forward": 0.2, "lateral": 0.0, "turn": -0.7}
    if "left" in c:
        return {"forward": 0.3, "lateral": 0.0, "turn": 0.5}
    if "right" in c:
        return {"forward": 0.3, "lateral": 0.0, "turn": -0.5}
    if "backward" in c or "reverse" in c or "back" in c:
        return {"forward": -0.5, "lateral": 0.0, "turn": 0.0}

    # Speed keywords
    if "run" in c or "fast" in c or "sprint" in c:
        return {"forward": 1.0, "lateral": 0.0, "turn": 0.0}
    if "slow" in c:
        return {"forward": 0.3, "lateral": 0.0, "turn": 0.0}
    if "trot" in c:
        return {"forward": 0.6, "lateral": 0.0, "turn": 0.0}

    # "forward" / "walk" / "go" with no target
    for kw in ("forward", "ahead", "straight", "walk", "go"):
        if kw in c:
            return {"forward": 0.6, "lateral": 0.0, "turn": 0.0}

    # Fallback: treat as unknown, gentle walk
    return {"forward": 0.4, "lateral": 0.0, "turn": 0.0}


# ── Gait generator (imported from env but inlined for standalone) ────────

from sim.quadruped.env import TrotGaitGenerator


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Go2 Live Interactive GUI")
    parser.add_argument("--command", "-c", type=str, default=None,
                        help="Initial command to execute")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────
    scene_path = repo_root / "sim" / "quadruped" / "assets" / "outdoor_scene.xml"
    print(f"Loading scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # ── Gait & PD controller ─────────────────────────────────────────
    gait = TrotGaitGenerator(frequency=3.0, dt=model.opt.timestep)
    kp = np.tile([30.0, 60.0, 60.0], 4)
    kd = np.tile([1.0, 3.0, 3.0], 4)

    # ── Shared state (thread-safe via GIL for simple floats) ─────────
    vel_cmd = {"forward": 0.0, "lateral": 0.0, "turn": 0.0}
    nav_target: Optional[np.ndarray] = None  # (x, y) or None
    status_text = "Standing. Type a command below."
    running = True

    def get_base_pos():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        return data.xpos[bid].copy()

    def get_base_yaw():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        q = data.xquat[bid]
        siny = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1 - 2 * (q[2]**2 + q[3]**2)
        return math.atan2(siny, cosy)

    # ── Input thread ─────────────────────────────────────────────────
    def input_loop():
        nonlocal vel_cmd, nav_target, status_text, running

        # Apply initial command if provided
        if args.command:
            process_command(args.command)

        print("\n" + "=" * 55)
        print("  Go2 Live GUI — Type commands in plain English")
        print("=" * 55)
        print("Examples:")
        print('  walk forward      trot to the red marker')
        print('  turn left         explore the trees')
        print('  run fast          come back to center')
        print('  stop              spin around')
        print("Type 'quit' to exit.")
        print("=" * 55 + "\n")

        while running:
            try:
                cmd = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                running = False
                break
            if cmd.lower() in ("quit", "exit", "q"):
                running = False
                break
            if cmd:
                process_command(cmd)

    def process_command(cmd: str):
        nonlocal vel_cmd, nav_target, status_text

        # Check for landmark navigation
        nav_prefixes = ["go to", "walk to", "move to", "navigate to",
                        "head to", "run to", "trot to", "approach"]
        target_text = None
        for prefix in nav_prefixes:
            if prefix in cmd.lower():
                target_text = cmd.lower().split(prefix, 1)[1].strip()
                break

        if "come back" in cmd.lower() or "return" in cmd.lower():
            target_text = "center"

        if target_text is not None:
            pos = resolve_landmark(target_text)
            if pos is not None:
                nav_target = pos
                status_text = f"Navigating to {target_text}..."
                print(f"  → Navigating to {target_text} at ({pos[0]:.1f}, {pos[1]:.1f})")
                return
            else:
                print(f"  → Unknown landmark: '{target_text}'")

        # Simple velocity command
        v = parse_command(cmd)
        vel_cmd.update(v)
        nav_target = None
        status_text = f"Command: {cmd}"
        print(f"  → fwd={v['forward']:.1f}  turn={v['turn']:.1f}")

    # ── Physics + control callback ───────────────────────────────────
    def controller(m, d):
        nonlocal vel_cmd, nav_target, status_text

        # If navigating to a target, compute heading control
        if nav_target is not None:
            pos2d = get_base_pos()[:2]
            yaw = get_base_yaw()
            to_target = nav_target - pos2d
            dist = np.linalg.norm(to_target)

            if dist < 0.35:
                # Arrived
                nav_target_ref = nav_target  # capture before clearing
                vel_cmd.update({"forward": 0.0, "lateral": 0.0, "turn": 0.0})
                status_text = "Arrived!"
                # We can't set nav_target=None inside controller easily,
                # so set speed to zero; the command is effectively done.
                # Next user command will override.
            else:
                desired_yaw = math.atan2(to_target[1], to_target[0])
                yaw_err = math.atan2(math.sin(desired_yaw - yaw),
                                     math.cos(desired_yaw - yaw))
                turn = float(np.clip(yaw_err * 2.5, -1.0, 1.0))
                alignment = math.cos(yaw_err)
                fwd = float(np.clip(alignment * 0.7, -0.1, 0.8))
                if dist < 1.0:
                    fwd *= dist
                vel_cmd.update({"forward": fwd, "lateral": 0.0, "turn": turn})

        # Generate gait targets
        targets = gait.get_targets(
            forward_speed=vel_cmd["forward"],
            lateral_speed=vel_cmd["lateral"],
            turn_rate=vel_cmd["turn"],
        )

        # PD control
        qpos_joints = d.qpos[7:19]
        qvel_joints = d.qvel[6:18]
        pos_err = targets - qpos_joints
        vel_err = -qvel_joints
        torques = kp * pos_err + kd * vel_err
        d.ctrl[:12] = np.clip(torques, -23.7, 23.7)

    # ── Launch viewer ────────────────────────────────────────────────
    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    print("\nOpening MuJoCo viewer window...")
    print("(Drag mouse to orbit, scroll to zoom, double-click to re-center)\n")

    # mujoco.viewer.launch_passive gives us a handle to the viewer
    # while we control the physics loop ourselves.
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Sync the controller callback into the physics step
        # We'll step manually and call controller ourselves.
        while viewer.is_running() and running:
            step_start = time.time()

            # Run controller + physics
            controller(model, data)
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Maintain real-time
            elapsed = time.time() - step_start
            dt = model.opt.timestep
            if elapsed < dt:
                time.sleep(dt - elapsed)

    running = False
    print("\nViewer closed. Goodbye!")


if __name__ == "__main__":
    main()

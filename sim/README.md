# Factory Simulation for Cosmos-Predict2.5 World Model Demo

This module provides simulated factory environments for demonstrating how robots
can be augmented with world model capabilities using Cosmos-Predict2.5.

## Overview

The demo showcases **planning-via-imagination**: a mobile robot uses a learned
world model to imagine multiple possible futures based on different action
sequences, then selects the best trajectory to reach its goal.

## Components

### Simulation Environments

Two environment implementations are provided:

1. **`SimpleFactoryEnv`** (no dependencies)
   - Lightweight 2D simulation using only NumPy
   - Works anywhere without additional packages
   - Good for quick demos and testing

2. **`FactoryEnv`** (requires MuJoCo)
   - Full 3D MuJoCo-based simulation
   - Realistic physics and rendering
   - Multiple camera views
   - Robot arm for manipulation tasks

### Environment Features

- **4m x 4m factory floor** with workstations, storage areas, and delivery zones
- **Mobile robot** with differential drive kinematics
- **Goal-directed navigation** tasks
- **RGB observations** (configurable resolution)
- **Gym-like API**: `reset()`, `step()`, `render()`

## Quick Start

```python
# Using the simple environment (works without MuJoCo)
from sim.factory import SimpleFactoryEnv

env = SimpleFactoryEnv(render_size=256)
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space_sample()  # [vx, vy, vyaw]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print("Goal reached!")
        break

env.close()
```

## Running Demos

### Planning-via-Imagination Demo

Shows how a world model augments robot planning:

```bash
python demo/run_factory_simulation.py --mode planning
```

This demonstrates:
1. Generating 6 candidate action sequences
2. Imagining outcomes for each (using simulation as proxy for world model)
3. Selecting the best trajectory
4. Executing the chosen plan

### Random Actions Demo

```bash
python demo/run_factory_simulation.py --mode random
```

### Goal-Directed Demo

```bash
python demo/run_factory_simulation.py --mode goal
```

### Saving Outputs

Add `--save-video` to save MP4 videos and PNG visualizations:

```bash
python demo/run_factory_simulation.py --mode planning --save-video --output demo_output/
```

## Integration with Cosmos-Predict2.5

### Data Collection

Generate training data for the action-conditioned world model:

```bash
python scripts/collect_factory_data.py --output_dir datasets/factory --num_episodes 1000
```

Data format:
```
datasets/factory/
├── annotation/
│   ├── train/
│   │   ├── episode_00000.json
│   │   └── ...
│   └── val/
└── videos/
    ├── episode_00000.mp4
    └── ...
```

### Dataset Adapter

The `FactoryVideoDataset` class adapts collected data for Cosmos-Predict2.5 training:

```python
from cosmos_predict2.datasets import FactoryVideoDataset

dataset = FactoryVideoDataset(
    annotation_path="datasets/factory/annotation/train",
    video_path="datasets/factory",
    num_action_per_chunk=15,
    video_size=[256, 256],
)
```

### World Model Demo

With a trained checkpoint:

```bash
python demo/factory_world_model_demo.py --checkpoint path/to/checkpoint
```

Without a checkpoint (simulation-only mode):

```bash
python demo/factory_world_model_demo.py --simulation-only
```

## Installation

### Minimal (SimpleFactoryEnv only)

No additional packages required beyond NumPy.

### Full Installation (MuJoCo support)

```bash
pip install mujoco
```

### Visualization Support

```bash
pip install imageio[ffmpeg] opencv-python
```

## API Reference

### Action Space

Navigation mode (3D): `[vx, vy, vyaw]`
- `vx`: Forward velocity (-0.5 to 0.5 m/s)
- `vy`: Lateral velocity (-0.3 to 0.3 m/s)
- `vyaw`: Angular velocity (-1.0 to 1.0 rad/s)

### Observation Space

RGB image of shape `(render_size, render_size, 3)` as uint8.

### Info Dictionary

```python
{
    "robot_position": np.array([x, y]),
    "robot_yaw": float,
    "robot_state": np.array([x, y, yaw]),
    "goal_position": np.array([x, y]),
    "distance_to_goal": float,
    "step": int,
    "success": bool,
}
```

## Architecture

```
sim/
├── factory/
│   ├── __init__.py          # Module exports
│   ├── env.py               # MuJoCo-based FactoryEnv
│   ├── simple_env.py        # Pure-Python SimpleFactoryEnv
│   └── assets/
│       └── factory.xml      # MuJoCo scene definition
```

## License

Apache 2.0 - See LICENSE file in repository root.

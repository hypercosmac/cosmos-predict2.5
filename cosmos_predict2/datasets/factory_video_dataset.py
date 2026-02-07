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
Factory Video Dataset for Cosmos-Predict2.5 Action-Conditioned Training

This dataset adapter loads factory simulation data and formats it for
training the Cosmos-Predict2.5 action-conditioned world model.

The dataset follows the pattern established by Dataset_3D in:
cosmos_predict2/_src/predict2/action/datasets/dataset_local.py

Key differences from Dataset_3D:
- Action dimension is 3 (vx, vy, vyaw) instead of 7 (robot arm)
- State is (x, y, yaw) instead of 6-DoF pose
- No rotation matrix conversions needed
"""

import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import imageio


class Resize_Preprocess:
    """Resize video frames to target size with bilinear interpolation."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: Tensor of shape (C, T, H, W) or (T, C, H, W)

        Returns:
            Resized video tensor
        """
        # Handle different input formats
        if video.dim() == 4:
            if video.shape[0] in [1, 3]:  # (C, T, H, W)
                video = video.permute(1, 0, 2, 3)  # -> (T, C, H, W)
                was_ctchw = True
            else:
                was_ctchw = False

            # Resize each frame
            resized = torch.nn.functional.interpolate(
                video.float(),
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )

            if was_ctchw:
                resized = resized.permute(1, 0, 2, 3)  # -> (C, T, H, W)

            return resized
        else:
            raise ValueError(f"Expected 4D tensor, got {video.dim()}D")


class ToTensorVideo:
    """Convert video numpy array to tensor."""

    def __call__(self, video: np.ndarray) -> torch.Tensor:
        """
        Args:
            video: numpy array of shape (T, H, W, C) uint8

        Returns:
            Tensor of shape (C, T, H, W) float32 in [0, 1]
        """
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        # Rearrange from (T, H, W, C) to (C, T, H, W)
        if video.dim() == 4 and video.shape[-1] in [1, 3]:
            video = video.permute(3, 0, 1, 2)

        # Convert to float [0, 1]
        if video.dtype == torch.uint8:
            video = video.float() / 255.0

        return video


class FactoryVideoDataset(Dataset):
    """
    Dataset for factory simulation action-conditioned training.

    Loads video sequences and corresponding actions from the factory
    environment data, formatted for Cosmos-Predict2.5 training.

    Args:
        annotation_path: Path to annotation JSON files (train/val split)
        video_path: Base path to video files
        num_action_per_chunk: Number of actions per training sample (prediction horizon)
        video_size: Target (H, W) for video frames
        fps_downsample_ratio: Sample every N frames
        mode: 'train' or 'val'
        normalize: If True, normalize to [-1, 1]
        action_scaler: Scale factor for actions (default 1.0 for nav actions)
    """

    # Action dimension for navigation: [vx, vy, vyaw]
    ACTION_DIM = 3

    def __init__(
        self,
        annotation_path: str,
        video_path: str,
        num_action_per_chunk: int = 15,
        video_size: List[int] = [256, 256],
        fps_downsample_ratio: int = 1,
        mode: str = "train",
        normalize: bool = False,
        debug: bool = False,
        action_scaler: float = 1.0,
        val_start_frame_interval: int = 1,
    ):
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.video_path = Path(video_path)
        self.num_action_per_chunk = num_action_per_chunk
        self.sequence_length = 1 + num_action_per_chunk  # 1 initial frame + prediction frames
        self.video_size = video_size
        self.fps_downsample_ratio = fps_downsample_ratio
        self.mode = mode
        self.normalize = normalize
        self.action_scaler = action_scaler

        # Frame interval for sampling start positions
        self.start_frame_interval = 1 if mode == "train" else val_start_frame_interval

        # Action scaling
        self.c_act_scaler = np.array([action_scaler] * self.ACTION_DIM, dtype=np.float32)

        # Load annotations
        self.ann_files = self._init_anns(self.annotation_path)
        print(f"[FactoryVideoDataset] Found {len(self.ann_files)} annotation files in {self.annotation_path}")

        # Build sample index
        self.samples = self._init_sequences(self.ann_files)
        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))

        if debug:
            self.samples = self.samples[:10]

        print(f"[FactoryVideoDataset] {len(self.samples)} samples in total")

        # Transforms
        self.to_tensor = ToTensorVideo()
        self.resize = Resize_Preprocess(tuple(video_size))

        if normalize:
            self.final_transform = T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            )
        else:
            self.final_transform = None

    def _init_anns(self, data_dir: Path) -> List[str]:
        """Load annotation file paths."""
        if not data_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {data_dir}")

        ann_files = sorted([
            str(f) for f in data_dir.glob("*.json")
        ])
        return ann_files

    def _init_sequences(self, ann_files: List[str]) -> List[Dict]:
        """Build index of valid training samples."""
        samples = []

        def process_ann_file(ann_file: str) -> List[Dict]:
            file_samples = []
            try:
                with open(ann_file, "r") as f:
                    ann = json.load(f)

                n_frames = ann.get("num_frames", len(ann.get("state", [])))

                # Generate samples with sliding window
                for frame_i in range(0, n_frames, self.start_frame_interval):
                    sample = {"ann_file": ann_file, "frame_ids": []}
                    curr_frame_i = frame_i

                    while True:
                        if curr_frame_i > (n_frames - 1):
                            break
                        sample["frame_ids"].append(curr_frame_i)
                        if len(sample["frame_ids"]) == self.sequence_length:
                            break
                        curr_frame_i += self.fps_downsample_ratio

                    # Only keep samples with full sequence length
                    if len(sample["frame_ids"]) == self.sequence_length:
                        file_samples.append(sample)

            except Exception as e:
                warnings.warn(f"Error processing {ann_file}: {e}")

            return file_samples

        # Process files with thread pool for speed
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(process_ann_file, f): f for f in ann_files}
            for future in tqdm(as_completed(futures), total=len(ann_files), desc="Indexing"):
                samples.extend(future.result())

        return samples

    def _load_video(self, video_path: str, frame_ids: List[int]) -> np.ndarray:
        """Load video frames at specified indices."""
        if HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            vr.seek(0)
            frames = vr.get_batch(frame_ids).asnumpy()
        else:
            # Fallback to imageio
            reader = imageio.get_reader(video_path)
            frames = []
            for idx in frame_ids:
                frame = reader.get_data(idx)
                frames.append(frame)
            frames = np.stack(frames)
            reader.close()

        return frames  # (T, H, W, C)

    def _get_frames(self, ann: Dict, frame_ids: List[int]) -> torch.Tensor:
        """Load and preprocess video frames."""
        video_info = ann["videos"]["0"]
        video_path = self.video_path / video_info["video_path"]

        frames = self._load_video(str(video_path), frame_ids)
        frames = frames.astype(np.uint8)

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames_tensor = self.to_tensor(frames)

        # Resize
        frames_tensor = self.resize(frames_tensor)

        # Optionally normalize
        if self.final_transform is not None:
            frames_tensor = self.final_transform(frames_tensor)
        else:
            # Keep as float [0, 1] or convert back to uint8
            frames_tensor = torch.clamp(frames_tensor * 255.0, 0, 255).to(torch.uint8)

        return frames_tensor

    def _get_actions(self, ann: Dict, frame_ids: List[int]) -> torch.Tensor:
        """
        Extract actions for the selected frames.

        Actions are the velocities applied between consecutive frames.
        For frame_ids [0, 1, 2, ...], we need actions [a0, a1, ...] where
        a_i is the action taken from frame i to frame i+1.
        """
        all_actions = np.array(ann["action"])  # (total_actions, 3)

        # Get actions corresponding to transitions between our frames
        # frame_ids gives us frames [f0, f1, f2, ...]
        # We need actions [a_f0, a_f1, ...] up to second-to-last frame
        action_indices = frame_ids[:-1]  # Don't need action for last frame

        # Handle case where frame indices might exceed action array
        action_indices = [min(i, len(all_actions) - 1) for i in action_indices]

        actions = all_actions[action_indices]
        actions = actions * self.c_act_scaler

        return torch.from_numpy(actions).float()

    def _get_states(self, ann: Dict, frame_ids: List[int]) -> np.ndarray:
        """Extract robot states (x, y, yaw) for selected frames."""
        all_states = np.array(ann["state"])  # (total_frames, 3)
        states = all_states[frame_ids]
        return states

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        """
        Get a training sample.

        Returns dict compatible with Cosmos action-conditioned training:
            - video: (C, T, H, W) uint8 or normalized float
            - action: (T-1, 3) float32 actions
            - t5_text_embeddings: (512, 1024) dummy embeddings
            - t5_text_mask: (512,) ones
            - fps: int
            - padding_mask: (1, H, W) zeros
            - num_frames: int
            - __key__: str episode identifier
        """
        if self.mode != "train":
            # Deterministic sampling for validation
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                ann = json.load(f)

            # Load video frames
            video = self._get_frames(ann, frame_ids)  # (C, T, H, W)

            # Load actions
            actions = self._get_actions(ann, frame_ids)  # (T-1, 3)

            # Build output dict
            data = {
                "video": video,
                "action": actions,
                # Dummy text embeddings (not used for action-conditioned)
                "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
                "t5_text_mask": torch.ones(512, dtype=torch.int64),
                "ai_caption": "",
                "fps": 20,
                "image_size": torch.tensor([self.video_size[0]] * 4, dtype=torch.float32),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, self.video_size[0], self.video_size[1]),
                "__key__": ann.get("episode_id", Path(ann_file).stem),
                "annotation_file": ann_file,
                # Extra info for debugging
                "goal": torch.tensor(ann.get("goal", [0, 0]), dtype=torch.float32),
            }

            return data

        except Exception as e:
            warnings.warn(f"Error loading sample {index} from {self.samples[index]['ann_file']}: {e}")
            # Return a random other sample
            return self[np.random.randint(len(self.samples))]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    """Test the dataset loader."""
    import sys

    # Test with dummy or real data
    data_path = Path("datasets/factory")

    if not data_path.exists():
        print(f"Test data not found at {data_path}")
        print("Run scripts/collect_factory_data.py first to generate data")
        sys.exit(1)

    print("Loading dataset...")
    dataset = FactoryVideoDataset(
        annotation_path=str(data_path / "annotation" / "train"),
        video_path=str(data_path),
        num_action_per_chunk=15,
        video_size=[256, 256],
        mode="train",
        debug=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a sample
    print("\nLoading sample 0...")
    sample = dataset[0]

    print(f"Video shape: {sample['video'].shape}")
    print(f"Video dtype: {sample['video'].dtype}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Action dtype: {sample['action'].dtype}")
    print(f"Episode key: {sample['__key__']}")
    print(f"Goal: {sample['goal']}")

    # Verify dimensions
    assert sample["video"].shape[0] == 3, "Video should have 3 channels"
    assert sample["video"].shape[1] == 16, f"Video should have 16 frames, got {sample['video'].shape[1]}"
    assert sample["action"].shape[0] == 15, f"Should have 15 actions, got {sample['action'].shape[0]}"
    assert sample["action"].shape[1] == 3, f"Actions should be 3D, got {sample['action'].shape[1]}"

    print("\nAll checks passed!")

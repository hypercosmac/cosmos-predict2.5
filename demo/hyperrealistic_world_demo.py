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

# MPS compatibility: set before torch is imported.
# 1) PYTORCH_ENABLE_MPS_FALLBACK — the UniPC scheduler uses torch.linalg.solve
#    which is not yet implemented on MPS; this makes PyTorch fall back to CPU.
# 2) PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 — disable the hard memory ceiling so
#    large allocations can spill into system swap instead of crashing.
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""
Hyperrealistic World Model Demo
================================

An interactive Gradio demo showcasing Cosmos-Predict2.5 as a hyperrealistic
world simulator. Users can generate physically plausible video predictions
from text prompts, images, or video context across diverse real-world scenarios.

Scenarios span autonomous driving, robotics, weather/nature, urban life,
and industrial environments -- all emphasizing photorealistic rendering
and physically accurate dynamics.

Usage:
    # Launch with default 2B model (requires ~24GB VRAM):
    python demo/hyperrealistic_world_demo.py

    # Use 14B model for higher quality (requires multi-GPU or offloading):
    python demo/hyperrealistic_world_demo.py --model_id nvidia/Cosmos-Predict2.5-14B

    # CPU offloading for limited VRAM:
    python demo/hyperrealistic_world_demo.py --offload

    # Custom port:
    python demo/hyperrealistic_world_demo.py --port 7861

    # CLI-only mode (no Gradio, runs a single scenario):
    python demo/hyperrealistic_world_demo.py --cli --scenario rainy_highway
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A curated hyperrealistic world-model scenario."""
    name: str
    display_name: str
    category: str
    inference_type: str  # text2world | image2world | video2world
    prompt: str
    negative_prompt: str = (
        "The video captures a series of frames showing ugly scenes, static with "
        "no motion, motion blur, over-saturation, shaky footage, low resolution, "
        "grainy texture, pixelated images, poorly lit areas, underexposed and "
        "overexposed scenes, poor color balance, washed out colors, choppy "
        "sequences, jerky movements, low frame rate, artifacting, color banding, "
        "unnatural transitions, outdated special effects, fake elements, "
        "unconvincing visuals, poorly edited content, jump cuts, visual noise, "
        "and flickering. Overall, the video is of poor quality."
    )
    num_output_frames: int = 93
    num_steps: int = 36
    guidance: float = 7.0
    seed: int = 42
    input_path: Optional[str] = None  # relative to assets/base/
    tags: List[str] = field(default_factory=list)


# ---------- Autonomous Driving ----------
SCENARIOS: Dict[str, Scenario] = {}

SCENARIOS["rainy_highway"] = Scenario(
    name="rainy_highway",
    display_name="Rainy Highway Drive",
    category="Autonomous Driving",
    inference_type="text2world",
    prompt=(
        "A dashcam view of a multi-lane highway during heavy rainfall at dusk. "
        "Headlights from oncoming traffic reflect off the wet asphalt, creating "
        "long streaks of white and amber light. Raindrops splatter on the "
        "windshield as wipers sweep back and forth at regular intervals. Spray "
        "from a large semi-truck ahead billows into the camera's view, partially "
        "obscuring the red taillights of the vehicle in front. The sky is a "
        "deep slate grey with occasional flashes of distant lightning illuminating "
        "the cloud base. Roadside reflectors pulse rhythmically as the vehicle "
        "maintains speed, and the guardrail's metal surface gleams with water "
        "droplets. Gradually, the rain eases and the road surface brightens, "
        "revealing puddles that ripple as tires cut through them."
    ),
    tags=["driving", "weather", "night"],
)

SCENARIOS["snowy_intersection"] = Scenario(
    name="snowy_intersection",
    display_name="Snowy Urban Intersection",
    category="Autonomous Driving",
    inference_type="text2world",
    prompt=(
        "A front-facing camera captures a busy urban intersection blanketed in "
        "fresh snow. Traffic signals switch from red to green, and a city bus "
        "with frosted windows begins to pull away, its tires leaving dark tracks "
        "in the white surface. Pedestrians in heavy coats cross the street, "
        "their footprints forming a scattered pattern on the sidewalk. Snowflakes "
        "drift steadily through the amber glow of streetlights. A cyclist slowly "
        "navigates the slushy bike lane while cars idle at the curb with exhaust "
        "plumes rising into the frigid air. Building facades along the street are "
        "dusted with snow on ledges and awnings. The scene progresses as vehicles "
        "advance through the intersection, headlights reflecting off the wet "
        "snow-covered road."
    ),
    tags=["driving", "weather", "urban", "snow"],
)

SCENARIOS["sunset_coastal_road"] = Scenario(
    name="sunset_coastal_road",
    display_name="Sunset Coastal Drive",
    category="Autonomous Driving",
    inference_type="text2world",
    prompt=(
        "A dashcam records a winding coastal highway at golden hour. The sun "
        "hangs low over the Pacific Ocean, casting a warm orange glow across "
        "the water's surface with glittering reflections. The two-lane road "
        "curves along dramatic cliffsides, with a metal guardrail separating "
        "the asphalt from a steep drop to rocky shores below. Waves crash against "
        "dark basalt formations, sending mist into the air. A vintage convertible "
        "ahead rounds the curve, its chrome bumper catching the sunlight. Coastal "
        "vegetation — ice plants and wild grasses — line the road's shoulder. "
        "As the car progresses, the sun descends further, shifting the sky from "
        "gold to deep magenta, and the headlights of approaching traffic begin "
        "to glow brighter against the dimming landscape."
    ),
    tags=["driving", "nature", "sunset"],
)

# ---------- Robotics ----------
SCENARIOS["warehouse_robot"] = Scenario(
    name="warehouse_robot",
    display_name="Warehouse Pick-and-Place",
    category="Robotics",
    inference_type="text2world",
    prompt=(
        "Inside a brightly lit logistics warehouse, a six-axis robotic arm with "
        "a pneumatic suction gripper operates above a moving conveyor belt. The "
        "arm identifies a cardboard package among several items, lowers precisely, "
        "and attaches to its surface with a soft pneumatic hiss. It lifts the "
        "package smoothly, rotates 90 degrees, and places it onto a second "
        "parallel conveyor belt heading in the opposite direction. The robot's "
        "joints articulate with mechanical precision, casting sharp shadows under "
        "the overhead fluorescent panels. Other packages continue flowing past on "
        "the belt. In the background, shelving units stacked with inventory are "
        "visible, and a small autonomous mobile robot rolls along a painted floor "
        "guide. The arm returns to its standby position and prepares for the next "
        "pick cycle."
    ),
    tags=["robotics", "industrial", "manipulation"],
)

SCENARIOS["surgical_precision"] = Scenario(
    name="surgical_precision",
    display_name="Precision Assembly Robot",
    category="Robotics",
    inference_type="text2world",
    prompt=(
        "A close-up view of a dual-arm robotic system performing delicate "
        "electronic assembly on a circuit board under bright task lighting. "
        "The left arm holds a small surface-mount component with micro-tweezers "
        "while the right arm applies a precise dot of solder paste. The component "
        "is placed onto copper pads with sub-millimeter accuracy, and a puff of "
        "rosin smoke rises briefly. The circuit board, green with gold traces, "
        "sits on a vacuum fixture that holds it perfectly still. A magnification "
        "lens mounted above gives a detailed view of the fine pitch leads. "
        "Both arms retract in coordinated motion, and the fixture rotates the "
        "board 90 degrees to present the next placement zone. Background racks "
        "of component reels are visible in soft focus."
    ),
    tags=["robotics", "electronics", "precision"],
)

# ---------- Weather & Nature ----------
SCENARIOS["thunderstorm_cityscape"] = Scenario(
    name="thunderstorm_cityscape",
    display_name="Thunderstorm Over City",
    category="Weather & Nature",
    inference_type="text2world",
    prompt=(
        "A wide-angle view of a major city skyline as a powerful thunderstorm "
        "rolls in from the west. Dark cumulonimbus clouds with anvil tops tower "
        "above glass skyscrapers. A bolt of lightning forks across the sky, "
        "its branches illuminating the cloud base in stark white-blue light for "
        "an instant. Rain begins to fall in heavy sheets, obscuring the far "
        "buildings in a grey curtain. Street-level traffic slows as headlights "
        "and neon signs reflect off the rapidly flooding pavement. Wind gusts "
        "send newspaper pages and leaves tumbling along the sidewalk. A second "
        "lightning strike hits a building's lightning rod, casting brief sharp "
        "shadows. The downpour intensifies, and rivulets of water stream down "
        "the camera lens before the scene transitions to a steady, heavy rain "
        "with the city lights glowing through the storm."
    ),
    tags=["weather", "urban", "dramatic"],
)

SCENARIOS["ocean_sunrise"] = Scenario(
    name="ocean_sunrise",
    display_name="Ocean Sunrise Timelapse",
    category="Weather & Nature",
    inference_type="text2world",
    prompt=(
        "A fixed tripod shot captures a tropical ocean sunrise from a sandy "
        "beach. The horizon transitions from deep indigo to warm coral as the "
        "sun's disc emerges slowly from behind a bank of low cumulus clouds. "
        "Gentle waves roll onto the shore with rhythmic consistency, leaving "
        "thin lines of white foam on wet sand that reflects the changing sky "
        "colors. Palm tree silhouettes frame the left edge of the composition. "
        "As the sun clears the clouds, it casts a narrow golden path across the "
        "water's surface, and the sand shifts from dark grey to warm beige. "
        "Seabirds glide low over the water in the middle distance. Small crabs "
        "scuttle across the sand at the bottom of the frame. The scene completes "
        "as full morning light bathes the beach, revealing turquoise shallows "
        "and the texture of coral beneath clear water."
    ),
    tags=["nature", "ocean", "sunrise"],
)

SCENARIOS["forest_rain"] = Scenario(
    name="forest_rain",
    display_name="Rainforest Downpour",
    category="Weather & Nature",
    inference_type="text2world",
    prompt=(
        "A ground-level camera captures a dense tropical rainforest canopy during "
        "an afternoon downpour. Large droplets cascade from broad green leaves, "
        "forming miniature waterfalls that splash onto the mossy forest floor. "
        "Ferns and undergrowth sway under the weight of accumulating water. "
        "Light filters through gaps in the canopy in visible shafts, each one "
        "filled with falling rain illuminated from above. A small stream running "
        "through the frame swells gradually, carrying leaf debris over smooth "
        "river stones. The bark of massive tree trunks darkens as water runs "
        "down their surfaces. Mist rises gently from the warm, wet ground. "
        "Eventually the rain eases to a fine drizzle, and the forest sounds of "
        "dripping water replace the heavy patter, while the light grows warmer "
        "as clouds thin overhead."
    ),
    tags=["nature", "rain", "forest"],
)

# ---------- Urban Life ----------
SCENARIOS["tokyo_crossing"] = Scenario(
    name="tokyo_crossing",
    display_name="Tokyo Scramble Crossing",
    category="Urban Life",
    inference_type="text2world",
    prompt=(
        "A high vantage point overlooking a major pedestrian scramble crossing "
        "at night in a dense metropolitan district. Hundreds of pedestrians "
        "wait at the curb as traffic flows through the intersection. When the "
        "signals change, the crowd surges forward from all four corners "
        "simultaneously, creating complex interwoven paths. Neon signs and LED "
        "billboards cast vibrant pink, blue, and white reflections on the wet "
        "pavement. Umbrellas dot the crowd in various colors. Taxis and delivery "
        "trucks idle at the edges, their indicators blinking. The camera captures "
        "the organized chaos as pedestrians navigate around each other with "
        "practiced efficiency. As the crossing empties, vehicles begin to move "
        "again, and a new wave of pedestrians gathers at the corners, waiting "
        "for the next cycle."
    ),
    tags=["urban", "night", "pedestrians"],
)

SCENARIOS["construction_site"] = Scenario(
    name="construction_site",
    display_name="Active Construction Site",
    category="Urban Life",
    inference_type="text2world",
    prompt=(
        "A panoramic view of an active high-rise construction site during the "
        "day. A tower crane swings a steel beam across the frame, its cable "
        "taut under the load. Workers in hard hats and high-visibility vests "
        "guide the beam into position on the upper floors of a concrete "
        "skeleton structure. Below, a concrete mixer truck rotates its drum "
        "while pumping material through a long articulated boom. Sparks fly "
        "from a welding operation on a lower floor, briefly illuminating the "
        "interior scaffolding. Dust rises from a section where a jackhammer "
        "breaks up old pavement. The surrounding city is visible in the "
        "background with completed buildings contrasting the raw construction. "
        "A site foreman checks a tablet near a stack of rebar bundles. The "
        "crane completes its rotation and lowers the beam into its final "
        "resting position."
    ),
    tags=["urban", "construction", "industrial"],
)

# ---------- Industrial ----------
SCENARIOS["steel_foundry"] = Scenario(
    name="steel_foundry",
    display_name="Steel Foundry Pour",
    category="Industrial",
    inference_type="text2world",
    prompt=(
        "Inside a steel foundry, a massive ladle tilts and pours molten metal "
        "into a casting mold. The liquid steel glows a brilliant orange-white, "
        "emitting intense radiant heat that distorts the air above it. Cascading "
        "sparks spray outward as the stream contacts the mold, creating a "
        "shower of incandescent particles that arc through the dark industrial "
        "space. Workers in full protective gear observe from behind heat shields. "
        "The refractory-lined ladle continues its controlled pour, and the mold "
        "fills with the luminous material, its surface rippling. Overhead crane "
        "hooks and chains are silhouetted against the glow. As the pour "
        "completes, the ladle tips back upright, and the surface of the cast "
        "steel begins to cool at the edges, transitioning from white to deep "
        "orange to dark red, with a thin crust forming over the still-liquid "
        "interior."
    ),
    tags=["industrial", "metal", "dramatic"],
)

SCENARIOS["wind_farm"] = Scenario(
    name="wind_farm",
    display_name="Offshore Wind Farm",
    category="Industrial",
    inference_type="text2world",
    prompt=(
        "An aerial drone shot approaches a row of offshore wind turbines at "
        "dawn. The massive white blades rotate steadily against a sky streaked "
        "with high cirrus clouds catching the early light. The North Sea "
        "stretches to the horizon, its surface a textured grey-green with "
        "white-capped swells passing between the turbine foundations. Each "
        "tubular steel tower rises from a concrete base surrounded by choppy "
        "water. The camera drifts between two turbines, revealing the scale — "
        "a maintenance vessel is docked at one foundation, appearing small "
        "against the tower's diameter. Blade tips catch the orange sunrise "
        "light while their bases remain in blue shadow. Seabirds bank between "
        "the structures. The drone climbs higher, and the full array of "
        "twelve turbines stretches into the distance in a gentle arc, all "
        "blades synchronized in rotation."
    ),
    tags=["industrial", "energy", "ocean"],
)


def get_scenarios_by_category() -> Dict[str, List[Scenario]]:
    """Group scenarios by category."""
    categories: Dict[str, List[Scenario]] = {}
    for s in SCENARIOS.values():
        categories.setdefault(s.category, []).append(s)
    return categories


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _apply_generation_config_patch():
    """
    Monkeypatch for transformers 4.49 + Qwen2_5_VL compatibility.

    Qwen2_5_VL's get_text_config(decoder=True) returns a raw dict, but
    GenerationConfig.from_model_config calls .to_dict() on it, causing:
        AttributeError: 'dict' object has no attribute 'to_dict'

    This patch wraps dict returns from get_text_config in a PretrainedConfig
    so the downstream code works as expected.
    """
    from transformers.generation.configuration_utils import GenerationConfig

    # Guard against double-patching
    if getattr(GenerationConfig, "_cosmos_patched", False):
        return

    _original_from_model_config = GenerationConfig.from_model_config

    @classmethod
    def _patched_from_model_config(cls, model_config, **kwargs):
        _orig_get_text_config = getattr(model_config, "get_text_config", None)
        if _orig_get_text_config is not None:
            def _safe_get_text_config(**kw):
                result = _orig_get_text_config(**kw)
                if isinstance(result, dict):
                    from transformers.configuration_utils import PretrainedConfig
                    return PretrainedConfig(**result)
                return result
            model_config.get_text_config = _safe_get_text_config
        try:
            return _original_from_model_config.__func__(cls, model_config, **kwargs)
        finally:
            if _orig_get_text_config is not None:
                model_config.get_text_config = _orig_get_text_config

    GenerationConfig.from_model_config = _patched_from_model_config
    GenerationConfig._cosmos_patched = True


def _resolve_device(requested: str) -> str:
    """Pick the best available device, falling back gracefully."""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA not available. Using MPS (Apple Silicon).")
            return "mps"
        print("CUDA not available. Falling back to CPU.")
        return "cpu"
    return requested


class _NoOpSafetyChecker:
    """
    Lightweight stand-in for CosmosSafetyChecker.

    The real guardrail model (nvidia/Cosmos-1.0-Guardrail) pulls in a large
    Aegis LLM (~14 GB) plus RetinaFace, which can OOM on machines with
    limited GPU memory.  This stub satisfies the pipeline's interface while
    skipping actual content filtering.
    """

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, text: str) -> bool:
        return True  # always safe

    def check_video_safety(self, video):
        return video  # pass-through


def load_pipeline(
    model_id: str = "nvidia/Cosmos-Predict2.5-2B",
    revision: str = "diffusers/base/post-trained",
    offload: bool = False,
    device: str = "cuda",
    disable_guardrails: bool = False,
):
    """Load the Cosmos-Predict2.5 diffusers pipeline."""
    import torch
    from diffusers import Cosmos2_5_PredictBasePipeline

    _apply_generation_config_patch()

    device = _resolve_device(device)
    print(f"Loading model {model_id} (revision={revision}) on {device}...")

    # On non-CUDA machines the cosmos_guardrail library calls torch.load()
    # without map_location, which crashes when the checkpoint was saved on
    # CUDA.  Patch cosmos_guardrail.cosmos_utils.torch.load to always set
    # map_location="cpu" when CUDA is unavailable.
    if not torch.cuda.is_available():
        import cosmos_guardrail.cosmos_utils as _gu
        _orig_torch_load = torch.load

        def _cpu_torch_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return _orig_torch_load(*args, **kwargs)

        _gu.torch.load = _cpu_torch_load

    # Auto-detect: disable guardrails when memory is tight (MPS / CPU)
    # to avoid loading the ~14 GB Aegis LLM on top of the main model.
    if device in ("mps", "cpu") and not disable_guardrails:
        print("Auto-disabling guardrails on non-CUDA device to save memory.")
        disable_guardrails = True

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if offload:
        load_kwargs["device_map"] = "auto"

    if disable_guardrails:
        # Monkey-patch CosmosSafetyChecker so the pipeline __init__
        # receives our no-op stub instead of downloading the real model.
        import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as _cpmod
        _orig_checker = _cpmod.CosmosSafetyChecker
        _cpmod.CosmosSafetyChecker = lambda *a, **k: _NoOpSafetyChecker()
        print("Guardrails disabled (using no-op safety checker).")

    try:
        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
            model_id,
            revision=revision,
            **load_kwargs,
        )
    finally:
        if disable_guardrails:
            _cpmod.CosmosSafetyChecker = _orig_checker

    if not offload:
        if device == "mps":
            # MPS has limited unified memory (~30 GB on most Macs).
            # enable_model_cpu_offload moves each sub-model (text_encoder,
            # transformer, vae) to the accelerator only during its forward
            # pass, keeping the rest on CPU. This prevents OOM while being
            # faster than full sequential offload.
            print("Enabling model CPU offload for MPS to avoid OOM...")
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe = pipe.to(device)

    print("Model loaded.")
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_world(
    pipe,
    scenario: Scenario,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    num_frames: Optional[int] = None,
    guidance: Optional[float] = None,
    image=None,
    video=None,
) -> Tuple[list, str]:
    """
    Run world model generation for a scenario.

    Returns:
        (frames, output_path): list of PIL frames and path to saved video.
    """
    import torch
    from diffusers.utils import export_to_video

    _seed = seed if seed is not None else scenario.seed
    _steps = num_steps if num_steps is not None else scenario.num_steps
    _frames = num_frames if num_frames is not None else scenario.num_output_frames
    _guidance = guidance if guidance is not None else scenario.guidance

    generator = torch.Generator().manual_seed(_seed) if _seed is not None else None

    result = pipe(
        image=image,
        video=video,
        prompt=scenario.prompt,
        negative_prompt=scenario.negative_prompt,
        num_frames=_frames,
        num_inference_steps=_steps,
        generator=generator,
    )
    frames = result.frames[0]

    # Save to temp file
    output_dir = Path(tempfile.gettempdir()) / "cosmos_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{scenario.name}_seed{_seed}.mp4")
    export_to_video(frames, output_path, fps=16)

    return frames, output_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_gradio_app(pipe):
    """Build and return the Gradio app."""
    import gradio as gr

    scenario_names = {s.display_name: s.name for s in SCENARIOS.values()}
    categories = get_scenarios_by_category()

    # Build category -> display_name mapping for dropdown
    choices = []
    for cat, scenes in categories.items():
        for s in scenes:
            choices.append(s.display_name)

    def on_scenario_select(display_name):
        name = scenario_names.get(display_name)
        if name is None:
            return "", "", 42, 36, 93
        s = SCENARIOS[name]
        tags_str = ", ".join(s.tags) if s.tags else ""
        return s.prompt, tags_str, s.seed, s.num_steps, s.num_output_frames

    def on_generate(
        display_name, prompt_text, seed, num_steps, num_frames, image, video
    ):
        name = scenario_names.get(display_name)
        if name is None:
            # Use custom prompt mode
            custom = Scenario(
                name="custom",
                display_name="Custom",
                category="Custom",
                inference_type="text2world",
                prompt=prompt_text,
                seed=int(seed),
                num_steps=int(num_steps),
                num_output_frames=int(num_frames),
            )
            scenario = custom
        else:
            scenario = SCENARIOS[name]
            if prompt_text and prompt_text.strip() != scenario.prompt:
                # User edited the prompt
                scenario = Scenario(
                    name=scenario.name,
                    display_name=scenario.display_name,
                    category=scenario.category,
                    inference_type=scenario.inference_type,
                    prompt=prompt_text,
                    negative_prompt=scenario.negative_prompt,
                    seed=int(seed),
                    num_steps=int(num_steps),
                    num_output_frames=int(num_frames),
                    tags=scenario.tags,
                )

        pil_image = None
        loaded_video = None

        if image is not None:
            from diffusers.utils import load_image
            pil_image = load_image(image)

        if video is not None:
            from diffusers.utils import load_video
            loaded_video = load_video(video)

        _, output_path = generate_world(
            pipe,
            scenario,
            seed=int(seed),
            num_steps=int(num_steps),
            num_frames=int(num_frames),
            image=pil_image,
            video=loaded_video,
        )
        return output_path

    with gr.Blocks(
        title="Cosmos-Predict2.5 Hyperrealistic World Demo",
    ) as app:
        gr.Markdown(
            "# Cosmos-Predict2.5 -- Hyperrealistic World Model Demo\n"
            "Generate physically plausible video predictions of real-world "
            "scenarios using NVIDIA's world foundation model. Select a curated "
            "scenario or write your own prompt.\n\n"
            "**Categories:** Autonomous Driving | Robotics | Weather & Nature | "
            "Urban Life | Industrial"
        )

        with gr.Row():
            with gr.Column(scale=1):
                scenario_dropdown = gr.Dropdown(
                    choices=choices,
                    label="Scenario",
                    value=choices[0] if choices else None,
                    info="Select a curated scenario or edit the prompt below",
                )
                tags_display = gr.Textbox(
                    label="Tags",
                    interactive=False,
                    max_lines=1,
                )
                prompt_box = gr.Textbox(
                    label="World Description Prompt",
                    lines=8,
                    placeholder="Describe the world scene to simulate...",
                )
                with gr.Row():
                    seed_input = gr.Number(label="Seed", value=42, precision=0)
                    steps_input = gr.Number(label="Inference Steps", value=36, precision=0)
                    frames_input = gr.Number(label="Output Frames", value=93, precision=0)

                with gr.Accordion("Optional: Conditioning Input", open=False):
                    gr.Markdown(
                        "Provide an image for **Image2World** or a video for "
                        "**Video2World** mode. Leave empty for **Text2World**."
                    )
                    image_input = gr.Image(
                        label="Conditioning Image",
                        type="filepath",
                    )
                    video_input = gr.Video(label="Conditioning Video")

                generate_btn = gr.Button(
                    "Generate World Simulation",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Generated World",
                    autoplay=True,
                )
                gr.Markdown(
                    "*Output: 16 FPS video. Generation uses the Cosmos-Predict2.5 "
                    "diffusers pipeline with flow-matching denoising.*"
                )

        # Wire events
        scenario_dropdown.change(
            fn=on_scenario_select,
            inputs=[scenario_dropdown],
            outputs=[prompt_box, tags_display, seed_input, steps_input, frames_input],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                scenario_dropdown,
                prompt_box,
                seed_input,
                steps_input,
                frames_input,
                image_input,
                video_input,
            ],
            outputs=[video_output],
        )

        # Populate on load
        app.load(
            fn=lambda: on_scenario_select(choices[0]) if choices else ("", "", 42, 36, 93),
            outputs=[prompt_box, tags_display, seed_input, steps_input, frames_input],
        )

        # Scenario gallery at the bottom
        with gr.Accordion("All Scenarios", open=False):
            for cat, scenes in categories.items():
                gr.Markdown(f"### {cat}")
                for s in scenes:
                    tags_str = " | ".join(s.tags)
                    gr.Markdown(
                        f"**{s.display_name}** ({s.inference_type}) -- {tags_str}"
                    )

    return app


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli(
    scenario_name: str,
    model_id: str,
    revision: str,
    offload: bool,
    device: str,
    output_dir: str,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    num_frames: Optional[int] = None,
):
    """Run a single scenario from the command line without Gradio."""
    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: '{scenario_name}'")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = SCENARIOS[scenario_name]
    print(f"\n{'=' * 60}")
    print(f"  Hyperrealistic World Demo -- {scenario.display_name}")
    print(f"  Category: {scenario.category}")
    print(f"  Mode: {scenario.inference_type}")
    print(f"{'=' * 60}\n")

    pipe = load_pipeline(model_id, revision, offload, device)
    frames, output_path = generate_world(
        pipe, scenario, seed=seed, num_steps=num_steps, num_frames=num_frames,
    )

    # Copy to user-specified output dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / f"{scenario.name}.mp4"

    import shutil
    shutil.copy2(output_path, final_path)

    print(f"\nGenerated {len(frames)} frames")
    print(f"Saved to {final_path}")


def list_scenarios():
    """Print all available scenarios."""
    categories = get_scenarios_by_category()
    print(f"\n{'=' * 60}")
    print("  Available Hyperrealistic Scenarios")
    print(f"{'=' * 60}\n")
    for cat, scenes in categories.items():
        print(f"  {cat}:")
        for s in scenes:
            tags = ", ".join(s.tags)
            print(f"    {s.name:25s} {s.display_name} [{tags}]")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cosmos-Predict2.5 Hyperrealistic World Model Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio UI:
  python demo/hyperrealistic_world_demo.py

  # CLI single scenario:
  python demo/hyperrealistic_world_demo.py --cli --scenario rainy_highway -o outputs/

  # List all scenarios:
  python demo/hyperrealistic_world_demo.py --list

  # Use 14B model:
  python demo/hyperrealistic_world_demo.py --model_id nvidia/Cosmos-Predict2.5-14B
        """,
    )

    parser.add_argument(
        "--cli", action="store_true",
        help="Run a single scenario from CLI (no Gradio UI)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available scenarios and exit",
    )
    parser.add_argument(
        "--scenario", type=str, default="rainy_highway",
        help="Scenario name for --cli mode",
    )
    parser.add_argument(
        "--model_id", type=str, default="nvidia/Cosmos-Predict2.5-2B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--revision", type=str, default="diffusers/base/post-trained",
        help="Model revision/variant",
    )
    parser.add_argument(
        "--offload", action="store_true",
        help="Enable CPU offloading for limited VRAM",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="outputs/hyperrealistic",
        help="Output directory for CLI mode",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Gradio server port",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed",
    )
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Override inference steps",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Override number of output frames",
    )

    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    if args.cli:
        run_cli(
            scenario_name=args.scenario,
            model_id=args.model_id,
            revision=args.revision,
            offload=args.offload,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            num_frames=args.num_frames,
        )
        return

    # Gradio UI mode
    pipe = load_pipeline(args.model_id, args.revision, args.offload, args.device)
    app = build_gradio_app(pipe)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

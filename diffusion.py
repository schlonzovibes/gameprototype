"""Diffusion-Backend (diffusers). Laedt Repo-Ordner oder Single-File-Checkpoint."""

from __future__ import annotations

import os

from discovery import Model

STEPS = int(os.environ.get("AIGAME_STEPS", "22"))
GUIDANCE = float(os.environ.get("AIGAME_GUIDANCE", "6.5"))
WIDTH = int(os.environ.get("AIGAME_WIDTH", "512"))    # SD 1.5 nativ 512x512
HEIGHT = int(os.environ.get("AIGAME_HEIGHT", "512"))

# Konstante Bildsprache - der "Capture Layer" auf Prompt-Ebene.
STYLE = os.environ.get(
    "AIGAME_STYLE",
    "cinematic still, anamorphic lens, soft volumetric light, muted desaturated palette, fine grain",
)
NEGATIVE = os.environ.get(
    "AIGAME_NEGATIVE",
    "text, watermark, signature, letters, ui, hud, frame, border, cartoon, lowres, deformed",
)


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class Diffusion:
    def __init__(self, model: Model):
        self.model = model
        self.pipe = None

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline

        device, dtype = _device()
        # Safety-Checker deaktiviert: verhindert Fehlalarme (schwarze Bilder)
        no_nsfw = {"safety_checker": None, "requires_safety_checker": False}
        if self.model.backend == "diffusers-file":
            pipe = StableDiffusionPipeline.from_single_file(
                self.model.ref, dtype=dtype, **no_nsfw
            )
        else:
            try:
                # Erst fp16-Varianten versuchen (*.fp16.safetensors) ...
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model.ref, dtype=dtype, variant="fp16", **no_nsfw
                )
            except OSError:
                # ... sonst Standardgewichte (Repos ohne fp16-Dateien).
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model.ref, dtype=dtype, **no_nsfw
                )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def render(self, visual_prompt: str):
        prompt = f"{visual_prompt.strip()}, {STYLE}" if visual_prompt.strip() else STYLE
        result = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            width=WIDTH,
            height=HEIGHT,
        )
        return result.images[0]

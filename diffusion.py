"""Diffusion-Backend (diffusers). Laedt Repo-Ordner oder Single-File-Checkpoint.

Unterstützt klassische SD-Pipelines (AutoPipeline/StableDiffusion) sowie
FLUX.2-Repos (Flux2KleinPipeline, Erkennung ueber model_index.json).
Die Generierungsparameter kommen aus Backend-Profilen; AIGAME_STEPS,
AIGAME_GUIDANCE, AIGAME_WIDTH und AIGAME_HEIGHT uebersteuern sie, wenn gesetzt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from discovery import Model

# Profile pro Backend - Default, wenn die Env-Variablen nichts setzen.
SD_PROFILE = {"steps": 22, "guidance": 6.5, "width": 512, "height": 512}
FLUX_PROFILE = {"steps": 28, "guidance": 4.0, "width": 1024, "height": 1024}
FLUX_DISTILLED_PROFILE = {"steps": 8, "guidance": 2.5}  # klein-4B: destilliert

OFFLOAD = os.environ.get("AIGAME_OFFLOAD", "").lower() in ("1", "true", "on", "yes")

# Konstante Bildsprache - der "Capture Layer" auf Prompt-Ebene.
STYLE = os.environ.get(
    "AIGAME_STYLE",
    "cinematic still, anamorphic lens, soft volumetric light, muted desaturated palette, fine grain",
)
NEGATIVE = os.environ.get(
    "AIGAME_NEGATIVE",
    "text, watermark, signature, letters, ui, hud, frame, border, cartoon, lowres, deformed",
)


def _env(key: str, default):
    """Env-Wert oder Default; der Zahlentyp folgt dem Default."""
    raw = os.environ.get(key)
    return type(default)(raw) if raw else default


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _flux_class(repo: str) -> str:
    """_class_name aus model_index.json - leer bei fehlender Datei."""
    try:
        idx = Path(repo) / "model_index.json"
        return json.loads(idx.read_text(encoding="utf-8")).get("_class_name", "")
    except (OSError, ValueError):
        return ""


class Diffusion:
    def __init__(self, model: Model):
        self.model = model
        self.pipe = None
        self.flux = False
        self.profile = dict(SD_PROFILE)

    def load(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline

        device, dtype = _device()
        # Safety-Checker deaktiviert: verhindert Fehlalarme (schwarze Bilder)
        no_nsfw = {"safety_checker": None, "requires_safety_checker": False}
        if _flux_class(self.model.ref).startswith("Flux2"):
            from diffusers import Flux2KleinPipeline

            pipe = Flux2KleinPipeline.from_pretrained(self.model.ref,
                                                      dtype=torch.bfloat16)
            self.flux = True
            self.profile = dict(FLUX_PROFILE)
            # FLUX.2-klein-base braucht die volle Schrittanzahl, das
            # destillierte klein-4B kommt mit wenigen aus. Der Check läuft
            # über den ganzen Pfad: im HF-Cache ist der Ordnername ein Commit-Sha.
            if "base" not in self.model.ref.lower():
                self.profile.update(FLUX_DISTILLED_PROFILE)
        elif self.model.backend == "diffusers-file":
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
        if OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def render(self, visual_prompt: str):
        prompt = f"{visual_prompt.strip()}, {STYLE}" if visual_prompt.strip() else STYLE
        call = {
            "prompt": prompt,
            "num_inference_steps": _env("AIGAME_STEPS", self.profile["steps"]),
            "guidance_scale": _env("AIGAME_GUIDANCE", self.profile["guidance"]),
            "width": _env("AIGAME_WIDTH", self.profile["width"]),
            "height": _env("AIGAME_HEIGHT", self.profile["height"]),
        }
        if not self.flux:
            # Flux-Pipelines haben keinen CFG-Negative-Pfad.
            call["negative_prompt"] = NEGATIVE
        result = self.pipe(**call)
        return result.images[0]

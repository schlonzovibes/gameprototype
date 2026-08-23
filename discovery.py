"""Findet lokal verfuegbare Modelle.

LLM:        Ollama-Server, GGUF-Dateien (llama.cpp / LM Studio)
Diffusion:  Diffusers-Repos im HF-Cache, Single-File-Checkpoints (.safetensors/.ckpt)

Zusaetzliche Suchpfade per Umgebungsvariable:
    AIGAME_MODEL_DIRS=/pfad/a:/pfad/b
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

DIMSEP = "\u00b7 "   # Trenner fuer Groessenangaben in Labels

_DEFAULT_ROOTS = [
    Path.cwd() / "models",
    Path.home() / "models",
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "lm-studio" / "models",
    Path.home() / ".lmstudio" / "models",
    Path.home() / "stable-diffusion-webui" / "models",
    Path.home() / "ComfyUI" / "models",
    Path("/opt/models"),
]


@dataclass(frozen=True)
class Model:
    kind: str      # "llm" | "diffusion"
    backend: str   # "ollama" | "llama-cpp" | "diffusers-repo" | "diffusers-file"
    ref: str       # Modellname oder Pfad
    label: str     # Anzeigetext


def _roots() -> list[Path]:
    roots = list(_DEFAULT_ROOTS)
    extra = os.environ.get("AIGAME_MODEL_DIRS", "")
    roots += [Path(p).expanduser() for p in extra.split(os.pathsep) if p]
    seen, out = set(), []
    for r in roots:
        try:
            r = r.expanduser().resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _human(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def _scan(root: Path, patterns: tuple[str, ...], limit: int = 400) -> list[Path]:
    hits: list[Path] = []
    for pat in patterns:
        try:
            for p in root.rglob(pat):
                if p.is_file():
                    hits.append(p)
                    if len(hits) >= limit:
                        return hits
        except (OSError, PermissionError):
            continue
    return hits


# --------------------------------------------------------------------------- LLM

def _ollama_models() -> list[Model]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return []
    out = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if not name:
            continue
        size = _human(m.get("size", 0)) if m.get("size") else ""
        out.append(Model("llm", "ollama", name, f"[ollama]     {name}  {DIMSEP}{size}"))
    return out


def _gguf_models() -> list[Model]:
    out, seen = [], set()
    for root in _roots():
        for p in _scan(root, ("*.gguf",)):
            key = p.name.lower()
            if key in seen:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            seen.add(key)
            out.append(Model(
                "llm", "llama-cpp", str(p),
                f"[llama.cpp]  {p.stem}  {DIMSEP}{_human(size)}",
            ))
    return out


# ---------------------------------------------------------------------- Diffusion

_DIFFUSION_HINTS = ("stable-diffusion", "sdxl", "sd-", "sd3", "flux", "kandinsky",
                    "playground", "pixart", "dreamshaper", "juggernaut", "realvis")


def _repo_label(repo: Path) -> str:
    """Sprechender Name - loest die HF-Cache-Struktur auf
    (models--org--name/snapshots/<sha> -> org/name)."""
    for parent in repo.parents:
        if parent.name.startswith("models--"):
            return parent.name.replace("models--", "").replace("--", "/")
    return repo.name


def _diffusers_repos() -> list[Model]:
    out, seen = [], set()
    for root in _roots():
        try:
            candidates = list(root.rglob("model_index.json"))
        except (OSError, PermissionError):
            continue
        for idx in candidates:
            repo = idx.parent
            name = _repo_label(repo)
            if name in seen:
                continue
            seen.add(name)
            out.append(Model("diffusion", "diffusers-repo", str(repo),
                             f"[diffusers]  {name}"))
    return out


_COMPONENT_DIRS = ("/unet/", "/vae/", "/transformer/", "/scheduler/",
                   "/text_encoder/", "/tokenizer/", "/safety_checker/",
                   "/feature_extractor/")


def _diffusion_files() -> list[Model]:
    out, seen = [], set()
    for root in _roots():
        try:
            repos = [d.parent.resolve() for d in root.rglob("model_index.json")]
        except (OSError, PermissionError):
            repos = []
        for p in _scan(root, ("*.safetensors", "*.ckpt")):
            low = str(p).lower()
            # Komponentengewichte eines Diffusers-Repos gehoeren nicht in die Liste
            if any(c in low for c in _COMPONENT_DIRS):
                continue
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if any(resolved.is_relative_to(d) for d in repos):
                continue
            # Heuristik: entweder in einem Checkpoint-Ordner oder mit sprechendem Namen
            in_ckpt_dir = any(d in low for d in ("checkpoint", "stable-diffusion", "unet-single"))
            named = any(h in low for h in _DIFFUSION_HINTS)
            try:
                size = p.stat().st_size
            except OSError:
                continue
            big = size > 1.2e9
            if not (in_ckpt_dir or (named and big)):
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            out.append(Model("diffusion", "diffusers-file", str(p),
                             f"[checkpoint] {p.stem}  {DIMSEP}{_human(size)}"))
    return out


def find_llms() -> list[Model]:
    return _ollama_models() + _gguf_models()


def find_diffusion() -> list[Model]:
    return _diffusers_repos() + _diffusion_files()

"""Bildgenerierung (diffusers).

=== Was hier passiert ===

Ein Diffusionsmodell erzeugt ein Bild aus einer Textbeschreibung. Es
beginnt mit reinem Rauschen und entfernt dieses schrittweise, bis ein Bild
uebrigbleibt, das zur Beschreibung passt. Jeder Schritt ist ein
"inference step" - mehr Schritte heisst meist besser, aber langsamer.

Alle Bildmodelle kommen als Diffusers-Repo aus dem HF-Cache. Welche
Pipeline zustaendig ist, steht in dessen model_index.json. FLUX.2 bekommt
eine eigene Klasse und ein eigenes Profil, alles andere laeuft ueber
AutoPipeline, die selbst herausfindet, was zu tun ist.

=== Die Parameter ===

    steps     Rechenschritte - mehr = detaillierter, aber langsamer
    guidance  wie streng sich das Modell an den Text haelt. Zu hoch wirkt
              ueberzeichnet, zu niedrig ignoriert es die Beschreibung.
    width/height   Bildgroesse in Pixeln

Die Werte kommen aus Profilen pro Modelltyp; die Umgebungsvariablen
AIGAME_STEPS, AIGAME_GUIDANCE, AIGAME_WIDTH und AIGAME_HEIGHT uebersteuern
sie, wenn gesetzt.

Python-Konzepte hier: Imports innerhalb von Funktionen, dict-Entpacken mit
**, und dynamische Typumwandlung.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from models import Model

# dict(...) kopiert spaeter diese Vorlagen, damit ein veraendertes Profil
# nicht die Vorlage selbst ueberschreibt.
SD_PROFILE = {"steps": 22, "guidance": 6.5, "width": 512, "height": 512}
FLUX_PROFILE = {"steps": 28, "guidance": 4.0, "width": 1024, "height": 1024}
FLUX_DISTILLED = {"steps": 8, "guidance": 2.5}   # klein-4B ist destilliert
# "Destilliert" heisst: ein grosses Modell wurde einem kleinen beigebracht.
# Solche Modelle brauchen deutlich weniger Schritte fuer dasselbe Ergebnis.

# CPU-Offload: Modellteile bleiben im RAM und wandern nur zum Rechnen auf
# die GPU. Spart VRAM, kostet Zeit. Nur einschalten, wenn es sonst nicht passt.
OFFLOAD = os.environ.get("AIGAME_OFFLOAD", "").lower() in ("1", "true", "on", "yes")

# Konstante Bildsprache - der "Capture Layer" auf Prompt-Ebene. Dieser Text
# haengt an JEDER Bildbeschreibung. Dadurch sehen alle Szenen aus wie aus
# demselben Film, statt bei jedem Bild den Stil zu wechseln.
STYLE = os.environ.get(
    "AIGAME_STYLE",
    "cinematic still, anamorphic lens, soft volumetric light, "
    "muted desaturated palette, fine grain",
)

# Negativ-Prompt: was NICHT im Bild sein soll. "text" und "letters" stehen
# drin, weil Diffusionsmodelle gern unleserliche Pseudo-Schrift einbauen.
NEGATIVE = os.environ.get(
    "AIGAME_NEGATIVE",
    "text, watermark, signature, letters, ui, hud, frame, border, "
    "cartoon, lowres, deformed",
)


def _env(key: str, default):
    """Env-Wert oder Default; der Zahlentyp folgt dem Default.

    Der Trick: type(default) ist die Klasse des Standardwerts - int bei 22,
    float bei 6.5. Diese Klasse wird dann als Umwandlungsfunktion benutzt.
    Bei steps kommt also int("30") heraus, bei guidance float("3.5").
    """
    raw = os.environ.get(key)
    return type(default)(raw) if raw else default


def _device():
    """Worauf rechnen wir? (Geraet, Zahlenformat)

    cuda = NVIDIA-GPU, mps = Apple Silicon, cpu = Notloesung (sehr langsam).
    float16 braucht halb so viel Speicher wie float32 und ist auf GPUs
    schneller; die CPU kann damit nichts anfangen, daher dort float32.
    """
    # Import innerhalb der Funktion: torch zu laden dauert mehrere Sekunden.
    # Beim Modul-Import ganz oben wuerde jeder Programmstart darauf warten,
    # auch wenn nie ein Bild erzeugt wird.
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    # getattr(objekt, name, default) fragt sicher nach einem Attribut, das
    # es vielleicht gar nicht gibt - aeltere torch-Versionen kennen mps nicht.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _silence_libraries() -> None:
    """diffusers und transformers auf "nur echte Fehler" stellen.

    Nicht aus Bequemlichkeit: die beiden schreiben ihre Meldungen direkt auf
    den Bildschirm, mitten in unser sorgfaeltig positioniertes Layout. Eine
    Zeile wie "Guidance scale 2.5 is ignored for step-wise distilled models."
    landet dann quer im Bild und zerstoert Kopfzeile oder Szene.

    Echte Fehler kommen weiterhin durch - nur Hinweise und Warnungen nicht.
    Wenn beim Bild etwas schiefgeht, faengt paint() in main.py das ohnehin
    als Exception ab.
    """
    # Jede Bibliothek einzeln in try/except: schlaegt eine fehl (andere
    # Version, umbenannte API), sollen die anderen trotzdem still werden.
    try:
        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _pipeline_class(repo: str) -> str:
    """Welche Pipeline-Klasse nennt model_index.json? Leer, wenn unlesbar."""
    try:
        index = Path(repo) / "model_index.json"
        return json.loads(index.read_text(encoding="utf-8")).get("_class_name", "")
    except (OSError, ValueError):
        return ""


class Diffusion:
    def __init__(self, model: Model):
        self.model = model
        self.pipe = None                   # die Pipeline, erst nach load()
        self.flux = False                  # ist es ein FLUX-Modell?
        self.profile = dict(SD_PROFILE)    # Kopie, nicht die Vorlage selbst

    def load(self) -> None:
        """Das Modell in den Speicher laden. Dauert je nach Groesse Minuten."""
        import torch
        from diffusers import AutoPipelineForText2Image

        _silence_libraries()
        device, dtype = _device()

        # startswith("Flux2") trifft Flux2KleinPipeline und Verwandte.
        if _pipeline_class(self.model.ref).startswith("Flux2"):
            from diffusers import Flux2KleinPipeline

            # bfloat16: anderes 16-Bit-Format als float16. Groesserer
            # Wertebereich bei weniger Nachkommagenauigkeit - FLUX braucht das.
            pipe = Flux2KleinPipeline.from_pretrained(self.model.ref,
                                                      dtype=torch.bfloat16)
            self.flux = True
            self.profile = dict(FLUX_PROFILE)

            # klein-base braucht die volle Schrittzahl, das destillierte
            # klein-4B kommt mit wenigen aus. Der Check laeuft ueber den
            # ganzen Pfad: im HF-Cache ist der Ordnername ein Commit-Sha,
            # der Modellname steckt weiter oben im Pfad.
            if "base" not in self.model.ref.lower():
                # .update() ueberschreibt nur die genannten Schluessel;
                # width und height aus FLUX_PROFILE bleiben stehen.
                self.profile.update(FLUX_DISTILLED)
        else:
            # Safety-Checker aus: der schlaegt bei harmlosen Szenen oft
            # faelschlich an und liefert dann ein komplett schwarzes Bild.
            no_nsfw = {"safety_checker": None, "requires_safety_checker": False}
            try:
                # Erst fp16-Varianten (*.fp16.safetensors) - halb so gross.
                # Das ** packt das dict als einzelne Argumente aus:
                # aus {"safety_checker": None} wird safety_checker=None.
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model.ref, dtype=dtype, variant="fp16", **no_nsfw)
            except OSError:
                # ... sonst die Standardgewichte. Nicht jedes Repo hat fp16.
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model.ref, dtype=dtype, **no_nsfw)

        if OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)   # alles auf die GPU schieben

        # Fortschrittsbalken von diffusers abschalten - er wuerde mitten in
        # unser Terminal-Layout schreiben und den Rahmen zerstoeren.
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def render(self, visual_prompt: str):
        """Ein Bild erzeugen. Gibt ein PIL-Image zurueck."""
        # Beschreibung plus Stil. Ist die Beschreibung leer, nur der Stil -
        # dann kommt wenigstens ein stimmungsvolles Bild statt einer Exception.
        prompt = f"{visual_prompt.strip()}, {STYLE}" if visual_prompt.strip() else STYLE

        # Die Argumente erst als dict sammeln, weil eines davon je nach
        # Modelltyp dazukommt oder wegfaellt.
        call = {
            "prompt": prompt,
            "num_inference_steps": _env("AIGAME_STEPS", self.profile["steps"]),
            "guidance_scale": _env("AIGAME_GUIDANCE", self.profile["guidance"]),
            "width": _env("AIGAME_WIDTH", self.profile["width"]),
            "height": _env("AIGAME_HEIGHT", self.profile["height"]),
        }
        if not self.flux:
            # FLUX-Pipelines haben keinen CFG-Negative-Pfad - sie wuerden
            # dieses Argument mit einem Fehler zurueckweisen.
            call["negative_prompt"] = NEGATIVE

        # ** packt das dict wieder in einzelne Argumente aus.
        # .images ist eine Liste (man kann mehrere Bilder auf einmal
        # erzeugen); [0] nimmt das einzige, das wir angefordert haben.
        return self.pipe(**call).images[0]


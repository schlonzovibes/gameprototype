#!/usr/bin/env python3
"""AI-Game-Prototyp - Terminal-Runner.

Ablauf:
    1. LLM auswaehlen (Pfeiltasten) -> laden
    2. Diffusion-Modell auswaehlen -> laden
    3. game_prompt.txt einlesen, erste Szene erzeugen
    4. Schleife: Bild (Viewport) / Erzaehltext / Trennlinie / Prompt

Start:  python3 main.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import diffusion
import discovery
import llm
import termimage
import ui

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "game_prompt.txt"

VIEWPORT_COLS_CAP = 80  # max. Bildbreite in Zellen
TEXT_ROWS = 10          # Zeilen-Reserve fuer Erzaehltext + Prompt
TEXT_WIDTH = 72
MARGIN = 2
HISTORY_TURNS = 12      # wie viele Zuege im Kontext bleiben


# ------------------------------------------------------------------ Setup

def pick_models() -> tuple[llm.LLM, diffusion.Diffusion]:
    ui.clear()

    with ui.Status("suche verf\u00fcgbare Sprachmodelle"):
        llms = discovery.find_llms()
    if not llms:
        sys.exit("Keine LLMs gefunden. Ollama starten oder AIGAME_MODEL_DIRS setzen.")
    choice = llms[ui.select("Sprachmodell", [m.label for m in llms])]
    engine = llm.build(choice)
    with ui.Status(f"lade {choice.ref}"):
        engine.load()

    ui.clear()
    with ui.Status("suche verf\u00fcgbare Diffusionsmodelle"):
        diffs = discovery.find_diffusion()
    if not diffs:
        sys.exit("Keine Diffusionsmodelle gefunden. AIGAME_MODEL_DIRS setzen.")
    picked = diffs[ui.select("Bildmodell", [m.label for m in diffs])]
    painter = diffusion.Diffusion(picked)
    with ui.Status(f"lade {Path(picked.ref).name}"):
        painter.load()

    return engine, painter


def read_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        sys.exit(f"{PROMPT_FILE.name} fehlt.")
    return PROMPT_FILE.read_text(encoding="utf-8").strip() + "\n\n" + llm.CONTRACT


# ------------------------------------------------------------------ Render

def render(image, narration: str) -> None:
    ui.clear()
    print()
    if image is not None:
        size = shutil.get_terminal_size((80, 24))
        cols = max(1, min(VIEWPORT_COLS_CAP, size.columns - 2 * MARGIN))
        rows = max(4, size.lines - TEXT_ROWS)
        termimage.show(image, cols, rows, margin=MARGIN)
    print()
    print(ui.wrap(narration, TEXT_WIDTH, indent=" " * MARGIN))
    print()
    ui.rule()
    print()


# ------------------------------------------------------------------ Loop

def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("Bitte in einem interaktiven Terminal starten.")

    engine, painter = pick_models()
    system = read_system_prompt()
    messages: list[dict] = [{"role": "system", "content": system}]
    turn = "Beginne. Erzeuge die erste Szene."

    try:
        while True:
            messages.append({"role": "user", "content": turn})

            with ui.Status("das Sprachmodell arbeitet"):
                scene = engine.scene(_trim(messages))
            # Vollstaendiges JSON im Verlauf behalten: die persistente
            # Welt (state_update) lebt von diesen Nachrichten.
            messages.append({"role": "assistant", "content": scene["raw"]})

            image = None
            if scene["visual"]:
                with ui.Status("das Bildmodell arbeitet"):
                    try:
                        image = painter.render(scene["visual"])
                    except Exception:
                        image = None

            render(image, scene["narration"])
            if scene["completed"]:
                print(ui.wrap("Die Reise ist zu Ende.", TEXT_WIDTH,
                              indent=" " * MARGIN))
                break
            turn = ui.ask()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        pass
    finally:
        ui.show_cursor()
        print()


def _trim(messages: list[dict]) -> list[dict]:
    head, tail = messages[:1], messages[1:]
    return head + tail[-(HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()

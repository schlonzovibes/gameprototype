#!/usr/bin/env python3
"""AI-Game-Prototyp - Terminal-Runner.

Ablauf:
    1. LLM auswaehlen (Pfeiltasten) -> laden
    2. Diffusion-Modell auswaehlen -> laden
    3. Titel anzeigen, START_PROMPT eingeben (ersetzt $START_PROMPT$
       in game_prompt.txt)
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
TITLE_FILE = HERE / "title_screen.txt"
DEBUG_DIR = HERE / "debug"

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


def read_system_prompt(start_prompt: str) -> str:
    if not PROMPT_FILE.exists():
        sys.exit(f"{PROMPT_FILE.name} fehlt.")
    text = PROMPT_FILE.read_text(encoding="utf-8")
    if "$START_PROMPT$" not in text:
        sys.exit("$START_PROMPT$ fehlt in game_prompt.txt.")
    return (text.replace("$START_PROMPT$", start_prompt).strip()
            + "\n\n" + llm.CONTRACT)


def show_title() -> None:
    """ASCII-Titel drucken. Fehlt die Datei: stilistisch ueberspringen."""
    if not TITLE_FILE.exists():
        return
    sys.stdout.write(ui.DIM)
    for line in TITLE_FILE.read_text(encoding="utf-8").splitlines():
        sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.write(ui.RESET)


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


# ------------------------------------------------------------------ Debug

def _debug_file(start_prompt: str) -> Path:
    """Dateiname aus dem Start-Prompt: bereinigt, gekürzt, kollisionsfrei."""
    import re

    slug = re.sub(r"[^\w\s.-]", "", start_prompt, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "_", slug)[:80].strip("._-") or "ohne_titel"
    path = DEBUG_DIR / f"{slug}.txt"
    n = 2
    while path.exists():
        path = DEBUG_DIR / f"{slug}-{n}.txt"
        n += 1
    return path


def _log_scene(log, engine, scene: dict, number: int) -> None:
    """LLM-Output (inkl. Thinking) 1x pro Zug in die Debug-Datei schreiben."""
    log.write(f"==================== Szene {number} ====================\n\n")
    if getattr(engine, "last_thinking", ""):
        log.write("[THINKING]\n" + engine.last_thinking.strip() + "\n\n")
    log.write("[OUTPUT]\n" + scene["raw"].strip() + "\n\n")
    log.write("-" * 60 + "\n\n")
    log.flush()


# ------------------------------------------------------------------ Loop

def main() -> None:
    if not sys.stdin.isatty():
        sys.exit("Bitte in einem interaktiven Terminal starten.")

    engine, painter = pick_models()

    ui.clear()
    show_title()
    sys.stdout.write("\n" + ui.DIM)
    sys.stdout.write(" REPLOT YOUR STORY:\n" + ui.RESET)
    start_prompt = ui.ask()

    system = read_system_prompt(start_prompt)
    messages: list[dict] = [{"role": "system", "content": system}]
    turn = "Beginne. Erzeuge die erste Szene."

    DEBUG_DIR.mkdir(exist_ok=True)
    log = _debug_file(start_prompt).open("w", encoding="utf-8")

    try:
        scene_number = 0
        while True:
            scene_number += 1
            messages.append({"role": "user", "content": turn})

            try:
                with ui.Status("das Sprachmodell arbeitet"):
                    scene = engine.scene(_trim(messages))
            except Exception as e:
                # Fehler sichtbar machen statt still abzuwarten.
                print()
                print(ui.wrap(f"Sprachmodell-Fehler: {e}", TEXT_WIDTH,
                              indent=" " * MARGIN))
                print()
                messages.pop()   # Turn nicht doppelt im Verlauf behalten
                turn = ui.ask()
                continue

            if not scene["narration"] and not scene["visual"]:
                print(ui.wrap("Unbrauchbare Antwort des Sprachmodells - "
                              "bitte erneut versuchen.", TEXT_WIDTH,
                              indent=" " * MARGIN))
                messages.pop()
                turn = ui.ask()
                continue

            _log_scene(log, engine, scene, scene_number)

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
        log.close()
        ui.show_cursor()
        print()


def _trim(messages: list[dict]) -> list[dict]:
    head, tail = messages[:1], messages[1:]
    return head + tail[-(HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()

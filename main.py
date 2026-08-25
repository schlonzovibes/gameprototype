#!/usr/bin/env python3
"""RePlot - Terminal-Runner.

Das ist die Einstiegsdatei. Sie enthaelt bewusst keine Spiellogik, kein
Modell-Wissen und keine Terminal-Tricks - sie bringt nur die anderen Module
in die richtige Reihenfolge.

=== Wer macht was ===

    main.py        der Ablauf (diese Datei)
    story.py       Spielzustand: Gespraechsverlauf, Szenenstand, Log
    llm.py         Sprachmodelle: Ollama und vLLM, JSON auswerten
    models.py      welche Modelle gibt es ueberhaupt
    diffusion.py   Bildmodell laden und Bilder erzeugen
    termimage.py   Bild -> Text fuers Terminal
    ui.py          alles, was auf den Bildschirm geht
    gpu.py         VRAM- und Auslastungswerte fuer die Fusszeile

=== Ablauf ===

    0. Rahmen starten: Kopfzeile (Zeile 1) und GPU-Fusszeile (letzte Zeile)
       laufen ab hier durchgehend, bis das Programm endet
    1. Inferenz-Backend waehlen (Ollama auf dem Host / vLLM im Container)
    2. Sprachmodell aus diesem Backend waehlen -> laden (Name erscheint
       sofort in der Kopfzeile)
    3. Bildmodell aus dem HF-Cache waehlen -> laden
    4. Titelgrafik, Start-Prompt eingeben
    5. Bild / Erzaehltext / Eingabe zwischen Kopf- und Fusszeile, Szenenstand
       wandert ab hier zusaetzlich in die Kopfzeile

Ein Durchlauf endet, wenn das Modell game.status auf "completed" setzt
(spaetestens Szene 15) oder bei "/restart". Beides fuehrt zurueck zum
Titelbildschirm - die Modelle bleiben dafuer geladen, weil das Laden
Minuten dauert und die Geschichte davon unabhaengig ist.

=== Befehle an jeder Eingabezeile ===

    /restart   zurueck zum Titelbildschirm, neue Geschichte
    unload     Sprachmodell freigeben, Programm laeuft weiter - damit man
               in der Fusszeile zusehen kann, ob der Speicher zurueckkommt
    Strg+C     Programm beenden (gibt das Modell dabei ebenfalls frei)

Start:  python3 main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import diffusion
import llm
import models
import story
import termimage
import ui

TITLE = "RePlot"
TITLE_FILE = Path(__file__).resolve().parent / "title_screen.txt"

VIEWPORT_COLS = 80   # max. Bildbreite in Zellen - breiter wirkt es zerlaufen
TEXT_ROWS = 12       # Reserve: Kopf- und Fusszeile, Erzaehltext, Eingabe
TEXT_WIDTH = 72      # Umbruchbreite des Erzaehltexts
MARGIN = 2           # Einrueckung links

RESTART = "/restart"

# "unload" gibt das Sprachmodell frei, ohne das Programm zu beenden - man
# kann danach in der Fusszeile zusehen, wie der Speicher zurueckkommt.
# Beendet wird mit Strg+C. Beide Schreibweisen, weil /restart einen
# Schraegstrich hat und der Finger ihn hier gern mittippt.
UNLOAD_WORDS = {"unload", "/unload"}


# ------------------------------------------------------------------ Setup

def pick_models(frame: ui.Frame) -> tuple[llm.LLM, diffusion.Diffusion]:
    """Backend, Sprachmodell, Bildmodell - nur einmal pro Sitzung.

    frame laeuft hier schon: die Fusszeile mit VRAM/GPU ist waehrend des
    gesamten Ladens sichtbar, nicht erst ab der ersten Szene. Deshalb
    ui.clear_body() statt ui.clear() - das leert nur den Bereich zwischen
    Kopf- und Fusszeile und laesst beide unangetastet.
    """
    ui.clear_body()

    # ui.select() gibt einen Index zurueck (0 oder 1). Das Tupel davor wird
    # damit indiziert: ("ollama", "vllm")[0] ergibt "ollama".
    backend = ("ollama", "vllm")[ui.select(
        "Inference backend",
        [f"ollama   {models.OLLAMA_URL}", f"vllm     {models.VLLM_URL}"])]

    # indent=ui.INDENT: waehrend Backend/Modell-Auswahl steht auch der
    # Spinner buendig mit "RePlot" und "RAM", nicht am linken Rand wie der
    # Spinner spaeter im Spiel (der bewusst unter der "›"-Eingabe bleibt).
    with ui.Status(f"searching {backend} models", indent=ui.INDENT):
        # Hier wird die FUNKTION ausgewaehlt (ohne Klammern), und erst das
        # "()" ganz am Ende ruft die gewaehlte auf. Sonst wuerden beide
        # laufen, obwohl nur eine gebraucht wird.
        found = (models.ollama_models if backend == "ollama"
                 else models.vllm_models)()

    if not found:
        sys.exit(_no_models(backend))   # beendet das Programm mit Meldung

    choice = found[ui.select("Language model", [m.label for m in found])]
    engine = llm.build(choice)

    # Ab jetzt steht der Name in der Kopfzeile - auch waehrend das
    # Bildmodell danach noch laedt.
    frame.set_model(choice.ref)

    # status.update wird als Rueckruf durchgereicht: vLLM meldet damit den
    # Shard-Fortschritt in die Statuszeile. Ollama nutzt es nicht.
    with ui.Status(f"loading {choice.ref}", indent=ui.INDENT) as status:
        engine.load(status.update)

    ui.clear_body()
    with ui.Status("searching image models", indent=ui.INDENT):
        images = models.image_models()
    if not images:
        # "a or b" auch hier als Rueckfallebene: gibt es gar keinen
        # Cache-Ordner, liefert cache_root() None und der Text springt ein.
        sys.exit(f"No image models in {models.cache_root() or 'the HF cache'}.")

    picked = images[ui.select("Image model", [m.label for m in images])]
    painter = diffusion.Diffusion(picked)
    # Das Label ist "name · groesse"; split(SEP)[0] holt nur den Namen.
    with ui.Status(f"loading {picked.label.split(models.SEP)[0]}", indent=ui.INDENT):
        painter.load()

    return engine, painter


def _no_models(backend: str) -> str:
    """Eine Fehlermeldung, die auch sagt, was zu tun ist."""
    if backend == "ollama":
        return (f"No models reported by Ollama at {models.OLLAMA_URL}. "
                "Is it running with OLLAMA_HOST=0.0.0.0?")
    return (f"No language models in {models.cache_root() or 'the HF cache'}. "
            "Set AIGAME_CACHE if it lives elsewhere.")


# ------------------------------------------------------------------ Render

def show_title() -> None:
    """ASCII-Titel vor der ersten Eingabe. Fehlt die Datei: ueberspringen.

    Bewusst kein Fehler, wenn title_screen.txt fehlt - eine fehlende
    Zierde soll niemanden am Spielen hindern.

    Die Leerzeile Abstand zur Kopfzeile liefert schon ui.clear_body() (der
    Aufrufer raeumt davor immer damit auf) - hier nur noch dieselbe
    Einrueckung wie Bild und Erzaehltext (" " * MARGIN), statt am linken
    Rand zu kleben wie urspruenglich.
    """
    if not TITLE_FILE.exists():
        return
    indent = " " * MARGIN
    art = "\n".join(indent + line for line in
                    TITLE_FILE.read_text(encoding="utf-8").rstrip().splitlines())
    ui.write(f"{ui.GRAY}{art}{ui.RESET}\n")


def render(image, narration: str) -> None:
    """Szene in die Scroll-Region zwischen Kopf- und Fusszeile zeichnen.

    Frueher endete das hier mit einer Trennlinie (ui.rule()) unter dem
    Erzaehltext. Die ist weg - die Eingabezeile, die der Aufrufer direkt im
    Anschluss zeichnet (ask_turn() -> prompt()), uebernimmt jetzt selbst die
    Rolle des Abschlusses. Eine Leerzeile Abstand bleibt trotzdem, damit
    Text und Prompt nicht aneinander kleben.

    Die Leerzeile Abstand ZUR KOPFZEILE (oberhalb von art) liefert schon
    ui.clear_body() - deshalb kein fuehrendes "\\n" mehr im write() unten.
    """
    ui.clear_body()
    width, height = ui.size()

    art = ""
    if image is not None:
        art = termimage.render(
            image,
            # max/min klemmen die Werte: nie breiter als VIEWPORT_COLS, nie
            # breiter als das Fenster minus Rand, und nie kleiner als 1.
            max(1, min(VIEWPORT_COLS, width - 2 * MARGIN)),
            max(4, height - TEXT_ROWS),
            MARGIN,
        )

    # " " * MARGIN erzeugt die Einrueckung als Leerzeichen-String.
    ui.write(f"{art}\n{ui.wrap(narration, TEXT_WIDTH, ' ' * MARGIN)}\n\n")


def paint(painter: diffusion.Diffusion, visual: str):
    """Bild erzeugen - oder None, wenn es nicht klappt.

    Ein fehlgeschlagenes Bild darf die Geschichte nicht anhalten. Deshalb
    wird hier jede Exception geschluckt und stattdessen None geliefert;
    render() zeigt die Szene dann eben nur als Text.
    """
    if not visual:
        return None
    with ui.Status("the image model is working"):
        try:
            return painter.render(visual)
        except Exception:
            return None


# ------------------------------------------------------------------ Story

def run_story(engine: llm.LLM, painter: diffusion.Diffusion, frame: ui.Frame) -> bool:
    """Ein kompletter Durchlauf. Rueckgabe True: noch eine Geschichte starten.

    Der Rahmen existiert schon (er laeuft seit main() gestartet ist) und
    wird hier nur benutzt, nicht erzeugt oder gestoppt - das macht main().
    """
    ui.clear_body()
    frame.reset_scene()   # "Szene 15/15" der letzten Geschichte ausblenden
    show_title()
    ui.write(f"\n{ui.GRAY} REPLOT YOUR STORY:{ui.RESET}\n")
    start_prompt = prompt(engine)   # nicht ui.ask(): "unload" soll auch hier gehen

    tale = story.Story(engine, start_prompt)
    turn = tale.first_turn   # der erste "Zug" kommt vom Programm, nicht vom Spieler

    try:
        while True:
            try:
                with ui.Status("the language model is working"):
                    scene = tale.advance(turn)
            except story.SceneError as e:
                # Zug gescheitert. story.Story hat den Verlauf sauber
                # gehalten, der Spieler darf einfach nochmal.
                ui.write("\n" + ui.wrap(str(e), TEXT_WIDTH, " " * MARGIN) + "\n\n")
                turn = ask_turn(engine)
                if turn is None:
                    return True     # /restart
                continue            # zurueck an den Schleifenanfang

            frame.update(scene.number, scene.max_scenes)
            # Von innen nach aussen gelesen: erst malen, dann anzeigen.
            render(paint(painter, scene.visual), scene.narration)

            if scene.completed:
                ui.write(ui.wrap("The journey has ended.", TEXT_WIDTH,
                                 " " * MARGIN) + "\n\n")
                # "== 0" macht aus dem Index ein True/False: 0 ist
                # "start a new story", also weitermachen.
                return ui.select("What next?", ["start a new story", "quit"]) == 0

            turn = ask_turn(engine)
            if turn is None:
                return True
    finally:
        tale.close()


def unload_model(engine: llm.LLM) -> None:
    """Sprachmodell freigeben, ohne das Programm zu beenden.

    Genau das ist der Zweck: danach laeuft alles weiter, und man kann in
    der Fusszeile beobachten, ob der Speicher wirklich zurueckkommt.
    """
    with ui.Status("unloading language model", indent=ui.INDENT):
        freed = engine.unload()

    ui.write(ui.wrap(
        "Language model released - watch the RAM bar. Ctrl+C to quit."
        if freed else
        "Could not confirm the release - check with 'nvidia-smi'.",
        TEXT_WIDTH, " " * MARGIN) + "\n\n")


def prompt(engine: llm.LLM) -> str:
    """Eingabe holen und dabei "unload" abfangen.

    Die Schleife sorgt dafuer, dass "unload" KEINE Spieleingabe wird: nach
    dem Freigeben wird einfach erneut gefragt. Hier statt in ui.ask(), weil
    ui.py nichts von Sprachmodellen wissen soll.
    """
    while True:
        value = ui.ask()
        if value.lower() in UNLOAD_WORDS:
            unload_model(engine)
            continue
        return value


def ask_turn(engine: llm.LLM) -> str | None:
    """Naechste Eingabe; None bedeutet: Neustart angefordert.

    None als Sonderwert statt einer eigenen Exception - bei genau einem
    Sonderfall ist das die einfachere Loesung.
    """
    value = prompt(engine)
    return None if value.lower() == RESTART else value


# ------------------------------------------------------------------ Loop

def main() -> None:
    # isatty() = "haengt hier ein echtes Terminal dran?". Bei "python main.py
    # < datei.txt" waere das nein - und ohne Terminal gibt es keine
    # Pfeiltasten-Auswahl und keine Scroll-Region.
    if not sys.stdin.isatty():
        sys.exit("Please run inside an interactive terminal.")

    # Ein Rahmen fuer das gesamte Programm - nicht erst ab der ersten Szene.
    # So sind Kopf- und Fusszeile (VRAM/GPU) schon waehrend Sprach- und
    # Bildmodell laden sichtbar. ui.clear() einmal davor, solange noch keine
    # Scroll-Region aktiv ist - danach uebernimmt clear_body() das Aufraeumen
    # zwischen den Zeilen, die dem Rahmen gehoeren.
    ui.clear()
    frame = ui.Frame(TITLE).start()

    # None, bis pick_models() zurueckkommt: schlaegt das Laden selbst fehl
    # (sys.exit in pick_models), gibt es noch kein Modell zum Entladen -
    # der finally-Block unten prueft das ab.
    engine = None

    try:
        engine, painter = pick_models(frame)
        # Die Schleife laeuft, solange run_story() True liefert. Der Rumpf
        # ist leer - die ganze Arbeit steckt in der Bedingung.
        while run_story(engine, painter, frame):
            pass   # Neustart: gleiche Modelle, neue Geschichte
    except KeyboardInterrupt:
        pass       # Strg+C ist ein normaler Weg zu gehen, kein Fehler
    finally:
        # finally laeuft IMMER: bei return, bei Strg+C, bei jedem Fehler
        # (auch bei sys.exit() aus pick_models und beim SystemExit, das
        # ui.ask()/ui.select() fuer 'q'/Strg+C/Strg+D werfen). Deshalb steht
        # engine.unload() hier und nicht an einem einzelnen Ausstiegsweg:
        # bei JEDEM Programmende soll der Speicher zurueckkommen.
        #
        # Der "unload"-Befehl ist davon unabhaengig - er dient dem Zusehen
        # waehrend das Programm laeuft. Hat der Spieler ihn schon benutzt,
        # ist dieser Aufruf ein harmloser Leerlauf: es gibt dann keinen
        # Prozess mehr zu beenden.
        freed = True
        if engine is not None:
            with ui.Status("unloading model", indent=ui.INDENT):
                freed = engine.unload()
        # Ohne das bliebe die Scroll-Region gesetzt und das Terminal waere
        # nach dem Spiel kaputt.
        frame.stop()
        ui.show_cursor()
        ui.write("\n")

        # Erst NACH frame.stop(): der Rahmen haelt sonst die letzte Zeile
        # besetzt und ueberschreibt die Meldung eine Sekunde spaeter wieder.
        if not freed:
            ui.write(ui.wrap(
                "Warning: vLLM may still be holding GPU memory. Check with "
                "'nvidia-smi'; if needed: "
                f"docker exec {llm.VLLM.CONTAINER} pkill -KILL -f 'vllm'",
                indent=" " * MARGIN) + "\n\n")


# Dieser Block laeuft nur, wenn die Datei direkt gestartet wird
# ("python3 main.py") - nicht, wenn ein anderes Modul sie importiert.
# Ohne ihn wuerde schon ein "import main" das ganze Spiel starten.
if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""RePlot - Terminal-Runner.

Das ist die Einstiegsdatei. Sie enthaelt bewusst keine Spiellogik, kein
Modell-Wissen und keine Terminal-Tricks - sie bringt nur die anderen Module
in die richtige Reihenfolge.

=== Wer macht was ===

    main.py        der Ablauf (diese Datei)
    story.py       eine Partie: Modellaufrufe, Szenenstand, Log
    state.py       der Weltzustand - Graph, Figuren, Spuren
    schema.py      die einzige Quelle des Ausgabeformats
    llm.py         Sprachmodelle: Ollama und vLLM
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
    4. Spielsystem aus game_prompts/ waehlen - zuletzt, weil hier nichts
       geladen wird und die Entscheidung so direkt vor dem Spiel faellt
    5. Titelgrafik, Start-Prompt eingeben
    6. Bild / Erzaehltext / Eingabe zwischen Kopf- und Fusszeile, Szenenstand
       wandert ab hier zusaetzlich in die Kopfzeile

Ein Durchlauf endet, wenn der CLIENT ihn beendet - nach Szene 15, oder
frueher, wenn das Modell die Situation als abgeschlossen meldet UND genug
Szenen gespielt sind (story.MAX_SCENES / MIN_SCENES). "/restart" tut
dasselbe von Hand. Beides fuehrt zurueck zum Titelbildschirm; die Modelle
bleiben geladen, weil das Laden Minuten dauert und die Geschichte davon
unabhaengig ist.

=== Befehle an jeder Eingabezeile ===

    /restart   zurueck zum Titelbildschirm, neue Geschichte
    unload     Sprachmodell freigeben, Programm laeuft weiter - damit man
               in der Fusszeile zusehen kann, ob der Speicher zurueckkommt
    Strg+C     Programm beenden. Das Modell bleibt dabei ABSICHTLICH
               geladen: der naechste Start ist dadurch sofort da. Wer den
               Speicher zurueck will, tippt vorher "unload".

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

def pick_models(frame: ui.Frame) -> tuple[llm.LLM, diffusion.Diffusion, Path]:
    """Backend, Sprachmodell, Bildmodell, Spielprompt - einmal pro Sitzung.

    Das Spielsystem kommt bewusst zuletzt, nach dem Laden des Bildmodells:
    die beiden Modelle brauchen zusammen viele Minuten, und in dieser Zeit
    soll der Spieler nicht schon eine Entscheidung getroffen haben, die er
    bis zum Spielstart wieder vergessen hat.

    frame laeuft hier schon: die Fusszeile mit RAM/GPU ist waehrend des
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

    # Sofort pruefen, ob dieses Backend ueberhaupt betriebsbereit ist. Ohne
    # das faellt ein fehlendes docker-Paket erst auf, nachdem der Spieler
    # schon ein Modell ausgesucht hat und das Laden anlaeuft - er hat dann
    # eine Auswahl umsonst getroffen.
    if backend == "vllm":
        llm.VLLM.check_requirements()

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

    # Zuletzt das Spielsystem. Nichts zu laden - deshalb kein Spinner.
    ui.clear_body()
    systems = story.available_systems()
    if not systems:
        sys.exit(f"No game systems in {story.PROMPTS_DIR}. Each one is a "
                 f"folder containing {', '.join(story.PARTS)}.")

    system_dir = systems[ui.select("Game system",
                                   [_system_label(d) for d in systems])]

    return engine, painter, system_dir


def _system_label(path: Path) -> str:
    """Anzeigetext eines Spielsystems: Ordnername und ungefaehre Tokenzahl.

    Gleiches Muster wie bei den Modellen ("name · groesse"), damit alle vier
    Auswahllisten gleich aussehen.

    Tokens statt Kilobyte, weil nur die Tokenzahl etwas Brauchbares aussagt:
    sie ist der Anteil am Kontextfenster (NUM_CTX), den dieser Prompt in
    JEDER Runde belegt. Kilobyte sagen darueber wenig - eine Datei voller
    Trennlinien ist gross, kostet aber kaum Kontext.

    Die Tilde ist Absicht: der Wert ist geschaetzt, nicht gezaehlt.
    """
    tokens = story.estimate_tokens(path)
    if tokens is None:
        return path.name   # nicht lesbar - wenigstens der Name
    return f"{path.name}{models.SEP}~{_compact(tokens)} tokens"


def _compact(number: int) -> str:
    """6464 -> "6.5k", 850 -> "850". Haelt die Auswahlliste schmal."""
    return f"{number / 1000:.1f}k" if number >= 1000 else str(number)


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

    Der Erzaehltext bricht auf der BILDBREITE um, nicht auf einer festen
    Spaltenzahl: beide Bloecke enden dadurch buendig an derselben Kante.
    """
    ui.clear_body()
    width, height = ui.size()

    art = ""
    text_width = TEXT_WIDTH   # Rueckfall fuer Szenen ohne Bild
    if image is not None:
        # max/min klemmen die Werte: nie breiter als VIEWPORT_COLS, nie
        # breiter als das Fenster minus Rand, und nie kleiner als 1.
        max_cols = max(1, min(VIEWPORT_COLS, width - 2 * MARGIN))
        max_rows = max(4, height - TEXT_ROWS)

        # fit() liefert die Masse, die das Bild WIRKLICH bekommt. Bei
        # niedrigem Terminal begrenzt die Hoehe, und die Breite schrumpft
        # mit, damit das Seitenverhaeltnis stimmt - max_cols waere dann zu
        # optimistisch und der Text ragte ueber das Bild hinaus.
        cols, _ = termimage.fit(image, max_cols, max_rows)
        art = termimage.render(image, max_cols, max_rows, MARGIN)

        # textwrap rechnet die Einrueckung in die Breite mit ein, das Bild
        # steht zusaetzlich um MARGIN eingerueckt - daher die Summe, damit
        # beide an derselben Spalte enden.
        text_width = MARGIN + cols

    # " " * MARGIN erzeugt die Einrueckung als Leerzeichen-String.
    ui.write(f"{art}\n{ui.wrap(narration, text_width, ' ' * MARGIN)}\n\n")


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

def run_story(engine: llm.LLM, painter: diffusion.Diffusion, frame: ui.Frame,
              system_dir: Path) -> bool:
    """Ein kompletter Durchlauf. Rueckgabe True: noch eine Geschichte starten.

    Der Rahmen existiert schon (er laeuft seit main() gestartet ist) und
    wird hier nur benutzt, nicht erzeugt oder gestoppt - das macht main().
    """
    ui.clear_body()
    frame.reset_scene()   # "Szene 15/15" der letzten Geschichte ausblenden
    show_title()
    ui.write(f"\n{ui.GRAY} REPLOT YOUR STORY:{ui.RESET}\n")
    start_prompt = prompt(engine)   # nicht ui.ask(): "unload" soll auch hier gehen

    tale = story.Game(engine, start_prompt, system_dir)

    # Szene 1 kommt aus der Welterzeugung, nicht aus einem Spielerzug. Der
    # frueher hier erfundene erste Zug ("Beginne. Erzeuge die erste Szene.")
    # entfaellt damit - begin() liefert Welt und Eroeffnung in einem Aufruf.
    #
    # step/argument halten fest, WAS als Naechstes zu rufen ist. Ohne diesen
    # Kniff braeuchte es einen Sonderfall vor der Schleife und denselben
    # Fehlerbehandlungsblock zweimal.
    step, argument = tale.begin, ()

    try:
        while True:
            try:
                with ui.Status("the language model is working"):
                    beat = step(*argument)
            except story.SceneError as e:
                # Zug gescheitert. Der Weltzustand ist dabei garantiert
                # unveraendert - state.World.apply() laeuft erst NACH den
                # Modellaufrufen. Der Spieler darf einfach nochmal.
                ui.write("\n" + ui.wrap(str(e), TEXT_WIDTH, " " * MARGIN) + "\n\n")
                turn = ask_turn(engine)
                if turn is None:
                    return True     # /restart

                # War es begin() selbst, das gescheitert ist, gibt es noch
                # keine Welt (tale.world ist dann None) - advance() wuerde
                # darauf sofort mit "begin() was never called." abbrechen,
                # und die naechste Runde durch denselben Zweig liefe in
                # eine Endlosschleife, aus der nur /restart herausfuehrt.
                # Stattdessen: begin() erneut versuchen, mit dem gerade
                # eingegebenen Text als (moeglicherweise korrigiertem)
                # Startprompt.
                if tale.world is None:
                    tale.start_prompt = turn
                    step, argument = tale.begin, ()
                else:
                    step, argument = tale.advance, (turn,)
                continue            # zurueck an den Schleifenanfang

            frame.update(beat.number, beat.max_scenes)
            # Von innen nach aussen gelesen: erst malen, dann anzeigen.
            render(paint(painter, beat.image_prompt), beat.narration)

            if beat.completed:
                ui.write(ui.wrap("The journey has ended.", TEXT_WIDTH,
                                 " " * MARGIN) + "\n\n")
                # "== 0" macht aus dem Index ein True/False: 0 ist
                # "start a new story", also weitermachen.
                return ui.select("What next?", ["start a new story", "quit"]) == 0

            turn = ask_turn(engine)
            if turn is None:
                return True
            step, argument = tale.advance, (turn,)
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

    # Fehlertext zwischenspeichern statt sofort ausgeben: solange der Rahmen
    # laeuft, gehoeren ihm Kopf- und Fusszeile, und er wuerde die Meldung eine
    # Sekunde spaeter wieder ueberschreiben. Ausgegeben wird deshalb erst
    # NACH frame.stop(), unterhalb des try/finally.
    error = None

    try:
        engine, painter, system_dir = pick_models(frame)
        # Die Schleife laeuft, solange run_story() True liefert. Der Rumpf
        # ist leer - die ganze Arbeit steckt in der Bedingung.
        while run_story(engine, painter, frame, system_dir):
            pass   # Neustart: gleiche Modelle, gleicher Prompt, neue Geschichte
    except KeyboardInterrupt:
        pass       # Strg+C ist ein normaler Weg zu gehen, kein Fehler
    except RuntimeError as e:
        # Vorhergesehene Betriebsfehler: fehlendes docker-Paket, vLLM nicht
        # erreichbar, Server beim Laden gestorben. Die tragen alle bereits
        # eine erklaerende Meldung - ein roher Traceback macht sie nur
        # unleserlich.
        #
        # BEWUSST nur RuntimeError: ein Programmierfehler (AttributeError,
        # TypeError, ...) soll weiterhin mit vollem Traceback durchschlagen,
        # sonst sucht man ihn spaeter im Dunkeln.
        error = str(e)
    finally:
        # Hier wird BEWUSST nicht entladen. Das Sprachmodell freizugeben ist
        # eine ausdrueckliche Entscheidung des Spielers ("unload"), kein
        # Nebeneffekt des Beendens.
        #
        # Das hat auch einen praktischen Vorteil: laeuft vLLM weiter, ist der
        # naechste Programmstart sofort da - VLLM.load() sieht das Modell in
        # /v1/models und kehrt direkt zurueck, statt es minutenlang neu zu
        # laden. Wer den Speicher wirklich zurueck will, tippt vorher
        # "unload" und sieht in der Fusszeile zu.
        #
        # Aufgeraeumt wird hier nur das Terminal - ohne das bliebe die
        # Scroll-Region gesetzt und die Shell danach unbrauchbar.
        frame.stop()
        ui.show_cursor()
        ui.write("\n")

    # Erst jetzt - der Rahmen ist gestoppt, die Meldung bleibt stehen.
    # sys.exit(str) schreibt den Text nach stderr und endet mit Code 1.
    if error:
        sys.exit(ui.wrap(error, TEXT_WIDTH, " " * MARGIN))


# Dieser Block laeuft nur, wenn die Datei direkt gestartet wird
# ("python3 main.py") - nicht, wenn ein anderes Modul sie importiert.
# Ohne ihn wuerde schon ein "import main" das ganze Spiel starten.
if __name__ == "__main__":
    main()


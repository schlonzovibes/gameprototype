"""Eine laufende Geschichte: Konversation, Szenenstand, Debug-Log.

=== Warum es dieses Modul gibt ===

Ein Sprachmodell hat kein Gedaechtnis. Es bekommt bei JEDER Anfrage den
kompletten bisherigen Gespraechsverlauf mitgeschickt und leitet daraus ab,
was als Naechstes passiert. Dieser Verlauf ist der Zustand des Spiels -
und dieses Modul verwaltet ihn.

Es trennt den Spielzustand von der Terminal-Darstellung. main.py fragt nur
noch advance() und bekommt eine Scene zurueck - oder eine Exception, wenn
der Zug nicht zustande kam. Der Verlauf bleibt in beiden Faellen konsistent.

=== Wie der Verlauf aussieht ===

Eine Liste von Nachrichten, jede mit einer Rolle:

    [{"role": "system",    "content": "<game_prompt.txt + CONTRACT>"},
     {"role": "user",      "content": "Beginne. Erzeuge die erste Szene."},
     {"role": "assistant", "content": "<JSON der Szene 1>"},
     {"role": "user",      "content": "Ich oeffne die Tuer."},
     {"role": "assistant", "content": "<JSON der Szene 2>"},
     ...]

system    = die Spielregeln, steht genau einmal ganz vorne
user      = was der Spieler eingibt
assistant = was das Modell geantwortet hat

Wichtig: wir speichern das VOLLSTAENDIGE JSON der Antwort, nicht nur den
Erzaehltext. Darin steckt state_update - die persistente Welt. Wuerden wir
nur die Erzaehlung behalten, vergaesse das Spiel bei jedem Zug, was es
ueber seine eigene Welt weiss.
"""

from __future__ import annotations

import re
from pathlib import Path

import llm

# __file__ ist der Pfad dieser Datei. .resolve() macht ihn absolut,
# .parent nimmt den Ordner darum. Ergebnis: der Projektordner - egal, von
# welchem Verzeichnis aus das Spiel gestartet wurde.
HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "game_prompt.txt"
DEBUG_DIR = HERE / "debug"

# Wie viele Zuege im Kontext bleiben. Der Verlauf waechst mit jeder Szene;
# irgendwann sprengt er das Kontextfenster (NUM_CTX in llm.py). Deshalb
# schneiden wir vorne ab - der system-Prompt bleibt aber immer erhalten.
HISTORY_TURNS = 12

FIRST_TURN = "Beginne. Erzeuge die erste Szene."


class SceneError(RuntimeError):
    """Der Zug ist gescheitert - der Spieler darf ihn wiederholen.

    Eine eigene Exception-Klasse. Der Rumpf ist nur der Docstring, mehr
    braucht es nicht: sie erbt alles von RuntimeError. Der Sinn ist die
    Unterscheidbarkeit - main.py kann gezielt "except story.SceneError"
    schreiben und faengt damit nur Spielfehler, keine Programmierfehler.
    """


def system_prompt(start_prompt: str) -> str:
    """game_prompt.txt laden, den Start-Prompt einsetzen, Contract anhaengen.

    In der Datei steht der Platzhalter $START_PROMPT$. Genau dort landet,
    was der Spieler eingegeben hat.
    """
    if not PROMPT_FILE.exists():
        raise SystemExit(f"{PROMPT_FILE.name} is missing.")

    text = PROMPT_FILE.read_text(encoding="utf-8")

    # Ohne den Platzhalter wuerde das Spiel ohne Startsituation laufen und
    # voellig beliebig erzaehlen. Lieber sofort und deutlich abbrechen.
    if "$START_PROMPT$" not in text:
        raise SystemExit("$START_PROMPT$ missing in game_prompt.txt.")

    return text.replace("$START_PROMPT$", start_prompt).strip() + "\n\n" + llm.CONTRACT


def _log_path(start_prompt: str) -> Path:
    """Dateiname aus dem Start-Prompt: bereinigt, gekuerzt, kollisionsfrei."""
    # re.sub(muster, ersatz, text) ersetzt alles, was passt.
    # [^\w\s.-] heisst: alles, was NICHT (^) Buchstabe/Ziffer (\w),
    # Leerraum (\s), Punkt oder Bindestrich ist. Also Schraegstriche,
    # Doppelpunkte und aehnliches, die in Dateinamen Aerger machen.
    slug = re.sub(r"[^\w\s.-]", "", start_prompt, flags=re.UNICODE).strip()

    # \s+ = eine oder mehrere Leerstellen -> ein Unterstrich.
    # [:80] schneidet nach 80 Zeichen ab, strip("._-") raeumt Reste am Rand.
    # "or 'untitled'" faengt den Fall ab, dass nichts uebrig bleibt.
    slug = re.sub(r"\s+", "_", slug)[:80].strip("._-") or "untitled"

    # Gibt es die Datei schon, wird durchgezaehlt: name.txt, name-2.txt, ...
    path, n = DEBUG_DIR / f"{slug}.txt", 2
    while path.exists():
        path = DEBUG_DIR / f"{slug}-{n}.txt"
        n += 1
    return path


class Story:
    """Eine Geschichte von der ersten bis zur letzten Szene."""

    def __init__(self, engine: llm.LLM, start_prompt: str):
        self.engine = engine

        # Der Verlauf beginnt mit genau einer system-Nachricht.
        self.messages = [{"role": "system", "content": system_prompt(start_prompt)}]

        self.number = 0                    # noch keine Szene gespielt
        self.max_scenes = llm.MAX_SCENES

        # exist_ok=True: kein Fehler, wenn der Ordner schon da ist.
        DEBUG_DIR.mkdir(exist_ok=True)
        # "w" = schreiben (vorhandene Datei wuerde geleert - kann hier nicht
        # passieren, _log_path() sucht ja einen freien Namen).
        self.log = _log_path(start_prompt).open("w", encoding="utf-8")

    # @property laesst eine Methode wie ein Attribut aussehen: man schreibt
    # tale.first_turn, nicht tale.first_turn(). Fuer einfache Abfragen ohne
    # Argumente ist das die lesbarere Form.
    @property
    def first_turn(self) -> str:
        return FIRST_TURN

    def advance(self, player_input: str) -> llm.Scene:
        """Einen Zug spielen. Bei Fehlschlag bleibt der Verlauf unveraendert.

        Das ist die zentrale Zusage dieser Klasse: entweder es kommt eine
        Scene zurueck UND der Verlauf ist gewachsen, oder es fliegt eine
        SceneError UND der Verlauf ist exakt wie vorher. Nie etwas dazwischen.

        Ohne diese Zusage wuerde eine gescheiterte Anfrage die Spielereingabe
        im Verlauf zuruecklassen. Beim naechsten Versuch stuende sie doppelt
        drin, und das Modell wuerde sich fragen, warum der Spieler alles
        zweimal sagt.
        """
        self.messages.append({"role": "user", "content": player_input})

        try:
            # self.number + 1 ist die Rueckfall-Szenennummer, falls das
            # Modell game.scene_number nicht mitschickt.
            scene = self.engine.scene(self._context(), self.number + 1)
        except Exception as e:
            self.messages.pop()   # Eingabe wieder entfernen - Verlauf sauber
            # Aus jedem Fehler wird eine SceneError, damit main.py nur eine
            # Sorte fangen muss. "from e" behaelt die Originalursache.
            raise SceneError(str(e)) from e

        # Antwort kam an, war aber unbrauchbar - genauso behandeln.
        if not scene.narration and not scene.visual:
            self.messages.pop()
            raise SceneError("Unusable response from the language model.")

        # Vollstaendiges JSON in den Verlauf, nicht nur die Erzaehlung:
        # die persistente Welt (state_update) lebt von diesen Nachrichten.
        self.messages.append({"role": "assistant", "content": scene.raw})

        self.number, self.max_scenes = scene.number, scene.max_scenes
        self._write_log(scene)
        return scene

    def close(self) -> None:
        """Log-Datei schliessen. main.py ruft das im finally auf."""
        self.log.close()

    def _context(self) -> list[dict]:
        """Der Verlauf, zurechtgeschnitten aufs Kontextfenster.

        Der system-Prompt (die Spielregeln) muss IMMER dabei sein, sonst
        vergisst das Modell mitten im Spiel, was es eigentlich tut. Deshalb
        wird der Kopf abgetrennt, nur der Rest gekuerzt und beides wieder
        zusammengesetzt.
        """
        head, tail = self.messages[:1], self.messages[1:]
        # Negative Indizes zaehlen von hinten: [-24:] sind die letzten 24
        # Eintraege. Mal zwei, weil jeder Zug aus zwei Nachrichten besteht
        # (Spielereingabe + Modellantwort).
        return head + tail[-(HISTORY_TURNS * 2):]

    def _write_log(self, scene: llm.Scene) -> None:
        """Die Rohantwort ins Debug-Log schreiben.

        Wenn eine Szene seltsam wird, steht hier, was das Modell wirklich
        geliefert hat - inklusive seines Denkprozesses.
        """
        # "=" * 20 ergibt eine Reihe von zwanzig Gleichheitszeichen.
        self.log.write(f"{'=' * 20} scene {scene.number} {'=' * 20}\n\n")

        # [THINKING] steht IMMER da - eine fehlende Sektion waere sonst nicht
        # von "Modell hat nicht gedacht" zu unterscheiden. Ist sie leer,
        # obwohl THINK an ist, ist das selbst ein Debug-Signal (Reasoning-
        # Parser aus? Grammatik wuergt das Denken? Token-Limit im Denken?).
        thinking = (self.engine.last_thinking or "").strip()
        if thinking:
            self.log.write("[THINKING]\n" + thinking + "\n\n")
        elif llm.THINK:
            self.log.write("[THINKING]\n(leer - kein Denkprozess empfangen)\n\n")
        else:
            self.log.write("[THINKING]\n(deaktiviert - AIGAME_THINK=0)\n\n")

        self.log.write("[OUTPUT]\n" + scene.raw.strip() + "\n\n" + "-" * 60 + "\n\n")

        # flush() schreibt sofort auf die Platte. Ohne das sammelt Python
        # den Text und bei einem Absturz waere genau die interessante letzte
        # Szene weg - also die, die den Absturz verursacht hat.
        self.log.flush()


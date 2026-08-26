"""Eine laufende Partie: Aufrufe an das Modell, Zustand, Debug-Log.

=== Was sich gegenueber frueher geaendert hat ===

Frueher war DIESE Datei der Weltzustand: eine wachsende Liste von
Chatnachrichten, aus der HISTORY_TURNS vorne abschnitt, sobald sie zu lang
wurde. Die Welt vergass damit ihren eigenen Anfang.

Jetzt liegt der Zustand in state.World, und jeder Aufruf besteht aus GENAU
ZWEI Nachrichten:

    system   Spielsystem-Prompt + Feldbeschreibung aus schema.describe()
             - ueber den ganzen Lauf byteweise identisch
    user     der gerenderte Zustand + die Spieleraktion
             - der einzige Teil, der sich pro Zug aendert

Das hat zwei Folgen. Erstens vergisst nichts mehr: der volle Zustand geht
jedes Mal frisch mit. Zweitens ist der System-Prompt vollstaendig
prefix-cachebar - der Server prefillt ihn einmal und danach nie wieder.

=== Ein Spielsystem ist ein ORDNER ===

    game_prompts/<name>/init.txt      Weltgenerator, laeuft genau einmal
    game_prompts/<name>/intent.txt    eine Figur entscheidet fuer sich
    game_prompts/<name>/resolve.txt   Aufloesung und Erzaehlung

Keine dieser Dateien enthaelt eine Feldliste. Die kommt zur Laufzeit aus
schema.describe() - wer sie in die Datei schriebe, haette die Drift wieder
eingebaut, gegen die der ganze Umbau geht.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import llm
import schema
from state import World

HERE = Path(__file__).resolve().parent
DEBUG_DIR = HERE / "debug"
PROMPTS_DIR = HERE / "game_prompts"

# Die drei Dateien, aus denen ein Spielsystem besteht. Fehlt eine, ist der
# Ordner keins - available_systems() zeigt ihn dann gar nicht erst an.
PARTS = ("init.txt", "intent.txt", "resolve.txt")

# Die Szenengrenze liegt jetzt hier, nicht mehr im Modelloutput. Frueher
# meldete das Modell game.scene_number und game.status, und der Client
# zaehlte "hilfsweise" selbst mit - er wusste die Zahl also die ganze Zeit
# und fragte trotzdem.
MAX_SCENES = 15

# Vor dieser Szene wird can_end ignoriert. Ein Modell, das die Situation
# gern aufloest, koennte sonst in Szene 3 abschliessen - technisch richtig,
# als Spiel wertlos.
MIN_SCENES = 8

# Grobe Token-Schaetzung: Tokens ~ Woerter * 1.4. Die verbreitete Regel
# "Zeichen / 4" liegt bei Prompt-Dateien dieser Machart um rund ein Drittel
# daneben, weil Leerzeilen und Trennlinien viele Zeichen, aber kaum Tokens
# erzeugen. Die Wortzahl ignoriert diesen Fuellstoff von selbst.
TOKENS_PER_WORD = 1.4


class SceneError(RuntimeError):
    """Der Zug ist gescheitert - der Spieler darf ihn wiederholen.

    Eine eigene Klasse, damit main.py gezielt nur Spielfehler faengt und
    Programmierfehler weiterhin mit Traceback durchschlagen.
    """


@dataclass(frozen=True)
class Beat:
    """Was main.py von einem Zug braucht - und sonst nichts.

    Ersetzt die frueherer llm.Scene. Der Unterschied ist nicht die Form,
    sondern die Herkunft: number und completed rechnet jetzt der Client,
    statt sie beim Modell zu erfragen.
    """
    narration: str
    image_prompt: str
    number: int
    max_scenes: int
    completed: bool


# --------------------------------------------------------- Spielsysteme

def available_systems() -> list[Path]:
    """Alle vollstaendigen Spielsysteme, alphabetisch.

    Vollstaendig heisst: der Ordner enthaelt alle drei Dateien. Ein halb
    angelegtes System taucht gar nicht erst auf - besser, als es waehlbar
    zu machen und erst beim ersten Aufruf zu scheitern.
    """
    try:
        return sorted(d for d in PROMPTS_DIR.iterdir()
                      if d.is_dir() and all((d / p).is_file() for p in PARTS))
    except OSError:
        return []


def estimate_tokens(system_dir: Path) -> int | None:
    """Ungefaehre Tokenzahl aller drei Dateien zusammen.

    Die Summe, nicht das Maximum: sie ist zwar keine Kontextgroesse (die
    drei Prompts laufen nie gleichzeitig), aber ein ehrliches Mass fuer den
    Umfang eines Spielsystems - und darum geht es in der Auswahlliste.
    """
    try:
        words = sum(len((system_dir / part).read_text(encoding="utf-8").split())
                    for part in PARTS)
    except OSError:
        return None
    return round(words * TOKENS_PER_WORD)


# ---------------------------------------------------------------- Partie

class Game:
    """Eine Partie von der Welterzeugung bis zur letzten Szene."""

    def __init__(self, engine: llm.LLM, start_prompt: str, system_dir: Path):
        self.engine = engine
        self.start_prompt = start_prompt

        try:
            self.prompts = {part: (system_dir / part).read_text(encoding="utf-8")
                            for part in PARTS}
        except OSError as e:
            raise SystemExit(f"Cannot read game system {system_dir.name}: {e}")

        # Die Welt entsteht erst in begin(). Bis dahin gibt es sie nicht -
        # und das soll man auch sehen koennen.
        self.world: World | None = None

        DEBUG_DIR.mkdir(exist_ok=True)
        self.log = _log_path(start_prompt).open("w", encoding="utf-8")

    # ------------------------------------------------------------ Aufrufe

    def begin(self) -> Beat:
        """Die Welt erzeugen und Szene 1 zurueckgeben.

        Ein einziger Aufruf, der beides liefert. Frueher schickte das
        Programm dafuer einen erfundenen ersten Spielerzug ("Beginne.
        Erzeuge die erste Szene.") - eine Nachricht, die so tat, als haette
        jemand etwas eingegeben.
        """
        # init.txt traegt den Platzhalter $START_PROMPT$ (siehe dort, Abschnitt
        # LANGUAGE) - er muss vor dem Versand ersetzt sein, sonst liest das
        # Modell im System-Prompt woertlich "$START_PROMPT$" statt des Texts,
        # dessen Sprache es bestimmen soll. Erst hier statt schon in
        # __init__(): begin() liest self.start_prompt bei jedem Aufruf frisch,
        # ein Neuversuch nach einem gescheiterten ersten begin() (main.py)
        # kann den Startprompt also aendern, ohne dass die Ersetzung veraltet.
        init_prompt = self.prompts["init.txt"].replace(
            "$START_PROMPT$", self.start_prompt)
        system = self._system(init_prompt, schema.InitWorld)
        user = f"START PROMPT:\n{self.start_prompt}"

        self._log_block("INIT / system", system)
        self._log_block("INIT / user", user)

        try:
            init = self.engine.structured(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                schema.InitWorld)
        except Exception as e:
            raise SceneError(str(e)) from e

        self._log_thinking()
        self._log_block("INIT / result", init.model_dump_json(indent=2))

        narration, image_prompt = schema.opening_text(init)

        self.world = World.from_init(init)
        self.world.scene_number = 1
        self.world.recent.append(narration)

        self._log_block("STATE after init", self.world.render())

        return Beat(
            narration=narration,
            image_prompt=image_prompt,
            number=1,
            max_scenes=MAX_SCENES,
            # Szene 1 kann nicht die letzte sein - sie hat noch keine
            # Spieleraktion gesehen.
            completed=False,
        )

    def advance(self, player_input: str) -> Beat:
        """Einen Zug spielen.

        Der Zustand aendert sich AUSSCHLIESSLICH in Schritt 3 (apply). Wirft
        einer der Modellaufrufe davor, ist die Welt garantiert unveraendert -
        das ist strenger als die fruehere Zusage "der Chatverlauf bleibt
        konsistent" und braucht kein Aufraeumen im Fehlerfall.
        """
        world = self.world
        if world is None:                      # pragma: no cover
            raise SceneError("begin() was never called.")

        try:
            intent_text = self._intent(world)
        except Exception as e:
            raise SceneError(str(e)) from e

        turn_cls = schema.turn_model(world.node_ids(), world.active_ids(),
                                     world.exits_from(world.player_at))

        system = self._system(self.prompts["resolve.txt"], turn_cls)
        user = (f"WORLD STATE:\n{world.render()}\n\n"
                + intent_text
                + f"PLAYER ACTION:\n{player_input}")

        self._log_block(f"TURN {world.scene_number + 1} / user", user)

        try:
            turn = self.engine.structured(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                turn_cls)
        except Exception as e:
            raise SceneError(str(e)) from e

        self._log_thinking()
        self._log_block("RESOLVE / result", turn.model_dump_json(indent=2))

        rejected = world.apply(turn)
        if rejected:
            self._log_block("REJECTED", "\n".join(rejected))

        # Der Client entscheidet ueber das Ende, nicht das Modell. can_end
        # ist ein Vorschlag: er zaehlt erst ab MIN_SCENES, und die harte
        # Grenze bei MAX_SCENES gilt unabhaengig davon.
        completed = (world.scene_number >= MAX_SCENES
                     or (turn.can_end and world.scene_number >= MIN_SCENES))

        # Der gerenderte Zustand ist beim Debuggen wertvoller als alles
        # andere: an ihm sieht man, was das Modell im NAECHSTEN Zug
        # tatsaechlich zu sehen bekommt.
        self._log_block("STATE", world.render())

        narration, image_prompt = schema.scene_text(turn)
        return Beat(
            narration=narration,
            image_prompt=image_prompt,
            number=world.scene_number,
            max_scenes=MAX_SCENES,
            completed=completed,
        )

    def close(self) -> None:
        self.log.close()

    # ------------------------------------------------------------ intern

    def _intent(self, world: World) -> str:
        """Eine anwesende Figur fuer sich entscheiden lassen.

        Gewaehlt wird der erste aktive Charakter am Spielerknoten. Gibt es
        keinen, entfaellt die Stufe ersatzlos - dann steht dort niemand, der
        etwas wahrnehmen koennte.

        DER SINN DER GANZEN STUFE: Diese Figur bekommt NICHT den Weltzustand,
        sondern world.render_for(sie) - nur ihren eigenen Knoten, ihr eigenes
        Ziel, ihr eigenes Gedaechtnis. Ihr Nichtwissen ist damit eine
        Eigenschaft des Kontextfensters und keine Bitte im Prompt. Sie kann
        nicht "vergessen", was sie nicht wissen soll, weil es nie da war.

        Kein Thinking noetig: kleine Frage, drei Felder, kleine Antwort.

        Rueckgabe ist bereits der fertige Textblock fuer den Resolver -
        leer, wenn niemand da war. So bleibt advance() frei von Sonderfaellen.
        """
        companions = world.companions()
        if not companions:
            return ""

        char = companions[0]
        intent_cls = schema.intent_model(world.exits_from(char.at) or ("stay",))

        system = self._system(self.prompts["intent.txt"], intent_cls)
        user = world.render_for(char)

        self._log_block(f"INTENT / {char.id} context", user)

        intent = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            intent_cls)

        self._log_block("INTENT / result", intent.model_dump_json(indent=2))

        # Als Klartext an den Resolver, nicht als JSON - aus demselben
        # Grund, aus dem der Zustand als Klartext geht (siehe
        # World.render()). Der Resolver soll das lesen, nicht spiegeln.
        return (f"CHARACTER INTENT ({char.id} {char.name}):\n"
                f"  wants to: {intent.intent}\n"
                f"  says: {intent.utterance or '(nothing)'}\n"
                f"  moving to: {intent.move_to}\n\n")

    def _system(self, prompt: str, model_cls) -> str:
        """Spielsystem-Text plus die Feldbeschreibung aus dem Schema.

        Die Grammatik erzwingt die Form, aber das Modell sieht sie nie - sie
        wirkt beim Server. Die BEDEUTUNG der Felder muss deshalb hier in den
        Prompt. Beides stammt aus derselben Klasse und kann nicht
        auseinanderlaufen.
        """
        return (prompt.rstrip()
                + "\n\n"
                + "--------------------------------------------------\n"
                + "OUTPUT FIELDS\n"
                + "--------------------------------------------------\n\n"
                + schema.describe(model_cls))

    def _log_block(self, title: str, body: str) -> None:
        self.log.write(f"===== {title} =====\n{body}\n\n")
        # Sofort schreiben: bei einem Absturz ist gerade der letzte Block
        # der interessante, und der stuende sonst noch im Puffer.
        self.log.flush()

    def _log_thinking(self) -> None:
        if self.engine.last_thinking:
            self._log_block("THINKING", self.engine.last_thinking.strip())


def _log_path(start_prompt: str) -> Path:
    """Dateiname aus dem Start-Prompt: bereinigt, gekuerzt, kollisionsfrei."""
    slug = re.sub(r"[^\w\s.-]", "", start_prompt, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "_", slug)[:80].strip("._-") or "untitled"

    path, n = DEBUG_DIR / f"{slug}.txt", 2
    while path.exists():
        path = DEBUG_DIR / f"{slug}-{n}.txt"
        n += 1
    return path

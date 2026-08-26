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
    game_prompts/<name>/decide.txt    eine Figur entscheidet fuer sich
    game_prompts/<name>/resolve.txt   loest EINEN Akteurzug zu einem Delta auf
    game_prompts/<name>/narrate.txt   erzaehlt, was der Spieler wahrnehmen konnte

Keine dieser Dateien enthaelt eine Feldliste. Die kommt zur Laufzeit aus
schema.describe() - wer sie in die Datei schriebe, haette die Drift wieder
eingebaut, gegen die der ganze Umbau geht.

=== Die Runde (agentische NPCs) ===

Eine Runde besteht aus mehreren Akteurzuegen, nicht mehr aus einem: zuerst
der Spieler, dann jeder aktive NPC in stabiler Reihenfolge (DECIDE -> aus
seiner eigenen, gefilterten Wahrnehmung heraus; RESOLVE -> mit vollem
Weltwissen), zuletzt EIN Erzaehl-Aufruf (NARRATE), der nur die Ereignisse
sieht, die am Spielerknoten passiert sind. Der Resolver erzaehlt selbst
nichts mehr - er liefert nur noch strukturierte events (Ort + Klausel), aus
denen sich Erinnerungen und Erzaehlmaterial deterministisch ergeben (siehe
state.World.apply_turn/visible). Die ganze Runde laeuft auf einer Kopie der
Welt (World.copy()) und wird erst nach erfolgreichem NARRATE committet -
schlaegt etwas davor fehl, ist self.world exakt wie vorher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import llm
import schema
import state
from state import World

HERE = Path(__file__).resolve().parent
DEBUG_DIR = HERE / "debug"
PROMPTS_DIR = HERE / "game_prompts"

# Die vier Dateien, aus denen ein Spielsystem besteht. Fehlt eine, ist der
# Ordner keins - available_systems() zeigt ihn dann gar nicht erst an.
PARTS = ("init.txt", "decide.txt", "resolve.txt", "narrate.txt")

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

    Vollstaendig heisst: der Ordner enthaelt alle vier Dateien. Ein halb
    angelegtes System taucht gar nicht erst auf - besser, als es waehlbar
    zu machen und erst beim ersten Aufruf zu scheitern.
    """
    try:
        return sorted(d for d in PROMPTS_DIR.iterdir()
                      if d.is_dir() and all((d / p).is_file() for p in PARTS))
    except OSError:
        return []


def estimate_tokens(system_dir: Path) -> int | None:
    """Ungefaehre Tokenzahl aller vier Dateien zusammen.

    Die Summe, nicht das Maximum: sie ist zwar keine Kontextgroesse (die
    vier Prompts laufen nie gleichzeitig), aber ein ehrliches Mass fuer den
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

    def begin(self, on_actor=None) -> Beat:
        """Die Welt erzeugen und Szene 1 zurueckgeben.

        Ein einziger Aufruf, der beides liefert. Frueher schickte das
        Programm dafuer einen erfundenen ersten Spielerzug ("Beginne.
        Erzeuge die erste Szene.") - eine Nachricht, die so tat, als haette
        jemand etwas eingegeben.

        on_actor existiert NUR der Signatur wegen (siehe advance()) - hier
        gibt es noch keine Runde und keine NPC-Zuege, ueber die main.py
        etwas anzeigen koennte. Der Parameter bleibt ungenutzt, damit
        main.py begin() und advance() einheitlich aufrufen kann, ohne
        zwischen beiden unterscheiden zu muessen.
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
        self.world.remember(narration)

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

    def advance(self, player_input: str, on_actor=None) -> Beat:
        """Eine Runde spielen: Spieler, dann jeder aktive NPC, dann Erzaehlung.

        Laeuft komplett auf einer KOPIE der Welt (World.copy()) und
        committet sie erst hier am Ende, NACH erfolgreichem NARRATE. Das ist
        strenger als die fruehere Zusage "der Chatverlauf bleibt konsistent":
        eine Runde besteht jetzt aus bis zu zehn Modellaufrufen, und ein
        Abbruch mittendrin darf keine halb gezogene Welt hinterlassen, in
        der zwei von drei NPCs schon gehandelt haben.

        Fehlerisolierung nach Wichtigkeit (siehe SceneError-Faelle unten):
        scheitert der Spieler-RESOLVE oder NARRATE, ist die ganze Runde
        verworfen und der Spieler darf es nochmal versuchen. Scheitert
        DECIDE oder RESOLVE fuer EINEN NPC, setzt NUR dieser NPC aus - bei
        bis zu vier NPCs pro Runde macht Null-Toleranz das Spiel unspielbar,
        und ein NPC, der einmal nicht handelt, ist im Ergebnis nicht von
        einem NPC zu unterscheiden, der abwartet.

        on_actor ist optional und wird - wenn uebergeben - mit einem
        Klartext-Label aufgerufen, sobald ein NPC an der Reihe ist ("Vogel
        is acting"). Passt auf main.py's ui.Status.update() als
        on_actor=lambda label: status.update(label=label).
        """
        if self.world is None:                      # pragma: no cover
            raise SceneError("begin() was never called.")
        world = self.world.copy()
        log: list[str] = []   # abgelehnte Operationen, fuers Debug-Log

        # --- 1. Spieler-Resolve ---
        try:
            delta = self._resolve(
                world, actor_id="player", actor_node=world.player_at,
                action_block=self._resolve_block_player(player_input))
        except Exception as e:
            raise SceneError(str(e)) from e
        log += world.apply_turn("player", delta)

        # --- 2. NPC-Zuege, stabile INIT-Reihenfolge ---
        for npc in world.active_npcs_in_order():
            if npc.status != "active":
                # Kann durch eine FRUEHERE Figur in DERSELBEN Runde
                # deaktiviert worden sein (status_changes) - active_npcs_
                # in_order() wurde vor der Schleife einmal ausgewertet, npc
                # ist aber eine lebende Referenz in world.characters, ihr
                # .status ist also aktuell.
                continue

            if on_actor:
                on_actor(f"{npc.name} is acting")

            try:
                decision = self._decide(world, npc)
            except Exception as e:
                self._log_block(f"DECIDE {npc.id} failed - skipped", str(e))
                continue

            npc.aim = decision.aim

            try:
                delta = self._resolve(
                    world, actor_id=npc.id, actor_node=npc.at,
                    action_block=self._resolve_block_npc(npc, decision))
            except Exception as e:
                self._log_block(f"RESOLVE {npc.id} failed - skipped", str(e))
                continue
            log += world.apply_turn(npc.id, delta)

        if log:
            self._log_block("REJECTED", "\n".join(log))

        # --- 3. Erzaehlen ---
        try:
            narrate = self._narrate(world, player_input)
        except Exception as e:
            raise SceneError(str(e)) from e

        world.scene_number += 1
        # Der Client entscheidet ueber das Ende, nicht das Modell. can_end
        # ist ein Vorschlag: er zaehlt erst ab MIN_SCENES, und die harte
        # Grenze bei MAX_SCENES gilt unabhaengig davon.
        completed = (world.scene_number >= MAX_SCENES
                     or (narrate.can_end and world.scene_number >= MIN_SCENES))

        narration, image_prompt = schema.narration_text(narrate)
        world.remember(narration)

        self.world = world   # erst jetzt committen

        # Der gerenderte Zustand ist beim Debuggen wertvoller als alles
        # andere: an ihm sieht man, was das Modell in der NAECHSTEN Runde
        # tatsaechlich zu sehen bekommt.
        self._log_block("STATE", world.render())

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

    def _resolve(self, world: World, actor_id: str, actor_node: str,
                action_block: str):
        """EINEN Akteurzug (Spieler oder ein NPC) zu einem Delta aufloesen.

        resolve_cls wird bei JEDEM Aufruf frisch gebaut, nicht einmal pro
        Runde: node_ids/active_ids/die Ausgaenge des Akteurs koennen sich
        MITTEN in der Runde aendern (ein fruehrer NPC kann einen anderen per
        status_changes deaktivieren - die Grammatik des naechsten Akteurs
        darf diese Figur dann nicht mehr als CharId anbieten).
        """
        resolve_cls = schema.resolve_model(
            world.node_ids(), world.active_ids(), world.exits_from(actor_node))

        system = self._system(self.prompts["resolve.txt"], resolve_cls)
        user = f"WORLD STATE:\n{world.render()}\n\n{action_block}"

        self._log_block(f"RESOLVE {actor_id} / user", user)

        result = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            resolve_cls)

        self._log_thinking()
        self._log_block(f"RESOLVE {actor_id} / result",
                        result.model_dump_json(indent=2))
        return result

    def _resolve_block_player(self, player_input: str) -> str:
        return f"ACTING: the player\nPLAYER ACTION:\n{player_input}\n"

    def _resolve_block_npc(self, npc, decision) -> str:
        return (f"ACTING: {npc.id} {npc.name}\n"
                f"THEIR INTENT: {decision.intent}\n"
                f"THEY SAY: {decision.utterance or '(nothing)'}\n"
                f"THEY MOVE TO: {decision.move_to}\n")

    def _decide(self, world: World, npc):
        """Eine Figur fuer sich entscheiden lassen, aus IHRER Wahrnehmung
        allein.

        DER SINN DER GANZEN STUFE: Diese Figur bekommt NICHT den
        Weltzustand, sondern world.render_for(sie) - nur ihren eigenen
        Knoten, ihren eigenen Antrieb, ihr eigenes Gedaechtnis. Ihr
        Nichtwissen ist damit eine Eigenschaft des Kontextfensters und
        keine Bitte im Prompt. Sie kann nicht "vergessen", was sie nicht
        wissen soll, weil es nie da war.
        """
        decide_cls = schema.decide_model(world.exits_from(npc.at) or ("stay",))

        system = self._system(self.prompts["decide.txt"], decide_cls)
        user = world.render_for(npc)

        self._log_block(f"DECIDE / {npc.id} context", user)

        result = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            decide_cls)

        self._log_thinking()
        self._log_block(f"DECIDE / {npc.id} result",
                        result.model_dump_json(indent=2))
        return result

    def _narrate(self, world: World, player_input: str):
        """Aus den fuer den Spieler SICHTBAREN Ereignissen dieser Runde eine
        Szene machen.

        Sichtbar heisst: state.visible(world.round_log) - nur Eintraege, bei
        denen der Ort des Events mit der Spielerposition ZUM ZEITPUNKT des
        Events uebereinstimmt. Ist die Liste leer, bekommt NARRATE einen
        ausdruecklichen Hinweis darauf, statt zu schweigen - sonst waere ein
        leerer Abschnitt eine Einladung, sich etwas auszudenken.
        """
        narrate_cls = schema.narrate_model()
        system = self._system(self.prompts["narrate.txt"], narrate_cls)

        visible_events = state.visible(world.round_log)
        lines = [f"YOUR PLACE:\n{world.render_player_place()}", ""]
        if visible_events:
            lines.append("WHAT HAPPENED HERE THIS ROUND:")
            lines.extend(f"  {e.clause}" for e in visible_events)
        else:
            lines.append("NOTHING VISIBLE HAPPENED HERE THIS ROUND.")
        lines.append(f"\nPLAYER ACTION:\n{player_input}")
        user = "\n".join(lines)

        self._log_block("NARRATE / user", user)

        result = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            narrate_cls)

        self._log_thinking()
        self._log_block("NARRATE / result", result.model_dump_json(indent=2))
        return result

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

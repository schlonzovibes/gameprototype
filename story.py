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

    game_prompts/<name>/init.txt            Weltgenerator - erzeugt NUR den
                                            Startraum, laeuft genau einmal
    game_prompts/<name>/decide.txt          eine agentische Figur entscheidet
                                            fuer sich
    game_prompts/<name>/resolve_player.txt  loest die Spieleraktion zu
                                            einem Delta auf
    game_prompts/<name>/resolve_agentic.txt loest den Zug einer agentischen
                                            Figur zu einem Delta auf
    game_prompts/<name>/narrate.txt         erzaehlt, was der Spieler
                                            wahrnehmen konnte

Keine dieser Dateien enthaelt eine Feldliste. Die kommt zur Laufzeit aus
schema.describe() - wer sie in die Datei schriebe, haette die Drift wieder
eingebaut, gegen die der ganze Umbau geht.

`game_prompts/<name>/npc_names.txt` ist eine fuenfte, OPTIONALE Datei (eine
Zeile je Name) - Fallback-Namenspool fuer den Fall, dass die
Charakterquote-Notbremse (siehe advance()) selbst nach mehreren
Wiederholungsversuchen keine Figur vom Modell bekommt und der Client eine
schmucklose NPC-Figur selbst anlegen muss. Fehlt die Datei, greift eine
kleine eingebaute Liste - ein Spielsystem ohne diese Datei bleibt spielbar.

=== Die wachsende Welt und die agentischen NPCs ===

INIT erzeugt nur einen Startraum, keinen Graphen - Figuren stehen 0-2 schon
darin. Die Levelgeometrie entsteht waehrend des Spielens: RESOLVE(mode=player)
darf einen neuen Raum vorschlagen, sobald die Erzaehlung ihn braucht
(persistent, kein Raum verschwindet je wieder). Ist WIRKLICH niemand da, muss
bis Runde 5 eine Figur auftauchen (siehe state.World.character_quota_status -
weiche Fuehrung, dann eine harte Notbremse mit begrenzten Versuchen).

Die ersten state.MAX_AGENTIC (Default 4, AIGAME_MAX_AGENTIC) eingefuehrten
Figuren werden agentisch: sie bekommen pro Runde einen eigenen Zug. Stirbt
eine, gibt sie ihren Slot frei und eine spaeter eingefuehrte rueckt nach.

Eine Runde: zuerst der Spieler (RESOLVE mode=player), dann je aktive
agentische Figur in Zug-Reihenfolge ein DECIDE (aus ihrer gefilterten
Wahrnehmung; die DECIDE-Aufrufe werden gegen vLLM GEFAECHERT, siehe
_decide_all) und danach - SERIELL, damit Figur N+1 das schon angewandte
Delta von Figur N sieht - ihr RESOLVE(mode=agentic) mit vollem Weltwissen.
Zuletzt EIN NARRATE, der nur die Ereignisse am Spielerort sieht. Der
Resolver erzaehlt nichts mehr - er liefert strukturierte events (Ort +
Klausel), aus denen sich Erinnerungen und Erzaehlmaterial deterministisch
ergeben (state.World.apply_turn/visible). Die ganze Runde laeuft auf einer
Kopie der Welt (World.copy()) und wird erst nach erfolgreichem NARRATE
committet - schlaegt etwas davor fehl, ist self.world exakt wie vorher.
Scheitert das DECIDE oder RESOLVE EINER agentischen Figur, entfaellt nur
ihr Zug.

Jede agentische Figur verfolgt zusaetzlich ein VERBORGENES Ziel (state.py) -
real und lenkend, aber nie ausgesprochen. Ein einfacher Teilstring-Check
nach jedem NARRATE protokolliert einen Verstoss ins Debug-Log, ohne das
Spiel abzubrechen.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import llm
import schema
import state
from state import World

HERE = Path(__file__).resolve().parent
DEBUG_DIR = HERE / "debug"
PROMPTS_DIR = HERE / "game_prompts"

# Die fuenf Pflichtdateien, aus denen ein Spielsystem besteht. Fehlt eine,
# ist der Ordner keins - available_systems() zeigt ihn dann gar nicht erst
# an. npc_names.txt ist bewusst NICHT hier drin - siehe Modul-Docstring,
# das bleibt optional.
PARTS = ("init.txt", "decide.txt", "resolve_player.txt",
        "resolve_agentic.txt", "narrate.txt")

# Greift nur, wenn ein Spielsystem kein eigenes npc_names.txt mitbringt
# (siehe Game.__init__) - kurz gehalten, weil die Notbremse sie hoechst
# selten je braucht.
_FALLBACK_NAMES = ["Vale", "Marrow", "Osei", "Brandt", "Iker"]

# Die Szenengrenze liegt jetzt hier, nicht mehr im Modelloutput. Frueher
# meldete das Modell game.scene_number und game.status, und der Client
# zaehlte "hilfsweise" selbst mit - er wusste die Zahl also die ganze Zeit
# und fragte trotzdem.
MAX_SCENES = 15

# Vor dieser Szene wird can_end ignoriert. Ein Modell, das die Situation
# gern aufloest, koennte sonst in Szene 3 abschliessen - technisch richtig,
# als Spiel wertlos.
MIN_SCENES = 8

# Wie viele DECIDE-Aufrufe der agentischen Figuren einer Runde gleichzeitig
# gegen vLLM laufen duerfen (_decide_all). Die Aufrufe sind unabhaengig -
# jeder liest nur world.render_for(seine Figur), keiner mutiert etwas
# Geteiltes. vLLM batcht mehrere Sequenzen und liefert dann aggregiert mehr
# Tokens/s. NICHT hoeher als --max-num-seqs des Servers setzen, sonst stauen
# sich die Requests statt zu batchen (AIGAME_VLLM_ARGS in docker-compose.yml).
LLM_CONCURRENCY = int(os.environ.get("AIGAME_LLM_CONCURRENCY", "4"))

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

    Vollstaendig heisst: der Ordner enthaelt alle fuenf Pflichtdateien aus
    PARTS. Ein halb angelegtes System taucht gar nicht erst auf - besser,
    als es waehlbar zu machen und erst beim ersten Aufruf zu scheitern.
    """
    try:
        return sorted(d for d in PROMPTS_DIR.iterdir()
                      if d.is_dir() and all((d / p).is_file() for p in PARTS))
    except OSError:
        return []


def estimate_tokens(system_dir: Path) -> int | None:
    """Ungefaehre Tokenzahl der Pflichtdateien zusammen.

    Die Summe, nicht das Maximum: sie ist zwar keine Kontextgroesse (die
    Prompts laufen nie gleichzeitig), aber ein ehrliches Mass fuer den
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

        # npc_names.txt ist optional (siehe Modul-Docstring) - anders als
        # die PARTS-Dateien also mit Rueckfall statt SystemExit.
        names_file = system_dir / "npc_names.txt"
        self.npc_names = (names_file.read_text(encoding="utf-8").split()
                          if names_file.is_file() else list(_FALLBACK_NAMES))

        # Die Welt entsteht erst in begin(). Bis dahin gibt es sie nicht -
        # und das soll man auch sehen koennen.
        self.world: World | None = None

        # Serialisiert die Schreibzugriffe aufs Debug-Log: _decide_all faechert
        # die DECIDE-Aufrufe in Threads, deren _log_block()-Aufrufe wuerden
        # sonst ineinanderlaufen.
        self._log_lock = threading.Lock()

        DEBUG_DIR.mkdir(exist_ok=True)
        log_path = _log_path(start_prompt)
        self.log = log_path.open("w", encoding="utf-8")
        # Zweites Log DANEBEN: der komplette Weltzustand als JSON, ein
        # Schnappschuss je abgeschlossener Runde (siehe _log_json). Gleicher
        # Dateiname mit Prefix "JSON_", also JSON_<slug>.txt - damit man die
        # ANHAeufung aller Werte ueber die Szenen nachvollziehen kann, nicht
        # nur den gerenderten Klartext des Haupt-Logs.
        self.json_log = log_path.with_name(
            f"JSON_{log_path.name}").open("w", encoding="utf-8")

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
        init_cls = schema.init_model()
        init_prompt = self.prompts["init.txt"].replace(
            "$START_PROMPT$", self.start_prompt)
        system = self._system(init_prompt, init_cls)
        user = f"START PROMPT:\n{self.start_prompt}"

        self._log_block("INIT / system", system)
        self._log_block("INIT / user", user)

        try:
            reply = self.engine.structured(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                init_cls)
        except Exception as e:
            self._log_thinking(getattr(e, "thinking", ""))
            raise SceneError(str(e)) from e

        init = reply.value
        self._log_thinking(reply.thinking)
        self._log_block("INIT / result", init.model_dump_json(indent=2))

        narration, image_prompt = schema.opening_text(init)

        self.world = World.from_init(init)
        self.world.scene_number = 1
        self.world.remember(narration)

        self._log_block("STATE after init", self.world.render())
        self._log_json("scene 1 / after init")

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
        """Eine Runde spielen: Spieler, dann die agentischen Zuege, dann
        Erzaehlung.

        Laeuft komplett auf einer KOPIE der Welt (World.copy()) und
        committet sie erst hier am Ende, NACH erfolgreichem NARRATE. Ein
        Abbruch mittendrin darf keine halb gezogene Welt hinterlassen.

        Fehlerisolierung nach Wichtigkeit: scheitert der Spieler-RESOLVE
        oder NARRATE, ist die ganze Runde verworfen und der Spieler darf es
        nochmal versuchen. Scheitert DECIDE oder RESOLVE einer EINZELNEN
        agentischen Figur, entfaellt nur ihr Zug - Spielerzug, die anderen
        Agenten und die Erzaehlung laufen trotzdem.

        on_actor ist optional und wird - wenn uebergeben - einmal je Runde
        mit einem Klartext-Label aufgerufen, sobald die agentische Phase
        beginnt ("Vogel is acting" bzw. "3 characters are acting"). Passt auf
        main.py's ui.Status.update() als
        on_actor=lambda label: status.update(label=label).
        """
        if self.world is None:                      # pragma: no cover
            raise SceneError("begin() was never called.")
        world = self.world.copy()
        log: list[str] = []   # abgelehnte Operationen, fuers Debug-Log
        turn_number = world.scene_number + 1

        # --- 1. Spieler-Resolve, mit Charakterquote-Notbremse ---
        # Ab Runde 6 (turn_number > 5) ist ein Charakter PFLICHT, solange
        # ueberhaupt keine Figur existiert (INIT setzt normalerweise schon
        # 0-2 in den Startraum, dann greift das nie). Bis zu zwei
        # Wiederholungsversuche generieren neue KANDIDATEN - erst der finale
        # wird angewendet (siehe unten, warum nicht frueher).
        direction = world.character_quota_status(turn_number)
        mandatory = turn_number > 5 and bool(direction)
        delta_p = None
        attempts = 0
        for attempts in range(1, 4):   # 1 regulaerer Versuch + max. 2 Wiederholungen
            try:
                delta_p = self._resolve(
                    world, actor_id="player", actor_node=world.player_at,
                    mode="player",
                    action_block=self._resolve_block_player(player_input),
                    direction=direction)
            except Exception as e:
                self._log_thinking(getattr(e, "thinking", ""))
                raise SceneError(str(e)) from e
            if not mandatory or delta_p.characters_introduced:
                break
            direction = world.character_quota_status(turn_number)  # bleibt MANDATORY

        # Generieren-dann-einmal-anwenden statt anwenden-und-bei-Bedarf-
        # wiederholen: ein zweiter RESOLVE-Aufruf wuerde sonst dieselbe
        # Spieleraktion gegen eine bereits von Versuch 1 veraenderte Welt
        # aufloesen (z.B. eine Tuer, die Versuch 1 schon geoeffnet hat) -
        # das koennte zu doppelten/widerspruechlichen Effekten fuehren.
        log += world.apply_turn("player", delta_p)

        # Quote NACH dem Anwenden geprueft, nicht nur "hat das Modell
        # ueberhaupt etwas versucht": ein Versuch kann durchaus vorliegen
        # (delta_p.characters_introduced nicht leer) und trotzdem an
        # apply_turn() scheitern (z.B. ein unbekannter Knoten) - dann waere
        # die Quote weiterhin unerfuellt, ohne dass die Notbremse greift,
        # wenn man nur auf den blossen Versuch schaute.
        if mandatory and world.character_quota_status(turn_number):
            fallback_id = world.spawn_fallback_character(self.npc_names)
            self._log_block("QUOTA FALLBACK",
                            f"client-spawned {fallback_id} after {attempts} attempt(s)")

        # --- 2. die agentischen Zuege ---
        # Je aktive agentische Figur (in Spawn-Reihenfolge) ein DECIDE, dann
        # ihr RESOLVE. Die DECIDE-Aufrufe sind unabhaengig und werden
        # gefaechert (_decide_all); die RESOLVE-Aufrufe laufen SERIELL, damit
        # Figur N+1 das schon angewandte Delta von Figur N sieht - der
        # "jemand war schneller"-Mechanismus (resolve_agentic.txt) haengt
        # daran. Ein gescheitertes DECIDE/RESOLVE entfernt nur den Zug dieser
        # einen Figur (Design-Doc §7).
        agents = world.agentic_actors()
        if agents:
            if on_actor:
                on_actor(self._acting_label(agents))
            decisions = self._decide_all(world, agents)
            for npc, decision in zip(agents, decisions):
                if decision is not None:
                    npc.aim = decision.aim
            for npc, decision in zip(agents, decisions):
                if decision is None:
                    continue
                if npc.status != "active":
                    self._log_block(
                        f"AGENTIC {npc.id} skipped",
                        f"no longer active ({npc.status}) after an earlier "
                        f"agent's turn this round")
                    continue
                try:
                    delta_n = self._resolve(
                        world, actor_id=npc.id, actor_node=npc.at,
                        mode="agentic",
                        action_block=self._resolve_block_npc(npc, decision))
                    log += world.apply_turn(npc.id, delta_n)
                except Exception as e:
                    self._log_thinking(getattr(e, "thinking", ""))
                    self._log_block(f"AGENTIC {npc.id} failed - skipped",
                                    str(e))

        if log:
            self._log_block("REJECTED", "\n".join(log))

        # --- 3. Erzaehlen ---
        try:
            narrate = self._narrate(world, player_input)
        except Exception as e:
            self._log_thinking(getattr(e, "thinking", ""))
            raise SceneError(str(e)) from e
        world.scene_number += 1
        # Der Client entscheidet ueber das Ende, nicht das Modell. can_end
        # ist ein Vorschlag: er zaehlt erst ab MIN_SCENES, und die harte
        # Grenze bei MAX_SCENES gilt unabhaengig davon.
        completed = (world.scene_number >= MAX_SCENES
                     or (narrate.can_end and world.scene_number >= MIN_SCENES))

        # schema.narration_text() statt das Feld direkt zu lesen: die
        # FELDNAMEN des Ausgabeformats sollen ausschliesslich in schema.py
        # stehen (siehe dort) - das gilt auch fuer die Leak-Pruefung unten.
        narration, image_prompt = schema.narration_text(narrate)
        self._check_narration_leak(narration, world)
        world.remember(narration)

        self.world = world   # erst jetzt committen

        # Der gerenderte Zustand ist beim Debuggen wertvoller als alles
        # andere: an ihm sieht man, was das Modell in der NAECHSTEN Runde
        # tatsaechlich zu sehen bekommt.
        self._log_block("STATE", world.render())
        self._log_json(f"scene {world.scene_number} / after round")

        return Beat(
            narration=narration,
            image_prompt=image_prompt,
            number=world.scene_number,
            max_scenes=MAX_SCENES,
            completed=completed,
        )

    def close(self) -> None:
        self.log.close()
        self.json_log.close()

    # ------------------------------------------------------------ intern

    def _resolve(self, world: World, actor_id: str, actor_node: str,
                mode: str, action_block: str, direction: str = ""):
        """EINEN Akteurzug (Spieler oder die agentische Figur) zu einem
        Delta aufloesen.

        resolve_cls wird bei JEDEM Aufruf frisch gebaut, nicht einmal pro
        Runde: node_ids/active_ids/die Ausgaenge des Akteurs koennen sich
        MITTEN in der Runde aendern (der Spielerzug kann neue Charaktere
        oder Raeume erzeugt haben - die Grammatik des agentischen Zuges muss
        das schon sehen).

        direction ist der optionale STORY-DIRECTION/MANDATORY-Regiehinweis
        (siehe advance()) - nur bei mode="player" jemals nicht-leer. Wird
        GETRENNT vom Weltzustand geloggt (nicht nur als Teil des user-
        Blocks): sonst liesse sich spaeter nicht mehr unterscheiden, was
        das Modell aus der Welt wusste und was ihm der Client zugefluestert
        hat.
        """
        resolve_cls = schema.resolve_model(
            world.node_ids(), world.active_ids(), world.exits_from(actor_node),
            mode)

        prompt_file = "resolve_player.txt" if mode == "player" else "resolve_agentic.txt"
        system = self._system(self.prompts[prompt_file], resolve_cls)

        direction_block = f"{direction}\n\n" if direction else ""
        user = f"WORLD STATE:\n{world.render()}\n\n{direction_block}{action_block}"

        if direction:
            self._log_block(f"RESOLVE {actor_id} / story direction", direction)
        self._log_block(f"RESOLVE {actor_id} / user", user)

        reply = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            resolve_cls)

        self._log_thinking(reply.thinking)
        self._log_block(f"RESOLVE {actor_id} / result",
                        reply.value.model_dump_json(indent=2))
        return reply.value

    def _resolve_block_player(self, player_input: str) -> str:
        return f"ACTING: the player\nPLAYER ACTION:\n{player_input}\n"

    def _resolve_block_npc(self, npc, decision) -> str:
        return (f"ACTING: {npc.id} {npc.name}\n"
                f"THEIR INTENT: {decision.intent}\n"
                f"THEY SAY: {decision.utterance or '(nothing)'}\n"
                f"THEY MOVE TO: {decision.move_to}\n")

    def _acting_label(self, agents) -> str:
        """Der on_actor-Text fuer die agentische Phase (main.py zeigt ihn im
        Spinner). Einmal pro Runde, nicht pro Figur - bei gefaecherten
        DECIDE-Aufrufen wuerde ein Text je Figur nur flackern."""
        if len(agents) == 1:
            return f"{agents[0].name} is acting"
        return f"{len(agents)} characters are acting"

    def _decide_all(self, world: World, agents: list):
        """Je Figur ihr DECIDE, gefaechert gegen vLLM. Rueckgabe: Liste in
        agents-Reihenfolge, None wo das DECIDE einer Figur gescheitert ist
        (nur ihr Zug entfaellt, Design-Doc §7).

        Die DECIDE-Aufrufe sind voneinander unabhaengig: jeder liest nur
        world.render_for(seine Figur) (rein lesend), keiner mutiert etwas
        Geteiltes - die aims werden erst NACH dieser Methode gesetzt, das
        world-mutierende RESOLVE laeuft danach seriell. structured() ist nach
        dem M4-Umbau re-entrant, _log_block() ist gelockt.

        Ein einzelner Agent laeuft ohne Pool - dann ist der Pfad byteweise
        der von vor dem Umbau.
        """
        if len(agents) == 1:
            return [self._decide_safe(world, agents[0])]

        workers = min(len(agents), LLM_CONCURRENCY)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._decide_safe, world, npc)
                       for npc in agents]
            return [f.result() for f in futures]

    def _decide_safe(self, world: World, npc):
        """_decide() mit Fehlerisolierung: ein gescheitertes DECIDE gibt None
        zurueck (und wird geloggt), statt die ganze Faecherung zu reissen."""
        try:
            return self._decide(world, npc)
        except Exception as e:
            self._log_thinking(getattr(e, "thinking", ""))
            self._log_block(f"DECIDE {npc.id} failed - skipped", str(e))
            return None

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

        reply = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            decide_cls)

        self._log_thinking(reply.thinking)
        self._log_block(f"DECIDE / {npc.id} result",
                        reply.value.model_dump_json(indent=2))
        return reply.value

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

        reply = self.engine.structured(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            narrate_cls)

        self._log_thinking(reply.thinking)
        self._log_block("NARRATE / result", reply.value.model_dump_json(indent=2))
        return reply.value

    def _check_narration_leak(self, narration: str, world: World) -> None:
        """Ist ein verborgenes Figuren-Ziel oder die Story-Richtung woertlich
        in die Erzaehlung durchgesickert?

        Die eigentliche Pruefung sitzt in World.secret_leaked() - story.py
        fragt nur, ohne den Feldnamen selbst zu kennen (siehe dort, warum das
        absichtlich getrennt ist). Ein Treffer ist kein Spielabbruch: er wird
        sichtbar ins Debug-Log geschrieben, damit man ihn beim Playtesting
        nachschaerfen kann, statt dass er unbemerkt durchrutscht.
        """
        if world.secret_leaked(narration):
            self._log_block("HIDDEN TARGET LEAK",
                            "the narration names the hidden shared target verbatim")

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
        # Unter dem Lock: parallele _decide()-Threads (siehe _decide_all)
        # schreiben sonst verschraenkt. Blockgranular - die Bloecke eines
        # Threads koennen mit denen eines anderen abwechseln, jeder traegt
        # aber seine npc-Id im Titel.
        with self._log_lock:
            self.log.write(f"===== {title} =====\n{body}\n\n")
            # Sofort schreiben: bei einem Absturz ist gerade der letzte Block
            # der interessante, und der stuende sonst noch im Puffer.
            self.log.flush()

    def _log_json(self, label: str) -> None:
        """Den vollen Weltzustand als JSON in die JSON_<slug>.txt schreiben.

        Ein Schnappschuss je abgeschlossener Runde (nach dem kompletten
        LLM-Durchlauf; die Bilderzeugung danach aendert den Zustand nicht
        mehr). Die Datei ist kein einzelnes JSON-Dokument, sondern eine Kette
        aus "===== label ====="-Bloecken mit je EINEM vollstaendigen,
        eingerueckten Objekt - so sieht man, wie Knoten, Figuren, Marks,
        Facts usw. ueber die Szenen anwachsen.

        dataclasses.asdict() klappt den ganzen World-Baum (Node, Character,
        Exit) auf; round_log traegt NamedTuples, die per _asdict() zu
        benannten Feldern werden statt zu positionellen Listen.
        ensure_ascii=False haelt die Spielsprache lesbar.
        """
        if self.world is None:                       # pragma: no cover
            return
        snapshot = asdict(self.world)
        snapshot["round_log"] = [e._asdict() for e in self.world.round_log]
        self.json_log.write(f"===== {label} =====\n")
        self.json_log.write(json.dumps(snapshot, indent=2, ensure_ascii=False))
        self.json_log.write("\n\n")
        self.json_log.flush()

    def _log_thinking(self, thinking: str) -> None:
        # THINKING steht IMMER da - eine fehlende Sektion waere sonst nicht
        # von "Modell hat nicht gedacht" zu unterscheiden. Leer, obwohl THINK
        # an ist, ist selbst ein Debug-Signal (Reasoning-Parser aus? Grammatik
        # wuergt das Denken? Denkprozess ins Token-Limit gelaufen?). Wird auch
        # im Fehlerpfad aufgerufen (mit getattr(e, "thinking", "")), damit ein
        # verrannter Denkprozess sichtbar ist, statt mit der Exception zu
        # verschwinden. thinking kommt jetzt aus dem Reply bzw. der Exception,
        # nicht mehr aus einem Engine-Attribut - das vertraegt parallele Calls.
        thinking = (thinking or "").strip()
        if thinking:
            self._log_block("THINKING", thinking)
        elif llm.THINK:
            self._log_block("THINKING", "(leer - kein Denkprozess empfangen)")
        else:
            self._log_block("THINKING", "(deaktiviert - AIGAME_THINK=0)")


def _log_path(start_prompt: str) -> Path:
    """Dateiname aus dem Start-Prompt: bereinigt, gekuerzt, kollisionsfrei."""
    slug = re.sub(r"[^\w\s.-]", "", start_prompt, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "_", slug)[:80].strip("._-") or "untitled"

    path, n = DEBUG_DIR / f"{slug}.txt", 2
    while path.exists():
        path = DEBUG_DIR / f"{slug}-{n}.txt"
        n += 1
    return path

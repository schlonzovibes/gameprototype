"""Unittests fuer story.Game.advance() - Fehlerisolierung und die
Charakterquote-Notbremse ueber eine Runde, gegen einen FakeEngine statt
eines echten Sprachmodells.
"""

import tempfile
import threading
import time
import typing
import unittest
from pathlib import Path

import llm
import story
from state import Character, Exit, Node, World

SYSTEM_DIR = Path(__file__).resolve().parent.parent / "game_prompts" / "default"


def _reply(value):
    """FakeEngine-Antworten tragen jetzt die Reply-Huelle wie das echte
    llm.LLM.structured() - value plus (hier leere) Metadaten."""
    return llm.Reply(value=value, thinking="", tokens_per_sec=None)


def _fill(model_cls, overrides=None):
    """Eine minimal gueltige Instanz von model_cls bauen.

    Iteriert model_fields und setzt fuer jeden Typ einen harmlosen
    Platzhalter - str -> "", bool -> False, list -> [], Literal -> den
    ersten erlaubten Wert (bevorzugt "stay", falls vorhanden), verschachtelte
    BaseModel-Klassen rekursiv (ein leerer new_room.name heisst dabei
    automatisch "kein neuer Raum diese Runde" - genau der beabsichtigte
    Default).
    """
    overrides = overrides or {}
    kwargs = {}
    for name, f in model_cls.model_fields.items():
        if name in overrides:
            kwargs[name] = overrides[name]
            continue
        ann = f.annotation
        origin = typing.get_origin(ann)
        if origin is list:
            kwargs[name] = []
        elif origin is typing.Literal:
            args = typing.get_args(ann)
            kwargs[name] = "stay" if "stay" in args else args[0]
        elif ann is bool:
            kwargs[name] = False
        elif ann is str:
            kwargs[name] = ""
        else:
            kwargs[name] = _fill(ann)
    return model_cls(**kwargs)


class FakeEngine:
    """Ersatz fuer llm.LLM: beantwortet jeden structured()-Aufruf mit einer
    minimal gueltigen Instanz - ausser die Nachricht traegt einen Marker aus
    fail_on, dann wird eine RuntimeError geworfen (simuliert einen
    gescheiterten Modellaufruf)."""

    def __init__(self, fail_on: set[str] = frozenset()):
        self.fail_on = set(fail_on)
        self.calls: list[str] = []   # model_cls.__name__ je Aufruf

    def structured(self, messages, model_cls, *, call="", retries=1):
        self.calls.append(model_cls.__name__)
        user = messages[1]["content"]
        for marker in self.fail_on:
            if marker in user:
                raise llm.StructuredError(
                    f"forced failure for test (marker: {marker!r})")
        return _reply(_fill(model_cls))


def _world_with_agents(*names) -> World:
    """Eine Welt mit zwei verbundenen Knoten und je einer agentischen Figur
    pro Name (alle an n1). Ohne Namen: eine Figur namens Vogel."""
    world = World(
        language="en",
        nodes={
            "n1": Node(id="n1", name="Hall", anchor="stone floor",
                      exits=[Exit(to="n2", one_way=False, justification="")]),
            "n2": Node(id="n2", name="Cellar", anchor="damp walls",
                      exits=[Exit(to="n1", one_way=False, justification="")]),
        },
        characters={}, facts=[], player_at="n1",
    )
    for name in (names or ("Vogel",)):
        world.add_character(name, "n1", f"{name} wants out", is_agentic=True)
    return world


def _world_with_agentic() -> World:
    return _world_with_agents("Vogel")


class GameTestCase(unittest.TestCase):
    """Gemeinsames Setup: DEBUG_DIR auf ein Temp-Verzeichnis umgebogen,
    damit Testlaeufe keine Dateien im echten debug/-Ordner hinterlassen."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_debug_dir = story.DEBUG_DIR
        story.DEBUG_DIR = Path(self._tmp.name)

    def tearDown(self):
        story.DEBUG_DIR = self._orig_debug_dir
        self._tmp.cleanup()

    def _game(self, engine) -> story.Game:
        return story.Game(engine, "a test story", SYSTEM_DIR)


class AgenticFailureIsolationTest(GameTestCase):
    """Ein scheiternder agentischer Zug (DECIDE oder RESOLVE) darf die
    Runde nicht abbrechen - nur er selbst entfaellt."""

    def test_failed_decide_does_not_abort_round(self):
        engine = FakeEngine(fail_on={"YOU ARE c1"})
        game = self._game(engine)
        game.world = _world_with_agentic()

        acted = []
        beat = game.advance("look around", on_actor=acted.append)

        self.assertIsNotNone(beat)
        self.assertEqual(game.world.characters["c1"].aim, "",
                         "DECIDE never completed, aim must be unchanged")
        self.assertEqual(acted, ["Vogel is acting"])
        game.close()

    def test_failed_resolve_does_not_abort_round(self):
        engine = FakeEngine(fail_on={"ACTING: c1"})
        game = self._game(engine)
        game.world = _world_with_agentic()

        beat = game.advance("look around")

        self.assertIsNotNone(beat)
        # DECIDE selbst lief durch (nur RESOLVE scheiterte - unschaedlich,
        # siehe Game.advance()-Docstring): "Decide" muss unter den
        # Aufrufen sein, "ResolveAgentic" wurde zwar versucht aber warf.
        self.assertIn("Decide", engine.calls)
        self.assertIn("ResolveAgentic", engine.calls)
        self.assertEqual(game.world.characters["c1"].status, "active")
        game.close()


class MultiAgentRoundTest(GameTestCase):
    """Die agentische Phase geht ueber ALLE aktiven agentischen Figuren -
    je Figur ein DECIDE, dann seriell ihr RESOLVE. Fehler/Tod einer Figur
    lassen die anderen unberuehrt."""

    def test_every_active_agent_gets_a_decide_and_a_resolve(self):
        engine = FakeEngine()
        game = self._game(engine)
        game.world = _world_with_agents("Vogel", "Renner", "Kranz")

        game.advance("wait")

        self.assertEqual(engine.calls.count("Decide"), 3)
        self.assertEqual(engine.calls.count("ResolveAgentic"), 3)
        game.close()

    def test_one_failed_decide_isolates_only_that_agent(self):
        engine = FakeEngine(fail_on={"YOU ARE c2"})   # nur Renners DECIDE
        game = self._game(engine)
        game.world = _world_with_agents("Vogel", "Renner", "Kranz")

        game.advance("wait")

        self.assertEqual(engine.calls.count("Decide"), 3)          # alle drei versucht
        self.assertEqual(engine.calls.count("ResolveAgentic"), 2)  # c2 faellt aus
        self.assertEqual(game.world.characters["c2"].aim, "",
                         "c2s DECIDE warf, sein aim bleibt unveraendert")
        game.close()

    def test_decide_calls_of_a_round_run_concurrently(self):
        """Die DECIDE-Faecherung (_decide_all) schickt die Aufrufe
        gleichzeitig gegen vLLM - nicht seriell nacheinander."""
        active = {"now": 0, "peak": 0}
        lock = threading.Lock()

        class SlowEngine(FakeEngine):
            def structured(self, messages, model_cls, *, call="", retries=1):
                self.calls.append(model_cls.__name__)
                if model_cls.__name__ == "Decide":
                    with lock:
                        active["now"] += 1
                        active["peak"] = max(active["peak"], active["now"])
                    time.sleep(0.05)
                    with lock:
                        active["now"] -= 1
                return _reply(_fill(model_cls))

        engine = SlowEngine()
        game = self._game(engine)
        game.world = _world_with_agents("A", "B", "C")

        game.advance("wait")

        self.assertGreaterEqual(active["peak"], 2,
                                "DECIDE-Aufrufe liefen seriell statt gefaechert")
        game.close()

    def test_agent_killed_by_an_earlier_agent_is_skipped(self):
        class KillEngine(FakeEngine):
            def structured(self, messages, model_cls, *, call="", retries=1):
                self.calls.append(model_cls.__name__)
                user = messages[1]["content"]
                if model_cls.__name__ == "ResolveAgentic" and "ACTING: c1" in user:
                    SC = model_cls.model_fields["status_changes"].annotation.__args__[0]
                    kill = _fill(SC, {"character": "c2", "status": "dead"})
                    return _reply(_fill(model_cls, {"status_changes": [kill]}))
                return _reply(_fill(model_cls))

        engine = KillEngine()
        game = self._game(engine)
        game.world = _world_with_agents("Vogel", "Renner", "Kranz")

        game.advance("wait")

        self.assertEqual(game.world.characters["c2"].status, "dead")
        self.assertFalse(game.world.characters["c2"].is_agentic,
                         "ein toter Agent verlaesst den Pool")
        self.assertEqual(engine.calls.count("Decide"), 3)
        self.assertEqual(engine.calls.count("ResolveAgentic"), 2,
                         "c2s RESOLVE wird uebersprungen, c1 und c3 laufen")
        game.close()


class NarrateFailureTest(GameTestCase):
    def test_failed_narrate_leaves_world_unchanged(self):
        engine = FakeEngine(fail_on={"YOUR PLACE"})
        game = self._game(engine)
        world = World(
            language="en",
            nodes={"n1": Node(id="n1", name="Hall", anchor="stone", exits=[])},
            characters={}, facts=[], player_at="n1",
        )
        game.world = world
        before = _snapshot(world)

        with self.assertRaises(story.SceneError):
            game.advance("look around")

        self.assertEqual(_snapshot(game.world), before)
        self.assertIs(game.world, world, "self.world must not be reassigned")
        game.close()


class PhaseDirectorTest(GameTestCase):
    """Der Phasen-Regieblock (Szenenzahl + setup/commit/escalate) geht an
    RESOLVE (beide Modi) und NARRATE - aber NICHT an DECIDE."""

    def test_phase_boundaries(self):
        self.assertEqual(
            [story._phase(t) for t in (1, 4, 5, 9, 10, 15, 99)],
            ["setup", "setup", "commit", "commit", "escalate", "escalate",
             "escalate"])

    def test_block_reaches_resolve_and_narrate_not_decide(self):
        seen = {}

        class Capturing(FakeEngine):
            def structured(self, messages, model_cls, *, call="", retries=1):
                seen[model_cls.__name__] = messages[1]["content"]
                return _reply(_fill(model_cls))

        game = self._game(Capturing())
        game.world = _world_with_agents("Vogel")
        game.world.scene_number = 10          # -> turn 11 -> escalate

        game.advance("wait")

        for kind in ("ResolvePlayer", "ResolveAgentic", "Narrate"):
            self.assertIn("SCENE 11 of 15", seen[kind], kind)
            self.assertIn("PHASE escalate", seen[kind], kind)
        self.assertNotIn("PHASE", seen["Decide"])
        self.assertNotIn("SCENE 11", seen["Decide"])
        game.close()


class NormalizeInputTest(GameTestCase):
    """Die Spieler-Eingabe wird vor dem Ledger in die Ich-Perspektive
    gebracht; woertliche Rede in "..." bleibt; das Ergebnis erreicht
    RESOLVE und NARRATE."""

    class _NormEngine(FakeEngine):
        def __init__(self, text="Ich nehme das Radio", **kw):
            super().__init__(**kw)
            self._text = text
            self.seen: dict[str, str] = {}

        def structured(self, messages, model_cls, *, call="", retries=1):
            self.calls.append(model_cls.__name__)
            self.seen[model_cls.__name__] = messages[1]["content"]
            if model_cls.__name__ == "Normalized":
                return _reply(_fill(model_cls, {"text": self._text}))
            return _reply(_fill(model_cls))

    def test_fast_path_leaves_first_person_untouched(self):
        engine = FakeEngine()
        game = self._game(engine)
        game.world = _world_with_agents("Vogel")
        self.assertEqual(game._normalize_input("Ich nehme das Radio"),
                         "Ich nehme das Radio")
        self.assertNotIn("Normalized", engine.calls)
        game.close()

    def test_pure_quote_kept_verbatim(self):
        game = self._game(FakeEngine())
        game.world = _world_with_agents("Vogel")
        self.assertEqual(game._normalize_input('"Hände hoch!"'), '"Hände hoch!"')
        game.close()

    def test_du_phrasing_is_rewritten_via_model(self):
        engine = self._NormEngine("Ich nehme das Radio vom Tisch")
        game = self._game(engine)
        game.world = _world_with_agents("Vogel")
        self.assertEqual(game._normalize_input("Du nimmst das Radio vom Tisch"),
                         "Ich nehme das Radio vom Tisch")
        game.close()

    def test_normalized_text_reaches_resolve_and_narrate(self):
        engine = self._NormEngine("Ich hebe das Radio auf")
        game = self._game(engine)
        game.world = _world_with_agents("Vogel")
        game.advance("Nimm das Radio")
        self.assertIn("Ich hebe das Radio auf", engine.seen["ResolvePlayer"])
        self.assertIn("Ich hebe das Radio auf", engine.seen["Narrate"])
        game.close()


class DecidePlayerActionTest(GameTestCase):
    """Die frische (normalisierte) Spieleraktion wird jeder co-lokierten
    Figur in den DECIDE-Kontext gehaengt, damit sie darauf reagiert."""

    def _run(self, npc_at):
        seen = {}

        class Capturing(FakeEngine):
            def structured(self, messages, model_cls, *, call="", retries=1):
                self.calls.append(model_cls.__name__)
                if model_cls.__name__ == "Decide":
                    seen["ctx"] = messages[1]["content"]
                if model_cls.__name__ == "Normalized":
                    return _reply(_fill(model_cls, {"text": "Ich sage Vogel Bescheid"}))
                return _reply(_fill(model_cls))

        game = self._game(Capturing())
        game.world = _world_with_agents("Vogel")
        game.world.characters["c1"].at = npc_at
        game.advance('Vogel, geh zur Tür')
        game.close()
        return seen.get("ctx", "")

    def test_block_present_when_co_located(self):
        ctx = self._run("n1")            # Spieler @ n1, Vogel @ n1
        self.assertIn("THIS ROUND THE PLAYER JUST DID / SAID", ctx)
        self.assertIn("Ich sage Vogel Bescheid", ctx)

    def test_block_absent_when_apart(self):
        ctx = self._run("n2")            # Vogel @ n2, Spieler @ n1
        self.assertNotIn("THIS ROUND THE PLAYER JUST DID", ctx)


class StrandedEndingTest(GameTestCase):
    """Hat der Spieler den Kernschauplatz ueber eine Einbahn verlassen und
    ist scene_number >= MIN_SCENES, ist die Runde completed."""

    def test_stranded_after_min_scenes_completes(self):
        game = self._game(FakeEngine())
        world = World(
            language="en",
            nodes={
                "n1": Node(id="n1", name="Hut", anchor="stone",
                           exits=[Exit(to="n2", one_way=True,
                                       justification="the ledge broke")]),
                "n2": Node(id="n2", name="Path", anchor="ice", exits=[]),
            },
            characters={}, facts=[], player_at="n2",
            scene_number=story.MIN_SCENES,
        )
        game.world = world
        beat = game.advance("keep walking")
        self.assertTrue(beat.completed)
        game.close()


class QuotaNotbremseTest(GameTestCase):
    def test_mandatory_retry_stops_after_three_attempts_and_spawns_fallback(self):
        # scene_number=5 -> turn_number=6 > 5 -> MANDATORY; die Engine
        # liefert bei JEDEM ResolvePlayer-Aufruf ein leeres
        # characters_introduced, die Notbremse muss also alle drei
        # Versuche ausschoepfen und danach selbst einen NPC anlegen.
        engine = FakeEngine()
        game = self._game(engine)
        world = World(
            language="en",
            nodes={"n1": Node(id="n1", name="Hall", anchor="stone", exits=[])},
            characters={}, facts=[], player_at="n1", scene_number=5,
        )
        game.world = world

        game.advance("do nothing")

        player_resolve_calls = sum(1 for c in engine.calls if c == "ResolvePlayer")
        self.assertEqual(player_resolve_calls, 3)
        self.assertEqual(len(game.world.characters), 1)
        self.assertTrue(game.world.agentic_actors())   # der Fallback ist agentisch
        game.close()

    def test_soft_direction_does_not_force_retries(self):
        # turn_number klein genug fuer weiche Fuehrung (kein MANDATORY) -
        # ein einziger ResolvePlayer-Aufruf reicht, auch wenn die Engine
        # keine Charaktere einfuehrt.
        engine = FakeEngine()
        game = self._game(engine)
        world = World(
            language="en",
            nodes={"n1": Node(id="n1", name="Hall", anchor="stone", exits=[])},
            characters={}, facts=[], player_at="n1",
        )
        game.world = world

        game.advance("look around")

        player_resolve_calls = sum(1 for c in engine.calls if c == "ResolvePlayer")
        self.assertEqual(player_resolve_calls, 1)
        game.close()

    def test_one_valid_introduction_satisfies_the_quota(self):
        """Ein einziger gueltig eingefuehrter Charakter erfuellt die Quote
        (1 Figur bis Zug 5) - die Schleife bricht nach dem ersten Versuch
        ab, und die Notbremse legt KEINEN zusaetzlichen Fallback an."""
        class OneCharacterEngine(FakeEngine):
            def structured(self, messages, model_cls, *, call="", retries=1):
                self.calls.append(model_cls.__name__)
                if model_cls.__name__ == "ResolvePlayer":
                    NC = model_cls.model_fields["characters_introduced"].annotation.__args__[0]
                    one = _fill(NC, {"name": "Sailor", "at": "n1",
                                     "agenda_draft": "x", "agenda_target_hint": "x"})
                    return _reply(_fill(model_cls, {"characters_introduced": [one]}))
                return _reply(_fill(model_cls))

        engine = OneCharacterEngine()
        game = self._game(engine)
        world = World(
            language="en",
            nodes={"n1": Node(id="n1", name="Hall", anchor="stone", exits=[])},
            characters={}, facts=[], player_at="n1", scene_number=5,
        )
        game.world = world

        game.advance("do nothing")

        player_resolve_calls = sum(1 for c in engine.calls if c == "ResolvePlayer")
        self.assertEqual(player_resolve_calls, 1)
        self.assertEqual([c.name for c in game.world.characters.values()], ["Sailor"])
        game.close()


def _snapshot(world: World):
    """Ein deep-copy-unabhaengiger Vergleichswert: World ist ein reines
    dataclass, repr() ist deshalb ein verlaesslicher Fingerabdruck seines
    gesamten Inhalts (nodes/characters/facts/scene_number/recent)."""
    return repr(world)


if __name__ == "__main__":
    unittest.main()

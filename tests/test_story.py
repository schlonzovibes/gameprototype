"""Unittests fuer story.Game.advance() - Fehlerisolierung ueber eine Runde.

Kriterien E (ein scheiternder DECIDE setzt nur die eine Figur aus, die
Runde laeuft weiter) und F (ein scheiterndes NARRATE laesst self.world
byteweise unveraendert), gegen einen FakeEngine statt eines echten
Sprachmodells.
"""

import tempfile
import typing
import unittest
from pathlib import Path

import story
from state import Character, Exit, Node, World

SYSTEM_DIR = Path(__file__).resolve().parent.parent / "game_prompts" / "default"


def _fill(model_cls, overrides=None):
    """Eine minimal gueltige Instanz von model_cls bauen.

    Iteriert model_fields und setzt fuer jeden Typ einen harmlosen
    Platzhalter - str -> "", bool -> False, list -> [], Literal -> den
    ersten erlaubten Wert (bevorzugt "stay", falls vorhanden - eine echte
    Bewegung waere ein Seiteneffekt, den die meisten Tests nicht wollen),
    verschachtelte BaseModel-Klassen rekursiv.
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
        self.last_thinking = ""
        self.fail_on = set(fail_on)

    def structured(self, messages, model_cls, retries=1):
        user = messages[1]["content"]
        for marker in self.fail_on:
            if marker in user:
                raise RuntimeError(f"forced failure for test (marker: {marker!r})")
        return _fill(model_cls)


def _three_npc_world() -> World:
    return World(
        language="en",
        nodes={
            "n1": Node(id="n1", name="Hall", anchor="stone floor",
                      exits=[Exit(to="n2", one_way=False, justification="")]),
            "n2": Node(id="n2", name="Cellar", anchor="damp walls",
                      exits=[Exit(to="n1", one_way=False, justification="")]),
        },
        characters={
            "c1": Character(id="c1", name="Vogel", at="n1",
                            agenda="get out", aim="find the door"),
            "c2": Character(id="c2", name="Renner", at="n1",
                            agenda="hide", aim="stay quiet"),
            "c3": Character(id="c3", name="Katz", at="n1",
                            agenda="watch", aim="observe"),
        },
        facts=[], player_at="n1",
    )


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


class DecideFailureIsolationTest(GameTestCase):
    """Kriterium E."""

    def test_failed_decide_skips_only_that_npc(self):
        engine = FakeEngine(fail_on={"YOU ARE c2"})
        game = self._game(engine)
        game.world = _three_npc_world()

        acted = []
        beat = game.advance("look around", on_actor=acted.append)

        self.assertIsNotNone(beat)
        self.assertEqual(game.world.characters["c2"].aim, "stay quiet",
                         "c2's DECIDE never completed, aim must be unchanged")
        self.assertNotEqual(game.world.characters["c3"].aim, "observe",
                            "c3 should have decided+resolved normally")
        self.assertEqual(acted, ["Vogel is acting", "Renner is acting",
                                 "Katz is acting"])
        game.close()

    def test_failed_resolve_skips_only_that_npc(self):
        engine = FakeEngine(fail_on={"ACTING: c1"})
        game = self._game(engine)
        game.world = _three_npc_world()

        game.advance("look around")

        # c1's DECIDE succeeded (aim changed) but its RESOLVE failed - the
        # aim change on the discarded... no, COMMITTED copy is harmless and
        # expected (see story.Game.advance docstring).
        self.assertNotEqual(game.world.characters["c1"].aim, "find the door")
        self.assertNotEqual(game.world.characters["c3"].aim, "observe")
        game.close()


class NarrateFailureTest(GameTestCase):
    """Kriterium F."""

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


def _snapshot(world: World):
    """Ein deep-copy-unabhaengiger Vergleichswert: World ist ein reines
    dataclass, repr() ist deshalb ein verlaesslicher Fingerabdruck seines
    gesamten Inhalts (nodes/characters/facts/scene_number/recent)."""
    return repr(world)


if __name__ == "__main__":
    unittest.main()

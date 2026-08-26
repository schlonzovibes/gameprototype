"""Reine Unittests fuer state.py - kein Sprachmodell, kein Game.

Deckt die Abnahmekriterien A-D aus dem Brief "REPLOT - AGENTISCHE
NPC-RUNDE" ab, plus ein paar Randfaelle, die beim Umsetzen selbst
aufgefallen sind (actor_char in receivers, moves gegen den Akteur
selbst, copy() setzt round_log zurueck).
"""

import types
import unittest

from state import Character, Exit, Node, RoundEntry, World, visible


def _world():
    """Eine kleine, feste Welt fuer alle Tests: n1 <-> n2, n3 hat keinen
    Ausgang. c1 an n1, c2 an n3."""
    return World(
        language="en",
        nodes={
            "n1": Node(id="n1", name="Hall", anchor="stone floor",
                      exits=[Exit(to="n2", one_way=False, justification="")]),
            "n2": Node(id="n2", name="Cellar", anchor="damp walls",
                      exits=[Exit(to="n1", one_way=False, justification="")]),
            "n3": Node(id="n3", name="Attic", anchor="dusty beams", exits=[]),
        },
        characters={
            "c1": Character(id="c1", name="Vogel", at="n1",
                            agenda="get out", aim="find the door"),
            "c2": Character(id="c2", name="Renner", at="n3",
                            agenda="hide", aim="stay quiet"),
        },
        facts=["the power is out"],
        player_at="n1",
    )


def _delta(**overrides):
    """Ein Resolve-Delta-Stub. Defaults sind ein leeres, folgenloses Delta -
    einzelne Felder ueberschreiben, was ein Test gerade braucht."""
    base = dict(actor_move_to="stay", moves=[], status_changes=[],
               marks_added=[], facts_added=[], events=[])
    base.update(overrides)
    return types.SimpleNamespace(**base)


class RenderForTest(unittest.TestCase):
    """Kriterium A: render_for() zeigt weder fremde agenda/aim noch fremde
    Knoten."""

    def test_excludes_other_agenda_and_other_nodes(self):
        world = _world()
        out = world.render_for(world.characters["c1"])
        self.assertNotIn("hide", out)          # c2s agenda
        self.assertNotIn("stay quiet", out)    # c2s aim
        self.assertNotIn("Attic", out)         # c2s Knoten
        self.assertNotIn("dusty beams", out)   # dessen anchor
        # Die eigenen Werte muessen dagegen drinstehen.
        self.assertIn("get out", out)
        self.assertIn("find the door", out)


class ApplyTurnMoveTest(unittest.TestCase):
    """Kriterium B: eine Bewegung ohne echte Kante wird abgelehnt."""

    def test_rejects_move_without_edge(self):
        world = _world()
        rejected = world.apply_turn("player", _delta(actor_move_to="n3"))
        self.assertEqual(world.player_at, "n1")
        self.assertTrue(any("no such exit" in r for r in rejected), rejected)

    def test_npc_move_without_edge_is_rejected_too(self):
        world = _world()
        # c2 steht an n3, das hat gar keinen Ausgang.
        rejected = world.apply_turn("c2", _delta(actor_move_to="n1"))
        self.assertEqual(world.characters["c2"].at, "n3")
        self.assertTrue(any("no such exit" in r for r in rejected), rejected)

    def test_moves_cannot_target_the_acting_character(self):
        world = _world()
        move = types.SimpleNamespace(character="c1", to="n2")
        rejected = world.apply_turn("c1", _delta(moves=[move]))
        self.assertEqual(world.characters["c1"].at, "n1")
        self.assertTrue(any("cannot re-move" in r for r in rejected), rejected)


class EventPerceptionTest(unittest.TestCase):
    """Kriterium C: ein Event erreicht nur die Anwesenden am eigenen Ort."""

    def test_event_reaches_only_perceivers_at_that_node(self):
        world = _world()
        world.characters["c1"].at = "n3"   # c1 zieht zu c2 nach n3
        event = types.SimpleNamespace(node="n3", clause="a shelf creaks")
        world.apply_turn("c2", _delta(events=[event]))
        self.assertIn("a shelf creaks", world.characters["c1"].memory)

    def test_event_does_not_reach_absent_character(self):
        world = _world()
        event = types.SimpleNamespace(node="n1", clause="a door slams")
        world.apply_turn("player", _delta(events=[event]))
        self.assertNotIn("a door slams", world.characters["c2"].memory)

    def test_acting_character_always_perceives_own_event(self):
        """Auch wenn das Event an einem ANDEREN Knoten passiert als dem, an
        dem der Akteur gerade steht (z.B. eine Fernwirkung)."""
        world = _world()
        event = types.SimpleNamespace(node="n1", clause="the alarm trips")
        world.apply_turn("c2", _delta(events=[event]))   # c2 steht an n3
        self.assertIn("the alarm trips", world.characters["c2"].memory)

    def test_events_beyond_max_are_dropped_and_logged(self):
        world = _world()
        events = [types.SimpleNamespace(node="n1", clause=f"event {i}")
                 for i in range(6)]
        rejected = world.apply_turn("player", _delta(events=events))
        self.assertEqual(len(world.round_log), 4)   # MAX_EVENTS
        self.assertTrue(any("dropped" in r for r in rejected), rejected)


class VisibleTest(unittest.TestCase):
    """Kriterium D: visible() filtert nach der Spielerposition zum
    Zeitpunkt des jeweiligen Events."""

    def test_filters_by_player_position_at_event_time(self):
        log = [RoundEntry("c1", "n7", "a", "n3"),
              RoundEntry("c1", "n7", "b", "n7")]
        self.assertEqual(visible(log), [RoundEntry("c1", "n7", "b", "n7")])

    def test_empty_log_stays_empty(self):
        self.assertEqual(visible([]), [])


class WorldCopyTest(unittest.TestCase):
    """copy() ist eine tiefe Kopie und setzt round_log zurueck."""

    def test_copy_is_independent_and_resets_round_log(self):
        world = _world()
        world.round_log.append(RoundEntry("player", "n1", "x", "n1"))

        clone = world.copy()
        self.assertEqual(clone.round_log, [])
        self.assertEqual(world.round_log, [RoundEntry("player", "n1", "x", "n1")])

        clone.characters["c1"].aim = "CHANGED"
        self.assertEqual(world.characters["c1"].aim, "find the door")


class ActiveNpcsInOrderTest(unittest.TestCase):
    """Stabile Reihenfolge = Einfuegereihenfolge, nicht sortiert."""

    def test_order_matches_insertion_order(self):
        world = _world()
        self.assertEqual([c.id for c in world.active_npcs_in_order()],
                         ["c1", "c2"])

    def test_inactive_characters_are_excluded(self):
        world = _world()
        world.characters["c1"].status = "dead"
        self.assertEqual([c.id for c in world.active_npcs_in_order()], ["c2"])


if __name__ == "__main__":
    unittest.main()

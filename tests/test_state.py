"""Reine Unittests fuer state.py - kein Sprachmodell, kein Game.

Deckt die Abnahmekriterien A-H aus dem Brief "REPLOT - WACHSENDE WELT UND
EIN EINZELNER AGENTISCHER NPC" ab, plus ein paar Randfaelle, die beim
Umsetzen selbst aufgefallen sind (actor_char in receivers, moves gegen den
Akteur selbst, copy() setzt round_log zurueck, new_room impliziert Ankunft).
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


def _no_room():
    """Ein leerer new_room-Stub - "kein neuer Raum diese Runde"."""
    return types.SimpleNamespace(name="", anchor="", one_way=False,
                                 justification="")


def _delta(**overrides):
    """Ein Resolve-Delta-Stub (mode="player"-Form: mit
    characters_introduced). Defaults sind ein leeres, folgenloses Delta -
    einzelne Felder ueberschreiben, was ein Test gerade braucht."""
    base = dict(new_room=_no_room(), actor_move_to="stay", moves=[],
               status_changes=[], marks_added=[], facts_added=[],
               characters_introduced=[], events=[])
    base.update(overrides)
    return types.SimpleNamespace(**base)


class RenderForTest(unittest.TestCase):
    """Kriterium (aus dem vorigen Brief, weiterhin gueltig): render_for()
    zeigt weder fremde agenda/aim noch fremde Knoten."""

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
    """Eine Bewegung ohne echte Kante wird abgelehnt."""

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
    """Ein Event erreicht nur die Anwesenden am eigenen Ort."""

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
    """visible() filtert nach der Spielerposition zum Zeitpunkt des
    jeweiligen Events."""

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


# ------------------------------------------------------------- A-H (neuer Brief)

def _init_stub(**over):
    """Ein INIT-Objekt-Stub: Startraum, leere Richtung, keine Figuren -
    einzelne Felder ueberschreiben, was ein Test braucht."""
    base = dict(language="en", start_node_name="Gas Station",
                start_node_anchor="fluorescent lit, wet floor",
                direction=types.SimpleNamespace(pull="", pressure=""),
                starting_characters=[])
    base.update(over)
    return types.SimpleNamespace(**base)


class FromInitTest(unittest.TestCase):
    """World.from_init() liefert einen Knoten, die abstrakte Richtung und
    0-2 Startfiguren."""

    def test_single_node_no_characters(self):
        world = World.from_init(_init_stub())
        self.assertEqual(list(world.nodes.keys()), ["n1"])
        self.assertEqual(world.characters, {})
        self.assertEqual(world.player_at, "n1")

    def test_direction_is_carried_over(self):
        world = World.from_init(_init_stub(
            direction=types.SimpleNamespace(pull="a way off this floor",
                                            pressure="the water keeps rising")))
        self.assertEqual(world.pull, "a way off this floor")
        self.assertEqual(world.pressure, "the water keeps rising")

    def test_starting_characters_land_in_n1_and_are_agentic(self):
        world = World.from_init(_init_stub(starting_characters=[
            types.SimpleNamespace(name="Rae", agenda_draft="rob the player",
                                  agenda_target_hint="the register"),
            types.SimpleNamespace(name="Bo", agenda_draft="warn the player off",
                                  agenda_target_hint="the door"),
        ]))
        self.assertEqual([c.name for c in world.characters.values()], ["Rae", "Bo"])
        self.assertTrue(all(c.at == "n1" for c in world.characters.values()))
        # Bis MAX_AGENTIC werden alle eingefuehrten Figuren agentisch, in
        # Spawn-Reihenfolge; jede mit ihrem eigenen hidden_target.
        self.assertEqual([c.name for c in world.agentic_actors()], ["Rae", "Bo"])
        self.assertEqual(world.characters["c1"].hidden_target, "the register")
        self.assertEqual(world.characters["c2"].hidden_target, "the door")
        self.assertEqual(world.character_quota_status(6), "")   # Quote erfuellt


class AddNodeTest(unittest.TestCase):
    """Kriterium B: World.add_node() vergibt fortlaufende Ids und
    verknuepft beidseitig, ausser bei one_way=True."""

    def test_sequential_ids_and_bidirectional_linking(self):
        world = _world()   # hat schon n1, n2, n3
        n4 = world.add_node("Roof", "tar paper", from_node="n1",
                            one_way=False, justification="")
        self.assertEqual(n4, "n4")
        self.assertIn("n4", [e.to for e in world.nodes["n1"].exits])
        self.assertIn("n1", [e.to for e in world.nodes["n4"].exits])

    def test_one_way_has_no_return_edge(self):
        world = _world()
        n4 = world.add_node("Shaft", "rusted ladder", from_node="n1",
                            one_way=True, justification="the ladder breaks")
        self.assertIn("n4", [e.to for e in world.nodes["n1"].exits])
        self.assertNotIn("n1", [e.to for e in world.nodes["n4"].exits])


class CanGrowTest(unittest.TestCase):
    """Kriterium C: can_grow() liefert False bei erreichter max_nodes-Grenze."""

    def test_false_at_max_nodes(self):
        world = _world()
        world.max_nodes = len(world.nodes)   # bereits am Limit (3)
        self.assertFalse(world.can_grow())

    def test_true_below_max_nodes(self):
        world = _world()
        world.max_nodes = len(world.nodes) + 1
        self.assertTrue(world.can_grow())


class AddCharacterTest(unittest.TestCase):
    """add_character() laesst bis max_agentic agentische Figuren zu und
    wirft erst beim Ueberschreiten."""

    def test_agentic_allowed_up_to_max_and_then_raises(self):
        world = _world()
        world.max_agentic = 2
        world.add_character("Vogel", "n1", "get out", is_agentic=True)
        world.add_character("Renner", "n1", "hide", is_agentic=True)   # noch ok
        self.assertEqual(world.agentic_count(), 2)
        with self.assertRaises(ValueError):
            world.add_character("Dritter", "n1", "watch", is_agentic=True)

    def test_dead_agentic_frees_a_slot(self):
        world = _world()
        world.max_agentic = 1
        vid = world.add_character("Vogel", "n1", "get out", is_agentic=True)
        # Vogel stirbt -> is_agentic wird False (apply_turn), Slot frei
        world.apply_turn("player", _delta(status_changes=[
            types.SimpleNamespace(character=vid, status="dead")]))
        self.assertEqual(world.agentic_count(), 0)
        rid = world.add_character("Renner", "n1", "hide", is_agentic=True)
        self.assertTrue(world.characters[rid].is_agentic)   # rueckt nach

    def test_new_character_aim_starts_empty(self):
        world = _world()
        cid = world.add_character("Stranger", "n1", "something", False)
        self.assertEqual(world.characters[cid].aim, "")


class CharacterQuotaStatusTest(unittest.TestCase):
    """Leer, sobald IRGENDEINE Figur existiert; sonst ein Hinweis -
    MANDATORY erst nach Zug 5."""

    def _empty_world(self):
        return World(language="en",
                     nodes={"n1": Node(id="n1", name="Hall", anchor="stone",
                                       exits=[])},
                     characters={}, facts=[], player_at="n1")

    def test_empty_when_any_character_exists(self):
        world = self._empty_world()
        world.add_character("One", "n1", "x", False)
        self.assertEqual(world.character_quota_status(1), "")
        self.assertEqual(world.character_quota_status(99), "")

    def test_soft_direction_when_nobody_and_early(self):
        status = self._empty_world().character_quota_status(4)
        self.assertNotEqual(status, "")
        self.assertNotIn("MANDATORY", status)   # Zug 4 <= 5, noch weiche Fuehrung

    def test_mandatory_when_nobody_after_turn_five(self):
        self.assertIn("MANDATORY", self._empty_world().character_quota_status(6))


class RenderForHiddenAimTest(unittest.TestCase):
    """render_for() enthaelt die "HIDDEN AIM"-Zeile NUR fuer eine agentische
    Figur mit gesetztem hidden_target, und nur in DEREN Kontext."""

    def test_hidden_aim_only_for_agentic(self):
        world = _world()
        agentic_id = world.add_character("Vogel", "n1", "get out", True)
        other_id = world.add_character("Renner", "n1", "hide", False)
        world.characters[agentic_id].hidden_target = "the ledger"

        out_agentic = world.render_for(world.characters[agentic_id])
        out_other = world.render_for(world.characters[other_id])

        self.assertIn("HIDDEN AIM", out_agentic)
        self.assertIn("the ledger", out_agentic)
        self.assertNotIn("HIDDEN AIM", out_other)
        self.assertNotIn("the ledger", out_other)

    def test_each_agentic_char_sees_only_its_own_hidden_target(self):
        world = _world()
        a = world.add_character("Vogel", "n1", "get out", True)
        b = world.add_character("Renner", "n1", "hide", True)
        world.characters[a].hidden_target = "the ledger"
        world.characters[b].hidden_target = "the back door"

        out_a = world.render_for(world.characters[a])
        out_b = world.render_for(world.characters[b])
        self.assertIn("the ledger", out_a)
        self.assertNotIn("the back door", out_a)
        self.assertIn("the back door", out_b)
        self.assertNotIn("the ledger", out_b)


class RenderNeverLeaksHiddenTargetTest(unittest.TestCase):
    """render() (voller Zustand, geht an RESOLVE + Spieler-Kontext) enthaelt
    ein hidden_target an keiner Stelle - nur render_for() der Figur selbst."""

    def test_render_excludes_hidden_target(self):
        world = _world()
        cid = world.add_character("Vogel", "n1", "get out", True)
        world.characters[cid].hidden_target = "the ledger"
        self.assertNotIn("the ledger", world.render())


class StoryDirectionVisibilityTest(unittest.TestCase):
    """pull/pressure gehen an RESOLVE (render) und DECIDE (render_for),
    aber NIE an NARRATE (render_player_place). render_player_place traegt
    ausserdem keine Knoten-Id mehr."""

    def _world_with_direction(self):
        world = _world()
        world.pull = "a way off this floor"
        world.pressure = "the water keeps rising"
        return world

    def test_resolve_and_decide_see_direction(self):
        world = self._world_with_direction()
        self.assertIn("a way off this floor", world.render())
        self.assertIn("the water keeps rising", world.render())
        self.assertIn("a way off this floor", world.render_for(world.characters["c1"]))

    def test_narrate_place_hides_direction_and_ids(self):
        world = self._world_with_direction()
        place = world.render_player_place()
        self.assertNotIn("a way off this floor", place)
        self.assertNotIn("the water keeps rising", place)
        self.assertNotIn("n1", place)   # keine Knoten-Id
        self.assertNotIn("n2", place)   # kein Exit-Id
        self.assertIn("Hall", place)    # aber der Name

    def test_leak_check_catches_pull_and_pressure(self):
        world = self._world_with_direction()
        self.assertTrue(world.secret_leaked(
            "You claw for a way off this floor as it groans."))
        self.assertTrue(world.secret_leaked(
            "The water keeps rising past your knees."))
        self.assertFalse(world.secret_leaked("The room is quiet."))


class NewRoomMechanicsTest(unittest.TestCase):
    """Regressionstests fuer die new_room-Klarstellung (siehe apply_turn):
    ein vorgeschlagener Raum impliziert IMMER die Ankunft dort, unabhaengig
    von actor_move_to; bei erreichter Obergrenze wird er verworfen."""

    def test_new_room_moves_actor_and_ignores_stale_move_to(self):
        world = _world()
        new_room = types.SimpleNamespace(name="Loft", anchor="hay bales",
                                         one_way=False, justification="")
        delta = _delta(new_room=new_room, actor_move_to="stay")
        world.apply_turn("player", delta)
        self.assertEqual(world.player_at, "n4")
        self.assertEqual(world.nodes["n4"].name, "Loft")

    def test_new_room_rejected_at_max_nodes(self):
        world = _world()
        world.max_nodes = len(world.nodes)
        new_room = types.SimpleNamespace(name="Loft", anchor="hay bales",
                                         one_way=False, justification="")
        rejected = world.apply_turn("player", _delta(new_room=new_room))
        self.assertEqual(world.player_at, "n1")
        self.assertEqual(len(world.nodes), 3)
        self.assertTrue(any("max_nodes" in r for r in rejected), rejected)


class AgenticPromotionTest(unittest.TestCase):
    """Eingefuehrte Figuren werden in Spawn-Reihenfolge agentisch, bis
    max_agentic voll ist - kein Vorsortieren, kein Modell-Urteil. Jede
    beförderte bekommt ihr eigenes hidden_target."""

    def test_all_introduced_get_promoted_in_order(self):
        world = _world()   # noch keine Figur agentisch
        cook = types.SimpleNamespace(name="Cook", at="n1",
                                     agenda_draft="wants food",
                                     agenda_target_hint="soup")
        sailor = types.SimpleNamespace(name="Sailor", at="n1",
                                       agenda_draft="wants the ship",
                                       agenda_target_hint="docks")
        world.apply_turn("player", _delta(characters_introduced=[cook, sailor]))

        agents = world.agentic_actors()
        self.assertEqual([c.name for c in agents], ["Cook", "Sailor"])
        self.assertEqual(agents[0].hidden_target, "soup")
        self.assertEqual(agents[1].hidden_target, "docks")

    def test_promotion_stops_at_max_agentic(self):
        world = _world()
        world.max_agentic = 1
        a = types.SimpleNamespace(name="A", at="n1", agenda_draft="x",
                                  agenda_target_hint="one")
        b = types.SimpleNamespace(name="B", at="n1", agenda_draft="x",
                                  agenda_target_hint="two")
        world.apply_turn("player", _delta(characters_introduced=[a, b]))
        self.assertEqual([c.name for c in world.agentic_actors()], ["A"])
        self.assertFalse(list(world.characters.values())[-1].is_agentic)  # B


class SpawnFallbackCharacterTest(unittest.TestCase):
    def test_uses_first_unused_name_and_marks_agentic_if_needed(self):
        world = _world()
        char_id = world.spawn_fallback_character(["Vale", "Marrow"])
        self.assertEqual(world.characters[char_id].name, "Vale")
        self.assertTrue(world.characters[char_id].is_agentic)
        self.assertEqual(world.characters[char_id].hidden_target, "")

    def test_no_promotion_when_agentic_pool_is_full(self):
        world = _world()
        world.max_agentic = 1
        world.add_character("Vogel", "n1", "get out", is_agentic=True)
        char_id = world.spawn_fallback_character(["Vale"])
        self.assertFalse(world.characters[char_id].is_agentic)


if __name__ == "__main__":
    unittest.main()

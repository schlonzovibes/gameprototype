"""Regressionstests fuer schema.py - Feldform und -reihenfolge.

Kein Sprachmodell noetig: model_json_schema() baut die Grammatik rein
lokal, genau das wird hier geprueft.
"""

import unittest

import schema


class InitModelTest(unittest.TestCase):
    def test_field_shape(self):
        """INIT erzeugt den Startraum, die abstrakte Richtung und 0-2
        Figuren, keinen Graphen. direction und starting_characters stehen
        VOR der Erzaehlung, damit die Erzaehlung sie aufgreifen kann."""
        Init = schema.init_model()
        self.assertEqual(
            list(Init.model_fields.keys()),
            ["language", "start_node_name", "start_node_anchor", "direction",
             "starting_characters", "opening_narration", "opening_image_prompt"])

    def test_start_character_fields(self):
        Init = schema.init_model()
        StartChar = Init.model_fields["starting_characters"].annotation.__args__[0]
        self.assertEqual(list(StartChar.model_fields.keys()),
                         ["name", "agenda_draft", "agenda_target_hint"])

    def test_direction_fields(self):
        Init = schema.init_model()
        Direction = Init.model_fields["direction"].annotation
        self.assertEqual(list(Direction.model_fields.keys()), ["pull", "pressure"])


class DecideModelTest(unittest.TestCase):
    def test_field_order_is_semantic(self):
        """aim steht vor intent - die Absicht wird auf dem frisch gesetzten
        Schritt konditioniert, nicht umgekehrt (siehe schema.decide_model)."""
        Decide = schema.decide_model(("n1", "n2"))
        self.assertEqual(list(Decide.model_fields.keys()),
                         ["aim", "intent", "utterance", "move_to"])

    def test_builds_json_schema(self):
        Decide = schema.decide_model(("n1",))
        Decide.model_json_schema()   # wirft bei einem kaputten Schema


class ResolveModelTest(unittest.TestCase):
    def test_player_mode_has_characters_introduced(self):
        Player = schema.resolve_model(("n1", "n2"), ("c1",), ("n2",), "player")
        self.assertIn("characters_introduced", Player.model_fields)

    def test_agentic_mode_has_no_characters_introduced(self):
        """Die Fuenf-Zuege-Garantie haengt am Spielerzug, nicht am
        Nebenschauplatz der agentischen Figur (Brief 5.1) - das Feld
        existiert bei mode="agentic" im Schema gar nicht, nicht bloss
        leer."""
        Agentic = schema.resolve_model(("n1", "n2"), ("c1",), ("n2",), "agentic")
        self.assertNotIn("characters_introduced", Agentic.model_fields)

    def test_field_order_player(self):
        Player = schema.resolve_model(("n1", "n2"), ("c1",), ("n2",), "player")
        self.assertEqual(
            list(Player.model_fields.keys()),
            ["new_room", "actor_move_to", "moves", "status_changes",
             "marks_added", "facts_added", "characters_introduced", "events"])

    def test_field_order_agentic(self):
        Agentic = schema.resolve_model(("n1", "n2"), ("c1",), ("n2",), "agentic")
        self.assertEqual(
            list(Agentic.model_fields.keys()),
            ["new_room", "actor_move_to", "moves", "status_changes",
             "marks_added", "facts_added", "events"])

    def test_new_room_is_required_not_optional(self):
        """Regression fuer die Optional-Korrektur ggue. dem Brief-Wortlaut:
        new_room ist ein Pflichtfeld (leerer name-String signalisiert
        "keiner"), kein echtes Optional/None-Feld."""
        Player = schema.resolve_model(("n1",), ("c1",), ("n1",), "player")
        self.assertTrue(Player.model_fields["new_room"].is_required())
        schema_dict = Player.model_json_schema()
        # Kein "anyOf"/null-Branch im generierten JSON-Schema fuer new_room.
        new_room_schema = schema_dict["properties"]["new_room"]
        self.assertNotIn("anyOf", new_room_schema)

    def test_actor_move_to_limited_to_actor_exits(self):
        """actor_move_to darf NUR die uebergebenen Ausgaenge des Akteurs
        annehmen, nicht jeden Knoten der Welt."""
        Player = schema.resolve_model(("n1", "n2", "n3"), ("c1",), ("n2",), "player")
        enum = Player.model_json_schema()["properties"]["actor_move_to"]["enum"]
        self.assertEqual(set(enum), {"n2", "stay"})

    def test_builds_without_characters(self):
        """Keine aktiven Figuren (char_ids leer) - darf nicht mit
        Literal[()]-TypeError abstuerzen."""
        Agentic = schema.resolve_model(("n1",), (), ("n1",), "agentic")
        Agentic.model_json_schema()


class NarrateModelTest(unittest.TestCase):
    def test_static_no_ids_needed(self):
        Narrate = schema.narrate_model()
        self.assertEqual(list(Narrate.model_fields.keys()),
                         ["can_end", "narrator_text", "image_prompt"])
        Narrate.model_json_schema()


if __name__ == "__main__":
    unittest.main()

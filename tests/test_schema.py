"""Regressionstests fuer schema.py - Feldform und -reihenfolge.

Kein Sprachmodell noetig: model_json_schema() baut die Grammatik rein
lokal, genau das wird hier geprueft.
"""

import unittest

import schema


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
    def test_field_order(self):
        Resolve = schema.resolve_model(("n1", "n2"), ("c1",), ("n2",))
        self.assertEqual(
            list(Resolve.model_fields.keys()),
            ["actor_move_to", "moves", "status_changes", "marks_added",
             "facts_added", "events"])

    def test_actor_move_to_limited_to_actor_exits(self):
        """actor_move_to darf NUR die uebergebenen Ausgaenge des Akteurs
        annehmen, nicht jeden Knoten der Welt (Regression fuer den
        PlayerMoveTo/ActorMoveTo-Fix)."""
        Resolve = schema.resolve_model(("n1", "n2", "n3"), ("c1",), ("n2",))
        enum = Resolve.model_json_schema()["properties"]["actor_move_to"]["enum"]
        self.assertEqual(set(enum), {"n2", "stay"})

    def test_builds_without_characters(self):
        """Keine aktiven Figuren (char_ids leer) - darf nicht mit
        Literal[()]-TypeError abstuerzen."""
        Resolve = schema.resolve_model(("n1",), (), ("n1",))
        Resolve.model_json_schema()


class NarrateModelTest(unittest.TestCase):
    def test_static_no_ids_needed(self):
        Narrate = schema.narrate_model()
        self.assertEqual(list(Narrate.model_fields.keys()),
                         ["can_end", "narrator_text", "image_prompt"])
        Narrate.model_json_schema()


class InitCharacterTest(unittest.TestCase):
    def test_has_agenda_and_aim_not_goal(self):
        self.assertEqual(list(schema.InitCharacter.model_fields.keys()),
                         ["id", "name", "at", "agenda", "aim"])


if __name__ == "__main__":
    unittest.main()

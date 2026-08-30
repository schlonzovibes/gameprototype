"""Die Feldnamen "narrator_text", "agenda", "hidden_target" und
"agenda_target_hint" duerfen ausserhalb von schema.py/state.py in keiner
Python-Datei des Projekts vorkommen (Kriterium G des vorigen Briefs,
Kriterium I des aktuellen).

tests/ selbst ist ausgenommen: Test-Fixtures muessen zwangslaeufig
Character(agenda=..., ...)/Character(hidden_target=..., ...) als Keyword-
Argument konstruieren, um die Kriterien ueberhaupt pruefen zu koennen - das
ist White-Box-Test der eigenen Datenklasse aus state.py, kein Durchsickern
des Vertrags in fremden Anwendungscode, was dieses Kriterium eigentlich
meint.

Bewusst nur diese vier Strings, wie in den Briefen spezifiziert - "aim" ist
als Teilstring ein zu haeufiges Alltagswort/Bezeichnerfragment fuer einen
sicheren Grep-Test. Damit story.py "hidden_target" nie selbst schreiben
muss, lebt die Leak-Pruefung komplett als Methode in state.py
(World.hidden_target_leaked) - story.py ruft sie nur auf.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_FILES = {"schema.py", "state.py"}
EXCLUDED_DIRS = {"tests", "cache", "diffusion_models", ".git"}
LEAKED_STRINGS = ("narrator_text", "agenda", "hidden_target", "agenda_target_hint")


def _project_python_files():
    for path in REPO_ROOT.glob("**/*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in EXCLUDED_DIRS:
            continue
        if path.name in EXCLUDED_FILES:
            continue
        yield path


class NoLeakedFieldNamesTest(unittest.TestCase):
    def test_narrator_text_and_agenda_stay_out_of_application_code(self):
        offenders = []
        for path in _project_python_files():
            text = path.read_text(encoding="utf-8")
            for needle in LEAKED_STRINGS:
                if needle in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {needle!r}")
        self.assertEqual(offenders, [],
                         "these field names should only appear in "
                         "schema.py/state.py:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()

"""Every table indexed by an action type must cover every action type.

ActionType's values used to be "G – Permanent", with an en dash. Changing them
to "G" fixed a real problem — the dash came out of json.dumps as \\u2013 and no
model could guess it — and quietly broke two maps that were keyed by the old
string:

* ``combinations._AT_TO_EC`` stopped matching, so every load case became
  EcLoadType.OTHER. A permanent action was combined as a variable one, and
  G + Q + W produced three ULS combinations instead of two.
* the application's table of partial factors stopped matching too, so a new
  permanent load case was given gamma_unf = 1.0 instead of 1.35 — through a
  fallback tuple that made the miss look like a default. That half is checked
  in the application's own suite: this file may not name it.

Neither raised. Both were found days later, and only because a test that runs
solely where eurocodepy is installed happened to be run there.

So: the members are the list, and the tables are checked against it here.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d.models import ActionType


class TestNoTableIsKeyedByTheLabel(unittest.TestCase):
    """The labels are for the interface. A table keyed by one is a table that
    breaks the next time the wording changes."""

    def test_the_value_is_a_single_letter(self):
        for at in ActionType:
            with self.subTest(action=at.name):
                self.assertEqual(at.value, at.name)
                self.assertEqual(len(at.value), 1)

    def test_the_label_is_not_the_value(self):
        self.assertNotEqual(ActionType.G.label, ActionType.G.value)

    def test_no_source_file_keys_a_dict_by_a_dashed_label(self):
        """The mistake itself, as a rule. Written as a search because both
        occurrences were in files nobody was looking at."""
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent / 'src' / 'xdfem2d'
        offenders = []
        for path in root.rglob('*.py'):
            if path.name == 'models.py':
                continue                     # where the labels are defined
            for n, line in enumerate(path.read_text(encoding='utf-8').
                                     splitlines(), 1):
                if re.search(r'["\'][GQWET]\s*[–-]\s*\w+["\']\s*:', line):
                    offenders.append(f'{path.name}:{n}: {line.strip()}')
        self.assertEqual(offenders, [], 'keyed by a display label')


class TestTheEurocodeMapping(unittest.TestCase):

    def _map(self):
        """The map lives inside the function, so it is rebuilt here from the
        same members rather than imported."""
        from xdfem2d.combinations import _generate_ec_combos  # noqa: F401
        import inspect
        src = inspect.getsource(_generate_ec_combos)
        return src

    def test_every_action_type_is_mapped(self):
        src = self._map()
        for at in ActionType:
            with self.subTest(action=at.name):
                self.assertIn(f'ActionType.{at.name}', src)

    def test_it_is_keyed_by_the_member_not_the_string(self):
        import re
        src = self._map()
        self.assertIn('ActionType.G: ', src)
        # Code only. The comment above the map quotes the old key to explain
        # what went wrong, and the first version of this test failed on that.
        code = re.sub(r'^\s*#.*$', '', src, flags=re.M)
        self.assertNotIn('– Permanent', code)

    def test_it_coerces_what_it_is_given(self):
        """So a file written before the values changed still maps."""
        self.assertIn('ActionType.coerce', self._map())


if __name__ == '__main__':
    unittest.main()

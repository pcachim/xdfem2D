"""The action type is a datum, not a label.

Its value used to be "G – Permanent", with an en dash, and that one string was
doing two jobs: what the interface shows and what goes into the file. The
consequences were not cosmetic.

json.dumps writes the dash as \\u2013. Nothing can guess it — a language model
asked for a permanent load case wrote action_type="Permanent" and was refused
by an error that named neither the valid values nor why. And every .x2d ever
saved carries the dash inside it.

So the value is now a single ASCII letter, `label` is what the interface shows,
and `coerce` accepts anything anyone reasonably meant — including every file
written before the change.
"""
from __future__ import annotations

import json
import unittest

from context import Structure2D
from xdfem2d.models import ActionType
from xdfem2d.structure_io import _to_dict


class TestTheValueIsPlain(unittest.TestCase):

    def test_every_value_is_a_single_ascii_letter(self):
        for member in ActionType:
            with self.subTest(action=member.name):
                self.assertEqual(member.value, member.name)
                self.assertTrue(member.value.isascii())
                self.assertEqual(len(member.value), 1)

    def test_no_value_survives_json_as_an_escape(self):
        """The symptom that started this: \\u2013 in a file nobody can type."""
        for member in ActionType:
            with self.subTest(action=member.name):
                self.assertNotIn('\\\\u', json.dumps(member.value))

    def test_the_label_still_exists_for_the_interface(self):
        self.assertEqual(ActionType.G.label, "G – Permanent")
        self.assertEqual(ActionType.T.label, "T – Temperature")

    def test_the_label_is_never_what_gets_stored(self):
        s = Structure2D()
        s.add_load_case('G', action_type='G')
        stored = _to_dict(s)['load_cases'][0]['action_type']
        self.assertEqual(stored, 'G')
        self.assertNotIn('–', stored)


class TestItUnderstandsWhatWasMeant(unittest.TestCase):

    def test_the_letter(self):
        self.assertIs(ActionType.coerce('Q'), ActionType.Q)
        self.assertIs(ActionType.coerce('q'), ActionType.Q)

    def test_the_word_a_model_reaches_for(self):
        """granite wrote 'Permanent'. It was refused; now it is understood."""
        self.assertIs(ActionType.coerce('Permanent'), ActionType.G)
        self.assertIs(ActionType.coerce('permanent'), ActionType.G)
        self.assertIs(ActionType.coerce('wind'), ActionType.W)
        self.assertIs(ActionType.coerce('SEISMIC'), ActionType.E)

    def test_the_old_dashed_label_from_every_existing_file(self):
        """The compatibility that makes the change safe: nine shipped examples
        carry these strings."""
        self.assertIs(ActionType.coerce('G – Permanent'), ActionType.G)
        self.assertIs(ActionType.coerce('T – Temperature'), ActionType.T)

    def test_the_same_label_typed_with_a_plain_hyphen(self):
        self.assertIs(ActionType.coerce('G - Permanent'), ActionType.G)

    def test_an_actiontype_passes_through(self):
        self.assertIs(ActionType.coerce(ActionType.W), ActionType.W)

    def test_nothing_becomes_permanent(self):
        self.assertIs(ActionType.coerce(''), ActionType.G)
        self.assertIs(ActionType.coerce(None), ActionType.G)

    def test_a_real_mistake_is_still_refused_and_told_what_to_use(self):
        """Tolerance is not silence: an error that lists neither the values nor
        the words is how this went wrong in the first place."""
        with self.assertRaises(ValueError) as cm:
            ActionType.coerce('Distributed')
        message = str(cm.exception)
        self.assertIn('G, Q, W, E, T', message)
        self.assertIn('permanent', message)


class TestExistingFilesStillOpen(unittest.TestCase):
    """The change alters what is written. It must not alter what can be read."""

    def test_every_shipped_example_loads(self):
        import glob
        from pathlib import Path

        from xdfem2d.file_io import load_x2d
        paths = sorted(glob.glob(str(Path(__file__).resolve().parent.parent.parent
                                     / 'examples' / '*.x2d')))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(model=Path(path).name):
                struc, _, _ = load_x2d(path)
                for case in struc.load_cases:
                    self.assertIsInstance(case.action_type, ActionType)

    def test_a_file_written_before_the_change_round_trips(self):
        from xdfem2d.structure_io import _from_dict
        old = {'load_cases': [{'id': 'W1', 'self_weight_factor': 0.0,
                               'action_type': 'W – Wind'}]}
        s = _from_dict(old)
        self.assertIs(s.load_cases[0].action_type, ActionType.W)
        self.assertEqual(_to_dict(s)['load_cases'][0]['action_type'], 'W')



class TestTheAmericanLetters(unittest.TestCase):
    """A model asked for a continuous beam wrote action_type "D" for dead and
    the file was refused. "Dead" was already understood and the letter for it
    was not, which is the wrong way round: the letter is what gets written."""

    def test_d_is_permanent(self):
        self.assertEqual(ActionType.coerce('D'), ActionType.G)
        self.assertEqual(ActionType.coerce('DL'), ActionType.G)

    def test_l_is_live(self):
        self.assertEqual(ActionType.coerce('L'), ActionType.Q)
        self.assertEqual(ActionType.coerce('LL'), ActionType.Q)

    def test_the_eurocode_letters_are_unchanged(self):
        for letter in ('G', 'Q', 'W', 'E', 'T'):
            with self.subTest(letter=letter):
                self.assertEqual(ActionType.coerce(letter).value, letter)

    def test_the_model_that_was_refused_now_opens(self):
        from xdfem2d import model_json
        d = model_json.template(as_text=False)
        d['load_cases'][0]['action_type'] = 'D'
        struc = model_json.load(d)
        self.assertEqual(struc.load_cases[0].action_type.value, 'G')

    def test_something_meaningless_is_still_refused(self):
        with self.assertRaises(ValueError):
            ActionType.coerce('ZZ')

    def test_the_error_names_the_letters_it_does_accept(self):
        try:
            ActionType.coerce('ZZ')
        except ValueError as e:
            self.assertIn('D and L', str(e))


if __name__ == '__main__':
    unittest.main()

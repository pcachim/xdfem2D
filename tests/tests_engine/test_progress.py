"""xdfem2d.progress: loops report (done, total, label) to a per-thread listener."""
import threading
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import progress


class TestProgress(unittest.TestCase):
    def test_track_is_transparent_without_listener(self):
        self.assertEqual(list(progress.track(iter([1, 2, 3]), "x")), [1, 2, 3])

    def test_track_reports_each_step_and_the_end(self):
        got = []
        with progress.listen(lambda d, t, l: got.append((d, t, l))):
            list(progress.track([10, 20], "stage"))
        self.assertEqual(got, [(0, 2, "stage"), (1, 2, "stage"), (2, 2, "stage")])

    def test_listener_is_per_thread_and_removed_afterwards(self):
        got = []
        with progress.listen(lambda *a: got.append(a)):
            t = threading.Thread(target=lambda: list(progress.track([1], "t")))
            t.start(); t.join()
        list(progress.track([1], "after"))
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()

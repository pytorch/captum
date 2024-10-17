#!/usr/bin/env python3

# pyre-unsafe

import io
import unittest
import unittest.mock

from captum._utils.progress import NullProgress, progress
from tests.helpers import BaseTest


class Test(BaseTest):
    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_nullprogress(self, mock_stderr) -> None:
        count = 0
        with NullProgress(["x", "y", "z"]) as np:
            for _ in np:
                for _ in NullProgress([1, 2, 3]):
                    count += 1

        self.assertEqual(count, 9)
        output = mock_stderr.getvalue()
        self.assertEqual(output, "")

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_nested_progress_tqdm(self, mock_stderr) -> None:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Skipping tqdm test, tqdm not available.")

        parent_data = ["x", "y", "z"]
        test_data = [1, 2, 3]
        with progress(parent_data, desc="parent progress") as parent:
            for item in parent:
                for _ in progress(test_data, desc=f"test progress {item}"):
                    pass
        output = mock_stderr.getvalue()
        self.assertIn("parent progress:", output)
        for item in parent_data:
            self.assertIn(f"test progress {item}:", output)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_nested_simple_progress(self, mock_stderr) -> None:
        parent_data = ["x", "y", "z"]
        test_data = [1, 2, 3]
        with progress(
            parent_data, desc="parent progress", use_tqdm=False, mininterval=0.0
        ) as parent:
            for item in parent:
                for _ in progress(
                    test_data, desc=f"test progress {item}", use_tqdm=False
                ):
                    pass

        output = mock_stderr.getvalue()
        self.assertEqual(
            output.count("parent progress:"), 5, "5 'parent' progress bar expected"
        )
        for item in parent_data:
            self.assertIn(f"test progress {item}:", output)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_progress_tqdm(self, mock_stderr) -> None:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Skipping tqdm test, tqdm not available.")

        test_data = [1, 3, 5]

        progressed = progress(test_data, desc="test progress")
        assert list(progressed) == test_data
        assert "test progress: " in mock_stderr.getvalue()

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_progress(self, mock_stderr) -> None:
        test_data = [1, 3, 5]
        desc = "test progress"

        progressed = progress(test_data, desc=desc, use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue().startswith(f"\r{desc}: 0% 0/3")
        assert mock_stderr.getvalue().endswith(f"\r{desc}: 100% 3/3\n")

        # progress iterable without len but explicitly specify total
        def gen():
            for n in test_data:
                yield n

        mock_stderr.seek(0)
        mock_stderr.truncate(0)

        progressed = progress(gen(), desc=desc, total=len(test_data), use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue().startswith(f"\r{desc}: 0% 0/3")
        assert mock_stderr.getvalue().endswith(f"\r{desc}: 100% 3/3\n")

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_progress_without_total(self, mock_stderr) -> None:
        test_data = [1, 3, 5]
        desc = "test progress"

        def gen():
            for n in test_data:
                yield n

        progressed = progress(gen(), desc=desc, use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue().startswith(f"\r{desc}: ")
        assert mock_stderr.getvalue().endswith(f"\r{desc}: ...\n")

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_progress_update_manually(self, mock_stderr) -> None:
        desc = "test progress"

        p = progress(total=5, desc=desc, use_tqdm=False)
        p.update(0)
        p.update(2)
        p.update(2)
        p.update(1)
        p.close()
        assert mock_stderr.getvalue().startswith(f"\r{desc}: 0% 0/5")
        assert mock_stderr.getvalue().endswith(f"\r{desc}: 100% 5/5\n")

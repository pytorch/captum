#!/usr/bin/env python3

import io
import unittest
import unittest.mock

from captum._utils.progress import progress
from tests.helpers.basic import BaseTest


class Test(BaseTest):
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

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
    def test_progress_without_tqdm(self, mock_stderr) -> None:
        test_data = [1, 3, 5]
        desc = "test progress"
        output = (
            f"\r{desc}: 0% 0/3\r{desc}: 33% 1/3\r{desc}: 66% 2/3\r{desc}: 100% 3/3\n"
        )

        progressed = progress(test_data, desc=desc, use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue() == output

        # progress iterable without len but explicitly specify total
        def gen():
            for n in test_data:
                yield n

        mock_stderr.seek(0)
        mock_stderr.truncate(0)

        progressed = progress(gen(), desc=desc, total=len(test_data), use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue() == output

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_progress_without_tqdm_no_total(self, mock_stderr) -> None:
        test_data = [1, 3, 5]
        desc = "test progress"
        output = f"\r{desc}: \r{desc}: .\r{desc}: ..\r{desc}: ...\n"

        def gen():
            for n in test_data:
                yield n

        progressed = progress(gen(), desc=desc, use_tqdm=False)

        assert list(progressed) == test_data
        assert mock_stderr.getvalue() == output

import unittest
import numpy as np
from lenses import mbx


class TestComponent(unittest.TestCase):
    def test_compute_bins(self):
        samples = np.array([0.0, 0.6, 0.5, 0.8, 1.0], dtype=np.float32)
        component = mbx.component.Component(
            None, 0, 1.0, samples, compute_sample_bins=True, num_sample_bins=2
        )
        self.assertEqual(len(component.sample_bins), 2)

        self.assertEqual(len(component.sample_bins[0].samples), 1)
        self.assertEqual(len(component.sample_bins[1].samples), 4)

        # wrapping scalar values in np.array(..., dtype=np.float32)
        # to avoid differences due to
        # single precision of float32 vs double precision of python floats
        sample_bin = component.sample_bins[0]
        self.assertEqual(
            sample_bin.samples[0].value, np.array(0.0, dtype=np.float32).item()
        )

        sample_bin = component.sample_bins[1]
        self.assertEqual(
            sample_bin.samples[0].value, np.array(0.5, dtype=np.float32).item()
        )
        self.assertEqual(
            sample_bin.samples[1].value, np.array(0.6, dtype=np.float32).item()
        )
        self.assertEqual(
            sample_bin.samples[2].value, np.array(0.8, dtype=np.float32).item()
        )
        self.assertEqual(
            sample_bin.samples[3].value, np.array(1.0, dtype=np.float32).item()
        )

        samples = None
        with self.assertRaises(ValueError):
            # samples are required to compute bins
            component = mbx.component.Component(
                None, 0, 1.0, samples, compute_sample_bins=True
            )

    def test_dict(self):
        samples = np.array([0.0, 0.6, 0.5, 0.8, 1.0], dtype=np.float32)
        src_component = mbx.component.Component(
            None, 0, 1.0, samples, compute_sample_bins=True, num_sample_bins=2
        )
        self.assertEqual(len(src_component.sample_bins), 2)

        data = src_component.to_dict()
        component = mbx.component.Component.from_dict(data, None, 0)
        self.assertEqual(src_component.value, component.value)
        self.assertEqual(len(src_component.sample_bins), len(component.sample_bins))


if __name__ == "__main__":
    unittest.main()

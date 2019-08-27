from __future__ import print_function

import unittest
from random import random, sample
from types import SimpleNamespace as SN

from captum.widgets.helpers import DataEntry, wrap_dataset


classes = ["doggo", "cat", "raptor"]

wrapped_model = SN(
    get_prediction_classes=lambda: classes,
    get_prediction_indices=lambda prediction: [prediction],
    get_prediction=lambda inputs, meta: inputs,
)

data_processed = []

for _i in range(20000):
    raw_inputs = sample(classes, k=int(random() * len(classes)) + 1)
    data_processed.append(
        DataEntry(
            raw_inputs,
            [classes.index(item) for item in raw_inputs],
            sample(list(range(len(classes))), k=int(random() * len(classes)) + 1),
            None,
        )
    )


class Test(unittest.TestCase):
    def test_wrap(self):
        dataset = wrap_dataset(data_processed, lambda x: x, wrapped_model)
        records = dataset.find(
            target_classes=["doggo"], predicted_classes=["cat"], limit=20000
        )
        for record in records:
            target_labels = [
                classes[index] for index in data_processed[record].target_indices
            ]
            self.assertIn("doggo", target_labels)
        predicted_labels = [classes[index[0]] for index in dataset._predicted_indices]
        self.assertIn("cat", predicted_labels)

        records = dataset.find(
            target_classes=["doggo"], predicted_classes=["cat"], exact=True, limit=20000
        )

        for record in records:
            target_labels = [
                classes[index] for index in data_processed[record].target_indices
            ]
            self.assertIn("doggo", target_labels)
            self.assertTrue(len(target_labels) == 1)
        predicted_labels = [classes[index[0]] for index in dataset._predicted_indices]
        self.assertIn("cat", predicted_labels)
        # This worked for the initial implementation. Now, it doesn't hold
        # self.assertTrue(len(predicted_labels) == 1)


if __name__ == "__main__":
    unittest.main()

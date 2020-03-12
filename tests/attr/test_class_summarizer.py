#!/usr/bin/env python3
import torch

from captum.attr import ClassSummarizer, CommonSummarizer

from .helpers.utils import BaseTest


class Test(BaseTest):
    def class_test(self, data, classes, x_sizes):
        summarizer = CommonSummarizer(ClassSummarizer)
        for x, y in data:
            summarizer.update(x, y)

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertIsInstance(summ, list)
        for s, size in zip(summ, x_sizes):
            self.assertIsInstance(s, dict)
            for key in s:
                self.assertEqual(s[key].size(), size)

        self.assertIsNotNone(summarizer.class_summaries)
        all_classes = torch.zeros(len(classes))
        class_summaries = summarizer.class_summaries
        all_keys = set(class_summaries.keys())
        for clazz in classes:
            self.assertTrue(clazz in class_summaries)
            all_keys.remove(clazz)

            summ = class_summaries[clazz]
            self.assertIsNotNone(summ)
            self.assertIsInstance(summ, list)

            for s, size in zip(summ, x_sizes):
                self.assertIsInstance(s, dict)
                for key in s:
                    self.assertEqual(s[key].size(), size)

        self.assertEqual(len(all_keys), 0)

    def test_classes(self):
        sizes_to_test = [
            ((1,),),
            ((3, 2, 10, 3), (1,)),
            ((20,)),
        ]
        classeses = [list(range(100)), ["%d" % i for i in range(100)]]
        for batch_size in [None, 1, 4]:
            for sizes, classes in zip(sizes_to_test, classeses):

                def create_batch_labels(batch_idx):
                    if batch_size is None:
                        # batch_size = 1
                        return classes[batch_idx]

                    return classes[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]

                sizes_plus_batch = sizes
                if batch_size is not None:
                    sizes_plus_batch = tuple((batch_size,) + si for si in sizes)

                # we want to test each possible element
                num_batches = len(classes)
                if batch_size is not None:
                    num_batches //= batch_size

                data = [
                    (
                        tuple(torch.randn(si) for si in sizes_plus_batch),
                        create_batch_labels(batch_idx),
                    )
                    for batch_idx in range(num_batches)
                ]

                with self.subTest(
                    batch_size=batch_size, sizes=sizes_plus_batch, classes=classes
                ):
                    self.class_test(data, classes, sizes)

    def test_no_class(self):
        size = (30, 20)
        summarizer = CommonSummarizer(ClassSummarizer)
        for _ in range(10):
            x = torch.randn(size)
            summarizer.update(x)

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertIsInstance(summ, dict)
        for key in summ:
            self.assertTrue(summ[key].size() == size)

        self.assertIsNotNone(summarizer.class_summaries)
        self.assertIsInstance(summarizer.class_summaries, dict)
        self.assertEqual(len(summarizer.class_summaries), 0)

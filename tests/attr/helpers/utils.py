#!/usr/bin/env python3


def assertArraysAlmostEqual(inputArr, refArr, delta=0.05):
    for index, (input, ref) in enumerate(zip(inputArr, refArr)):
        assert (
            abs(input - ref) <= delta
        ), "Values at index {}, {} and {}, \
            differ more than by {}".format(
            index, input, ref, delta
        )

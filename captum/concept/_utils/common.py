#!/usr/bin/env python3

from typing import List

from captum.concept._core.concept import Concept


def concepts_to_str(concepts: List[Concept]) -> str:
    r"""
    Returns a string of hyphen("-") concatenated concept names.
    Example output: "striped-random_0-random_1"

    Args:
        concepts (list[Concept]): a List of concept names to be
                concatenated and used as a concepts key. These concept
                names are respective to the Concept objects used for
                the classifier train.
    Returns:
        names_str (str): A string of hyphen("-") concatenated
                concept names. Ex.: "striped-random_0-random_1"
    """

    return "-".join([str(c.id) for c in concepts])

#!/usr/bin/env python3
import numpy as np
from enum import Enum


class Riemann(Enum):
    left = 1
    right = 2
    middle = 3
    trapezoid = 4


SUPPORTED_RIEMANN_METHODS = [
    "riemann_left",
    "riemann_right",
    "riemann_middle",
    "riemann_trapezoid",
]

SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ["gausslegendre"]


def approximation_parameters(method):
    r"""Retrieves parameters for the input approximation `method`

        Args
            method: The name of the approximation method. Currently only `riemann`
                    and gauss legendre are
    """
    if method in SUPPORTED_RIEMANN_METHODS:
        return riemann_builders(method=Riemann[method.split("_")[-1]])
    if method == "gausslegendre":
        return gauss_legendre_builders()
    raise ValueError("Invalid integral approximation method name: {}".format(method))


def riemann_builders(method=Riemann.trapezoid):
    r"""Step sizes are identical and alphas are scaled in [0, 1]

        Args
             n: The number of steps
             method: `left`, `right`, `middle` and `trapezoid` riemann
    """

    def w_list_func(n):
        assert n > 1, "The number of steps has to be larger than one"
        deltas = [1 / n] * n
        if method == Riemann.trapezoid:
            deltas[0] /= 2
            deltas[-1] /= 2
        return deltas

    def alpha_list_func(n):
        assert n > 1, "The number of steps has to be larger than one"
        if method == Riemann.trapezoid:
            return list(np.linspace(0, 1, n))
        if method == Riemann.left:
            return list(np.linspace(0, 1 - 1 / n, n))
        if method == Riemann.middle:
            return list(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
        if method == Riemann.right:
            return list(np.linspace(1 / n, 1, n))
        # This is not a standard riemann method but in many cases it
        # leades to faster approaximation. Test cases for small number of steps
        # do not make sense but for larger number of steps the approximation is
        # better therefore leaving this option available
        # if method == 'riemann_include_endpoints':
        #     return [i / (n - 1) for i in range(n)]

    return w_list_func, alpha_list_func


def gauss_legendre_builders():
    r""" For integration using gauss legendre
    numpy makes our life easier, but it integrates in [-1, 1]. Need to rescale.
    The weights returned by numpy also aggregate to 2. Need to rescale.

    Args
        n: The number of steps
    """

    def w_list_func(n):
        assert n > 0, "The number of steps has to be larger than zero"
        # Scale from 2 -> 1
        return list(0.5 * np.polynomial.legendre.leggauss(n)[1])

    def alpha_list_func(n):
        assert n > 0, "The number of steps has to be larger than zero"
        # [-1, 1] -> [0, 1] scaling is necessary.
        return list(0.5 * (1 + np.polynomial.legendre.leggauss(n)[0]))

    return w_list_func, alpha_list_func

# #!/usr/bin/env python3

import unittest

from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from tests.helpers.basic import BaseTest
from tests.influence._utils.common import (
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInRegression1DNumerical,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInGetKMostInfluential,
    _TestTracInSelfInfluence,
    _TestTracInDataLoader,
)


class TestTracInCPFast(
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInRegression1DNumerical,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInGetKMostInfluential,
    _TestTracInSelfInfluence,
    _TestTracInDataLoader,
    BaseTest,
):
    def setUp(self):
        super().setUp()
        self.reduction = "sum"
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn: TracInCPFast(
                net,
                list(net.children())[-1],
                dataset,
                tmpdir,
                loss_fn=loss_fn,
                batch_size=batch_size,
            )
        )


class TestTracInCPFastRandProjNoProjection(
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInRegression1DNumerical,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInGetKMostInfluential,
    _TestTracInDataLoader,
    BaseTest,
):
    def setUp(self):
        super().setUp()
        try:
            import annoy  # noqa
        except ImportError:
            raise unittest.SkipTest(
                (
                    "Skipping tests for TracInCPFastRandProj, "
                    "because it requires the Annoy module."
                )
            )
        self.reduction = "sum"
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn: TracInCPFastRandProj(
                net,
                list(net.children())[-1],
                dataset,
                tmpdir,
                loss_fn=loss_fn,
                batch_size=batch_size,
                projection_dim=None,
            )
        )


class TestTracInCPFastRandProj1DimensionalProjection(
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInDataLoader,
    BaseTest,
):
    def setUp(self):
        super().setUp()
        try:
            import annoy  # noqa
        except ImportError:
            raise unittest.SkipTest(
                "Skipping tests for TracInCPFastRandProj, "
                "because it requires the Annoy module."
            )
        self.reduction = "sum"
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn: TracInCPFastRandProj(
                net,
                list(net.children())[-1],
                dataset,
                tmpdir,
                loss_fn=loss_fn,
                batch_size=batch_size,
                projection_dim=1,
            )
        )

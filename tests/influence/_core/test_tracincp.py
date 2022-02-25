# #!/usr/bin/env python3

from captum.influence._core.tracincp import TracInCP
from tests.helpers.basic import BaseTest
from tests.influence._utils.common import (
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInXORCheckIdx,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInRegression1DCheckAutogradHacks,
    _TestTracInRegression20DCheckAutogradHacks,
    _TestTracInXORCheckAutogradHacks,
    _TestTracInIdentityRegressionCheckAutogradHacks,
    _TestTracInRegression1DNumerical,
    _TestTracInGetKMostInfluential,
    _TestTracInSelfInfluence,
    _TestTracInDataLoader,
)


class TestTracInCP(
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInXORCheckIdx,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInGetKMostInfluential,
    _TestTracInSelfInfluence,
    _TestTracInDataLoader,
    BaseTest,
):
    def setUp(self):
        super().setUp()
        self.reduction = "none"
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn: TracInCP(
                net,
                dataset,
                tmpdir,
                batch_size=batch_size,
                loss_fn=loss_fn,
                use_autograd_hacks=False,
            )
        )
        super(TestTracInCP, self).setUp()


class TestTracInCPCheckAutogradHacks(
    _TestTracInRegression1DCheckAutogradHacks,
    _TestTracInRegression20DCheckAutogradHacks,
    _TestTracInXORCheckAutogradHacks,
    _TestTracInIdentityRegressionCheckAutogradHacks,
    BaseTest,
):
    def setUp(self):
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn, use_autograd_hacks: (
                TracInCP(
                    net,
                    dataset,
                    tmpdir,
                    batch_size=batch_size,
                    loss_fn=loss_fn,
                    use_autograd_hacks=use_autograd_hacks,
                )
            )
        )


class TestTracInCPFastRandProjTests(
    _TestTracInRegression1DNumerical,
    _TestTracInGetKMostInfluential,
    _TestTracInSelfInfluence,
    BaseTest,
):
    def setUp(self):
        super().setUp()
        self.reduction = "sum"
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn: TracInCP(
                net,
                dataset,
                tmpdir,
                batch_size=batch_size,
                loss_fn=loss_fn,
                use_autograd_hacks=True,
            )
        )

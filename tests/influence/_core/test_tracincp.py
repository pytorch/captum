# #!/usr/bin/env python3

from captum.influence._core.tracincp import TracInCP
from tests.helpers.basic import BaseTest
from tests.influence._utils.common import (
    _TestTracInRegression1DCheckIdx,
    _TestTracInRegression20DCheckIdx,
    _TestTracInXORCheckIdx,
    _TestTracInIdentityRegressionCheckIdx,
    _TestTracInRegression1DCheckSampleWiseTrick,
    _TestTracInRegression20DCheckSampleWiseTrick,
    _TestTracInXORCheckSampleWiseTrick,
    _TestTracInIdentityRegressionCheckSampleWiseTrick,
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
                sample_wise_grads_per_batch=False,
            )
        )
        super(TestTracInCP, self).setUp()


class TestTracInCPCheckSampleWiseTrick(
    _TestTracInRegression1DCheckSampleWiseTrick,
    _TestTracInRegression20DCheckSampleWiseTrick,
    _TestTracInXORCheckSampleWiseTrick,
    _TestTracInIdentityRegressionCheckSampleWiseTrick,
    BaseTest,
):
    def setUp(self):
        self.tracin_constructor = (
            lambda net, dataset, tmpdir, batch_size, loss_fn, sample_wise_trick: (
                TracInCP(
                    net,
                    dataset,
                    tmpdir,
                    batch_size=batch_size,
                    loss_fn=loss_fn,
                    sample_wise_grads_per_batch=sample_wise_trick,
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
                sample_wise_grads_per_batch=True,
            )
        )

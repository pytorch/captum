#!/usr/bin/env python3

import torch
from captum._utils.models.linear_model.model import (
    SGDLasso,
    SGDLinearRegression,
    SGDRidge,
)
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


def _evaluate(test_data, classifier):
    classifier.eval()

    l1_loss = 0.0
    l2_loss = 0.0
    n = 0
    l2_losses = []
    with torch.no_grad():
        for data in test_data:
            if len(data) == 2:
                x, y = data
                w = None
            else:
                x, y, w = data

            out = classifier(x)

            y = y.view(x.shape[0], -1)
            assert y.shape == out.shape

            if w is None:
                l1_loss += (out - y).abs().sum(0).to(dtype=torch.float64)
                l2_loss += ((out - y) ** 2).sum(0).to(dtype=torch.float64)
                l2_losses.append(((out - y) ** 2).to(dtype=torch.float64))
            else:
                l1_loss += (
                    (w.view(-1, 1) * (out - y)).abs().sum(0).to(dtype=torch.float64)
                )
                l2_loss += (
                    (w.view(-1, 1) * ((out - y) ** 2)).sum(0).to(dtype=torch.float64)
                )
                l2_losses.append(
                    (w.view(-1, 1) * ((out - y) ** 2)).to(dtype=torch.float64)
                )

            n += x.shape[0]

    l2_losses = torch.cat(l2_losses, dim=0)
    assert n > 0

    # just to double check
    assert ((l2_losses.mean(0) - l2_loss / n).abs() <= 0.1).all()

    classifier.train()
    return {"l1": l1_loss / n, "l2": l2_loss / n}


class TestLinearModel(BaseTest):
    MAX_POINTS: int = 3

    def train_and_compare(
        self,
        model_type,
        xs,
        ys,
        expected_loss,
        expected_reg=0.0,
        expected_hyperplane=None,
        norm_hyperplane=True,
        weights=None,
        delta=0.1,
        init_scheme="zeros",
        objective="lasso",
        bias=True,
    ):
        assert objective in ["lasso", "ridge", "ols"]

        if weights is None:
            train_dataset = torch.utils.data.TensorDataset(xs, ys)
        else:
            train_dataset = torch.utils.data.TensorDataset(xs, ys, weights)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), num_workers=0
        )

        model = model_type(bias=bias)
        model.fit(
            train_loader,
            init_scheme=init_scheme,
            max_epoch=150,
            initial_lr=0.1,
            patience=5,
        )

        self.assertTrue(model.bias() is not None if bias else model.bias() is None)

        l2_loss = _evaluate(train_loader, model)["l2"]

        if objective == "lasso":
            reg = model.representation().norm(p=1).view_as(l2_loss)
        elif objective == "ridge":
            reg = model.representation().norm(p=2).view_as(l2_loss)
        else:
            assert objective == "ols"
            reg = torch.zeros_like(l2_loss)

        if not isinstance(expected_loss, torch.Tensor):
            expected_loss = torch.tensor([expected_loss], dtype=l2_loss.dtype).view(1)

        if not isinstance(expected_reg, torch.Tensor):
            expected_reg = torch.tensor([expected_reg], dtype=reg.dtype)

        assertTensorAlmostEqual(self, l2_loss, expected_loss, delta=delta)
        assertTensorAlmostEqual(self, reg, expected_reg, delta=delta)

        if expected_hyperplane is not None:
            h = model.representation()
            if norm_hyperplane:
                h /= h.norm(p=2)
            assertTensorAlmostEqual(self, h, expected_hyperplane, delta=delta)

    def test_simple_linear_regression(self):
        xs = torch.randn(TestLinearModel.MAX_POINTS, 1)
        ys = 3 * xs + 1

        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=0,
            expected_reg=0,
            objective="ols",
        )
        self.train_and_compare(
            SGDLasso,
            xs,
            ys,
            expected_loss=3,
            expected_reg=0,
            objective="lasso",
            delta=0.2,
        )
        self.train_and_compare(
            SGDRidge,
            xs,
            ys,
            expected_loss=3,
            expected_reg=0,
            objective="ridge",
            delta=0.2,
        )

    def test_simple_multi_output(self):
        xs = torch.randn(TestLinearModel.MAX_POINTS, 1)
        y1 = 3 * xs + 1
        y2 = -5 * xs
        ys = torch.stack((y1, y2), dim=1).squeeze()

        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=torch.DoubleTensor([0, 0]),
            expected_reg=torch.DoubleTensor([0, 0]),
            objective="ols",
        )

    def test_simple_linear_classification(self):
        xs = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
        ys = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=0,
            expected_reg=0,
            objective="ols",
        )
        self.train_and_compare(
            SGDLasso, xs, ys, expected_loss=1, expected_reg=0.0, objective="lasso"
        )
        self.train_and_compare(
            SGDRidge, xs, ys, expected_loss=1, expected_reg=0.0, objective="ridge"
        )

        ys = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=0,
            expected_reg=0,
            objective="ols",
        )
        self.train_and_compare(
            SGDLasso, xs, ys, expected_loss=0.25, expected_reg=0, objective="lasso"
        )
        self.train_and_compare(
            SGDRidge, xs, ys, expected_loss=0.25, expected_reg=0, objective="ridge"
        )

    def test_simple_xor_problem(self):
        r"""
           ^
         o | x
        ---|--->
         x | o
        """
        xs = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
        ys = torch.tensor([1.0, 1.0, -1.0, -1.0])

        expected_hyperplane = torch.Tensor([[0, 0]])
        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=1,
            expected_reg=0,
            objective="ols",
            expected_hyperplane=expected_hyperplane,
            norm_hyperplane=False,
            bias=False,
        )
        self.train_and_compare(
            SGDLasso,
            xs,
            ys,
            expected_loss=1,
            expected_reg=0,
            objective="lasso",
            expected_hyperplane=expected_hyperplane,
            norm_hyperplane=False,
            bias=False,
        )
        self.train_and_compare(
            SGDRidge,
            xs,
            ys,
            expected_loss=1,
            expected_reg=0,
            objective="ridge",
            expected_hyperplane=expected_hyperplane,
            norm_hyperplane=False,
            bias=False,
        )

    def test_weighted_problem(self):
        r"""
           ^
         0 | x
        ---|--->
         0 | o
        """
        xs = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
        ys = torch.tensor([1.0, 1.0, -1.0, -1.0])
        weights = torch.tensor([1.0, 0.0, 1.0, 0.0])

        self.train_and_compare(
            SGDLinearRegression,
            xs,
            ys,
            expected_loss=0,
            expected_reg=0,
            expected_hyperplane=torch.Tensor([[0.0, 1.0]]),
            weights=weights,
            norm_hyperplane=True,
            init_scheme="zeros",
            objective="ols",
            bias=False,
        )
        self.train_and_compare(
            SGDLasso,
            xs,
            ys,
            expected_loss=0.5,
            expected_reg=0,
            expected_hyperplane=torch.Tensor([[0.0, 0.0]]),
            weights=weights,
            norm_hyperplane=False,
            init_scheme="zeros",
            objective="lasso",
            bias=False,
        )
        self.train_and_compare(
            SGDRidge,
            xs,
            ys,
            expected_loss=0.5,
            expected_reg=0,
            expected_hyperplane=torch.Tensor([[0.0, 0.0]]),
            weights=weights,
            norm_hyperplane=False,
            init_scheme="zeros",
            objective="ridge",
            bias=False,
        )

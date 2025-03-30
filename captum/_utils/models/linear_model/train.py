import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from captum._utils.models.linear_model.model import LinearModel
from torch.utils.data import DataLoader


def l2_loss(x1, x2, weights=None):
    if weights is None:
        return torch.mean((x1 - x2) ** 2) / 2.0
    else:
        return torch.sum((weights / weights.norm(p=1)) * ((x1 - x2) ** 2)) / 2.0


def sgd_train_linear_model(
    model: LinearModel,
    dataloader: DataLoader,
    construct_kwargs: Dict[str, Any],
    max_epoch: int = 100,
    reduce_lr: bool = True,
    initial_lr: float = 0.01,
    alpha: float = 1.0,
    loss_fn: Callable = l2_loss,
    reg_term: Optional[int] = 1,
    patience: int = 10,
    threshold: float = 1e-4,
    running_loss_window: Optional[int] = None,
    device: Optional[str] = None,
    init_scheme: str = "zeros",
    debug: bool = False,
) -> Dict[str, float]:
    r"""
    Trains a linear model with SGD. This will continue to iterate your
    dataloader until we converged to a solution or alternatively until we have
    exhausted `max_epoch`.

    Convergence is defined by the loss not changing by `threshold` amount for
    `patience` number of iterations.

    Args:
        model
            The model to train
        dataloader
            The data to train it with. We will assume the dataloader produces
            either pairs or triples of the form (x, y) or (x, y, w). Where x and
            y are typical pairs for supervised learning and w is a weight
            vector.

            We will call `model._construct_model_params` with construct_kwargs
            and the input features set to `x.shape[1]` (`x.shape[0]` corresponds
            to the batch size). We assume that `len(x.shape) == 2`, i.e. the
            tensor is flat. The number of output features will be set to
            y.shape[1] or 1 (if `len(y.shape) == 1`); we require `len(y.shape)
            <= 2`.
        max_epoch
            The maximum number of epochs to exhaust
        reduce_lr
            Whether or not to reduce the learning rate as iterations progress.
            Halves the learning rate when the training loss does not move. This
            uses torch.optim.lr_scheduler.ReduceLROnPlateau and uses the
            parameters `patience` and `threshold`
        initial_lr
            The initial learning rate to use.
        alpha
            A constant for the regularization term.
        loss_fn
            The loss to optimise for. This must accept three parameters:
            x1 (predicted), x2 (labels) and a weight vector
        reg_term
            Regularization is defined by the `reg_term` norm of the weights.
            Please use `None` if you do not wish to use regularization.
        patience
            Defines the number of iterations in a row the loss must remain
            within `threshold` in order to be classified as converged.
        threshold
            Threshold for convergence detection.
        running_loss_window
            Used to report the training loss once we have finished training and
            to determine when we have converged (along with reducing the
            learning rate).

            The reported training loss will take the last `running_loss_window`
            iterations and average them.

            If `None` we will approximate this to be the number of examples in
            an epoch.
        init_scheme
            Initialization to use prior to training the linear model.
        device
            The device to send the model and data to. If None then no `.to` call
            will be used.
        debug
            Whether to print the loss, learning rate per iteration

    Returns
        This will return the final training loss (averaged with
        `running_loss_window`)
    """
    loss_window: List[torch.Tensor] = []
    min_avg_loss = None
    convergence_counter = 0
    converged = False

    def get_point(datapoint):
        if len(datapoint) == 2:
            x, y = datapoint
            w = None
        else:
            x, y, w = datapoint

        if device is not None:
            x = x.to(device)
            y = y.to(device)
            if w is not None:
                w = w.to(device)

        return x, y, w

    # get a point and construct the model
    data_iter = iter(dataloader)
    x, y, w = get_point(next(data_iter))

    model._construct_model_params(
        in_features=x.shape[1],
        out_features=y.shape[1] if len(y.shape) == 2 else 1,
        **construct_kwargs,
    )
    model.train()

    assert model.linear is not None

    if init_scheme is not None:
        assert init_scheme in ["xavier", "zeros"]

        with torch.no_grad():
            if init_scheme == "xavier":
                torch.nn.init.xavier_uniform_(model.linear.weight)
            else:
                model.linear.weight.zero_()

            if model.linear.bias is not None:
                model.linear.bias.zero_()

    with torch.enable_grad():
        optim = torch.optim.SGD(model.parameters(), lr=initial_lr)
        if reduce_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, factor=0.5, patience=patience, threshold=threshold
            )

        t1 = time.time()
        epoch = 0
        i = 0
        while epoch < max_epoch:
            while True:  # for x, y, w in dataloader
                if running_loss_window is None:
                    running_loss_window = x.shape[0] * len(dataloader)

                y = y.view(x.shape[0], -1)
                if w is not None:
                    w = w.view(x.shape[0], -1)

                i += 1

                out = model(x)

                loss = loss_fn(y, out, w)
                if reg_term is not None:
                    reg = torch.norm(model.linear.weight, p=reg_term)
                    loss += reg.sum() * alpha

                if len(loss_window) >= running_loss_window:
                    loss_window = loss_window[1:]
                loss_window.append(loss.clone().detach())
                assert len(loss_window) <= running_loss_window

                average_loss = torch.mean(torch.stack(loss_window))
                if min_avg_loss is not None:
                    # if we haven't improved by at least `threshold`
                    if average_loss > min_avg_loss or torch.isclose(
                        min_avg_loss, average_loss, atol=threshold
                    ):
                        convergence_counter += 1
                        if convergence_counter >= patience:
                            converged = True
                            break
                    else:
                        convergence_counter = 0
                if min_avg_loss is None or min_avg_loss >= average_loss:
                    min_avg_loss = average_loss.clone()

                if debug:
                    print(
                        f"lr={optim.param_groups[0]['lr']}, Loss={loss},"
                        + "Aloss={average_loss}, min_avg_loss={min_avg_loss}"
                    )

                loss.backward()
                optim.step()
                model.zero_grad()
                if scheduler:
                    scheduler.step(average_loss)

                temp = next(data_iter, None)
                if temp is None:
                    break
                x, y, w = get_point(temp)

            if converged:
                break

            epoch += 1
            data_iter = iter(dataloader)
            x, y, w = get_point(next(data_iter))

    t2 = time.time()
    return {
        "train_time": t2 - t1,
        "train_loss": torch.mean(torch.stack(loss_window)).item(),
        "train_iter": i,
        "train_epoch": epoch,
    }


class NormLayer(nn.Module):
    def __init__(self, mean, std, n=None, eps=1e-8) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def sklearn_train_linear_model(
    model: LinearModel,
    dataloader: DataLoader,
    construct_kwargs: Dict[str, Any],
    sklearn_trainer: str = "Lasso",
    norm_input: bool = False,
    **fit_kwargs,
):
    r"""
    Alternative method to train with sklearn. This does introduce some slight
    overhead as we convert the tensors to numpy and then convert the resulting
    trained model to a `LinearModel` object. However, this conversion
    should be negligible.

    Please note that this assumes:

    0. You have sklearn and numpy installed
    1. The dataset can fit into memory

    Args
        model
            The model to train.
        dataloader
            The data to use. This will be exhausted and converted to numpy
            arrays. Therefore please do not feed an infinite dataloader.
        norm_input
            Whether or not to normalize the input
        sklearn_trainer
            The sklearn model to use to train the model. Please refer to
            sklearn.linear_model for a list of modules to use.
        construct_kwargs
            Additional arguments provided to the `sklearn_trainer` constructor
        fit_kwargs
            Other arguments to send to `sklearn_trainer`'s `.fit` method
    """
    from functools import reduce

    try:
        import numpy as np
    except ImportError:
        raise ValueError("numpy is not available. Please install numpy.")

    try:
        import sklearn
        import sklearn.linear_model
        import sklearn.svm
    except ImportError:
        raise ValueError("sklearn is not available. Please install sklearn >= 0.23")

    if not sklearn.__version__ >= "0.23.0":
        warnings.warn(
            "Must have sklearn version 0.23.0 or higher to use "
            "sample_weight in Lasso regression."
        )

    num_batches = 0
    xs, ys, ws = [], [], []
    for data in dataloader:
        if len(data) == 3:
            x, y, w = data
        else:
            assert len(data) == 2
            x, y = data
            w = None

        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        if w is not None:
            ws.append(w.cpu().numpy())
        num_batches += 1

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(ws) > 0:
        w = np.concatenate(ws, axis=0)
    else:
        w = None

    if norm_input:
        mean, std = x.mean(0), x.std(0)
        x -= mean
        x /= std

    t1 = time.time()
    sklearn_model = reduce(
        lambda val, el: getattr(val, el), [sklearn] + sklearn_trainer.split(".")
    )(**construct_kwargs)
    try:
        sklearn_model.fit(x, y, sample_weight=w, **fit_kwargs)
    except TypeError:
        sklearn_model.fit(x, y, **fit_kwargs)
        warnings.warn(
            "Sample weight is not supported for the provided linear model!"
            " Trained model without weighting inputs. For Lasso, please"
            " upgrade sklearn to a version >= 0.23.0."
        )

    t2 = time.time()

    # Convert weights to pytorch
    classes = (
        torch.IntTensor(sklearn_model.classes_)
        if hasattr(sklearn_model, "classes_")
        else None
    )

    # extract model device
    device = model.device if hasattr(model, "device") else "cpu"

    num_outputs = sklearn_model.coef_.shape[0] if sklearn_model.coef_.ndim > 1 else 1
    weight_values = torch.FloatTensor(sklearn_model.coef_).to(device)  # type: ignore
    bias_values = torch.FloatTensor([sklearn_model.intercept_]).to(  # type: ignore
        device  # type: ignore
    )  # type: ignore
    model._construct_model_params(
        norm_type=None,
        weight_values=weight_values.view(num_outputs, -1),
        bias_value=bias_values.squeeze().unsqueeze(0),
        classes=classes,
    )

    if norm_input:
        model.norm = NormLayer(mean, std)

    return {"train_time": t2 - t1}

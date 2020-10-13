from contextlib import suppress
from typing import Dict, Callable, Iterable, Optional, List, Union
from typing_extensions import Protocol
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from clarity.pytorch import StopCriteria, Objective


def optimize(
    objective: Objective,
    stop_criteria: Optional[StopCriteria] = None,
    optimizer: Optional[optim.Optimizer] = None,
):
    stop_criteria = stop_criteria or n_steps(1024)
    optimizer = optimizer or optim.Adam(objective.parameters(), lr=0.025)
    assert isinstance(optimizer, optim.Optimizer)

    history = []
    step = 0
    while stop_criteria(step, objective, history, optimizer):
        optimizer.zero_grad()

        loss_value = objective.loss()
        history.append(loss_value.cpu().detach().numpy())
        (-1 * loss_value.mean()).backward()
        optimizer.step()
        step += 1

    objective.cleanup()
    return history


def n_steps(n: int) -> StopCriteria:
    pbar = tqdm(total=n, unit="step")

    def continue_while(step, obj, history, optim):
        if len(history) > 0:
            pbar.set_postfix({"Objective": f"{history[-1].mean():.1f}"}, refresh=False)
        if step < n:
            pbar.update()
            return True
        else:
            pbar.close()
            return False

    return continue_while

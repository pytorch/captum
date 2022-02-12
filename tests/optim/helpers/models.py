from typing import Type

import torch


def _check_layer_in_model(
    self,
    model: torch.nn.Module,
    layer: Type[torch.nn.Module],
) -> None:
    def check_for_layer_in_model(model, layer) -> bool:
        for name, child in model._modules.items():
            if child is not None:
                if isinstance(child, layer):
                    return True
                if check_for_layer_in_model(child, layer):
                    return True
        return False

    self.assertTrue(check_for_layer_in_model(model, layer))


def _check_layer_not_in_model(
    self, model: torch.nn.Module, layer: Type[torch.nn.Module]
) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            _check_layer_not_in_model(self, child, layer)

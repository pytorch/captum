import torch
import torch.nn as nn
from captum.optim.transform import (RandomScale, RandomSpatialJitter,
                                    rand_select)
from tests.helpers.basic import BaseTest


class TestRandSelect(BaseTest):
    def test_rand_select(self):
        a = (1, 2, 3, 4, 5)
        b = torch.Tensor([0.1, -5, 56.7, 99.0])

        assert rand_select(a) in a
        assert rand_select(b) in b


class TestRandomScale(BaseTest):
    def test_random_scale(self):
        scale_module = RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)

        # Test rescaling
        assert torch.all(
            scale_module.scale_tensor(test_tensor, 0.5).eq(
                torch.tensor(
                    [
                        [
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        ]
                    ]
                )
            )
        )
        assert torch.all(
            scale_module.scale_tensor(test_tensor, 1.5).eq(
                torch.tensor(
                    [
                        [
                            [
                                [0.2500, 0.5000, 0.2500],
                                [0.5000, 1.0000, 0.5000],
                                [0.2500, 0.5000, 0.2500],
                            ],
                            [
                                [0.2500, 0.5000, 0.2500],
                                [0.5000, 1.0000, 0.5000],
                                [0.2500, 0.5000, 0.2500],
                            ],
                            [
                                [0.2500, 0.5000, 0.2500],
                                [0.5000, 1.0000, 0.5000],
                                [0.2500, 0.5000, 0.2500],
                            ],
                        ]
                    ]
                )
            )
        )

        # Test scale matrices
        assert torch.all(
            scale_module.get_scale_mat(0.5, test_tensor.device, test_tensor.dtype).eq(
                torch.tensor([[0.5000, 0.0000, 0.0000], [0.0000, 0.5000, 0.0000]])
            )
        )
        assert torch.all(
            scale_module.get_scale_mat(1.24, test_tensor.device, test_tensor.dtype).eq(
                torch.tensor([[1.2400, 0.0000, 0.0000], [0.0000, 1.2400, 0.0000]])
            )
        )


class TestRandomSpatialJitter(BaseTest):
    def test_random_spatial_jitter(self):

        spatialjitter = RandomSpatialJitter(3)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)

        assert torch.all(
            spatialjitter.translate_tensor(test_input, [4, 4]).eq(
                torch.tensor(
                    [
                        [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                            ],
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                            ],
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 1.0],
                            ],
                        ]
                    ]
                )
            )
        )

        spatialjitter = RandomSpatialJitter(2)

        torch.all(
            spatialjitter.translate_tensor(test_input, [0, 3]).eq(
                torch.tensor(
                    [
                        [
                            [
                                [0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                            ],
                            [
                                [0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                            ],
                        ]
                    ]
                )
            )
        )

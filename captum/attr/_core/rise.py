import math

from typing import Any, Callable, Generator, List, Tuple

import torch
import torch.nn.functional as F

from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from captum.log import log_usage

TupleOfTensors = Tuple[torch.Tensor]
InputShape = Tuple[int]
InputShapes = List[InputShape]
InputTypes = List[torch.dtype]


def ensure_tuple(x):
    if not isinstance(x, Tuple):
        x = (x,)
    return x


class MaskSetConfig:
    r"""

    Will generate 2 input mask sets
    """

    @classmethod
    def from_input(
        cls,
        xs: TensorOrTupleOfTensorsGeneric,
        initial_mask_shapes: InputShapes,
        ignore_initial_dims: int = 2,
    ):
        """Generates as MaskSetConfig from a tuple of inputs and a tuple of initial mask shapes.
        By default, assumes each input is of shape [B,C,D1,D2,D3]], where B and C are the batch and channel dimensions which are ignored, and the final mask sizes are extracted as [D1,D2,D3]. Dimensions D2 and D3 are optional.
        Because of pytorch's `interpolate` limitations, this only supports 5D inputs (3D masks) at most.

        Returns:
            MaskSetConfig with the config to generate masks
        """
        xs = ensure_tuple(xs)
        input_shapes = tuple(x.shape[ignore_initial_dims:] for x in xs)
        input_types = tuple(x.dtype for x in xs)
        return MaskSetConfig(input_shapes, input_types, initial_mask_shapes)

    def __init__(
        self,
        final_mask_shapes: InputShapes,
        input_types: InputTypes,
        initial_mask_shapes: InputShapes,
    ) -> None:

        # validate having same amount of info for all inputs
        ni, ns, nt = len(initial_mask_shapes), len(final_mask_shapes), len(input_types)
        assert (
            ni == ns == nt
        ), f"Number of final shapes, input types and initial shapes must match, found {ns}, {nt} and {ni} respectively."

        # validate shapes
        for input_shape, initial_mask_shape in zip(
            final_mask_shapes, initial_mask_shapes
        ):
            assert (
                len(input_shape) <= 5
            ), f"Mask generation only supports up to 5D inputs, found input shape {input_shape}."
            assert len(initial_mask_shape) == len(
                input_shape
            ), f"The mask shape must have the same dimensions as the input shape, because it must include at least the batch dimension and 'channel' dimension"

        self.input_shapes = final_mask_shapes
        self.input_types = input_types
        self.initial_mask_shapes = initial_mask_shapes

    def mask_configs(self) -> List[Tuple[InputShape, torch.dtype, InputShape]]:
        """Returns a list of the config of each mask for each different input of the model
        Each config contains the shape of the input, its type and the initial shape of the mask
        """
        return list(zip(self.input_shapes, self.input_types, self.initial_mask_shapes))

    def generate(self):
        """Generates a set of masks for the preconfigured input shapes, types and initial mask shapes.

        Returns:

        - A mask set. Each input mask set actually contains a tuple with the masks for each different input of the model. Therefore, the length of the tuple must match len(self.input_shapes)==len(self.input_types)==len(self.initial_mask_shapes). The mask for each input have the same size as the input itself.
        """

        return tuple(
            self.generate_mask(*mask_config) for mask_config in self.mask_configs()
        )

    def generate_mask(
        self,
        input_shape: InputShape,
        input_type: torch.dtype,
        initial_mask_shape: InputShape,
    ):
        """Generates a single mask for a given input shape, type and initial mask shape.

        Examples::

        >>> generate_mask((25,30),torch.float64,(5,6))
        Will generate a random mask of size (5,6) and float64 dtype, upsample it to a value of than (25+5,30+6), then crop it to (25,30).
        """

        # input_shape = (HxW), initial_mask_shape = (hxw)

        # upsample_shape = (h+1)*CH x (w+1)*CW (where CHxCW = ceil(H/h) x ceil(W/w))
        upsample_shape = tuple(
            (shape + 1) * math.ceil(input_shape / shape)
            for (shape, input_shape) in zip(initial_mask_shape, input_shape)
        )

        mask_def = torch.empty(initial_mask_shape, dtype=input_type)

        for i in range(initial_mask_shape[0]):
            for j in range(initial_mask_shape[1]):
                mask_def[i, j] = torch.randint(0, 2, (1,))

        # Billinear interpolation
        mask_def = mask_def[None, None, :]

        upsampled_mask = F.interpolate(
            mask_def,
            upsample_shape,
            mode="bilinear",
            align_corners=True,
        )

        upsampled_mask = upsampled_mask[0, 0, :, :]

        cropped_mask = self.random_crop(upsampled_mask, input_shape)
        # print(cropped_mask.shape)
        return cropped_mask

    def random_crop(self, mask, input_shape):
        mask_shape = mask.shape

        # Ensure mask size is greater or equal to input size
        for ms, ins in zip(mask_shape, input_shape):
            assert ins <= ms

        # Compute size differences between mask and input shapes
        differences = [ms - ins for ms, ins in zip(mask_shape, input_shape)]
        # Compute random offsets based on the differences
        offsets = [int(torch.randint(0, d, (1,))) for d in differences]
        # Generate slices in terms of the offsets to crop
        slices = tuple(
            slice(offset, offset + dim) for offset, dim in zip(offsets, input_shape)
        )
        # crop mask
        mask = mask[slices]
        return mask


def tuple_to_device(t, device):
    return tuple(x.to(device) for x in t)


class RISE(FeatureAblation):
    r"""
    RISE: Randomized Input Sampling for Explanation of Black-box Models

    A perturbation based approach to compute attribution, involving
    a monte-carlo approach to detecting the sensitivity of the output with
    respect to features. RISE estimates the sensitivity of each input feature
    by sampling `n_masks` random occlusion masks, and computing the output for each
    correspondingly occluded input image. Each mask is assigned a score based on
    the output of the model. Afterwards, masks are averaged, using the score
    as a weight.

    To sample occlusion masks, RISE assumes a strong spatial structure in the
    feature space, so that features that are close to each other are more likely
    to be correlated.


    More details regarding  method can be found in the original paper and in the
    DeepExplain implementation.
    https://arxiv.org/abs/1806.07421
    https://github.com/eclique/RISE
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    @log_usage()
    def attribute(  # type: ignore
        self,
        input_set: TensorOrTupleOfTensorsGeneric,
        n_masks: int,
        initial_mask_shapes: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

                inputs (Tensor or tuple[Tensor, ...]): Input for which RISE
                            attributions are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
        """
        # Generate mask sets
        input_set = ensure_tuple(input_set)
        mask_set_config = MaskSetConfig.from_input(input_set, initial_mask_shapes)
        mask_sets = generate_mask_sets(n_masks, mask_set_config)

        # initialize heatmap set
        batch_size = input_set[0].shape[0]
        heatmap_set = tuple(
            torch.zeros(batch_size, *input_shape)
            for input_shape in mask_set_config.input_shapes
        )

        # send heatmaps to same device as inputs
        input_device = input_set[0].device
        heatmap_set = tuple_to_device(heatmap_set, input_device)

        if show_progress:
            rise_progress = progress(
                total=n_masks,
                desc=f"{self.get_name()} mask",
            )
            rise_progress.update(0)
        # calculate weights for masks
        for i, mask_set in enumerate(mask_sets):
            # send mask to same device as inputs
            mask_set = tuple_to_device(mask_set, input_device)
            # generate masked inputs
            masket_input_set = tuple(m * input for m, input in zip(mask_set, input_set))
            # compute scores, obtain score for each sample in batch
            # detach to avoid computing backward and using more memory
            # TODO find a way to avoid forward_func from being in training state and returning the grad_fn
            output = self.forward_func(*masket_input_set).detach()
            mask_weight = output[range(batch_size), target]

            # update heatmaps with weight of mask
            for heatmap, mask in zip(heatmap_set, mask_set):
                # Monte Carlo approximation
                # heatmap: batch_size x input_shape
                # mask: input_shape
                # mask_weight: batch_size
                fill_dims = (1,) * len(mask.shape)
                m2 = mask[None, :]
                m1 = mask_weight.view(-1, *fill_dims)
                # batch_mask_weights: batch_size * input_shape
                batch_mask_weights = m1 * m2
                heatmap += batch_mask_weights / n_masks

            if show_progress:
                rise_progress.update()
        heatmap_set = tuple_to_device(heatmap_set, "cpu")
        if show_progress:
            rise_progress.close()

        if len(heatmap_set) == 1:
            heatmap_set = heatmap_set[0]

        return heatmap_set


def generate_mask_sets(
    n_masks: int, mask_set_config: MaskSetConfig
) -> Generator[Tuple[torch.Tensor], None, None]:

    r"""returns a generator for of n_masks
    Args:
    n_masks: Number of mask sets to generate
    mask_set_config: Configuration of the mask sets

    Returns:

    - Generator of a set of n_masks input mask sets. Each input mask set actually contains a tuple with the masks for each different input of the model. Therefore, the length of the tuple must match len(input_shapes)==len(input_types). The mask for each input has the same size as the input itself.
    """
    for _ in range(n_masks):
        yield mask_set_config.generate()

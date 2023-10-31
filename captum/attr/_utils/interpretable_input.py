from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


def _scatter_itp_attr_by_mask(
    itp_attr: Tensor,
    input_shape: Tuple[int, ...],
    mask: Tensor,
) -> Tensor:
    """
    Scatter the attribution of the interpretable features to the model input shape
    by mask, if the interpretable features are the mask groups of the raw
    input elements,
    """

    # itp_attr in shape(*output_dims, n_itp_features)
    output_dims = itp_attr.shape[:-1]
    n_itp_features = itp_attr.shape[-1]

    # input_shape in shape(batch_size, *inp_feature_dims)
    # attribute in shape(*output_dims, *inp_feature_dims)
    attr_shape = (*output_dims, *input_shape[1:])

    expanded_feature_indices = mask.expand(attr_shape)

    if len(input_shape) > 2:
        # exclude batch_size & last of actual value
        extra_inp_dims = list(input_shape[1:-1])

        # unsqueeze itp_attr to have same number of dims as input
        # (*output_dims, 1..., 1, n_itp_features)
        # then broadcast to (*output_dims, *inp.shape[1:-1], n_itp_features)
        n_extra_dims = len(extra_inp_dims)
        unsqueezed_shape = (*output_dims, *(1,) * n_extra_dims, n_itp_features)
        expanded_shape = (*output_dims, *extra_inp_dims, n_itp_features)
        expanded_itp_attr = itp_attr.reshape(unsqueezed_shape).expand(expanded_shape)
    else:
        expanded_itp_attr = itp_attr

    # gather from (*output_dims, *inp.shape[1:-1], n_itp_features)
    attr = torch.gather(expanded_itp_attr, -1, expanded_feature_indices)

    return attr


class InterpretableInput:
    """
    InterpretableInput is an adapter for different kinds of model inputs to
    work in Captum's attribution methods. Generally, attribution methods of Captum
    assume the inputs are numerical PyTorch tensors whose 1st dimension must be batch
    size and each index in the rest of dimensions is an interpretable feature. But this
    is not always true in practice. First, the model may take inputs of formats other
    than tensor that also require attributions. For example, a model with encapsulated
    tokenizer can directly take string as input. Second, what is considered as
    an interpretable feature always depends on the actual application and the user's
    desire. For example, the interpretable feature of an image tensor can either be
    each pixel or some segments. For text, users may see the entire string as one
    interpretable feature or view each word as one interpretable feature. This class
    provides a place to define what is the actual model input and the corresponding
    interpretable format for attribution, and the transformation between them.
    It serves as a common interface to be used inthe attribution methods to make
    Captum understand how to perturb various inputs.

    The concept Interpretable Input mainly comes from the following two papers:

    `"Why Should I Trust You?": Explaining the Predictions of Any Classifier
    <https://arxiv.org/abs/1602.04938>`_

    `A Unified Approach to Interpreting Model Predictions
    <https://arxiv.org/abs/1705.07874>`_

    which is also referred to as interpretable representation or simplified
    input. It can be represented as a mapping function:

    .. math::
        x = h_x(x')

    where :math:`x` is the model input, which can be anything that the model consumes;
    :math:`x'` is the interpretable input used in the attribution algorithms
    (it must be a PyTorch tensor in Captum), which is often
    binary indicating the “presence” or “absence”; :math:`h_x` is the
    transformer. It is supposed to work with perturbation-based attribution methods,
    but if :math:`h_x` is differentiable, it may also be used
    in gradient-based methods.

    InterpretableInput is the abstract class defining the interface. Captum provides
    the child implementations for some common input formats,
    like text and sparse features. Users can inherit this
    class to create other types of customized input.

    (We expect to support InterpretableInput in all attribution methods, but it
    is only allowed in certain attribution classes like LLMAttribution for now.)
    """

    def to_tensor(self) -> Tensor:
        """
        Return the interpretable representation of this input as a tensor

        Returns:

            itp_tensor (Tensor): interpretable tensor
        """
        pass

    def to_model_input(self, itp_tensor: Optional[Tensor] = None) -> Any:
        """
        Get the (perturbed) input in the format required by the model
        based on the given (perturbed) interpretable representation.

        Args:

            itp_tensor (Tensor, optional): tensor of the interpretable representation
                    of this input. If it is None, assume the interpretable
                    representation is pristine and return the original model input
                    Default: None.


        Returns:

            model_input (Any): model input passed to the forward function
        """
        pass

    def format_attr(self, itp_attr: Tensor) -> Tensor:
        """
        Format the attribution of the interpretable feature if needed.
        The way of formatting depends on the specific interpretable input type.
        A common use is if the interpretable features are the mask groups of the raw
        input elements, the attribution of the interpretable features can be scattered
        back to the model input shape.

        Args:

                itp_attr (Tensor): attributions of the interpretable features

        Returns:

                attr (Tensor): formatted attribution
        """
        return itp_attr


class TextTemplateInput(InterpretableInput):
    """
    TextTemplateInput is an implementation of InterpretableInput for text inputs, whose
    interpretable features are certain segments (e.g., words, phrases) of the text.
    It takes a template string (or function) to define the feature segmentats
    of the input text. Its input format to the model will be the completed text,
    while its interpretable representation will be a binary tensor of the number of
    the segment features whose values indicates if the feature is
    “presence” or “absence”.

    Args:

        template (str or Callable): template string or function that takes
                the text segments and format them into the text input for the model
        values (List[str] or Dict[str, str]): the values of the segments. it is
                the input to the template.
        baselines (List[str] or Dict[str, str] or Callable or None, optional): the
                baseline values for the segment features. If it is None, emptry string
                will be used as the baseline.
                Default: None
        mask (List[int] or Dict[str, int] or None, optional): the mask to group the
                segment features. It must be in the same format as the values
                and assign each segment a mask index. Segments with the same
                index will be seen as a single interpretable feature, which means
                they must be perturbed together and end with same attributions.
                Default: None

    Examples::

        >>> text_inp = TextTemplateInput(
        >>>     template="{} feels {} right now",
        >>>     values=["He", "depressed"],
        >>>     baselines=["It", "neutral"],
        >>> )
        >>>
        >>> text_inp.to_tensor()
        >>> # torch.tensor([[1, 1]])
        >>>
        >>> text_inp.to_model_input(torch.tensor([[0, 1]]))
        >>> # "It feels depressed right now"

    """

    def __init__(
        self,
        template: Union[str, Callable],
        values: Union[List[str], Dict[str, str]],
        baselines: Union[List[str], Dict[str, str], Callable, None] = None,
        mask: Union[List[int], Dict[str, int], None] = None,
    ):
        # convert values dict to list
        if isinstance(values, dict):
            dict_keys = list(values.keys())
            values = [values[k] for k in dict_keys]
        else:
            assert isinstance(
                values, list
            ), f"the values must be either a list or a dict, received: {type(values)}"
            dict_keys = []

        self.values = values
        self.dict_keys = dict_keys

        n_features = len(values)

        if baselines is None:
            # default baseline is to remove the element
            baselines = [""] * len(values)
        elif not callable(baselines):
            if dict_keys:
                assert isinstance(baselines, dict), (
                    "if values is a dict, the baselines must also be a dict "
                    "or a callable which return a dict, "
                    f"received: {type(baselines)}"
                )

                # convert dict to list
                baselines = [baselines[k] for k in dict_keys]
            else:
                assert isinstance(baselines, list), (
                    "if values is a list, the baselines must also be a list "
                    "or a callable which return a list, "
                    f"received: {type(baselines)}"
                )

        self.baselines = baselines

        if mask is None:
            n_itp_features = n_features
        else:
            if self.dict_keys:
                assert isinstance(mask, dict), (
                    "if values is dict, the mask must also be a dict, "
                    f"received: {type(mask)}"
                )

                # convert dict to list
                mask = [mask[k] for k in self.dict_keys]

            mask_ids = set(mask)
            mask_id_to_idx = {mid: i for i, mid in enumerate(mask_ids)}

            # internal compressed mask of continuous interpretable indices from 0
            # cannot replace original mask of ids for grouping across values externally
            self.formatted_mask = [mask_id_to_idx[mid] for mid in mask]

            n_itp_features = len(mask_ids)

        # number of raw features and intepretable features
        self.n_features = n_features
        self.n_itp_features = n_itp_features

        if isinstance(template, str):
            template = template.format
        else:
            assert callable(template), (
                "the template must be either a string or a callable, "
                f"received: {type(template)}"
            )
            template = template
        self.format_fn = template

        self.mask = mask

    def to_tensor(self) -> torch.Tensor:
        # Interpretable representation in shape(1, n_itp_features)
        return torch.tensor([[1.0] * self.n_itp_features])

    def to_model_input(self, perturbed_tensor: Optional[Tensor] = None) -> str:
        values = list(self.values)  # clone

        if perturbed_tensor is not None:
            if callable(self.baselines):
                # a placeholder for advanced baselines
                # TODO: support callable baselines
                baselines = self.baselines()
                if self.dict_keys:
                    assert isinstance(baselines, dict), (
                        "if values is a dict and the baselines is a callable"
                        f"it must return a dict, received: {type(baselines)}"
                    )
                    baselines = [baselines[k] for k in self.dict_keys]
                else:
                    assert isinstance(baselines, list), (
                        "if values is a list and the baselines is a callable"
                        f"it must return a list, received: {type(baselines)}"
                    )
            else:
                baselines = self.baselines

            for i in range(len(values)):
                itp_idx = i
                if self.mask:
                    itp_idx = self.formatted_mask[i]

                itp_val = perturbed_tensor[0][itp_idx]

                if not itp_val:
                    values[i] = baselines[i]

        if self.dict_keys:
            dict_values = dict(zip(self.dict_keys, values))
            input_str = self.format_fn(**dict_values)
        else:
            input_str = self.format_fn(*values)

        return input_str

    def format_attr(self, itp_attr: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return itp_attr

        device = itp_attr.device

        formatted_attr = _scatter_itp_attr_by_mask(
            itp_attr,  # shape(*output_dims, n_itp_features)
            (1, self.n_features),
            torch.tensor([self.formatted_mask], device=device),
        )
        return formatted_attr

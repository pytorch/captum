from typing import Any, Optional

from torch import Tensor


class InterpretableInput:
    """
    InterpretableInput is an adapter for different kinds of model inputs to
    work in Captum's attribution methods. Generally, attribution methods of Captum
    assume the inputs are numerical PyTorch tensors whose 1st dimension must be batch
    size and each index in the rest of dimensions is an interpretable feature. But this
    is not always true in practice. First, the model may take inputs of formats other
    than tensor that also require attributions. For example, a model with encapsulated
    tokenizer can directly take string as input. Second, what is considered as
    an interpretable feature is always depended on the actual application and the user's
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

    def format_attr(self, itp_attr) -> Tensor:
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

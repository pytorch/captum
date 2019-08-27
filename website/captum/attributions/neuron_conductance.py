#!/usr/bin/env python3
import torch
from .utils.approximation_methods import approximation_parameters
from .utils.attribution import NeuronAttribution
from .utils.common import _reshape_and_sum, _extend_index_list
from .utils.gradient import compute_layer_gradients_and_eval


class NeuronConductance(NeuronAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def attribute(
        self,
        inputs,
        neuron_index,
        baselines=None,
        target=None,
        n_steps=500,
        method="riemann_trapezoid",
    ):
        r"""
            Computes conductance with respect to particular hidden neurons. The
            returned output is in the shape of the input, showing the attribution
            / conductance of each input feature to the selected hidden layer neuron.
            The details of the approach can be found here:
            https://arxiv.org/abs/1805.12233

            Args

                inputs:     A single high dimensional input tensor, in which
                            dimension 0 corresponds to number of examples.
                neuron_index: Tuple providing index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                baselines:   A single high dimensional baseline tensor,
                            which has the same shape as the input
                target:     Predicted class index. This is necessary only for
                            classification use cases
                n_steps:    The number of steps used by the approximation method
                method:     Method for integral approximation, one of `riemann_right`,
                            `riemann_left`, `riemann_middle`, `riemann_trapezoid`
                            or `gausslegendre`

            Return

                attributions: Total conductance with respect to each neuron in
                              output of given layer
        """
        if baselines is None:
            baselines = 0

        num_examples = inputs.shape[0]

        # Retrieve scaling factors for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # Compute scaled inputs from baseline to final input.
        scaled_features = torch.cat(
            [baselines + alpha * (inputs - baselines) for alpha in alphas], dim=0
        )

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval = compute_layer_gradients_and_eval(
            self.forward_func, self.layer, scaled_features, target
        )
        # Creates list of target neuron across batched examples (dimension 0)
        indices = _extend_index_list(layer_eval.shape[0], neuron_index)

        # Computes gradients of target neurons with respect to input
        # input_grads shape is (batch_size*#steps x inputs.shape[1:])
        with torch.autograd.set_grad_enabled(True):
            input_grads = torch.autograd.grad(
                [layer_eval[index] for index in indices], scaled_features
            )[0]

        # Multiplies by appropriate gradient of output with respect to hidden neurons
        # mid_grads is a 1D Tensor of length num_steps*batch_size, containing
        # mid layer gradient for each input step.
        mid_grads = torch.stack([layer_gradients[index] for index in indices])

        scaled_input_gradients = input_grads * mid_grads.reshape(
            (len(indices),) + (1,) * (len(input_grads.shape) - 1)
        )

        # Mutliplies by appropriate step size.
        scaled_grads = scaled_input_gradients.contiguous().view(
            n_steps, -1
        ) * torch.tensor(step_sizes).view(n_steps, 1).to(scaled_input_gradients.device)

        # Aggregates across all steps for each tensor in the input tuple
        total_grads = _reshape_and_sum(
            scaled_grads, n_steps, num_examples, input_grads.shape[1:]
        )
        return total_grads * (inputs - baselines)

import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from captum.attr import Attribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Parameter


DEFAULT_COEFFS = {
    "edge_size": 0.005,
    "edge_ent": 1.0,
    "node_feat_size": 1.0,
    "node_feat_ent": 0.1,
}


class GNNExplainer(Attribution):
    r"""
    GNNExplainer a model interpretability technique for Graph Neural Networks
    (GNNs) as described in the paper:
    `GNNExplainer: Generating Explanations for Graph Neural Networks
    <https://arxiv.org/abs/1903.03894>`_

    GNNExplainer aims to identify a compact subgraph structure and a small
    subset of node features that are influential to the GNN's prediction.
    """

    def __init__(self, model: Callable) -> None:
        r"""
        Args:
            model (Callable): The GNN model that is being explained.
                    The model should take at least two inputs:
                    node features (inputs) and edge_index.
                    It should return a single tensor output.
                    The model can optionally take edge_weights as a third input.
                    If edge_weights are used, they should be multiplied by the
                    learned edge_mask.
        """
        self.model = model
        super().__init__(model)

    def _init_masks(self, inputs: Tensor, edge_index: Tensor) -> None:
        """Initialize learnable masks."""
        num_nodes, num_node_feats = inputs.shape
        num_edges = edge_index.shape[1]

        self.node_feat_mask = Parameter(torch.randn(num_node_feats) * 0.1)
        # Edge mask is initialized for each edge.
        self.edge_mask = Parameter(torch.randn(num_edges) * 0.1)


    def _clear_masks(self) -> None:
        """Clear masks that were stored as attributes."""
        self.node_feat_mask = None
        self.edge_mask = None


    def _get_masked_inputs(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        edge_mask_value: Tensor,
        node_feat_mask_value: Tensor,
        apply_sigmoid: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Applies the masks to the inputs and edge_index.
        Assumes edge_mask_value and node_feat_mask_value are the raw parameters.
        """
        if apply_sigmoid:
            node_feat_m = torch.sigmoid(node_feat_mask_value)
            edge_m = torch.sigmoid(edge_mask_value)
        else:
            node_feat_m = node_feat_mask_value
            edge_m = edge_mask_value

        masked_inputs = inputs * node_feat_m
        # The edge_mask is used as edge_weights in the forward pass
        return masked_inputs, edge_m


    def _loss_fn(
        self,
        masked_pred: Tensor,
        original_pred: Tensor,
        edge_mask_value: Tensor, # raw mask before sigmoid
        node_feat_mask_value: Tensor, # raw mask before sigmoid
        coeffs: Dict[str, float],
        target_class: Optional[int] = None,
    ) -> Tensor:
        """
        Computes the loss for GNNExplainer.
        """
        if target_class is None:
            # Use the predicted class if not specified
            target_class = torch.argmax(original_pred, dim=-1)

        # 1. Prediction Loss (Negative Log-Likelihood for the target class)
        # Ensure masked_pred is in log scale if model output isn't already
        # Assuming model output is raw logits, apply log_softmax
        log_probs = F.log_softmax(masked_pred, dim=-1)
        pred_loss = -log_probs[0, target_class] # Assuming batch size 1 or explaining for one node

        # 2. Edge Mask Sparsity Loss
        edge_m = torch.sigmoid(edge_mask_value)
        loss_edge_size = coeffs["edge_size"] * torch.sum(edge_m)

        # 3. Edge Mask Entropy Loss (to encourage binary values)
        ent_edge = -edge_m * torch.log2(edge_m + 1e-12) - (1 - edge_m) * torch.log2(
            1 - edge_m + 1e-12
        )
        loss_edge_ent = coeffs["edge_ent"] * torch.mean(ent_edge)


        # 4. Node Feature Mask Sparsity Loss
        node_feat_m = torch.sigmoid(node_feat_mask_value)
        loss_node_feat_size = coeffs["node_feat_size"] * torch.sum(node_feat_m)

        # 5. Node Feature Mask Entropy Loss
        ent_node_feat = -node_feat_m * torch.log2(node_feat_m + 1e-12) - \
                        (1-node_feat_m) * torch.log2(1-node_feat_m + 1e-12)
        loss_node_feat_ent = coeffs["node_feat_ent"] * torch.mean(ent_node_feat)

        total_loss = (
            pred_loss
            + loss_edge_size
            + loss_edge_ent
            + loss_node_feat_size
            + loss_node_feat_ent
        )
        return total_loss


    @log_usage()
    def attribute(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        target_node: Optional[int] = None, # Specific node to explain
        target_class: Optional[int] = None, # Specific class to explain
        num_epochs: int = 100,
        lr: float = 0.01,
        coeffs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Explains the GNN's prediction for a given node or graph.

        Args:
            inputs (Tensor): The node features. Shape: (num_nodes, num_node_features)
            edge_index (Tensor): The edge index of the graph.
                        Shape: (2, num_edges)
            target_node (int, optional): The node for which the explanation
                        is generated. If None, the explanation is for the
                        entire graph's prediction. Default: None
            target_class (int, optional): The specific class for which the
                        explanation is generated. If None, the class with the
                        highest prediction score is chosen. Default: None
            num_epochs (int): The number of epochs to train the masks.
                        Default: 100
            lr (float): The learning rate for optimizing the masks.
                        Default: 0.01
            coeffs (Dict[str, float], optional): Coefficients for the different
                        loss terms (edge_size, edge_ent, node_feat_size,
                        node_feat_ent). Default: Predefined DEFAULT_COEFFS.
            **kwargs (Any): Additional arguments that are passed to the GNN model
                        during the forward pass.

        Returns:
            Tuple[Tensor, Tensor]:
            - node_feat_mask (Tensor): The learned node feature mask
                                    Shape: (num_node_features,)
            - edge_mask (Tensor): The learned edge_mask (attributions for edges)
                                    Shape: (num_edges,)
        """
        if coeffs is None:
            coeffs = DEFAULT_COEFFS

        self._init_masks(inputs, edge_index)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=lr)

        # --- Improved edge_weight handling: Check model signature ---
        model_accepts_edge_weight = False
        # Check if the model is a torch.nn.Module and has a forward method
        if isinstance(self.model, torch.nn.Module) and hasattr(self.model, "forward"):
            sig = inspect.signature(self.model.forward)
            if "edge_weight" in sig.parameters:
                model_accepts_edge_weight = True
        elif callable(self.model) and not isinstance(self.model, torch.nn.Module):
            # For general callables (not nn.Module), inspect directly.
            try:
                sig = inspect.signature(self.model)
                if "edge_weight" in sig.parameters:
                    model_accepts_edge_weight = True
            except ValueError:
                # Some callables (e.g. built-ins) may not have a signature.
                pass
        # --- End of improved edge_weight handling ---

        # Get original model prediction (logits)
        # For the original prediction, we don't use the edge_mask yet.
        original_pred = self.model(inputs, edge_index, **kwargs)
        if isinstance(original_pred, tuple): # Handle models returning multiple outputs
            original_pred = original_pred[0]


        # Determine target_class if not provided
        if target_class is None:
            if target_node is not None:
                # Explain the prediction for the target_node
                _target_class = torch.argmax(original_pred[target_node]).item()
            else:
                # Explain the graph-level prediction (e.g., max of summed node embeddings)
                # This might require a different handling of original_pred
                # For now, let's assume original_pred is [num_nodes, num_classes]
                # and we take the class with max score for the graph.
                # This part might need refinement based on specific GNN model for graph pred.
                _target_class = torch.argmax(original_pred.sum(dim=0)).item()
        else:
            _target_class = target_class


        for epoch in range(num_epochs):
            optimizer.zero_grad()

            masked_inputs, current_edge_mask_weights = self._get_masked_inputs(
                inputs,
                edge_index,
                self.edge_mask,
                self.node_feat_mask,
                apply_sigmoid=True, # Sigmoid is applied within _get_masked_inputs for weights
            )

            model_kwargs = kwargs.copy()
            if model_accepts_edge_weight:
                model_kwargs["edge_weight"] = current_edge_mask_weights
                masked_pred = self.model(masked_inputs, edge_index, **model_kwargs)
            else:
                if epoch == 0 and not getattr(self, "_warned_edge_weight_this_call", False):
                    warnings.warn(
                        "The GNN model's forward method does not explicitly accept "
                        "'edge_weight'. The learned edge mask might not be utilized by "
                        "the model. Please ensure your model can utilize edge weights "
                        "for the GNNExplainer to be fully effective.", UserWarning
                    )
                    self._warned_edge_weight_this_call = True # Flag for current call
                masked_pred = self.model(masked_inputs, edge_index, **kwargs)

            if isinstance(masked_pred, tuple):
                masked_pred = masked_pred[0]

            # We need to decide how to get the relevant prediction for the loss
            # If target_node is set, use its prediction. Otherwise, use graph-level pred.
            pred_for_loss = masked_pred
            original_pred_for_loss = original_pred
            if target_node is not None:
                pred_for_loss = masked_pred[target_node].unsqueeze(0)
                original_pred_for_loss = original_pred[target_node].unsqueeze(0)
            # If explaining graph and model output is per node, sum up for graph prediction
            # This is a common approach but might need to be adapted based on the GCN.
            elif masked_pred.ndim > 1 and masked_pred.shape[0] > 1 : #  num_nodes x num_classes
                pred_for_loss = masked_pred.sum(dim=0).unsqueeze(0)
                original_pred_for_loss = original_pred.sum(dim=0).unsqueeze(0)


            loss = self._loss_fn(
                pred_for_loss,
                original_pred_for_loss,
                self.edge_mask, # Pass raw mask
                self.node_feat_mask, # Pass raw mask
                coeffs,
                _target_class,
            )

            loss.backward()
            optimizer.step()

        final_edge_mask = torch.sigmoid(self.edge_mask).detach()
        final_node_feat_mask = torch.sigmoid(self.node_feat_mask).detach()

        # Clear the warning flag for this call if it was set
        if hasattr(self, "_warned_edge_weight_this_call"):
            delattr(self, "_warned_edge_weight_this_call")

        self._clear_masks() # Clean up masks from attributes

        return final_node_feat_mask, final_edge_mask


    def __deepcopy__(self, memo) -> "GNNExplainer":
        """
        Custom deepcopy implementation for GNNExplainer.
        This method is called by `copy.deepcopy`.
        It ensures that the GNNExplainer instance, including its model,
        is correctly copied. Learnable masks (node_feat_mask, edge_mask)
        are not part of the explainer's persistent state as they are
        initialized within the `attribute` method and cleared afterwards;
        therefore, they don't need special handling here.
        """
        # The `Attribution` base class's __deepcopy__ handles the `model` attribute.
        # If GNNExplainer had other attributes requiring deep copying,
        # they would be handled here. For example:
        # new_copy = self.__class__(self.model)
        # memo[id(self)] = new_copy
        # new_copy.some_other_attribute = copy.deepcopy(self.some_other_attribute, memo)
        # return new_copy
        return super().__deepcopy__(memo) # type: ignore

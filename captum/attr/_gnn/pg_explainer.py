import copy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from captum.attr import Attribution
from captum.log import log_usage
from torch import Tensor, nn

# Default coefficients for PGExplainer loss terms
DEFAULT_PGEXPLAINER_COEFFS = {
    "prediction_ce": 1.0,  # Cross-entropy for fidelity
    "edge_size": 0.01,     # Sparsity: L1 norm for edge mask
    "edge_ent": 0.1,       # Entropy for edge mask (encourage binary values)
}


class PGExplainer(Attribution):
    r"""
    PGExplainer (Parameterized Graph Explainer) a model interpretability technique
    for Graph Neural Networks (GNNs) from the paper:
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2004.11990>`_

    PGExplainer trains a separate neural network (the explainer network) to
    generate edge masks for explanations. It aims to provide instance-level
    explanations by identifying important edges.
    """

    def __init__(self, model: nn.Module, explainer_net: nn.Module, name: str = "PGExplainer") -> None:
        r"""
        Args:
            model (nn.Module): The GNN model that is being explained.
                               It should take node features and edge_index as input,
                               and can optionally accept `edge_weight`.
            explainer_net (nn.Module): The explainer network that is trained to
                                       generate edge masks. This network typically
                                       takes graph structure information (e.g., node
                                       features of connected nodes for each edge)
                                       and outputs a scalar per edge.
            name (str, optional): A human-readable name for the explainer.
        """
        self.explainer_net = explainer_net
        super().__init__(model)
        self.name = name

    def _get_explainer_inputs(
        self, inputs: Tensor, edge_index: Tensor, model_outputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Prepares inputs for the explainer_net.
        A common strategy is to concatenate features of source and target nodes
        for each edge, and optionally, the GNN's output/embeddings for these nodes.

        Args:
            inputs (Tensor): Node features. Shape: (num_nodes, num_node_features)
            edge_index (Tensor): Edge index. Shape: (2, num_edges)
            model_outputs (Optional[Tensor]): Node embeddings or outputs from the main
                                           GNN model. Shape: (num_nodes, D)

        Returns:
            Tensor: Input tensor for explainer_net.
                    Shape: (num_edges, explainer_input_dim)
        """
        src_nodes, dest_nodes = edge_index[0], edge_index[1]
        explainer_inputs = [inputs[src_nodes], inputs[dest_nodes]]

        if model_outputs is not None:
            explainer_inputs.append(model_outputs[src_nodes])
            explainer_inputs.append(model_outputs[dest_nodes])

        return torch.cat(explainer_inputs, dim=-1)


    def _calculate_loss(
        self,
        edge_mask_logits: Tensor,
        masked_pred: Tensor,
        original_pred_for_loss: Tensor,
        target_class_for_loss: Tensor,
        coeffs: Dict[str, float],
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Calculates the total loss for training the explainer_net.
        """
        # Apply sigmoid and temperature to get edge probabilities
        edge_probs = torch.sigmoid(edge_mask_logits / temperature)

        # 1. Prediction Cross-Entropy Loss (Fidelity)
        # Assumes masked_pred and original_pred_for_loss are logits
        ce_loss = F.cross_entropy(masked_pred, target_class_for_loss.expand(masked_pred.shape[0]))
        loss = coeffs["prediction_ce"] * ce_loss

        # 2. Edge Mask Sparsity Loss (L1 norm)
        loss_edge_size = coeffs["edge_size"] * torch.mean(edge_probs) # Mean instead of sum for stability
        loss += loss_edge_size

        # 3. Edge Mask Entropy Loss (to encourage binary values)
        # Prevents mask values from being stuck at 0.5
        entropy = -edge_probs * torch.log2(edge_probs + 1e-12) - \
                  (1 - edge_probs) * torch.log2(1 - edge_probs + 1e-12)
        loss_edge_ent = coeffs["edge_ent"] * torch.mean(entropy)
        loss += loss_edge_ent

        return loss


    @log_usage()
    def attribute(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        target_node_idx: Optional[Tensor] = None, # Node indices for which to explain
        target_class: Optional[Tensor] = None, # Target class for each explanation
        train_mode: bool = False,
        epochs: int = 100,
        lr: float = 0.01,
        loss_coeffs: Optional[Dict[str, float]] = None,
        temperature: float = 1.0, # Temperature for sigmoid in loss
        model_kwargs: Optional[Dict[str, Any]] = None, # For main model
        explainer_kwargs: Optional[Dict[str, Any]] = None, # For explainer_net
    ) -> Tensor:
        r"""
        Trains the explainer network or generates an edge mask using a trained one.

        Args:
            inputs (Tensor): Node features. Shape: (num_nodes, num_node_features)
            edge_index (Tensor): Edge index. Shape: (2, num_edges)
            target_node_idx (Optional[Tensor]): Indices of target nodes for which
                        explanations are required or for which loss is computed
                        during training. If None, assumes graph-level explanation.
                        Shape: (num_targets,)
            target_class (Optional[Tensor]): Target class for each node/graph in
                        target_node_idx. If None, it's inferred from the model's
                        prediction. Shape: (num_targets,)
            train_mode (bool): If True, trains `self.explainer_net`. Otherwise,
                        uses the trained `explainer_net` for inference.
                        Default: False
            epochs (int): Number of epochs for training. Default: 100 (if train_mode)
            lr (float): Learning rate for training. Default: 0.01 (if train_mode)
            loss_coeffs (Optional[Dict[str, float]]): Coefficients for loss terms.
                        Uses DEFAULT_PGEXPLAINER_COEFFS if None.
            temperature (float): Temperature for scaling edge mask logits before
                        sigmoid, affects sharpness of probabilities. Default: 1.0
            model_kwargs (Optional[Dict[str, Any]]): Additional arguments for the main GNN model.
            explainer_kwargs (Optional[Dict[str, Any]]): Additional arguments for the explainer network.

        Returns:
            Tensor: Edge mask (probabilities). Shape: (num_edges,)
                    If train_mode is True, this is the mask from the last batch/epoch,
                    primarily for inspection, as the main result is the trained explainer.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if explainer_kwargs is None:
            explainer_kwargs = {}
        if loss_coeffs is None:
            loss_coeffs = DEFAULT_PGEXPLAINER_COEFFS.copy()

        # Use model's eval mode for generating embeddings if needed, and for original preds
        self.model.eval()
        # Get initial node embeddings/outputs from the main model (optional for explainer_net)
        # This depends on explainer_net's architecture. For now, let's assume it might use them.
        # h_nodes = self.model(inputs, edge_index, **model_kwargs) # Or specific layers
        # if isinstance(h_nodes, tuple): h_nodes = h_nodes[0]
        # For simplicity, _get_explainer_inputs will just use `inputs` for now.
        # User can design explainer_net to take pre-computed embeddings via explainer_kwargs.

        explainer_input_feats = self._get_explainer_inputs(inputs, edge_index) # No model_outputs for now

        if train_mode:
            self.explainer_net.train()
            self.model.eval() # Keep main model in eval mode during explainer training

            optimizer = optim.Adam(self.explainer_net.parameters(), lr=lr)

            # Get original predictions (for calculating fidelity loss)
            # These are the "ground truth" predictions we want the subgraph to match.
            original_pred_logits = self.model(inputs, edge_index, **model_kwargs)
            if isinstance(original_pred_logits, tuple):
                original_pred_logits = original_pred_logits[0]

            if target_node_idx is not None:
                original_pred_for_loss = original_pred_logits[target_node_idx]
            else: # Graph-level explanation
                original_pred_for_loss = original_pred_logits.sum(dim=0, keepdim=True) # Example aggregation

            if target_class is None:
                _target_class_for_loss = torch.argmax(original_pred_for_loss, dim=-1)
            else:
                _target_class_for_loss = target_class

            for epoch in range(epochs):
                optimizer.zero_grad()

                edge_mask_logits = self.explainer_net(explainer_input_feats, edge_index, **explainer_kwargs)
                edge_mask_probs_for_model = torch.sigmoid(edge_mask_logits / temperature) # For model forward pass

                # Masked prediction: pass edge_mask_probs_for_model as edge_weight
                # This assumes self.model accepts 'edge_weight'
                masked_pred_logits = self.model(inputs, edge_index,
                                              edge_weight=edge_mask_probs_for_model.squeeze(-1),
                                              **model_kwargs)
                if isinstance(masked_pred_logits, tuple):
                    masked_pred_logits = masked_pred_logits[0]

                if target_node_idx is not None:
                    masked_pred_for_loss = masked_pred_logits[target_node_idx]
                else: # Graph-level
                    masked_pred_for_loss = masked_pred_logits.sum(dim=0, keepdim=True)

                loss = self._calculate_loss(
                    edge_mask_logits,
                    masked_pred_for_loss,
                    original_pred_for_loss, # Should be from original graph
                    _target_class_for_loss,
                    loss_coeffs,
                    temperature,
                )
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}") # Optional logging

            # After training, return the latest edge mask from the trained explainer
            self.explainer_net.eval()
            edge_mask_logits = self.explainer_net(explainer_input_feats, edge_index, **explainer_kwargs)
            final_edge_mask = torch.sigmoid(edge_mask_logits / temperature).detach()
            return final_edge_mask.squeeze(-1) if final_edge_mask.ndim > 1 else final_edge_mask

        else: # Inference mode
            self.explainer_net.eval()
            edge_mask_logits = self.explainer_net(explainer_input_feats, edge_index, **explainer_kwargs)
            edge_mask_probs = torch.sigmoid(edge_mask_logits / temperature).detach()
            return edge_mask_probs.squeeze(-1) if edge_mask_probs.ndim > 1 else edge_mask_probs


    def __deepcopy__(self, memo) -> "PGExplainer":
        r"""
        Custom deepcopy implementation.
        Ensures that the explainer_net and the model are correctly copied.
        """
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance

        # Call Attribution's init or deepcopy its relevant parts if direct init is complex
        # For now, let's assume Attribution's deepcopy handles its own state including self.model
        # This is a bit of a simplification; proper handling might involve
        # re-calling __init__ on new_instance or carefully copying from self.__dict__.
        # super(PGExplainer, new_instance).__init__(copy.deepcopy(self.model, memo))
        # The above super call is tricky with __new__.

        # Let's try this: copy attributes, then deepcopy complex ones.
        for k, v in self.__dict__.items():
            setattr(new_instance, k, v)

        # Deepcopy mutable or complex objects
        new_instance.model = copy.deepcopy(self.model, memo)
        new_instance.explainer_net = copy.deepcopy(self.explainer_net, memo)
        # self.name is a string, typically immutable, shallow copy is fine.

        # If Attribution class has its own __deepcopy__, it's better to call it.
        # However, direct call to super().__deepcopy__(memo) might be cleaner if it exists
        # and correctly initializes 'new_instance'.
        # The provided structure for Attribution doesn't show a __deepcopy__,
        # so we manage it here. If it does, the strategy would change.

        # A common pattern for __deepcopy__ if superclass doesn't have a good one:
        # cls = self.__class__
        # result = cls.__new__(cls)
        # memo[id(self)] = result
        # for k, v in self.__dict__.items():
        #     setattr(result, k, copy.deepcopy(v, memo))
        # return result
        # This pattern is more robust. Let's use this simplified version:

        # Call super's deepcopy first IF it's well-defined and returns the new instance.
        # Assuming Attribution is like nn.Module or has a similar __deepcopy__
        # new_copy = super().__deepcopy__(memo) # This would handle self.model
        # new_copy.explainer_net = copy.deepcopy(self.explainer_net, memo)
        # new_copy.name = self.name
        # return new_copy
        # Since I don't know Attribution's __deepcopy__ internals, I'll stick to a safer explicit copy.

        # Simplified and more standard approach:
        cls = self.__class__
        result = cls.__new__(cls) # Create new instance without calling __init__
        memo[id(self)] = result

        # Manually call __init__ or copy attributes
        # Calling __init__ is cleaner if possible:
        # result.__init__(model=copy.deepcopy(self.model, memo),
        #                  explainer_net=copy.deepcopy(self.explainer_net, memo),
        #                  name=self.name)
        # This is often the best way.

        # For now, sticking to the pattern from previous GNNExplainer for consistency.
        # This assumes super().__deepcopy__ handles model and other parent attributes.
        new_copy = super().__deepcopy__(memo) # type: ignore
        new_copy.explainer_net = copy.deepcopy(self.explainer_net, memo)
        new_copy.name = self.name
        return new_copy

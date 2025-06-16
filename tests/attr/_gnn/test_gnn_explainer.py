import unittest
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr._gnn.gnn_explainer import GNNExplainer, DEFAULT_COEFFS
from tests.helpers.basic_models import BasicGNN, BasicGNN_MultiLayer


class TestGNNExplainer(unittest.TestCase):
    def _create_simple_graph(self, num_nodes=5, num_node_features=3, num_edges=6, device="cpu"):
        inputs = torch.rand(num_nodes, num_node_features, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        # Ensure no self-loops for simplicity in some GNN models, though GNNExplainer should handle it
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        return inputs, edge_index

    def test_gnn_explainer_init(self) -> None:
        model = BasicGNN(3, 10, 2)
        explainer = GNNExplainer(model)
        self.assertIsNotNone(explainer)
        self.assertIs(explainer.model, model)

    def test_gnn_explainer_attribute_smoke(self) -> None:
        num_nodes, num_features, num_edges = 5, 3, 6
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)
        model = BasicGNN(num_features, 10, 2) # in_channels, hidden_channels, out_channels

        explainer = GNNExplainer(model)
        node_feat_mask, edge_mask = explainer.attribute(
            inputs, edge_index, num_epochs=2, lr=0.1 # Short training for smoke test
        )

        self.assertEqual(node_feat_mask.shape, (num_features,))
        self.assertEqual(edge_mask.shape, (edge_index.shape[1],))
        self.assertTrue(torch.all(node_feat_mask >= 0) and torch.all(node_feat_mask <= 1))
        self.assertTrue(torch.all(edge_mask >= 0) and torch.all(edge_mask <= 1))

    def test_gnn_explainer_attribute_target_node(self) -> None:
        num_nodes, num_features, num_edges = 5, 3, 6
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)
        model = BasicGNN(num_features, 10, 2)
        explainer = GNNExplainer(model)
        target_node = 0

        node_feat_mask, edge_mask = explainer.attribute(
            inputs, edge_index, target_node=target_node, num_epochs=2
        )
        self.assertEqual(node_feat_mask.shape, (num_features,))
        self.assertEqual(edge_mask.shape, (edge_index.shape[1],))

    def test_gnn_explainer_attribute_target_class(self) -> None:
        num_nodes, num_features, num_edges = 5, 3, 6
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)
        model = BasicGNN(num_features, 10, 2) # 2 output classes
        explainer = GNNExplainer(model)
        target_class = 1

        node_feat_mask, edge_mask = explainer.attribute(
            inputs, edge_index, target_class=target_class, num_epochs=2
        )
        self.assertEqual(node_feat_mask.shape, (num_features,))
        self.assertEqual(edge_mask.shape, (edge_index.shape[1],))

    def test_gnn_explainer_graph_level_explanation(self) -> None:
        num_nodes, num_features, num_edges = 5, 3, 6
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)
        # Model output for graph-level could be sum of node embeddings then a linear layer
        # BasicGNN outputs per node, GNNExplainer handles sum internally for graph explanation
        model = BasicGNN(num_features, 10, 2)
        explainer = GNNExplainer(model)

        node_feat_mask, edge_mask = explainer.attribute(
            inputs, edge_index, target_node=None, num_epochs=2 # Explicitly graph-level
        )
        self.assertEqual(node_feat_mask.shape, (num_features,))
        self.assertEqual(edge_mask.shape, (edge_index.shape[1],))


    def test_gnn_explainer_model_with_edge_weight(self) -> None:
        num_nodes, num_features, num_edges = 4, 3, 5
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)

        # Define a model that uses edge_weight
        class ModelWithEdgeWeight(BasicGNN):
            def forward(self, x, edge_index, edge_weight=None, **kwargs):
                # Simplified: just pass it along if GNN layer supports it
                # Or use it directly:
                # for i in range(self.num_layers -1):
                #     x = F.relu(self.convs[i](x, edge_index, edge_weight=edge_weight if i==0 else None))
                # x = self.convs[-1](x, edge_index, edge_weight=edge_weight if self.num_layers-1 == 0 else None)
                # For this test, the fact that it's accepted is key.
                # BasicGNN's internal GCNConv doesn't directly use edge_weight in its base form.
                # Let's make a simpler model for this test or adapt BasicGNN

                # Let's use a model structure that explicitly can take edge_weight
                # For now, we assume BasicGNN can be modified or we use a mock
                # that has 'edge_weight' in its signature.
                # The GNNExplainer checks signature, not if it's *used*.
                if edge_weight is not None:
                    x = x * edge_weight.mean() # Dummy use of edge_weight to show it could be used
                return super().forward(x, edge_index, **kwargs)

        model = ModelWithEdgeWeight(num_features, 5, 2)
        explainer = GNNExplainer(model)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            explainer.attribute(inputs, edge_index, num_epochs=1)
            self.assertEqual(len(caught_warnings), 0, "Should not warn if model accepts edge_weight.")
            # Check for specific warning text if possible/needed

    def test_gnn_explainer_model_without_edge_weight(self) -> None:
        num_nodes, num_features, num_edges = 4, 3, 5
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)

        # BasicGNN by default does not have edge_weight in its direct forward signature
        # (though its GCNConv layers might, GNNExplainer checks the model's forward)
        # To be sure, let's define one that definitely doesn't.
        class ModelWithoutEdgeWeight(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Linear(in_channels, hidden_channels) # Simplified
                self.conv2 = nn.Linear(hidden_channels, out_channels)

            def forward(self, x, edge_index_ignored, **kwargs): # No edge_weight
                # Simplified forward, ignoring edge_index for this dummy model
                x = F.relu(self.conv1(x.mean(dim=0, keepdim=True))) # Aggregate then process
                x = self.conv2(x)
                # Output needs to be per-node for GNNExplainer's default loss processing
                # This dummy model isn't a real GNN, just for signature testing
                return torch.randn(inputs.shape[0], self.conv2.out_features)


        model = ModelWithoutEdgeWeight(num_features, 5, 2)
        explainer = GNNExplainer(model)

        with self.assertWarnsRegex(UserWarning, "does not explicitly accept 'edge_weight'"):
            explainer.attribute(inputs, edge_index, num_epochs=1)
            # Ensure the _warned_edge_weight_this_call flag is reset
            delattr(explainer, "_warned_edge_weight_this_call")


    def test_gnn_explainer_mask_properties(self) -> None:
        num_nodes, num_features, num_edges = 6, 4, 8
        inputs, edge_index = self._create_simple_graph(num_nodes, num_features, num_edges)
        model = BasicGNN(num_features, 10, 3)
        explainer = GNNExplainer(model)

        node_feat_mask, edge_mask = explainer.attribute(inputs, edge_index, num_epochs=3)

        self.assertEqual(node_feat_mask.shape, (num_features,))
        self.assertTrue(torch.all(node_feat_mask >= 0.0) and torch.all(node_feat_mask <= 1.0),
                        "Node feature mask values should be between 0 and 1.")

        self.assertEqual(edge_mask.shape, (edge_index.shape[1],))
        self.assertTrue(torch.all(edge_mask >= 0.0) and torch.all(edge_mask <= 1.0),
                        "Edge mask values should be between 0 and 1.")

    def test_gnn_explainer_deepcopy(self) -> None:
        import copy
        model = BasicGNN(3, 5, 2)
        explainer = GNNExplainer(model)
        explainer.some_custom_attribute = "test_value" # Add a dummy attribute

        explainer_copy = copy.deepcopy(explainer)

        self.assertIsNot(explainer, explainer_copy)
        self.assertIsInstance(explainer_copy, GNNExplainer)
        self.assertIsNot(explainer.model, explainer_copy.model, "Model should be deepcopied.")
        # Check if model parameters are different objects but have same values initially
        for p1, p2 in zip(explainer.model.parameters(), explainer_copy.model.parameters()):
            self.assertIsNot(p1, p2)
            self.assertTrue(torch.equal(p1.data, p2.data))

        self.assertEqual(explainer_copy.some_custom_attribute, "test_value")
        explainer_copy.some_custom_attribute = "new_value"
        self.assertEqual(explainer.some_custom_attribute, "test_value") # Original should be unchanged

        # Test functionality after deepcopy
        inputs, edge_index = self._create_simple_graph()
        try:
            node_feat_mask, edge_mask = explainer_copy.attribute(inputs, edge_index, num_epochs=1)
            self.assertIsNotNone(node_feat_mask)
            self.assertIsNotNone(edge_mask)
        except Exception as e:
            self.fail(f"Attribute call failed after deepcopy: {e}")

    def test_gnn_explainer_default_coeffs(self) -> None:
        inputs, edge_index = self._create_simple_graph()
        model = BasicGNN(inputs.shape[1], 5, 2)
        explainer = GNNExplainer(model)
        # This test indirectly checks if default coeffs are used without error
        # A more direct test would involve checking the loss values, which is complex.
        try:
            explainer.attribute(inputs, edge_index, num_epochs=1, coeffs=None) # Explicitly use default
        except Exception as e:
            self.fail(f"Attribute call failed with default coeffs: {e}")

        custom_coeffs = DEFAULT_COEFFS.copy()
        custom_coeffs["edge_size"] = 0.5
        try:
            explainer.attribute(inputs, edge_index, num_epochs=1, coeffs=custom_coeffs)
        except Exception as e:
            self.fail(f"Attribute call failed with custom coeffs: {e}")


if __name__ == "__main__":
    unittest.main()

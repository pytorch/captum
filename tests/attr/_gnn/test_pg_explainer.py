import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from captum.attr._gnn.pg_explainer import PGExplainer, DEFAULT_PGEXPLAINER_COEFFS
from tests.helpers.basic_models import BasicGNN # Assuming this is available


# A simple explainer network for testing PGExplainer
class SimpleExplainerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        # input_dim is typically 2 * node_feature_dim (+ optional 2 * gnn_embedding_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Outputs a single logit per edge

    def forward(self, x, edge_index, **kwargs): # edge_index might be used for graph context
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestPGExplainer(unittest.TestCase):
    def _create_simple_graph(self, num_nodes=5, num_node_features=3, num_edges=6, device="cpu"):
        inputs = torch.rand(num_nodes, num_node_features, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]] # Ensure no self-loops
        # Ensure all nodes are part of at least one edge for some tests
        if edge_index.shape[1] > 0:
             for i in range(num_nodes):
                if i not in edge_index: # if a node is isolated
                    # add a dummy edge if possible, otherwise graph might be too small
                    if edge_index.shape[1] < num_edges:
                        if i != edge_index[0,0]:
                            new_edge = torch.tensor([[i],[edge_index[0,0]]], device=device, dtype=torch.long)
                        else: # if i is the only other node, connect to node 1 if possible
                            new_edge = torch.tensor([[i],[(i+1)%num_nodes]], device=device, dtype=torch.long)
                        edge_index = torch.cat([edge_index, new_edge], dim=1)

        return inputs, edge_index.to(torch.long)


    def setUp(self):
        self.num_nodes, self.num_features, self.num_edges = 6, 4, 8
        self.inputs, self.edge_index = self._create_simple_graph(
            self.num_nodes, self.num_features, self.num_edges
        )

        self.gnn_model = BasicGNN(self.num_features, hidden_channels=5, out_channels=2)

        # Determine explainer_input_dim based on _get_explainer_inputs
        # Default: 2 * num_node_features
        self.explainer_input_dim = 2 * self.num_features
        self.explainer_net = SimpleExplainerNet(self.explainer_input_dim)

        self.pg_explainer = PGExplainer(self.gnn_model, self.explainer_net)

    def test_pgexplainer_init(self):
        self.assertIsInstance(self.pg_explainer, PGExplainer)
        self.assertIs(self.pg_explainer.model, self.gnn_model)
        self.assertIs(self.pg_explainer.explainer_net, self.explainer_net)

    def test_pgexplainer_get_explainer_inputs(self):
        expl_inputs = self.pg_explainer._get_explainer_inputs(self.inputs, self.edge_index)
        self.assertEqual(expl_inputs.shape, (self.edge_index.shape[1], self.explainer_input_dim))

        # Test with model_outputs
        model_outputs_mock = torch.rand(self.num_nodes, 5) # 5 is embedding dim
        expl_inputs_with_emb = self.pg_explainer._get_explainer_inputs(
            self.inputs, self.edge_index, model_outputs=model_outputs_mock
        )
        expected_dim_with_emb = self.explainer_input_dim + 2 * 5
        self.assertEqual(expl_inputs_with_emb.shape, (self.edge_index.shape[1], expected_dim_with_emb))

    def test_pgexplainer_calculate_loss_smoke(self):
        edge_mask_logits = torch.randn(self.edge_index.shape[1], 1)
        # Assume target_node_idx is None for graph-level explanation for simplicity in this smoke test
        num_classes = self.gnn_model.convs[-1].out_features
        masked_pred = torch.randn(1, num_classes) # graph-level prediction
        original_pred = torch.randn(1, num_classes)
        target_class = torch.randint(0, num_classes, (1,))

        loss = self.pg_explainer._calculate_loss(
            edge_mask_logits, masked_pred, original_pred, target_class, DEFAULT_PGEXPLAINER_COEFFS
        )
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.ndim, 0) # scalar loss

    def test_pgexplainer_attribute_inference_mode_smoke(self):
        edge_mask = self.pg_explainer.attribute(self.inputs, self.edge_index, train_mode=False)
        self.assertEqual(edge_mask.shape, (self.edge_index.shape[1],))
        self.assertTrue(torch.all(edge_mask >= 0) and torch.all(edge_mask <= 1))

    def test_pgexplainer_attribute_train_mode_smoke(self):
        # Short training cycle
        edge_mask = self.pg_explainer.attribute(
            self.inputs, self.edge_index, train_mode=True, epochs=2, lr=0.01
        )
        self.assertEqual(edge_mask.shape, (self.edge_index.shape[1],))
        self.assertTrue(torch.all(edge_mask >= 0) and torch.all(edge_mask <= 1))


    def test_pgexplainer_training_updates_parameters(self):
        initial_params = [p.clone() for p in self.explainer_net.parameters()]

        self.pg_explainer.attribute(
            self.inputs, self.edge_index, train_mode=True, epochs=3, lr=0.01,
            # Provide target_node_idx for node-level loss calculation if model output is per node
            target_node_idx=torch.tensor([0, 1])
        )

        updated_params = list(self.explainer_net.parameters())
        self.assertGreater(len(initial_params), 0)
        self.assertEqual(len(initial_params), len(updated_params))

        params_changed = False
        for p_init, p_updated in zip(initial_params, updated_params):
            if not torch.equal(p_init, p_updated):
                params_changed = True
                break
        self.assertTrue(params_changed, "Explainer network parameters should change after training.")


    def test_pgexplainer_target_handling_train_mode(self):
        # Test with target_node_idx and target_class
        target_nodes = torch.tensor([0, 1, 2])
        # Assuming model has 2 output classes
        target_classes = torch.tensor([0, 1, 0])

        try:
            self.pg_explainer.attribute(
                self.inputs, self.edge_index,
                target_node_idx=target_nodes,
                target_class=target_classes,
                train_mode=True, epochs=1, lr=0.01
            )
        except Exception as e:
            self.fail(f"Training with target_node_idx and target_class failed: {e}")

    def test_pgexplainer_custom_loss_coeffs(self):
        custom_coeffs = DEFAULT_PGEXPLAINER_COEFFS.copy()
        custom_coeffs["edge_size"] = 0.5
        custom_coeffs["prediction_ce"] = 0.1

        try:
            self.pg_explainer.attribute(
                self.inputs, self.edge_index, train_mode=True, epochs=1,
                loss_coeffs=custom_coeffs, target_node_idx=torch.tensor([0])
            )
        except Exception as e:
            self.fail(f"Training with custom loss coefficients failed: {e}")

    def test_pgexplainer_deepcopy(self):
        explainer_copy = copy.deepcopy(self.pg_explainer)

        self.assertIsNot(self.pg_explainer, explainer_copy)
        self.assertIsInstance(explainer_copy, PGExplainer)

        # Check model and explainer_net are new instances
        self.assertIsNot(self.pg_explainer.model, explainer_copy.model)
        self.assertIsNot(self.pg_explainer.explainer_net, explainer_copy.explainer_net)

        # Check parameters are copied
        for p_orig, p_copy in zip(self.pg_explainer.model.parameters(), explainer_copy.model.parameters()):
            self.assertIsNot(p_orig, p_copy)
            self.assertTrue(torch.equal(p_orig.data, p_copy.data))

        for p_orig, p_copy in zip(self.pg_explainer.explainer_net.parameters(), explainer_copy.explainer_net.parameters()):
            self.assertIsNot(p_orig, p_copy)
            self.assertTrue(torch.equal(p_orig.data, p_copy.data))

        # Test functionality after deepcopy (inference mode)
        try:
            edge_mask = explainer_copy.attribute(self.inputs, self.edge_index, train_mode=False)
            self.assertIsNotNone(edge_mask)
            self.assertEqual(edge_mask.shape, (self.edge_index.shape[1],))
        except Exception as e:
            self.fail(f"Attribute call (inference) failed after deepcopy: {e}")

        # Test functionality after deepcopy (training mode)
        try:
            initial_params_copy = [p.clone() for p in explainer_copy.explainer_net.parameters()]
            explainer_copy.attribute(self.inputs, self.edge_index, train_mode=True, epochs=1, lr=0.01, target_node_idx=torch.tensor([0]))
            params_changed_copy = False
            for p_init, p_updated in zip(initial_params_copy, explainer_copy.explainer_net.parameters()):
                if not torch.equal(p_init, p_updated):
                    params_changed_copy = True
                    break
            self.assertTrue(params_changed_copy, "Copied explainer network parameters should change after training.")
        except Exception as e:
            self.fail(f"Attribute call (training) failed after deepcopy: {e}")

if __name__ == "__main__":
    unittest.main()

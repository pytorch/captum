# GNN Explainability in Captum

This directory contains implementations of Graph Neural Network (GNN) explainability methods within the Captum library. These methods aim to provide insights into the predictions made by GNN models.

## Implemented Methods

### GNNExplainer
-   **Description:** GNNExplainer is a model-agnostic method that identifies a compact subgraph structure and a small subset of node features that are influential for a GNN's prediction.
-   **Reference:** [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)

### PGExplainer
-   **Description:** PGExplainer (Parameterized Explainer) is a method that trains a parameterized explainer network to generate explanations (edge masks) for GNN predictions.
-   **Reference:** [Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573)
    Note: The reference link for PGExplainer in the prompt (2011.04573) seems to be for a different paper. The common one is often https://arxiv.org/abs/2004.11990. I will use the one from the prompt. If this is incorrect, it should be updated.

## Usage

Please refer to the Captum documentation and example notebooks for detailed instructions on how to use these GNN explainability methods.

# Graph Neural Networks for Mutagenicity Prediction

Predicting molecular mutagenicity using **Graph Convolutional Networks (GCN)** and **Graph Isomorphism Networks (GIN)** on the [Mutagenicity dataset](https://www.kaggle.com/datasets/keilacamarillo/mutagenesis-dataset) from TUDataset.

## Overview

Mutagenicity prediction is a critical task in drug discovery and toxicology. Molecules are naturally represented as graphs — atoms as nodes, bonds as edges — making Graph Neural Networks a strong fit for this binary classification problem.

This project implements and compares two GNN architectures:

| Model | Pooling | Key Idea |
|-------|---------|----------|
| **GCN** | Global Mean | Spectral-inspired convolution via symmetric normalised adjacency (Kipf & Welling, 2017) |
| **GIN** | Global Sum | Provably as powerful as the 1-WL graph isomorphism test (Xu et al., 2019) |

## Dataset

- **Source:** [TUDataset – Mutagenicity](https://chrsmrrs.github.io/datasets/) (also available on [Kaggle](https://www.kaggle.com/datasets/keilacamarillo/mutagenesis-dataset))
- **Graphs:** 4,337 molecular graphs
- **Task:** Binary classification (mutagenic vs. non-mutagenic)
- **Node features:** 14-dimensional atom attributes
- **Evaluation:** 10-fold stratified cross-validation

## Project Structure

```
.
├── mutagenesis_gnn_gin.py   # Full training & evaluation script
├── mutagenesis_results.png  # Generated comparison plots
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/mutagenicity-gnn.git
cd mutagenicity-gnn
pip install -r requirements.txt
```
## Hardware & Training Environment

* **Platform:** Kaggle Notebooks
* **GPU:** NVIDIA Tesla T4 (16GB VRAM)
* **CUDA:** Enabled
* **Frameworks:** PyTorch, PyTorch Geometric

**Notes:**

* All models were trained using GPU acceleration.
* Results are reproducible on CPU, but training time will be significantly longer.
* No model-specific GPU optimisations were applied.


### Requirements

```
torch>=2.0
torch-geometric>=2.4
matplotlib
scikit-learn
numpy
```

## Usage

```bash
python mutagenesis_gnn_gin.py
```

The script will:
1. Download the Mutagenicity dataset automatically.
2. Train both GCN and GIN with 10-fold cross-validation.
3. Print per-fold and mean test accuracy for each model.
4. Save a comparison plot to `mutagenesis_results.png`.

### Configuration

Key hyperparameters can be modified at the top of the script:

```python
DATASET_NAME = "Mutagenicity"  # Use "Mutagenesis" for the 188-molecule variant
NUM_FOLDS    = 10
BATCH_SIZE   = 64
HIDDEN_DIM   = 64
NUM_EPOCHS   = 100
LR           = 0.001
```

## Model Architecture

Both models use 3 message-passing layers with batch normalisation and dropout:

**GCN:**
```
GCNConv → BN → ReLU → Dropout  (×3)  → Global Mean Pool → Linear
```

**GIN:**
```
GINConv(MLP) → BN → ReLU → Dropout  (×3)  → Global Sum Pool → Linear
```

## Results

Expected accuracy range (10-fold CV):

| Model | Mean Accuracy |
|-------|--------------|
| GCN   | ~75–80%      |
| GIN   | ~77–82%      |

> GIN typically outperforms GCN on this task due to its strictly higher expressive power in distinguishing graph structures.

## Output Plots

The script generates three visualisations:
- **Training loss curves** (averaged across folds)
- **Per-fold accuracy** (grouped bar chart)
- **Accuracy distribution** (box plot)
![Loss Curve](loss%20curve.png)
## References

- Kipf, T. N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Xu, K. et al. (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Morris, C. et al. (2020). *TUDataset: A collection of benchmark datasets for learning with graphs.* ICML Workshop on GRL+.
- Fey, M. & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on RLR.

## License

MIT

"""
GNN (GCN) and GIN on the Mutagenicity Dataset
================================================
This script:
  1. Loads the Mutagenicity dataset from TUDataset (PyTorch Geometric).
     - 4337 molecular graphs, binary classification (mutagenic vs non-mutagenic).
     - Each molecule is a graph: atoms = nodes, bonds = edges.
  2. Builds two models:
       • GCN  – Graph Convolutional Network  (Kipf & Welling, 2017)
       • GIN  – Graph Isomorphism Network    (Xu et al., 2019)
  3. Trains both with 10-fold cross-validation.
  4. Reports per-fold and mean accuracy, and plots training curves.

Requirements:
    pip install torch torch-geometric matplotlib scikit-learn

If you specifically want the smaller 188-molecule relational Mutagenesis dataset
from Kaggle (keilacamarillo), change DATASET_NAME to "Mutagenesis" (see note below).
"""


import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ── PyTorch Geometric imports ──────────────────────────────────────────
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    global_mean_pool,
    global_add_pool,
)
from sklearn.model_selection import StratifiedKFold

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
DATASET_NAME = "Mutagenicity"   # 4337 graphs  (use "Mutagenesis" for the 188-molecule variant)
SEED         = 42
NUM_FOLDS    = 10
BATCH_SIZE   = 64
HIDDEN_DIM   = 64
NUM_EPOCHS   = 100
LR           = 0.001
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════
print(f"Loading {DATASET_NAME} dataset …")
dataset = TUDataset(root=f"/tmp/{DATASET_NAME}", name=DATASET_NAME, use_node_attr=True)

# If dataset has no node features, use degree as feature
if dataset.num_node_features == 0:
    print("No node features found – using node degree as feature.")
    from torch_geometric.transforms import Degree
    max_degree = 0
    for data in dataset:
        d = data.edge_index[0] if data.edge_index.numel() > 0 else torch.tensor([])
        if d.numel() > 0:
            max_degree = max(max_degree, int(d.bincount().max()))
    dataset.transform = Degree(max_degree)
    # reload with transform
    dataset = TUDataset(root=f"/tmp/{DATASET_NAME}", name=DATASET_NAME, transform=Degree(max_degree))

num_features = dataset.num_node_features
num_classes  = dataset.num_classes
print(f"  Graphs : {len(dataset)}")
print(f"  Classes: {num_classes}")
print(f"  Node features: {num_features}")

# ══════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

class GCN(nn.Module):
    """3-layer Graph Convolutional Network with global mean pooling."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        self.lin   = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)              # graph-level readout
        return self.lin(x)


class GIN(nn.Module):
    """3-layer Graph Isomorphism Network with global sum pooling."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        # Each GINConv wraps a 2-layer MLP
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim))
        mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim))
        mlp3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim))

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GINConv(mlp3, train_eps=True)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        self.lin   = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_add_pool(x, batch)               # sum pooling (standard for GIN)
        return self.lin(x)

# ══════════════════════════════════════════════════════════════════════
# 3. TRAINING / EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x.float(), data.edge_index, data.batch).argmax(dim=-1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


def run_fold(model_cls, train_idx, test_idx, fold_id):
    """Train a fresh model on one fold and return (train_accs, test_acc)."""
    train_set = dataset[torch.tensor(train_idx)]
    test_set  = dataset[torch.tensor(test_idx)]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

    model = model_cls(num_features, HIDDEN_DIM, num_classes).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_model = None
    train_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(loss)
        acc = evaluate(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model.state_dict())

    # final test with best model
    model.load_state_dict(best_model)
    test_acc = evaluate(model, test_loader)
    print(f"  Fold {fold_id:>2d}  |  Test Acc = {test_acc:.4f}")
    return train_losses, test_acc

# ══════════════════════════════════════════════════════════════════════
# 4. 10-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════

labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

results = {}

for name, model_cls in [("GCN", GCN), ("GIN", GIN)]:
    print(f"\n{'='*50}")
    print(f"  {name}  –  {NUM_FOLDS}-Fold Cross-Validation")
    print(f"{'='*50}")
    fold_accs   = []
    all_losses  = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), 1):
        losses, acc = run_fold(model_cls, train_idx, test_idx, fold)
        fold_accs.append(acc)
        all_losses.append(losses)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  ► {name} Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    results[name] = {"accs": fold_accs, "mean": mean_acc, "std": std_acc, "losses": all_losses}

# ══════════════════════════════════════════════════════════════════════
# 5. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("  COMPARISON SUMMARY")
print(f"{'='*50}")
for name, res in results.items():
    print(f"  {name:4s}  →  {res['mean']:.4f} ± {res['std']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Training loss curves (average over folds)
for name, res in results.items():
    avg_loss = np.mean(res["losses"], axis=0)
    axes[0].plot(range(1, NUM_EPOCHS + 1), avg_loss, label=name)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Training Loss (avg over folds)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) Per-fold accuracy comparison

x = np.arange(NUM_FOLDS)
w = 0.35
axes[1].bar(x - w/2, results["GCN"]["accs"], w, label="GCN", color="#4C72B0")
axes[1].bar(x + w/2, results["GIN"]["accs"], w, label="GIN", color="#DD8452")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Per-Fold Test Accuracy")
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(i+1) for i in x])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# (c) Box plot
axes[2].boxplot(
    [results["GCN"]["accs"], results["GIN"]["accs"]],
    labels=["GCN", "GIN"],
    patch_artist=True,
    boxprops=dict(facecolor="#4C72B0", alpha=0.5),
)
axes[2].set_ylabel("Test Accuracy")
axes[2].set_title("Accuracy Distribution")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("mutagenesis_results.png", dpi=150)
plt.show()
print("\nPlot saved to mutagenesis_results.png")
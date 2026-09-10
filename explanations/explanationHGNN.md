# HGNN-SA: Hypergraph Neural Network with Set Attention
### A Complete Theory, Math, and Code Walkthrough

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why a Hypergraph?](#2-why-a-hypergraph)
3. [Architecture Overview](#3-architecture-overview)
4. [Block 1 — SMILES Encoder](#4-block-1--smiles-encoder)
5. [Block 2 — Drug Node Feature Construction](#5-block-2--drug-node-feature-construction)
6. [Block 3 — Hypergraph Convolution](#6-block-3--hypergraph-convolution)
7. [Block 4 — Set Attention Pooling](#7-block-4--set-attention-pooling)
8. [Block 5 — Side Effect Interaction and Prediction](#8-block-5--side-effect-interaction-and-prediction)
9. [Data Preparation](#9-data-preparation)
10. [Training Loop](#10-training-loop)
11. [Evaluation](#11-evaluation)
12. [End-to-End Data Flow Summary](#12-end-to-end-data-flow-summary)

---

## 1. The Problem

**Polypharmacy** is when a patient takes multiple drugs simultaneously. This is extremely common — elderly patients often take 5–10 drugs at once. The danger is that drugs can interact with each other and cause **adverse side effects** that none of the individual drugs would cause alone.

The model answers a very specific binary question:

> **"Given this specific combination of drugs, will it cause this specific side effect?"**

- **Input:** A set of drugs (e.g. `{Warfarin, Aspirin, Ibuprofen}`) + one side effect (e.g. `Gastrointestinal Bleeding`)
- **Output:** A probability between 0 and 1

This is a **binary classification** task — not "which side effects does this combo cause?" but "does this combo cause *this particular* side effect?".

---

## 2. Why a Hypergraph?

### Standard Graphs vs Hypergraphs

A standard graph edge connects **exactly 2 nodes**. A drug-drug interaction graph would have one edge per drug pair. But polypharmacy involves groups of 3, 4, or more drugs simultaneously — a pair-wise graph fundamentally cannot represent this.

A **hyperedge** connects **any number of nodes at once**.

```
Standard graph edge:    DrugA ---- DrugB

Hyperedge:              DrugA
                           \
                            ●  ← one hyperedge (one prescription)
                           /
                        DrugB
                           \
                            DrugC
```

Each prescription in the dataset becomes one hyperedge connecting all the drugs in it.

### The Incidence Matrix

Mathematically, a hypergraph is defined by:
- A set of nodes V (drugs)
- A set of hyperedges E (prescriptions)
- An **incidence matrix** H ∈ {0,1}^(|V| × |E|), where H[v,e] = 1 if drug v is in prescription e

In code this is stored as a COO-format sparse tensor called `hyperedge_index`:

```
hyperedge_index = [[drug_0, drug_1, drug_2, drug_0, drug_3],   ← node indices
                   [edge_0, edge_0, edge_0, edge_1, edge_1]]   ← hyperedge indices
```

So `(drug_0, edge_0)`, `(drug_1, edge_0)`, `(drug_2, edge_0)` means prescription 0 contains drugs 0, 1, and 2.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HGNN-SA Forward Pass                     │
│                                                                 │
│  SMILES string ──► SmilesEncoder ──► 64-dim chemical vector    │
│                                              │                  │
│  Drug ID ──────────────────────────► 64-dim ID embedding       │
│                                              │                  │
│                                     [concat → 128-dim]         │
│                                              │                  │
│                                       node_proj (Linear)       │
│                                              │                  │
│                                    HypergraphConv × 2          │
│                                    (message passing)           │
│                                              │                  │
│                               Set Attention Pooling            │
│                               (per-hyperedge weighted sum)     │
│                                              │                  │
│  Side Effect ──────────────────► 128-dim SE embedding          │
│                                              │                  │
│                                     [concat → 256-dim]         │
│                                              │                  │
│                                       MLP → logit              │
│                                              │                  │
│                                     sigmoid → probability      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Block 1 — SMILES Encoder

### What is SMILES?

SMILES (Simplified Molecular Input Line Entry System) is a notation for representing molecular structure as a string. For example:

| Drug | SMILES |
|------|--------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Caffeine | `Cn1cnc2c1c(=O)n(c(=O)n2C)C` |
| Ethanol | `CCO` |

Every character encodes a structural feature: atoms (`C`, `N`, `O`), bonds (`=`, `#`), rings (`c`, ring-closure numbers), branches (`(`, `)`).

### The Intuition

Instead of using hand-crafted molecular fingerprints (which require domain expertise), we treat SMILES like a text sequence and let the model learn chemical features automatically through convolution — the same way a text CNN learns which n-grams matter.

### The Math

Given a SMILES string of length L with character vocabulary size V:

**Step 1 — Embedding:** Each character c_i is mapped to a dense vector:
```
e_i = Embedding(c_i) ∈ R^64
```
The full sequence becomes a matrix E ∈ R^(64 × L).

**Step 2 — 1D Convolution:** A convolution with kernel size 3 scans across the sequence. For each position j:
```
h1_j = ReLU( W1 * E[:, j-1:j+2] + b1 )    W1 ∈ R^(128×192)
h2_j = ReLU( W2 * h1[:, j-1:j+2] + b2 )   W2 ∈ R^(64×384)
```
Two conv layers allow the model to capture patterns spanning up to 5 characters (receptive field = 3 + 3 - 1 = 5).

**Step 3 — Global Max Pooling:** Collapse the sequence dimension:
```
f = MaxPool(h2)  ∈ R^64
```
This picks the strongest activation for each feature across the entire sequence — effectively asking "does this structural pattern appear anywhere in the molecule?".

### The Code

```python
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # padding_idx=0 means the zero-padding token contributes no gradient
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        # kernel_size=3: looks at trigrams (3 consecutive characters)
        # padding=1: preserves sequence length (same-padding)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)

    def forward(self, smiles_tokens):
        # smiles_tokens: [num_drugs, 256]  — all drugs tokenised to fixed length 256
        x = self.embed(smiles_tokens)       # [num_drugs, 256, 64]
        x = x.transpose(1, 2)              # [num_drugs, 64, 256] — Conv1d expects (N, C, L)
        x = F.relu(self.conv1(x))          # [num_drugs, 128, 256]
        x = F.relu(self.conv2(x))          # [num_drugs, 64, 256]
        return F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)
        # max_pool1d over full length → [num_drugs, 64]
        # One 64-dim chemical fingerprint per drug
```

**Output:** `smiles_feats` ∈ R^(num_drugs × 64)

---

## 5. Block 2 — Drug Node Feature Construction

### The Intuition

A drug has two sources of identity:
1. **Chemical structure** — what it's made of (from SMILES)
2. **Interaction history** — which drugs it tends to appear with (learned from the data)

Neither alone is sufficient. Two structurally similar drugs might behave very differently in combinations because of different metabolic pathways. And pure ID embeddings have no generalisation to unseen drugs. Combining both gives the model both molecular and relational signals.

### The Math

For each drug d:
```
drug_id_emb(d)  ∈ R^64     — learnable ID embedding (random init, trained)
smiles_feats(d) ∈ R^64     — chemical fingerprint from SmilesEncoder

concat_d = [drug_id_emb(d) ‖ smiles_feats(d)]  ∈ R^128
node_feat(d) = ReLU( W_proj · concat_d + b_proj )  ∈ R^128
```

W_proj ∈ R^(128×128) is a linear projection that lets the model learn how to weight the chemical vs relational information.

### The Code

```python
class HGNN_SA(nn.Module):
    def __init__(self, num_drugs, num_ses, smiles_vocab_size):
        super().__init__()
        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        # Learnable lookup table: one 64-dim vector per drug
        # Randomly initialised and updated during training
        # Captures relational/co-occurrence patterns

        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        # Produces chemical fingerprint from SMILES string

        self.node_proj = nn.Linear(128, 128)
        # Projects the concatenated 128-dim vector into a shared feature space
```

```python
def forward(self, hyperedge_index, se_indices, all_smiles_tokens):
    # all_smiles_tokens: [num_drugs, 256] — ALL drugs, not just the ones in this batch
    # This is done once per forward pass; the hypergraph conv selects what's relevant

    smiles_feats = self.smiles_enc(all_smiles_tokens)   # [num_drugs, 64]

    node_feats = F.relu(self.node_proj(
        torch.cat([self.drug_id_emb.weight, smiles_feats], dim=1)
        # .weight directly accesses the full embedding matrix [num_drugs, 64]
    ))
    # node_feats: [num_drugs, 128]
    # Every drug in the vocabulary now has a 128-dim feature vector
```

**Output:** `node_feats` ∈ R^(num_drugs × 128)

---

## 6. Block 3 — Hypergraph Convolution

This is the mathematical core of the model. It is how drugs "communicate" with each other.

### The Math

Standard GNN message passing aggregates features from direct 2-node neighbours. Hypergraph convolution generalises this to hyperedges. It proceeds in two stages per layer.

Given:
- Node feature matrix X ∈ R^(|V| × d)
- Incidence matrix H ∈ {0,1}^(|V| × |E|)
- Diagonal node degree matrix D_v where D_v[i,i] = number of hyperedges node i belongs to
- Diagonal hyperedge degree matrix D_e where D_e[j,j] = number of nodes in hyperedge j
- Learnable weight matrix W ∈ R^(d × d')

The full update equation is:

```
X' = D_v^{-1} · H · W_e · D_e^{-1} · H^T · X · W
```

Breaking this down step by step:

| Operation | Matrix | What it does |
|-----------|--------|--------------|
| H^T · X | R^(&#124;E&#124; × d) | Each hyperedge aggregates its member node features |
| D_e^{-1} · (H^T · X) | R^(&#124;E&#124; × d) | Normalise by hyperedge size (average, not sum) |
| H · (D_e^{-1} · H^T · X) | R^(&#124;V&#124; × d) | Each node aggregates from all its hyperedges |
| D_v^{-1} · ... | R^(&#124;V&#124; × d) | Normalise by node degree |
| (...) · W | R^(&#124;V&#124; × d') | Linear transformation |

The normalisation by D_v and D_e prevents nodes in many hyperedges from exploding in magnitude — it is the hypergraph analogue of symmetric normalisation in standard GCN.

**Intuition:** Drug A in prescription `{A, B, C}` and prescription `{A, D, E}`:
- Layer 1: A sees the average of B, C, D, E
- Layer 2: A sees drugs two hops away (e.g., what B co-occurs with in *other* prescriptions)

After 2 layers, each drug's feature vector is informed by the entire neighbourhood of prescriptions it appears in.

### The Code

```python
self.conv1 = HypergraphConv(128, 128)
self.conv2 = HypergraphConv(128, 128)
# From torch_geometric — implements the full normalised hypergraph convolution
# Args: (in_channels, out_channels)
# Both layers keep dimension at 128 — depth without expansion
```

```python
# In forward():
x = F.relu(self.conv1(node_feats, hyperedge_index))
# node_feats: [num_drugs, 128]
# hyperedge_index: [2, num_memberships]  — COO format incidence matrix
# x: [num_drugs, 128]  — drug features after 1 round of message passing

x = F.relu(self.conv2(x, hyperedge_index))
# x: [num_drugs, 128]  — drug features after 2 rounds of message passing
```

**The hyperedge_index format explained:**

```
Prescription 0: {DrugA=0, DrugB=1, DrugC=2}
Prescription 1: {DrugA=0, DrugD=3}

hyperedge_index = tensor([
    [0, 1, 2, 0, 3],   ← row 0: node (drug) indices
    [0, 0, 0, 1, 1]    ← row 1: hyperedge (prescription) indices
])
```

Each column is one (drug, prescription) membership. This is exactly the COO representation of the incidence matrix H.

**Output:** `x` ∈ R^(num_drugs × 128) — context-aware drug features

---

## 7. Block 4 — Set Attention Pooling

### The Problem

After message passing, we have one feature vector per drug node. But we need one feature vector per **drug combination** (per hyperedge). We need to pool the drug vectors belonging to each hyperedge into a single vector.

The naive approach is mean pooling: just average the drug vectors in each combination. But this treats all drugs in the combination as equally important. In reality, one drug might be the primary driver of an interaction while others play a minor role.

**Attention pooling** learns to assign importance weights.

### The Math

For a hyperedge e containing drugs {v_1, v_2, ..., v_k} with post-message-passing features {x_1, ..., x_k}:

**Step 1 — Score each drug:**
```
s_i = MLP_attn(x_i)  ∈ R^1
```
MLP_attn is a 2-layer network: Linear(128→64) → ReLU → Linear(64→1)

**Step 2 — Normalise scores within the hyperedge (softmax):**
```
α_i = exp(s_i) / Σ_{j∈e} exp(s_j)
```
This is a softmax applied *within* each hyperedge group, so weights sum to 1 per combination.

**Step 3 — Weighted sum:**
```
combo_vector_e = Σ_{i∈e} α_i · x_i  ∈ R^128
```

### The Code

```python
self.attn_layer = nn.Sequential(
    nn.Linear(128, 64),   # compress to 64
    nn.ReLU(),
    nn.Linear(64, 1)      # single scalar score per drug
)
```

```python
# In forward():
drugs = x[hyperedge_index[0]]
# hyperedge_index[0] contains drug indices for every (drug, hyperedge) pair
# drugs: [num_memberships, 128]
# e.g. if 3 prescriptions have 2, 3, 4 drugs → num_memberships = 9

attn_probs = scatter_softmax(
    self.attn_layer(drugs),   # raw scores: [num_memberships, 1]
    hyperedge_index[1],       # group by hyperedge index
    dim=0
)
# scatter_softmax computes softmax WITHIN each hyperedge group, not globally
# attn_probs: [num_memberships, 1]  — weights summing to 1 per hyperedge

combo_vector = scatter_add(
    drugs * attn_probs,       # weighted drug features: [num_memberships, 128]
    hyperedge_index[1],       # sum within each hyperedge group
    dim=0
)
# combo_vector: [num_hyperedges, 128]
# One 128-dim vector per drug combination
```

**Why scatter operations?** Each hyperedge can have a different number of drugs. `scatter_softmax` and `scatter_add` handle variable-size groups efficiently without padding.

**Output:** `combo_vector` ∈ R^(num_hyperedges × 128)

---

## 8. Block 5 — Side Effect Interaction and Prediction

### The Intuition

The `combo_vector` represents the combined effect of the drug set. To predict whether it causes a *specific* side effect, we need to compare that vector against a representation of the side effect itself.

Each side effect also gets a learnable 128-dim embedding. The model learns to place drug combinations and their associated side effects close together in this space.

### The Math

For a (drug combination e, side effect s) pair:

```
combo_vector_e  ∈ R^128    — from attention pooling
se_emb(s)       ∈ R^128    — learnable side effect embedding

input = [combo_vector_e ‖ se_emb(s)]  ∈ R^256

z = ReLU( W1 · input + b1 )    ∈ R^128
z = Dropout(z, p=0.3)
logit = W2 · z + b2             ∈ R^1

probability = σ(logit) = 1 / (1 + e^{-logit})
```

The loss during training is **Binary Cross-Entropy with Logits**:
```
L = -[ y · log(σ(logit)) + (1-y) · log(1 - σ(logit)) ]
```
where y ∈ {0, 1} is the ground truth label.

Using `BCEWithLogitsLoss` instead of `BCELoss(sigmoid(logit))` is numerically more stable because it uses the log-sum-exp trick internally.

### The Code

```python
self.se_emb = nn.Embedding(num_ses, 128)
# One 128-dim learnable vector per unique side effect
# Randomly initialised, trained alongside everything else

self.mlp = nn.Sequential(
    nn.Linear(256, 128),   # 256 = 128 (combo) + 128 (SE)
    nn.ReLU(),
    nn.Dropout(0.3),       # randomly zero 30% of activations during training
                           # prevents over-reliance on any single feature → regularisation
    nn.Linear(128, 1)      # single logit — no sigmoid here, applied in loss
)
```

```python
# In forward():
out = torch.cat([combo_vector, self.se_emb(se_indices)], dim=1)
# combo_vector:       [num_hyperedges, 128]
# self.se_emb(se_indices): [num_hyperedges, 128]
# out:                [num_hyperedges, 256]

return self.mlp(out).squeeze(-1)
# Returns [num_hyperedges] — one raw logit per (drug combo, side effect) sample
```

```python
# Training loss:
criterion = nn.BCEWithLogitsLoss()
loss = criterion(out, train_y)   # out: logits, train_y: 0.0 or 1.0

# Inference:
probs = torch.sigmoid(out_test).cpu().numpy()
y_pred = (probs > 0.5).astype(int)
```

**Output:** One probability per (drug combination, side effect) pair.

---

## 9. Data Preparation

### Dataset Structure

`hoddi_merged.csv` has one row per (drug combination, side effect, time quarter) triple:

| Column | Content |
|--------|---------|
| `DrugBankID` | List of DrugBank IDs e.g. `['DB00001', 'DB00002', 'DB00003']` |
| `SE_label` | UMLS CUI for the side effect e.g. `C0018681` |
| `hyperedge_label` | `1` = reported combination, `-1` = negative sample |
| `time` | Quarter e.g. `2019Q2` |

### Tokenising SMILES

```python
char_vocab = set()
for s in smiles_dict.values():
    if isinstance(s, str):
        char_vocab.update(list(s))
# Collect every unique character across all SMILES strings

char2idx = {c: i+1 for i, c in enumerate(sorted(char_vocab))}
# 1-indexed; 0 is reserved for padding (padding_idx=0 in the Embedding layer)
char2idx['<UNK>'] = len(char2idx) + 1
# Unknown characters (drugs with unusual notation) map to this token

MAX_SMILES_LEN = 256
all_smiles_tokens = np.zeros((len(drug2idx), MAX_SMILES_LEN), dtype=int)
# Pre-allocate as zeros (= padding) for all drugs

for drug, idx in drug2idx.items():
    smile = smiles_dict.get(drug, "C")  # "C" (methane) as fallback for missing SMILES
    tokens = [char2idx.get(c, char2idx['<UNK>']) for c in list(smile)[:MAX_SMILES_LEN]]
    all_smiles_tokens[idx, :len(tokens)] = tokens
    # Sequences shorter than 256 remain zero-padded at the end
```

### Chronological Split

Random splits would let the model "peek" at future prescription patterns during training. Instead, the dataset is split by time quarter:

```
Total quarters (e.g. 2014Q3 → 2024Q3 = 41 quarters)
├── Train: first 70% of quarters  → 2014Q3 – 2021Q4  (28 quarters)
├── Val:   next 15% of quarters   → 2022Q1 – 2022Q4  (6 quarters)
└── Test:  last 15% of quarters   → 2023Q1 – 2024Q3  (7 quarters)
```

```python
unique_quarters = sorted(df['time'].unique())   # sorted chronologically
num_q = len(unique_quarters)
train_end = int(0.70 * num_q)
val_end   = train_end + int(0.15 * num_q)

train_q = unique_quarters[:train_end]
val_q   = unique_quarters[train_end:val_end]
test_q  = unique_quarters[val_end:]
```

This is the correct evaluation strategy for a temporal dataset — the test set is genuinely unseen future data.

### Building the hyperedge_index

```python
def build_tensors(split_df):
    d_idx, r_idx, s_idx, labels = [], [], [], []

    for i, row in split_df.iterrows():
        for d in row['DrugBankID']:
            d_idx.append(drug2idx[d])   # drug node index
            r_idx.append(i)             # hyperedge index = row index
            # Every drug in the same row shares the same hyperedge index
            # This encodes the incidence relationship: drug d ∈ hyperedge i
        s_idx.append(se2idx[row['SE_label']])
        labels.append(row['target'])    # 1 (positive) or 0 (negative)

    return (
        torch.tensor([d_idx, r_idx], dtype=torch.long),  # hyperedge_index [2, E]
        torch.tensor(s_idx, dtype=torch.long),            # side effect index per hyperedge
        torch.tensor(labels, dtype=torch.float)           # binary labels
    )
```

**Concrete example:**

```
Row 5:  DrugBankID=['DB001','DB002','DB003'],  SE_label='C0018681',  target=1
Row 6:  DrugBankID=['DB001','DB004'],           SE_label='C0027051',  target=0

d_idx = [0,  1,  2,  0,  3 ]    ← drug indices (DB001=0, DB002=1, DB003=2, DB004=3)
r_idx = [5,  5,  5,  6,  6 ]    ← hyperedge indices
s_idx = [42, 17]                 ← side effect index (one per row)
labels= [1.0, 0.0]
```

---

## 10. Training Loop

### Optimiser and Scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
# AdamW = Adam + decoupled weight decay (L2 regularisation)
# weight_decay=0.001 penalises large weights → reduces overfitting
# lr=0.001 is a standard starting point for AdamW

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
# Watches validation AUC (mode='max' = higher is better)
# If AUC doesn't improve for 10 epochs → multiply lr by 0.1
# e.g. 0.001 → 0.0001 → 0.00001
# This allows coarse-to-fine optimisation
```

### Full-Batch Training

```python
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()

    out = model(train_h, train_s, smiles_t)
    # Passes THE ENTIRE training hypergraph in one call
    # No mini-batches — all training (drug, SE) pairs processed simultaneously
    # This works because the hypergraph conv is a sparse matrix operation

    loss = criterion(out, train_y)
    loss.backward()    # compute gradients for all parameters
    optimizer.step()   # update all parameters
```

### Early Stopping

```python
best_auc = 0
patience_counter = 0
early_stop_patience = 25

if auc > best_auc:
    best_auc = auc
    patience_counter = 0
    torch.save({...}, 'models/HGNN_model.pt')   # checkpoint best model
else:
    patience_counter += 1

if patience_counter >= early_stop_patience:
    break   # stop training — model has stopped improving
```

Early stopping prevents overfitting. The best checkpoint (highest val AUC across all epochs) is what gets loaded for test evaluation — not the final epoch's weights.

### What gets saved in the checkpoint

```python
torch.save({
    'model_state_dict': model.state_dict(),   # all learned weights
    'drug2idx': data['drug2idx'],             # drug string → integer index
    'se2idx': data['se2idx'],                 # SE string → integer index
    'char2idx': data['char2idx'],             # SMILES character → integer index
    'smiles_tensor': data['smiles_tensor']    # pre-tokenised SMILES for all drugs
}, 'models/HGNN_model.pt')
```

The vocabulary mappings are saved alongside the weights because inference (in `app.py`) needs them to convert new inputs into the same integer space the model was trained on.

---

## 11. Evaluation

The model is evaluated on the held-out test set (future quarters the model never saw):

```python
checkpoint = torch.load('models/HGNN_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()   # disables Dropout

with torch.no_grad():   # no gradient computation needed for inference
    out_test = model(test_h, test_s, smiles_t)
    probs = torch.sigmoid(out_test).cpu().numpy()   # convert logits to probabilities
    y_pred = (probs > 0.5).astype(int)              # threshold at 0.5
```

### Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **ROC-AUC** | Area under ROC curve | Ability to rank positives above negatives across all thresholds. 1.0 = perfect, 0.5 = random. Primary metric. |
| **PR-AUC** | Area under Precision-Recall curve | Performance on the positive class specifically. More informative than ROC-AUC when classes are imbalanced. |
| **F1-Score** | 2·(P·R)/(P+R) | Harmonic mean of precision and recall at threshold 0.5. |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correct predictions. |
| **Precision** | TP/(TP+FP) | Of all predicted toxic combinations, how many actually are? |
| **Recall** | TP/(TP+FN) | Of all actually toxic combinations, how many did we catch? |

In a safety-critical medical context, **recall is especially important** — a missed toxic combination (false negative) is more dangerous than a false alarm (false positive).

---

## 12. End-to-End Data Flow Summary

```
hoddi_merged.csv
      │
      ▼
 [Row: DB001+DB002+DB003, SE=C0018681, label=1, time=2019Q2]
      │
      ▼ prepare_sota_data()
      │
      ├─► drug2idx = {DB001:0, DB002:1, DB003:2, ...}
      ├─► se2idx   = {C0018681:42, ...}
      ├─► char2idx = {'C':1, 'N':2, 'O':3, ...}
      │
      ├─► all_smiles_tokens [num_drugs, 256]  ← tokenised SMILES
      │
      └─► hyperedge_index = [[0,1,2,...], [5,5,5,...]]
          se_indices      = [42, ...]
          labels          = [1.0, ...]
                │
                ▼ HGNN_SA.forward()
                │
      ┌─────────────────────────────────────┐
      │                                     │
      │  SmilesEncoder(all_smiles_tokens)   │
      │  → smiles_feats [num_drugs, 64]     │
      │                                     │
      │  drug_id_emb.weight [num_drugs, 64] │
      │                                     │
      │  cat + node_proj                    │
      │  → node_feats [num_drugs, 128]      │
      │                                     │
      │  HypergraphConv × 2                 │
      │  → x [num_drugs, 128]               │
      │                                     │
      │  Set Attention Pooling              │
      │  → combo_vector [num_edges, 128]    │
      │                                     │
      │  cat(combo_vector, se_emb)          │
      │  → [num_edges, 256]                 │
      │                                     │
      │  MLP → logits [num_edges]           │
      └─────────────────────────────────────┘
                │
                ▼
      BCEWithLogitsLoss(logits, labels)
                │
                ▼
      loss.backward() → optimizer.step()
                │
                ▼
      Best checkpoint → models/HGNN_model.pt
                │
                ▼
      Test set evaluation → results/evaluate_HGNN.txt
```

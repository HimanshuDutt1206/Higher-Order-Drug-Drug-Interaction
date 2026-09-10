# PolyFormer: A Deep Explanation

## Table of Contents

1. [The Problem Being Solved](#1-the-problem-being-solved)
2. [Why This is Hard](#2-why-this-is-hard)
3. [High-Level Architecture Overview](#3-high-level-architecture-overview)
4. [Building Blocks: Theory First](#4-building-blocks-theory-first)
   - 4.1 [Embeddings](#41-embeddings)
   - 4.2 [Convolutional Neural Networks on Sequences](#42-convolutional-neural-networks-on-sequences)
   - 4.3 [Self-Attention and the Transformer](#43-self-attention-and-the-transformer)
   - 4.4 [Cross-Attention](#44-cross-attention)
5. [Code Walkthrough: Every Block Explained](#5-code-walkthrough-every-block-explained)
   - 5.1 [Configuration Constants](#51-configuration-constants)
   - 5.2 [SmilesEncoder — CNN Drug Structure Encoder](#52-smilesencoder--cnn-drug-structure-encoder)
   - 5.3 [PolyFormerLearnableSE — The Full Model](#53-polyformerlearnablese--the-full-model)
   - 5.4 [Data Loading and Preparation](#54-data-loading-and-preparation)
   - 5.5 [Packing Drug Sets into Fixed-Size Tensors](#55-packing-drug-sets-into-fixed-size-tensors)
   - 5.6 [Set Invariance Augmentation](#56-set-invariance-augmentation)
   - 5.7 [The Training Loop](#57-the-training-loop)
   - 5.8 [Evaluation](#58-evaluation)
6. [End-to-End Data Flow (One Prediction)](#6-end-to-end-data-flow-one-prediction)
7. [Design Decisions and Why](#7-design-decisions-and-why)
8. [Limitations](#8-limitations)

---

## 1. The Problem Being Solved

When a patient takes **multiple drugs at the same time**, those drugs can interact with each other and cause **side effects** that none of them would cause individually. This is called a **Drug-Drug Interaction (DDI)**.

Most research only studies pairwise interactions (Drug A + Drug B). But in real clinical settings, patients often take 5, 10, or even 15 drugs simultaneously. The interactions in a group of drugs are called **Higher-Order Drug-Drug Interactions (HODDI)**.

**The goal of PolyFormer:**  
Given a *set* of drugs (any size, e.g. 3 or 7 drugs) and a candidate side effect, predict:

> Does this combination of drugs cause this side effect? → **Yes (1) or No (0)**

This is a **binary classification** task. The model is a scorer, not a generator — it does not name side effects from scratch. You give it a drug set and a side effect, and it tells you how likely that combination is to cause it.

---

## 2. Why This is Hard

| Challenge | Why it matters |
|---|---|
| **Variable-size inputs** | Standard neural nets expect fixed-size inputs. Drug sets can have 2 to 15+ drugs. |
| **Order doesn't matter** | {DrugA, DrugB, DrugC} is the same as {DrugC, DrugA, DrugB}. The model must be *set-invariant*. |
| **Sparse data** | Most drug combinations have never been tested together in clinical trials. |
| **High-order interactions** | You can't predict the effect of 5 drugs by just looking at all pairs — emergent effects exist. |
| **Rich drug representations** | A drug's molecular structure matters, not just its identity. |

PolyFormer addresses all of these in a single architecture.

---

## 3. High-Level Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INPUT                                       │
│                                                                      │
│   Drug Set: [DrugA, DrugB, DrugC, ...]    Side Effect: "nausea"    │
└───────────────────────────┬──────────────────────────┬──────────────┘
                            │                          │
                ┌───────────▼───────────┐   ┌──────────▼──────────┐
                │   For EACH drug:      │   │  SE Embedding       │
                │                       │   │                      │
                │  Drug ID ──► [64-dim] │   │  "nausea" ──► [128] │
                │  SMILES  ──► [64-dim] │   │  (learnable lookup) │
                │           (CNN)       │   └──────────┬──────────┘
                │                       │              │
                │  Concat → [128-dim]   │              │
                │  Linear → [128-dim]   │              │
                └───────────┬───────────┘              │
                            │                          │
                ┌───────────▼───────────┐              │
                │  Transformer          │              │
                │  Self-Attention       │              │
                │                       │              │
                │  Each drug attends    │              │
                │  to ALL other drugs   │              │
                │  → [K × 128-dim]      │              │
                └───────────┬───────────┘              │
                            │                          │
                ┌───────────▼──────────────────────────▼──────────┐
                │           Cross-Attention                        │
                │                                                  │
                │  SE embedding QUERIES the drug set               │
                │  "Which drugs are relevant to nausea?"           │
                │  → [128-dim] context vector                      │
                └───────────────────────┬──────────────────────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  MLP Classifier        │
                            │                        │
                            │  [context, SE] → [256] │
                            │  → [128] → [1]         │
                            │  → sigmoid → prob      │
                            └───────────┬────────────┘
                                        │
                                  0.73 (73% likely)
```

---

## 4. Building Blocks: Theory First

Before reading the code, you need to understand the four fundamental ideas the model uses.

---

### 4.1 Embeddings

An **embedding** is a way to represent a discrete item (like a drug ID or a character) as a continuous vector of numbers.

Imagine you have 500 drugs. You can't feed the string "DrugBank:DB00001" into a neural network. Instead, you assign each drug a vector of 64 floating-point numbers:

```
"DB00001"  →  [0.23, -0.11, 0.87, ..., 0.44]   (64 numbers)
"DB00002"  →  [-0.05, 0.72, 0.13, ..., -0.31]  (64 numbers)
```

These numbers are **learned during training**. Initially random, they shift so that drugs with similar interaction profiles end up with similar vectors (close in vector space).

This is implemented in PyTorch as `nn.Embedding(num_items, embedding_dim)`, which is essentially a lookup table — given an integer index, return the corresponding row.

**Mathematically:**  
Given item index `i` ∈ {0, ..., N-1}, and embedding matrix `E` of shape `[N, d]`:

```
embed(i) = E[i]     (just a row lookup)
```

Training updates `E` via gradient descent.

---

### 4.2 Convolutional Neural Networks on Sequences

You might know CNNs from image processing, but they work on any sequence. For a 1D sequence (like a string of characters), a convolution slides a small window across the sequence and computes a weighted sum at each position.

**Intuition:** A window of size 3 looks at 3 consecutive characters at a time. If the model learns to detect "C=O" (a carbonyl group, important in chemistry), the convolution fires strongly wherever it sees that pattern.

**Mathematically:**  
Given input sequence `x` of length `L`, where each position has `C_in` features, a Conv1D with kernel size 3 computes at each position `t`:

```
output[t] = ReLU( Σ_{k=0}^{2}  W[k] · x[t + k - 1]  +  b )
```

where `W` is the learned kernel (filter) of shape `[C_out, C_in, 3]`.

After two convolution layers, **max pooling** collapses the entire sequence into one vector by taking the maximum value at each feature dimension across all positions:

```
output[j] = max over t of  conv_output[t, j]
```

This gives a single fixed-size vector regardless of the input sequence length.

---

### 4.3 Self-Attention and the Transformer

Self-attention is the mechanism that lets each element in a sequence look at every other element and decide how much to "pay attention" to it.

**Intuition:** Imagine 4 drugs in a set. Drug A asks: "Given that I'm in this combination, which of the other drugs are most relevant to understanding how I contribute to the interaction?" The answer is computed dynamically based on the content of all drug representations.

**The Math:**

Each input vector `x_i` is linearly projected into three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I offer to others?"
- **Value (V):** "What information do I actually pass on?"

```
Q = X · W_Q       shape: [K, d_model]
K = X · W_K       shape: [K, d_model]
V = X · W_V       shape: [K, d_model]
```

where `X` is the matrix of all K drug embeddings stacked, and `W_Q, W_K, W_V` are learned weight matrices.

The attention score between drug `i` and drug `j` is:

```
score(i, j) = (Q[i] · K[j]) / √d_model
```

Dividing by `√d_model` prevents the dot products from getting too large (which would push the softmax into very flat or very sharp distributions, hurting gradient flow).

These scores are normalized with softmax to get attention weights that sum to 1:

```
α[i, j] = exp(score(i,j)) / Σ_k exp(score(i,k))
```

The new representation of drug `i` is a weighted sum of all values:

```
output[i] = Σ_j  α[i, j] · V[j]
```

**Multi-Head Attention** runs this process `nhead` times in parallel, each with its own set of `W_Q, W_K, W_V` matrices. Each "head" can specialize — one might learn "drugs with similar metabolism", another "drugs that share a receptor". The outputs of all heads are concatenated and projected back to `d_model`.

**The full Transformer Encoder Layer** is:

```
drug_ctx = LayerNorm( drug_feat + MultiHeadSelfAttention(drug_feat) )
drug_ctx = LayerNorm( drug_ctx  + FeedForward(drug_ctx) )
```

The **FeedForward** is a simple 2-layer MLP applied independently to each position. The **residual connections** (`+ drug_feat`) ensure gradients can flow directly to earlier layers (this is what makes deep networks trainable).

---

### 4.4 Cross-Attention

Cross-attention is the same mechanism as self-attention, but the Query comes from one source and the Keys/Values come from another.

Here, the **side effect embedding is the Query**, and the **drug set representations are the Keys and Values**:

```
Q = SE_embedding    (1 vector: what does "nausea" care about?)
K = V = drug_ctx    (K vectors: all drug representations)

output = softmax(Q · Kᵀ / √d) · V
```

The result is a single vector: a weighted summary of the drug set, where the weights reflect how relevant each drug is to *this specific side effect*.

This is the core insight of PolyFormer. A different side effect would produce a completely different weighting of the same drug set, and thus a different interaction vector.

---

## 5. Code Walkthrough: Every Block Explained

---

### 5.1 Configuration Constants

```python
MAX_DRUGS = 16
MAX_SMILES_LEN = 256

OUT_MODEL = "models/polyformer_model.pt"
OUT_REPORT = "results/Evaluate_PolyFormer.txt"
```

**`MAX_DRUGS = 16`**  
Transformers require fixed-size inputs in practice. The HODDI dataset has drug combos of varying sizes (2 to 15+). We pick 16 as the upper limit because it covers the vast majority of records. Combos with more than 16 drugs are deterministically truncated (sorted alphabetically, first 16 kept). Shorter combos are padded with a sentinel value of `-1`.

**`MAX_SMILES_LEN = 256`**  
SMILES strings also vary in length. 256 characters covers most drug structures. Shorter strings are zero-padded; longer ones are truncated.

**Why 16 and 256 specifically?**  
These are practical engineering choices — large enough to cover most real data, small enough to keep memory usage manageable during training. They were verified to cover the vast majority of the dataset with minimal truncation loss.

---

### 5.2 SmilesEncoder — CNN Drug Structure Encoder

```python
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)
```

**`nn.Embedding(vocab_size, embed_dim, padding_idx=0)`**  
Maps each character in the SMILES string to a 64-dim vector. The `padding_idx=0` means that character index 0 (used for padding) always gets the zero vector and its gradients are ignored — it contributes nothing to learning.

The `vocab_size` is the number of unique characters across all SMILES strings in the dataset (e.g. letters like C, N, O, digits, brackets, =, #, etc.) plus a special `<UNK>` token for characters not seen in training.

**`nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)`**  
A 1D convolution that:
- Takes `embed_dim=64` input channels (the character embedding dimension)
- Produces `128` output channels (128 different learned filters)
- Uses a kernel of size 3 (looks at 3 consecutive characters at a time)
- `padding=1` adds one zero on each side so the output length equals the input length (same-padding)

**`nn.Conv1d(128, out_dim, kernel_size=3, padding=1)`**  
Second convolution that compresses 128 channels down to `out_dim=64`.

```python
    def forward(self, smiles_tokens):  # [B, L]
        x = self.embed(smiles_tokens).transpose(1, 2)  # [B, E, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)  # [B, out_dim]
        return x
```

**Step-by-step tensor shapes:**

| Operation | Shape | Meaning |
|---|---|---|
| Input `smiles_tokens` | `[B, 256]` | B drugs, each as 256 character indices |
| After `embed` | `[B, 256, 64]` | Each character is now a 64-dim vector |
| After `transpose(1,2)` | `[B, 64, 256]` | Conv1D expects `[batch, channels, length]` |
| After `conv1 + ReLU` | `[B, 128, 256]` | 128 filters detect local patterns |
| After `conv2 + ReLU` | `[B, 64, 256]` | Compressed to 64 channels |
| After `max_pool1d` | `[B, 64, 1]` | Best activation across the whole sequence |
| After `squeeze(-1)` | `[B, 64]` | One 64-dim vector per drug |

**Why ReLU?**  
ReLU (Rectified Linear Unit): `f(x) = max(0, x)`. It introduces non-linearity — without it, stacking linear layers is mathematically equivalent to a single linear layer, and the model can't learn complex patterns.

**Why max pooling at the end?**  
We need a fixed-size output regardless of sequence length. Max pooling picks the most strongly activated feature across all positions — essentially asking "did this pattern appear anywhere in the molecule?" This is more appropriate than average pooling for detecting the *presence* of a chemical motif.

**Why CNN and not Transformer for SMILES?**  
CNNs are fast and local patterns in SMILES (functional groups, ring structures) are what carry chemical meaning. The 3-gram window is well-suited for this. Using a full Transformer on 256-character sequences for every drug in every batch would be computationally expensive and likely unnecessary.

---

### 5.3 PolyFormerLearnableSE — The Full Model

#### Constructor

```python
class PolyFormerLearnableSE(nn.Module):
    def __init__(self, num_drugs, smiles_vocab_size, num_ses,
                 d_model=128, nhead=8, num_layers=2):
        super().__init__()

        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.drug_proj = nn.Linear(128, d_model)
```

**`self.drug_id_emb`**: A learnable lookup table for drug identity. Each drug in the dataset gets a unique 64-dim vector. This captures "behavioral" information — how this drug has appeared in interaction patterns in the training data. It doesn't know anything about chemistry; it just knows which drug this is.

**`self.smiles_enc`**: The CNN from Section 5.2. This captures "structural" information from the drug's molecular formula. It doesn't know the drug's ID; it reads the chemical structure directly.

**Why use both?** They capture complementary information. The ID embedding learns from interaction patterns in the data. The SMILES CNN learns from molecular structure. A drug that just entered the dataset (new drug) would have a random ID embedding but a meaningful SMILES encoding. Together they're more robust.

**`self.drug_proj = nn.Linear(128, d_model)`**: After concatenating the 64-dim ID embedding and 64-dim SMILES vector, we have 128 dimensions. This linear layer projects it to `d_model=128`. This might seem like a no-op (128→128), but it serves as a learned "mixing" of the two sources — it's a full matrix multiplication, not just keeping the same numbers.

```python
        self.se_emb = nn.Embedding(num_ses, d_model)
```

**`self.se_emb`**: Learnable embedding for side effects. Each side effect (e.g., "nausea", "bradycardia") gets its own 128-dim vector. During training, the model learns to encode each side effect as a vector that captures what kinds of drug combinations are relevant to it.

This is called "learnable" (vs. using a fixed pre-computed similarity vector from a file) because the embedding is trained from scratch via gradient descent, making it fair to compare against the HGNN model.

```python
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
```

**`TransformerEncoderLayer`**: One layer of the Transformer. Internally it runs:
1. Multi-head self-attention
2. Add residual + LayerNorm
3. Feed-forward MLP (input → 4×d_model → d_model)
4. Add residual + LayerNorm

**`d_model=128`**: The working dimension everywhere in the model.

**`nhead=8`**: 8 attention heads. Each head uses `d_model / nhead = 128 / 8 = 16` dimensions. Having 8 heads means 8 different "views" of drug relationships can be learned simultaneously.

**`dim_feedforward=4 * d_model = 512`**: The feed-forward sublayer expands to 512 dims then back to 128. This expansion-then-compression is standard in Transformers — the wider intermediate layer gives the network more capacity to learn complex transformations.

**`dropout=0.1`**: During training, 10% of activations are randomly zeroed. This prevents the model from memorizing training data (overfitting). It's disabled at inference time.

**`batch_first=True`**: Tells PyTorch the input tensors are `[Batch, Sequence, Features]` rather than `[Sequence, Batch, Features]`. This is just a convention choice.

**`norm_first=False`**: Post-norm: normalization happens after the residual addition (original Transformer design). Pre-norm (norm first) is sometimes more stable for very deep networks, but with only 2 layers, post-norm is fine.

**`num_layers=2`**: Stack 2 of these encoder layers. Layer 1 lets each drug see its immediate neighbors. Layer 2 lets each drug see the enriched representations from layer 1 — effectively capturing second-order interactions.

```python
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True
        )
```

**`MultiheadAttention`**: Cross-attention layer. Unlike the encoder above, this one accepts separate `query`, `key`, and `value` arguments — the query will come from the side effect embedding, the key and value will come from the drug representations.

```python
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )
```

**The final classifier:**
- Input: 256 dims (128 from cross-attention output + 128 from SE embedding)
- Hidden: 128 dims with ReLU
- Dropout 0.3: stronger regularization here since this is the decision layer
- Output: 1 number (a logit, converted to probability via sigmoid outside)

**Why concatenate both the cross-attention output and the raw SE embedding?** The cross-attention output encodes "how the drug combination relates to this side effect." The raw SE embedding encodes "what this side effect is, independent of the drugs." The MLP gets both: the interaction and the prior.

---

#### Forward Pass

```python
    def forward(self, drugs_pad, mask, se_idx, smiles_tok_all):
        B, K = drugs_pad.shape
        drugs_safe = drugs_pad.clamp(min=0)
```

**`drugs_pad`**: Shape `[B, K]`. A batch of B samples, each containing up to K=16 drug indices. Padding positions contain `-1`.

**`mask`**: Shape `[B, K]`. 1 where a real drug exists, 0 where it's padding.

**`se_idx`**: Shape `[B]`. One side effect index per sample.

**`smiles_tok_all`**: Shape `[num_drugs, 256]`. Pre-tokenized SMILES for every drug in the vocabulary. Stored on GPU to avoid repeated transfers.

**`clamp(min=0)`**: Replaces all `-1` padding values with `0` before using them as embedding indices (you can't index with negative numbers). The mask ensures these padded positions are ignored by the attention layers, so it doesn't matter what index they use.

```python
        did = self.drug_id_emb(drugs_safe)  # [B, K, 64]
```

Look up ID embeddings for all drugs. Padding positions get the embedding of drug 0, but will be masked out later.

```python
        flat = drugs_safe.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        smiles_tokens = smiles_tok_all[uniq]          # [U, 256]
        smiles_uniq = self.smiles_enc(smiles_tokens)  # [U, 64]
        smiles_feat = smiles_uniq[inv].reshape(B, K, -1)  # [B, K, 64]
```

**This is an optimization.** In a batch of 512 samples, many samples share the same drugs. Instead of running the CNN 512×16=8192 times, we:
1. `reshape(-1)`: Flatten all drug indices into one list of B×K = 8192 entries
2. `torch.unique`: Find the unique drug indices (e.g., only 300 unique drugs)
3. Run the CNN only on those 300 unique SMILES → `[300, 64]`
4. `[inv]`: Use the inverse indices to map each of the 8192 entries back to its CNN output
5. `reshape(B, K, -1)`: Restore the batch×position structure

This can be **20-30x faster** than the naive approach.

```python
        drug_feat = torch.cat([did, smiles_feat], dim=-1)  # [B, K, 128]
        drug_feat = self.drug_proj(drug_feat)               # [B, K, d_model]
```

Concatenate ID and SMILES features along the last dimension, then project.

```python
        key_padding = (mask == 0)  # True where padding
        drug_ctx = self.encoder(drug_feat, src_key_padding_mask=key_padding)
```

**`src_key_padding_mask`**: Tells the Transformer which positions are padding. Where this mask is `True`, those positions are excluded from attention computation — their keys are effectively ignored. This prevents a drug with 3 real drugs from "attending" to 13 meaningless padding vectors.

After the encoder: `drug_ctx` has shape `[B, K, d_model]`. Each drug's representation now contains information about the whole drug set it belongs to.

```python
        q = self.se_emb(se_idx).unsqueeze(1)  # [B, 1, d_model]
        attn_out, _ = self.cross_attn(
            query=q, key=drug_ctx, value=drug_ctx,
            key_padding_mask=key_padding
        )
        combo = attn_out.squeeze(1)   # [B, d_model]
        se_token = q.squeeze(1)       # [B, d_model]
```

**Cross-attention step:**
- The side effect embedding `q` (shape `[B, 1, 128]`) is the single query
- The drug representations `drug_ctx` (shape `[B, K, 128]`) are the keys and values
- For each sample, this computes attention scores between the SE and all K drugs:

```
α[k] = softmax( q · drug_ctx[k]ᵀ / √128 )   for k = 1..K
combo = Σ_k α[k] · drug_ctx[k]
```

- `attn_out` has shape `[B, 1, 128]`; `.squeeze(1)` removes the length-1 sequence dimension to get `[B, 128]`
- The `_` second return value is the attention weight matrix itself — discarded here, but could be visualized for interpretability

```python
        logits = self.mlp(torch.cat([combo, se_token], dim=-1)).squeeze(-1)
        return logits
```

Concatenate the cross-attention output and SE embedding → pass through MLP → single logit per sample. The `.squeeze(-1)` removes the trailing dimension 1 to give shape `[B]`.

**Note:** The logit is a raw score (any real number). Converting it to a probability requires `sigmoid(logit)`. The model returns raw logits rather than probabilities because the loss function (`BCEWithLogitsLoss`) combines sigmoid and cross-entropy in one numerically stable operation.

---

### 5.4 Data Loading and Preparation

```python
def load_and_prepare():
    df = pd.read_csv(HODDI_CSV)
    df["DrugBankID"] = df["DrugBankID"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
```

The `DrugBankID` column stores drug lists as strings like `"['DB00001', 'DB00002']"`. `ast.literal_eval` safely parses this string into an actual Python list.

```python
    df["y"] = (df["hyperedge_label"] == 1).astype(np.int64)
```

Creates the binary target: 1 if the drug combination is labeled as causing the side effect (toxic interaction), 0 otherwise.

```python
    all_drugs = sorted({d for lst in df["DrugBankID"] for d in lst})
    drug2idx = {d: i for i, d in enumerate(all_drugs)}
```

Builds a vocabulary mapping: each unique DrugBank ID gets an integer index. Sorting ensures this mapping is deterministic across runs.

```python
    all_ses = sorted(df["SE_label"].unique().tolist())
    se2idx = {s: i for i, s in enumerate(all_ses)}
    df["se_idx"] = df["SE_label"].map(se2idx).astype(np.int64)
```

Same thing for side effects — each unique side effect name gets an integer index.

```python
    chars = set()
    for s in drug2smiles.values():
        if isinstance(s, str) and len(s) > 0:
            chars.update(list(s))
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char2idx["<UNK>"] = len(char2idx) + 1
    smiles_vocab_size = len(char2idx) + 1
```

Builds a character-level vocabulary for SMILES. Index 0 is reserved for padding (hence `i + 1`). `<UNK>` handles any character at inference time that wasn't in training data. `smiles_vocab_size` is +1 for the padding index 0.

```python
    smiles_tok = np.zeros((len(drug2idx), MAX_SMILES_LEN), dtype=np.int64)
    for drug, di in drug2idx.items():
        s = drug2smiles.get(drug, None)
        if not isinstance(s, str) or len(s) == 0:
            s = "C"   # fallback: carbon atom (simplest valid molecule)
        toks = [char2idx.get(ch, unk) for ch in s[:MAX_SMILES_LEN]]
        smiles_tok[di, :len(toks)] = toks
    smiles_tok = torch.from_numpy(smiles_tok)
```

Pre-tokenizes every drug's SMILES string into a `[num_drugs, 256]` integer tensor. This is done once and stored — the CNN reads from this table during every forward pass. The fallback `"C"` (a single carbon atom) handles drugs with missing SMILES data gracefully.

```python
def quarter_split(df, train_frac=0.70, val_frac=0.15):
    quarters = sorted(df["time"].unique())
    n = len(quarters)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return quarters[:train_end], quarters[train_end:val_end], quarters[val_end:]
```

**Chronological split**: Data is split by time quarter (Q1 2014, Q2 2014, ...) rather than randomly. 70% of quarters go to training, 15% to validation, 15% to test.

**Why chronological?** If you split randomly, the model can "see the future" — it might learn from drug reports from Q4 2023 while predicting reports from Q1 2023. A chronological split is a stricter and more realistic evaluation: the model must generalize to *future* data it has never seen.

---

### 5.5 Packing Drug Sets into Fixed-Size Tensors

```python
def pack_split(split_df, drug2idx):
    N = len(split_df)
    drugs_pad = np.full((N, MAX_DRUGS), -1, dtype=np.int64)
    mask = np.zeros((N, MAX_DRUGS), dtype=np.int64)

    for i, lst in enumerate(split_df["DrugBankID"].tolist()):
        lst = sorted(set(lst))              # deterministic, deduplicated
        idxs = [drug2idx[d] for d in lst[:MAX_DRUGS]]
        drugs_pad[i, :len(idxs)] = idxs
        mask[i, :len(idxs)] = 1

    return {
        "drugs_pad": torch.from_numpy(drugs_pad),
        "mask": torch.from_numpy(mask),
        "se_idx": torch.from_numpy(split_df["se_idx"].to_numpy(np.int64)),
        "y": torch.from_numpy(split_df["y"].to_numpy(np.int64)),
    }
```

Converts the variable-length drug lists into a fixed `[N, MAX_DRUGS]` matrix.

- `np.full(..., -1)`: Initialize everything to -1 (the padding sentinel)
- `sorted(set(lst))`: Deduplicate (in case a drug appears twice) and sort (for deterministic truncation)
- `lst[:MAX_DRUGS]`: Truncate combos longer than 16
- `mask[i, :len(idxs)] = 1`: Mark real positions

**Visualization for a 3-drug combo in a MAX_DRUGS=5 world:**

```
drugs_pad[i] = [  4,  17, 203,  -1,  -1 ]
mask[i]      = [  1,   1,   1,   0,   0 ]
```

---

### 5.6 Set Invariance Augmentation

```python
def permute_drug_order(drugs_pad, mask):
    B, K = drugs_pad.shape
    out = drugs_pad.clone()
    for i in range(B):
        m = mask[i].bool()
        vals = out[i, m]
        if vals.numel() > 1:
            perm = torch.randperm(vals.numel(), device=vals.device)
            out[i, m] = vals[perm]
    return out
```

Called once per training batch, this randomly shuffles the order of real drugs within each sample.

**Why?** Drug sets are mathematical sets — order is meaningless. {Aspirin, Ibuprofen} is the same as {Ibuprofen, Aspirin}. But when stored in a tensor, position matters. Without this augmentation, the model might learn "the first drug listed is most important" — a spurious pattern.

By randomly shuffling at every training step, the model is forced to learn representations that don't depend on order. This is called **set invariance** and is a key theoretical property for this task.

**Note:** At evaluation time, drugs are sorted deterministically (`sorted(set(lst))`), so results are reproducible.

---

### 5.7 The Training Loop

```python
model = PolyFormerLearnableSE(
    num_drugs=len(drug2idx),
    smiles_vocab_size=smiles_vocab_size,
    num_ses=len(se2idx),
    d_model=128,
    nhead=8,
    num_layers=2
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
crit = nn.BCEWithLogitsLoss()
sched = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=10)
```

**`AdamW` optimizer**: A gradient-based optimization algorithm. It maintains a running estimate of the gradient magnitude for each parameter (adaptive learning rates) and adds weight decay as a regularizer. Weight decay penalizes large weights, preventing overfitting:

```
loss_total = loss_task + weight_decay × Σ(w²)
```

**`BCEWithLogitsLoss`**: Binary Cross-Entropy Loss for binary classification. For a predicted logit `z` and true label `y` ∈ {0, 1}:

```
p = sigmoid(z) = 1 / (1 + e^(-z))

loss = -[ y · log(p) + (1 - y) · log(1 - p) ]
```

If the true label is 1 and the model predicts p=0.9: `loss = -log(0.9) ≈ 0.1` (small, good)  
If the true label is 1 and the model predicts p=0.1: `loss = -log(0.1) ≈ 2.3` (large, bad)

The model is penalized heavily for confident wrong predictions.

**`ReduceLROnPlateau`**: A learning rate scheduler. If the validation AUC doesn't improve for `patience=10` consecutive epochs, the learning rate is multiplied by `factor=0.1` (i.e., cut to 10% of current value). This helps the model make finer adjustments as training progresses.

```python
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
```

**Automatic Mixed Precision (AMP)**: Uses 16-bit floats (half precision) instead of 32-bit where safe, cutting memory usage roughly in half and speeding up GPU computation. The `GradScaler` compensates for the reduced precision during gradient computation to avoid numerical underflow.

```python
for epoch in range(1, max_epochs + 1):
    for batch in batch_iter(train_t, batch_size, shuffle=True):

        drugs_pad = permute_drug_order(drugs_pad, mask)   # set invariance

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(drugs_pad, mask, se_idx, smiles_tok_t)
            loss = crit(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
```

**The core training loop:**
1. `permute_drug_order`: Shuffle drug order for set invariance
2. `zero_grad`: Clear gradients from the previous step (PyTorch accumulates by default)
3. `autocast`: Use mixed precision for the forward pass
4. `model(...)`: Forward pass → logits
5. `crit(logits, y)`: Compute loss
6. `loss.backward()`: Backpropagation — compute gradients of the loss w.r.t. every parameter
7. `scaler.step(opt)`: Update all parameters using the computed gradients
8. `scaler.update()`: Update the gradient scaler for next iteration

```python
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            patience_ctr = 0
            torch.save({...}, OUT_MODEL)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:  # patience = 25
            print(f"Early stopping.")
            break
```

**Early stopping**: Save the model only when validation AUC improves. If it hasn't improved for 25 consecutive epochs, stop training entirely and use the best saved checkpoint. This prevents overfitting (the model memorizing training data at the expense of generalizing to new data).

---

### 5.8 Evaluation

```python
def eval_model(model, split_tensors, smiles_tok_t, device, batch_size=2048):
    model.eval()
    ...
    with torch.no_grad():
        ...
        probs = torch.sigmoid(logits)
```

**`model.eval()`**: Switches off dropout and batch normalization update behaviour. The model becomes deterministic.

**`torch.no_grad()`**: Disables gradient tracking during evaluation. Since we're not doing backpropagation, this saves memory and speeds up inference.

**Metrics computed:**

| Metric | What it measures |
|---|---|
| **ROC-AUC** | How well the model separates positives from negatives across all thresholds. 1.0 = perfect, 0.5 = random. This is the primary metric. |
| **PR-AUC** | Area under the Precision-Recall curve. More informative than ROC-AUC when the dataset is imbalanced (many more negatives than positives). |
| **F1-Score** | Harmonic mean of Precision and Recall. Penalizes models that sacrifice one for the other. |
| **Accuracy** | Fraction of correct predictions. Can be misleading with imbalanced data. |
| **Precision** | Of all samples predicted positive, what fraction actually are? |
| **Recall** | Of all actual positives, what fraction did the model find? |
| **Confusion Matrix** | Full breakdown of TP, FP, TN, FN counts. |

---

## 6. End-to-End Data Flow (One Prediction)

Let's trace a single prediction: *Do [DB00001, DB00002, DB00003] cause "nausea"?*

```
Input:
  drugs_pad = [[12, 47, 203, -1, -1, ..., -1]]   shape [1, 16]
  mask      = [[1,  1,  1,  0,  0, ...,  0 ]]    shape [1, 16]
  se_idx    = [7]                                  shape [1]   ("nausea" = index 7)

Step 1 — Drug ID embeddings:
  drug_id_emb([12, 47, 203, 0, 0, ..., 0])
  → did = [B, 16, 64]   (padding positions use drug-0 embedding, will be masked)

Step 2 — SMILES CNN:
  Look up SMILES for drugs 12, 47, 203 from smiles_tok_all
  Run CNN → [U, 64] for unique drugs
  Map back → smiles_feat = [1, 16, 64]

Step 3 — Concatenate and project:
  drug_feat = cat([did, smiles_feat], dim=-1) → [1, 16, 128]
  drug_feat = drug_proj(drug_feat)            → [1, 16, 128]

Step 4 — Transformer self-attention:
  (padding mask: positions 3..15 are ignored)
  drug_ctx = encoder(drug_feat)               → [1, 16, 128]
  Now drug 12's representation incorporates context from drugs 47 and 203.

Step 5 — Cross-attention:
  q = se_emb([7]).unsqueeze(1)                → [1, 1, 128]
  cross_attn(q, drug_ctx, drug_ctx)           → [1, 1, 128]
  combo = squeeze                             → [1, 128]

  Internally: "nausea" computes attention scores against each of the 3 real drugs,
  takes a weighted sum of their representations.

Step 6 — MLP:
  se_token = [1, 128]
  input = cat([combo, se_token])              → [1, 256]
  mlp(input)                                  → [1, 1]
  squeeze                                     → [1]
  logit = 1.06

Step 7 — Probability:
  prob = sigmoid(1.06) ≈ 0.74

Output: 74% probability that [DB00001, DB00002, DB00003] causes "nausea"
```

---

## 7. Design Decisions and Why

| Decision | Rationale |
|---|---|
| **Condition on side effect at input** | Allows side-effect-specific drug weighting via cross-attention. More expressive than a multi-label output head. |
| **Learnable SE embeddings** | Fair comparison with HGNN which also learns SE representations from scratch. Avoids privileging PolyFormer with external signal. |
| **CNN for SMILES** | Fast, local pattern detection is what chemistry needs. Avoids the quadratic cost of self-attention on 256-character sequences per drug. |
| **Two-source drug features (ID + SMILES)** | ID captures behavioral patterns from interaction data; SMILES captures chemical structure. Complementary, each fills the other's gaps. |
| **Quarter-wise chronological split** | Simulates real deployment: model trained on past data, evaluated on future data. Prevents temporal leakage. |
| **Set invariance via permutation** | Drug sets are unordered; augmentation teaches the model this fundamental property of the task. |
| **`MAX_DRUGS=16` truncation** | Covers ~99% of dataset. Deterministic truncation (sorted alphabetically) is stable and reproducible. |
| **Early stopping with patience=25** | Prevents overfitting; saves the best model rather than the final one. |
| **AMP (mixed precision)** | Halves GPU memory usage, typically 1.5–2× speedup with negligible accuracy loss. |

---

## 8. Limitations

**1. It is a scorer, not a generator.**  
You must query each side effect individually. To find all side effects a drug combo might cause, you run N_side_effects forward passes.

**2. Truncation of large combos.**  
Drug sets with more than 16 drugs are silently truncated. The first 16 (alphabetically sorted) are used. For clinical scenarios with very large polypharmacy regimens, this introduces information loss.

**3. Closed vocabulary.**  
The model cannot predict interactions for drugs not seen during training (no drug ID embedding exists for them). SMILES-only inference is possible but the ID embedding would be absent.

**4. No temporal dynamics.**  
The model treats all training data as static. It doesn't model how interaction patterns might shift over time, even though the data spans from 2014 to 2024.

**5. Binary labels.**  
Each (drug set, side effect) pair is labeled 0 or 1. Severity, frequency, and mechanism are not modeled.

---

*This document was written to explain the `polyformer_train.py` implementation found in `src/`.*  
*For the companion HGNN model, see `src/HGNN_train.py`.*

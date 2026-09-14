# PolyFormer: Set Transformer for Higher-Order DDIs

### A Complete Theory, Math, and Code Walkthrough

> **How to read this doc:** Same style as `explanationHGNN.md`. Assumes ML *ideas*, not formulas. Worked examples label where every number comes from.
>
> **Shared pieces:** SMILES CNN, embeddings, ID‖SMILES fusion, loss, chronological split, and basic attention pooling are already in the HGNN doc. This file **points there** and focuses on what PolyFormer does differently: **self-attention over a drug set** and **cross-attention from the side effect**.

### Actual PolyFormer dimension cheat-sheet (from the code)


| Piece                                  | Actual size                            |
| -------------------------------------- | -------------------------------------- |
| SMILES char embedding / fingerprint    | **64** / **64** (same CNN as HGNN)     |
| Drug ID embedding                      | **64**                                 |
| After concat + `drug_proj` (`d_model`) | **128**                                |
| SE embedding / combo vector            | **128**                                |
| Transformer heads                      | **8** (each head: 128/8 = **16** dims) |
| FFN hidden inside encoder              | **512** (= 4 × 128)                    |
| Encoder layers                         | **2**                                  |
| Final MLP                              | **256** → **128** → **1** logit        |
| Max drugs per sample (`MAX_DRUGS`)     | **16** (pad with −1)                   |
| SMILES pad length                      | **256**                                |


### Primer

Same ideas as HGNN (vector, weighted sum, ReLU, softmax, linear, embedding). See the **Primer** in `explanationHGNN.md`. Softmax turns scores into percentages that sum to 1 — that is the heart of attention.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why a Set Transformer (not a Hypergraph)?](#2-why-a-set-transformer-not-a-hypergraph)
3. [Architecture Overview](#3-architecture-overview)
4. [Block 1 — SMILES Encoder](#4-block-1--smiles-encoder)
5. [Block 2 — Drug Feature Construction](#5-block-2--drug-feature-construction)
6. [Block 3 — Self-Attention (drugs talk to each other)](#6-block-3--self-attention-drugs-talk-to-each-other)
7. [Block 4 — Cross-Attention (side effect queries drugs)](#7-block-4--cross-attention-side-effect-queries-drugs)
8. [Block 5 — Prediction Head](#8-block-5--prediction-head)
9. [Data Preparation](#9-data-preparation)
10. [Training Loop](#10-training-loop)
11. [Evaluation](#11-evaluation)
12. [End-to-End Data Flow Summary](#12-end-to-end-data-flow-summary)
13. [PolyFormer vs HGNN (quick)](#13-polyformer-vs-hgnn-quick)

---

## 1. The Problem

Same task as HGNN:

> **Given this set of drugs + this side effect, will the combination cause it?** → probability in (0, 1)

See `explanationHGNN.md` §1 for polypharmacy background. PolyFormer is another architecture for the **same binary question**.

---

## 2. Why a Set Transformer (not a Hypergraph)?

HGNN builds a **global hypergraph**: drugs are nodes; prescriptions are hyperedges; message passing mixes neighbourhoods across the training graph.

PolyFormer treats **each sample alone** as an unordered **set** of drug vectors:


| Challenge                              | PolyFormer approach                                                   |
| -------------------------------------- | --------------------------------------------------------------------- |
| Variable number of drugs               | Pad to `MAX_DRUGS=16`, mask pads                                      |
| Order should not matter                | Self-attention + random shuffle in training                           |
| Drugs must interact inside the combo   | **Self-attention**: each drug looks at every other drug in *this* set |
| Side effect should re-weight the combo | **Cross-attention**: SE vector queries the drug set                   |


No incidence matrix. No full-dataset graph in the forward pass — just one padded set per row.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PolyFormer Forward Pass                     │
│                                                                 │
│  For each drug in the set (up to 16):                           │
│    SMILES ──► SmilesEncoder ──► 64-dim  (same as HGNN)         │
│    Drug ID ──────────────────► 64-dim                           │
│              concat → drug_proj → 128-dim                       │
│                                                                 │
│  Stack K drug vectors → [K × 128]  (pads masked)                │
│                                                                 │
│  TransformerEncoder × 2  (self-attention)                       │
│  Each drug attends to all drugs in THIS set → [K × 128]         │
│                                                                 │
│  Side effect ──► 128-dim SE embedding                           │
│       │                                                         │
│       └─► Cross-Attention: SE queries drug set                  │
│           → one combo vector [128]                              │
│                                                                 │
│  concat(combo, SE) → [256] → MLP → logit → σ → probability      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Block 1 — SMILES Encoder

**Identical idea and code shape as HGNN** (`SmilesEncoder`: embed → Conv1d 64→128 → Conv1d 128→64 → max-pool → **64** numbers).

Do not re-learn the filter math here. Read:

- `explanationHGNN.md` §4 — full worked example on `CCO`
- what the **64** output means (pattern meters; position discarded by max-pool)

**Output here:** for every drug that appears, `smiles_feats[d] ∈ ℝ⁶⁴`.

---

## 5. Block 2 — Drug Feature Construction

### Same fusion as HGNN

```
ID emb (64)  ‖  SMILES (64)  →  concat (128)  →  Linear drug_proj (128→128)
```

Why both ID and SMILES, and why the linear mix: see `explanationHGNN.md` §5 (shared CNN vs private ID row).

### What is different: a set of vectors, not one graph of all drugs

HGNN builds features for **all** drugs, then pools per hyperedge.  
PolyFormer, for **one sample**, builds up to **16** drug vectors and stacks them:

```
Actual shapes for one batch of B samples:
  drugs_pad     [B, 16]      drug indices (−1 = pad)
  mask          [B, 16]      1 = real drug, 0 = pad
  drug_feat     [B, 16, 128] after ID‖SMILES + drug_proj
```

Pads still get some embedding after `clamp(min=0)`, but a **padding mask** tells attention to ignore them.

### Worked Example — one sample with 3 drugs

```
drugs_pad = [12, 47, 203, -1, -1, ..., -1]   # length 16, 12 and 47 and 203 are the drug index of lookup table
mask      = [ 1,  1,   1,  0,  0, ...,  0]

After Block 2 (toy 2-dim stand-in for 128):
  drug_feat[0] ≈ [0.45, 0.35]   # drug 12
  drug_feat[1] ≈ [0.10, 0.90]   # drug 47
  drug_feat[2] ≈ [0.80, 0.20]   # drug 203
  positions 3..15: ignored later by mask
```

### The Code

```python
did = self.drug_id_emb(drugs_safe)                    # [B, K, 64]
# SmilesEncoder on unique drugs in the batch, then map back
smiles_feat = ...                                     # [B, K, 64]
drug_feat = self.drug_proj(torch.cat([did, smiles_feat], dim=-1))  # [B, K, 128]
```

---

## 6. Block 3 — Self-Attention (drugs talk to each other)

This is the main PolyFormer-specific block. Read it slowly — each subsection builds on the last.

### 6.1 What we want

Before this block, each drug in the combo has its own 128-dim vector (from Block 2). Those vectors **do not yet know who else is in the combo**.

After this block, each drug’s vector should be **updated using the other drugs** in the same set.

> Drug A asks: “Given B and C are also here, what should *my* representation become?”

### 6.2 How Query, Key, Value are made (with numbers)

Start with three drug vectors.

| | Toy (hand math) | Actual model |
|---|---|---|
| Drug feature `x` | length **2** | length **128** |
| `W_Q`, `W_K`, `W_V` | **2×2** each | **128×128** each (learned) |
| Full Q, K, V before split | length **2** | length **128** |

```
Toy:
  Drug A  x_A = [1.0, 0.0]
  Drug B  x_B = [0.0, 1.0]
  Drug C  x_C = [1.0, 1.0]

Actual: each x is 128 numbers from Block 2 (ID‖SMILES + drug_proj).
```

The model does **not** use `x` raw for scoring. For each drug it builds **three** new vectors with three linear maps:

```
Q_full = x · W_Q     # Query   Actual: [128]
K_full = x · W_K     # Key     Actual: [128]
V_full = x · W_V     # Value   Actual: [128]
```

**Concrete toy weights** (stand-ins for the real 128×128 matrices):

```
W_Q = [[1, 0],      W_K = [[0, 1],      W_V = [[1, 0],
       [0, 1]]             [1, 0]]             [0, 1]]
```

For drug A (`x_A = [1, 0]`):

```
Q_A = [1, 0] · W_Q = [1, 0]     # “what A is looking for”
K_A = [1, 0] · W_K = [0, 1]     # “how A presents itself to others’ searches”
V_A = [1, 0] · W_V = [1, 0]     # “what content A will contribute if attended”
```

| Drug | x (toy / actual) | Q | K | V |
|---|---|---|---|---|
| A | [1,0] / ℝ¹²⁸ | [1,0] / ℝ¹²⁸ | [0,1] / ℝ¹²⁸ | [1,0] / ℝ¹²⁸ |
| B | [0,1] / ℝ¹²⁸ | [0,1] / ℝ¹²⁸ | [1,0] / ℝ¹²⁸ | [0,1] / ℝ¹²⁸ |
| C | [1,1] / ℝ¹²⁸ | [1,1] / ℝ¹²⁸ | [1,1] / ℝ¹²⁸ | [1,1] / ℝ¹²⁸ |

**Intuition:**

| Vector | Analogy |
|---|---|
| **Query** | The search text you type |
| **Key** | The title/tag on each book |
| **Value** | The book contents you get if you pick it |

Same drug → three different views because `W_Q`, `W_K`, `W_V` differ.

**Actual one-liner for one drug:**  
`x ∈ ℝ¹²⁸` → three matrices → **`Q_full, K_full, V_full` each ∈ ℝ¹²⁸** (not 16 yet).

### 6.3 What “matching” means

Matching = how well drug *i*’s **Query** lines up with drug *j*’s **Key** (dot product).

```
Toy (vectors length 2; scale √2):
  match(A, B) = Q_A · K_B = 1·1 + 0·0 = 1.0
  match(A, A) = Q_A · K_A = 1·0 + 0·1 = 0.0
  match(A, C) = Q_A · K_C = 1·1 + 0·1 = 1.0

Actual (before multi-head split, if you did one big head):
  match(i,j) = (Q_full_i · K_full_j) / √128
  # both vectors length 128; one scalar score
```

High match → pay more attention. Softmax over the other drugs → percentages:

```
Toy:  scores_A ≈ [0.00, 0.71, 0.71] → α_A ≈ [0.21, 0.40, 0.40]
Actual: α still sums to 1 over the real drugs in the set (pads masked to 0)
```

Matching uses **Q vs K only**. **V** is mixed afterward.

### 6.4 Mixing Values → new drug vector (single-head toy)

```
Toy (V length 2):
  new_A ≈ 0.21·[1,0] + 0.40·[0,1] + 0.40·[1,1] ≈ [0.61, 0.80]

Actual single-head analogue:
  new_A = Σ_j α_j · V_full_j     → length 128
```

In the **real** PolyFormer this full-128 attention is **not** run as one head — it is split into 8 heads (next subsection), then patched back to 128.

### 6.5 Multi-head (actual: 8 heads) — create 128, split, attend, patch back

This is what the code actually does (`nhead=8`, `d_model=128` → `d_head = 16`).

**Step 1 — Create full Q, K, V (length 128)**  
Same as §6.2: for each drug, `x ∈ ℝ¹²⁸` → `Q_full, K_full, V_full ∈ ℝ¹²⁸`.

**Step 2 — Split each into 8 heads of length 16**

```
Q_full [128] = [ Q₀(16) | Q₁(16) | Q₂(16) | … | Q₇(16) ]
K_full [128] = [ K₀(16) | K₁(16) | … ]
V_full [128] = [ V₀(16) | V₁(16) | … ]
```

| | Toy analogue | Actual |
|---|---|---|
| Full Q/K/V | length 2 (we don’t split the toy) | length **128** |
| Per head | — | length **16** |
| Number of heads | 1 | **8** |
| W that makes Q_full | 2×2 | **128×128** (not 16×16) |

So **16** = length of each head’s Q/K/V **vectors**, not a 16×16 weight matrix. The map is **128 → 128** then chop, or equivalently eight maps **128 → 16**.

**Step 3 — Attention inside each head** (same matching + softmax + mix as §6.3–6.4, but on 16-dim vectors)

```
For head h, drug A:
  score(A,j) = (Q_h_A · K_h_j) / √16      # both length 16
  α = softmax over drugs
  out_h_A = Σ_j α_j · V_h_j               # length 16
```

Do this for heads `h = 0..7` → eight outputs of length 16.

**Step 4 — Patch back to 128**

```
out_A = concat(out_0, out_1, …, out_7)    # 8×16 = 128
out_A = out_A · W_O                       # output projection, still 128
```

| Stage | Actual shape per drug |
|---|---|
| Input `x` | **[128]** |
| `Q_full`, `K_full`, `V_full` | **[128]** each |
| Per-head Q/K/V | **[16]** each × 8 heads |
| Per-head attention output | **[16]** × 8 |
| After concat (+ `W_O`) | **[128]** |

**Toy vs actual reminder:** the worked numbers in §6.2–6.4 use length-2 vectors so you can multiply by hand = **one tiny head**. The real model does that recipe **eight times** on length-16 chunks of the length-128 Q/K/V, then concatenates.

### 6.6 Full encoder layer — what “self-attn + residual + LN + FFN …” means

One `TransformerEncoderLayer` is **not** only attention. It is a small recipe applied to every drug vector:

```
Input:  x          Actual: 128-dim per drug   (toy above used 2-dim)

①  attn_out = MultiHeadSelfAttention(x)   # §6.5 → still 128-dim
②  x = LayerNorm( x + attn_out )          # residual: both terms 128-dim
③  ffn_out  = FFN(x)                      # Linear 128→512 → ReLU → 128
④  x = LayerNorm( x + ffn_out )           # again 128-dim

Output: still 128-dim per drug
        for one sample with pads: drug_ctx [16, 128]
```

**Residual (`x + attn_out`):** keep the original vector and **add** the attention result. So the drug never loses its own identity completely; attention only adds a delta. Also helps training (gradients flow through the `+ x` path).

Toy residual (length 2; actual both sides length **128**):

```
x_A before attn     = [1.0, 0.0]                 Actual: ℝ¹²⁸
attn new_A          = [0.61, 0.80]               Actual: ℝ¹²⁸ after concat heads
x_A after residual  = [1.61, 0.80]               Actual: ℝ¹²⁸
```

**LayerNorm:** re-scales that vector so its numbers stay in a stable range (across the **128** features). Think “keep sizes well-behaved.”

**FFN (feed-forward):** a tiny MLP on **each drug alone** (no mixing between drugs here):

```
Actual: 128 → 512 → ReLU → 128
```

Attention mixes **across drugs** (who to listen to + weighted average of Values). That mostly *combines* features; it is a weak *rewriter*. The FFN adds nonlinear capacity so each drug can reshape its already-mixed vector into something more useful for the next layer / cross-attention. Pattern: **mix across drugs → think per drug → repeat**.

Then another residual + LayerNorm.

### 6.7 Why stack 2 layers

```
Layer 1: each drug sees raw neighbours → richer vectors
Layer 2: each drug sees those already-enriched vectors → “second-order” mixing
```

After 2 layers → `drug_ctx` shape **`[B, 16, 128]`**.

### The Code

```python
enc_layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=8, dim_feedforward=512,
    dropout=0.1, batch_first=True, activation="relu",
)
self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

key_padding = (mask == 0)  # True = ignore pad slots
drug_ctx = self.encoder(drug_feat, src_key_padding_mask=key_padding)  # [B, K, 128]
```

**vs HGNN:** HGNN mixes drugs through a **hypergraph** over prescriptions. PolyFormer mixes them only **inside this sample’s set** via Q/K/V attention.

---

## 7. Block 4 — Cross-Attention (side effect queries drugs)

### 7.1 What we want

After Block 3 we still have **up to 16** drug vectors. We need **one** combo vector for the classifier — and the focus should depend on the **side effect**.

> SE asks: “For *GI bleeding*, which of these drugs should I weigh more?”

### 7.2 How Q, K, V are made here (different sources)

Same three roles, but **Query does not come from a drug**:

| Role | Where it comes from in cross-attn |
|---|---|
| **Query Q** | Side-effect embedding only (one vector) |
| **Key K** | Each drug’s `drug_ctx` (after self-attn) |
| **Value V** | Each drug’s `drug_ctx` (same as K in this code) |

In code, `MultiheadAttention` still applies learned projections, but conceptually:

```
Q_SE = se_emb("GI bleeding")     # one 128-dim query  (toy below: 2-dim)
K_A, K_B, K_C = drug_ctx vectors
V_A, V_B, V_C = same drug_ctx vectors
```

**Toy numbers** (2-dim):

```
Q_SE = [1.0, 0.0]          # “I care about slot 0”

drug_ctx after self-attn:
  A = [0.80, 0.60]
  B = [0.60, 0.80]
  C = [0.75, 0.75]
```

(For a minimal demo, treat K = V = drug_ctx, as if W_K = W_V = I.)

### 7.3 Matching (SE vs each drug) — full numbers

```
match(SE, A) = Q_SE · A = 1.0·0.80 + 0.0·0.60 = 0.80
match(SE, B) = Q_SE · B = 1.0·0.60 + 0.0·0.80 = 0.60
match(SE, C) = Q_SE · C = 1.0·0.75 + 0.0·0.75 = 0.75
```

Divide by `√d` (toy `√2 ≈ 1.414`; actual `√128`):

```
scores = [0.80, 0.60, 0.75] / 1.414 ≈ [0.566, 0.424, 0.530]
```

Softmax (same recipe as HGNN primer):

```
e^0.566 ≈ 1.761
e^0.424 ≈ 1.528
e^0.530 ≈ 1.700
sum ≈ 4.989

α_A = 1.761 / 4.989 ≈ 0.353
α_B = 1.528 / 4.989 ≈ 0.306
α_C = 1.700 / 4.989 ≈ 0.341
check: 0.353 + 0.306 + 0.341 = 1.000
```

So for “GI bleeding,” the model focuses **~35% A, ~31% B, ~34% C**.

**Matching still means:** high Q·K → that drug gets more weight. Here the searcher is the side effect.

**Actual:** Q is length **128**, each K is length **128**, score is still one scalar per drug; α still sums to 1 over real drugs (pads masked).

### 7.4 Mix Values → one combo vector (worked numbers)

The cross-attn **output** is not a rewrite of the stored `se_emb` table row. It is a **new** vector (same size as Q) = weighted mix of the drug **Values**. In this model that output is called `combo`.

Toy (V = drug_ctx, length 2; actual each V ∈ ℝ¹²⁸, combo ∈ ℝ¹²⁸):

```
combo = 0.353·A + 0.306·B + 0.341·C

slot0 = 0.353·0.80 + 0.306·0.60 + 0.341·0.75
      = 0.282 + 0.184 + 0.256
      = 0.722

slot1 = 0.353·0.60 + 0.306·0.80 + 0.341·0.75
      = 0.212 + 0.245 + 0.256
      = 0.713

combo ≈ [0.722, 0.713]
```

Compare to the raw SE query `[1.0, 0.0]`: the output has been **filled with drug content** according to α. That is what “the SE query gets answered / updated” means in the forward pass.

Unlike self-attention (one updated vector **per drug**), one SE query → **one** combo vector.

### 7.5 Different SE → different combo (full numbers)

Now SE query `Q_SE = [0.0, 1.0]` (“care about slot 1”):

```
match(SE, A) = 0·0.80 + 1·0.60 = 0.60
match(SE, B) = 0·0.60 + 1·0.80 = 0.80
match(SE, C) = 0·0.75 + 1·0.75 = 0.75

scores / √2 ≈ [0.424, 0.566, 0.530]

e^0.424 ≈ 1.528
e^0.566 ≈ 1.761
e^0.530 ≈ 1.700
sum ≈ 4.989

α_A ≈ 0.306   α_B ≈ 0.353   α_C ≈ 0.341
```

B is now the heaviest (was lightest for GI bleeding).

```
combo_hyp slot0 = 0.306·0.80 + 0.353·0.60 + 0.341·0.75
                = 0.245 + 0.212 + 0.256 = 0.713

combo_hyp slot1 = 0.306·0.60 + 0.353·0.80 + 0.341·0.75
                = 0.184 + 0.282 + 0.256 = 0.722

combo_hyp ≈ [0.713, 0.722]   ≠   combo_GI ≈ [0.722, 0.713]
```

Same drugs, different question → different α → different combo.

### 7.6 Residuals / FFN?

In **this** codebase, cross-attention is a single `nn.MultiheadAttention` call — **not** wrapped in a full Transformer encoder layer. So you get Q/K/V attention + the module’s internal projections, then go straight to the MLP classifier. The residual+LN+FFN stack is only on the **self-attention encoder** (Block 3).

### The Code

```python
self.cross_attn = nn.MultiheadAttention(
    embed_dim=128, num_heads=8, dropout=0.1, batch_first=True
)

q = self.se_emb(se_idx).unsqueeze(1)          # [B, 1, 128]  ← Query from SE
attn_out, _ = self.cross_attn(
    query=q, key=drug_ctx, value=drug_ctx,    # K,V from drugs
    key_padding_mask=key_padding,
)
combo = attn_out.squeeze(1)                   # [B, 128]
```

| | Self-attention (Block 3) | Cross-attention (Block 4) |
|---|---|---|
| Q from | every drug | side effect only |
| K, V from | every drug | every drug |
| Output | updated vector **per drug** `[K, 128]` | **one** combo `[128]` |

---

## 8. Block 5 — Prediction Head

Same pattern as HGNN §8:

```
concat(combo [128], SE [128]) → [256]
  → Linear 256→128 → ReLU → Dropout(0.3) → Linear 128→1 → logit
probability = σ(logit)
loss = BCEWithLogits  (worked numbers: explanationHGNN.md §8)
```

```python
logits = self.mlp(torch.cat([combo, se_token], dim=-1)).squeeze(-1)
```

Why concat combo **and** raw SE: combo = how the set relates to this SE; raw SE = what the SE is on its own.

---

## 9. Data Preparation

### Shared with HGNN

Dataset columns, chronological **70/15/15** quarter split, SMILES tokenisation → `explanationHGNN.md` §9.

### PolyFormer-specific: packing a set

```python
drugs_pad[i] = [idx0, idx1, ..., -1, -1]  # length MAX_DRUGS=16
mask[i]      = [1, 1, ..., 0, 0]
```

- `sorted(set(lst))` then truncate to 16 (deterministic)
- Combos longer than 16 lose the rest

### Set-invariance augmentation (train only)

Each batch, shuffle the **real** drug slots so the model cannot latch onto “first position = most important.” At eval, order stays sorted/fixed.

### Drop 1-drug rows

PolyFormer keeps rows with **≥ 2** drugs. HGNN does not apply that filter the same way.

---

## 10. Training Loop

Same spirit as HGNN: AdamW, `BCEWithLogitsLoss`, `ReduceLROnPlateau` on val AUC, early stop patience 25, save best checkpoint.

Differences:

- **Mini-batches** (e.g. 512), not one full-graph step
- Optional AMP on CUDA
- Logs train/val each epoch; evaluates **train / val / test** on the best checkpoint

**Outputs (after you run** `src/polyformer_train.py`**):**


| File                                   | Contents                              |
| -------------------------------------- | ------------------------------------- |
| `results/Evaluate_PolyFormer.txt`      | Train/val/test metrics + overfit gaps |
| `results/PolyFormer_train_history.csv` | Per-epoch train/val AUC etc.          |
| `results/PolyFormer_split_metrics.csv` | Final split summary table             |
| `models/polyformer_model.pt`           | Best weights + vocabs                 |


---

## 11. Evaluation

Load best checkpoint → `model.eval()` → sigmoid → threshold 0.5.  
Metric meanings: `explanationHGNN.md` §11.

Primary number: **test ROC-AUC** on future quarters. Compare to **train** AUC in the report to spot overfitting.

---

## 12. End-to-End Data Flow Summary

```
Row: drugs [DB001, DB002, DB003], SE=nausea
        │
        ▼ pack → drugs_pad [16], mask, se_idx
        │
        ├─ ID emb + SmilesEncoder → [16, 64] each
        ├─ cat + drug_proj → drug_feat [16, 128]
        │
        ▼ TransformerEncoder × 2 (self-attn, pads masked)
        │
        drug_ctx [16, 128]   # each drug sees the others
        │
        ▼ SE emb → query; cross-attn over drug_ctx
        │
        combo [128]
        │
        ▼ cat(combo, SE) → MLP → logit → σ → probability
```

Illustrative trail:

```
3 real drugs → self-attn mixes them
SE "nausea" → α e.g. [0.33, 0.28, 0.39] over those 3
combo → MLP → logit 1.06 → σ ≈ 0.74
```

---

## 13. PolyFormer vs HGNN (quick)


|                  | HGNN-SA                            | PolyFormer                                 |
| ---------------- | ---------------------------------- | ------------------------------------------ |
| Drug mixing      | Hypergraph conv over prescriptions | Self-attn inside each set                  |
| Pooling vs SE    | Attn-pool drugs **then** concat SE | SE **queries** drugs (cross-attn) then MLP |
| Batching         | Full train hypergraph / epoch      | Mini-batches of padded sets                |
| Shared front-end | SMILES CNN + ID + linear fuse      | Same idea                                  |


Same prediction task and chronological split; different inductive bias for how drugs talk and when the side effect enters.

---

*Implementation:* `src/polyformer_train.py`*. Companion:* `explanations/explanationHGNN.md`*.*
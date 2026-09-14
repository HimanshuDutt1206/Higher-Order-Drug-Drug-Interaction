# EHGNN-SA: Hypergraph Neural Network with Set Attention

### A Complete Theory, Math, and Code Walkthrough

> **How to read this doc:** Written for someone who knows ML *ideas* (layers, training, vectors) but not the formulas. Each *Worked Example* labels **where every number comes from**. Toy sizes shrink the real model so you can multiply by hand; **Actual** shapes are always stated.

### Actual HGNN dimension cheat-sheet (from the code)


| Piece                                                | Actual size                     |
| ---------------------------------------------------- | ------------------------------- |
| SMILES char embedding                                | **64**                          |
| Conv1 output channels                                | **128**                         |
| Conv2 / SMILES fingerprint                           | **64**                          |
| Drug ID embedding                                    | **64**                          |
| After concat ID‖SMILES                               | **128**                         |
| `node_proj` / HypergraphConv / combo vector / SE emb | **128**                         |
| Attn MLP hidden                                      | **64** → score **1**            |
| Final MLP                                            | **256** → **128** → **1** logit |
| SMILES pad length                                    | **256** chars                   |


### Primer — only math used in this file

You do not need linear algebra beyond these ideas:

1. **Vector** = a list of numbers, e.g. `[0.8, -0.2]`. “64-dim” means a list of 64 numbers.
2. **Weighted sum** = multiply pairs and add:
  `0.5·1.0 + 0.5·1.0 + 1.0·1.0 = 2.0`.  
   A convolution and attention both do this.
3. **ReLU** = “zero out negatives”: `ReLU(x) = x` if x>0, else `0`. So `ReLU(2)=2`, `ReLU(-1)=0`.
4. **Softmax** = turn any list of scores into positive weights that **sum to 1** (like percentages).
  Example scores `[2, 1, 0.5]`:
  - raise e to each power: `e²≈7.39`, `e¹≈2.72`, `e⁰·⁵≈1.65`
  - add them: `11.76`
  - divide each by the sum: `7.39/11.76≈0.63`, `2.72/11.76≈0.23`, `1.65/11.76≈0.14`
  - check: `0.63+0.23+0.14 = 1.00`
5. **Linear layer** = for each output slot, a weighted sum of all inputs (weights learned).
  If input is `[a,b,c,d]` and one output row of weights is `[0.5, 0, 0.5, 0]`, that output is `0.5·a + 0.5·c`.

An **embedding** is just a lookup table: drug #7 → row 7 of a big table of vectors. No formula — copy the row.

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

### Worked Example — Incidence Matrix

Suppose we have 4 drugs and 2 prescriptions:


| Prescription | Drugs                                |
| ------------ | ------------------------------------ |
| e₀           | {Warfarin=0, Aspirin=1, Ibuprofen=2} |
| e₁           | {Warfarin=0, Metformin=3}            |


The incidence matrix H (rows = drugs, columns = prescriptions):

```
         e₀  e₁
drug 0   1   1     ← Warfarin is in both prescriptions
drug 1   1   0
drug 2   1   0
drug 3   0   1
```

As COO `hyperedge_index` (only the 1-entries, column by column):

```
hyperedge_index = [[0, 1, 2, 0, 3],    ← which drug
                   [0, 0, 0, 1, 1]]    ← which prescription
```

Node degrees D_v (how many prescriptions each drug appears in):
`D_v = diag(2, 1, 1, 1)` — Warfarin appears in 2; the others in 1.

Hyperedge degrees D_e (how many drugs each prescription contains):
`D_e = diag(3, 2)` — e₀ has 3 drugs; e₁ has 2.

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


| Drug     | SMILES                       |
| -------- | ---------------------------- |
| Aspirin  | `CC(=O)Oc1ccccc1C(=O)O`      |
| Caffeine | `Cn1cnc2c1c(=O)n(c(=O)n2C)C` |
| Ethanol  | `CCO`                        |


Every character encodes a structural feature: atoms (`C`, `N`, `O`), bonds (`=`, `#`), rings (`c`, ring-closure numbers), branches (`(`, `)`).

### The Intuition

Instead of using hand-crafted molecular fingerprints (which require domain expertise), we treat SMILES like a text sequence and let the model learn chemical features automatically through convolution — the same way a text CNN learns which n-grams matter.

### The Math (plain English)

Three ideas only:

1. **Embedding:** turn each character into a short list of numbers (actual size: **64** numbers per character).
2. **1D convolution:** slide a window of 3 characters along the string. At each place, multiply the window’s numbers by a small set of **learned weights** (the “filter”) and add them up → one score. Do this with many filters (actual: **128**, then **64**).
3. **Max-pool:** for each filter, keep only its **highest** score along the whole string → one number per filter. That list is the fingerprint (actual: **64** numbers).

Formal notation (optional — the worked example below does not require it):

```
e_i = Embedding(c_i) ∈ R^64
h1_j = ReLU( conv1 over characters j-1, j, j+1 )
h2_j = ReLU( conv2 over h1 at j-1, j, j+1 )
f    = max_over_positions(h2) ∈ R^64
```

### Worked Example — Full SMILES Encoder on ethanol `CCO`

**Goal:** string `CCO` → one fingerprint vector.

We invent tiny numbers so you can see every multiply. The **procedure is identical** in the real model; only the lists are longer (64 / 128 instead of 2).


|                       | Toy (hand calculation) | Actual model             |
| --------------------- | ---------------------- | ------------------------ |
| Characters            | `C C O` (length 3)     | padded to length **256** |
| Numbers per character | **2**                  | **64**                   |
| Filters in conv1      | **2**                  | **128**                  |
| Filters in conv2      | **2**                  | **64**                   |
| Final fingerprint     | list of **2** numbers  | list of **64** numbers   |


---

#### Step 1 — Turn each character into numbers (embedding)

Think of a dictionary:


| Character | Toy vector we assigned | Meaning of the two slots                                     |
| --------- | ---------------------- | ------------------------------------------------------------ |
| `C`       | `[1.0, 0.0]`           | slot0=1 means “carbon-like”, slot1=0 means “not oxygen-like” |
| `O`       | `[0.0, 1.0]`           | opposite                                                     |


(In the real model these 64 numbers are **learned**, not hand-picked. Same idea: each character → a list.)

So for `C C O`:


| Position along the string | Character | Its vector   |
| ------------------------- | --------- | ------------ |
| pos0                      | C         | `[1.0, 0.0]` |
| pos1                      | C         | `[1.0, 0.0]` |
| pos2                      | O         | `[0.0, 1.0]` |


Rewrite the same data as a **table** (rows = the 2 slots of the vector, columns = positions). This is only a different layout — same six numbers:

```
              pos0 (C)   pos1 (C)   pos2 (O)
slot ch0         1.0        1.0        0.0
slot ch1         0.0        0.0        1.0
```

**Actual after this step:** table shape `[64 rows × 256 columns]` per drug.

---

#### Step 2 — What is a “filter”? What is a “window”?

A **filter** is a tiny pattern-detector: a fixed list of weights. During training the computer learns good weights; here we **choose** simple ones so the arithmetic is obvious.

Our filter 0 has **6 weights** because the window covers 3 positions × 2 slots:

```
Weights for filter 0 (we chose these by hand for the demo):

  For slot ch0, looking at (left, center, right):   0.5 ,  0.5 ,  0.0
  For slot ch1, looking at (left, center, right):   0.0 ,  0.0 ,  1.0
```

Read that as: “I care about carbon (ch0) on the left and center, and oxygen (ch1) on the right” — roughly the pattern `C C O`.

A **window** means: pick 3 consecutive columns of the table (and pretend there is a column of zeros past each end — that is **padding**).

```
Full table again (with pads drawn):

              pad     pos0    pos1    pos2    pad
ch0            0       1.0     1.0     0.0     0
ch1            0       0.0     0.0     1.0     0
```

---

#### Step 3 — Score at pos1 (center of the string) — every number traced

Center the window on **pos1**. Then:


| Window role | Which column? | ch0 value there | ch1 value there |
| ----------- | ------------- | --------------- | --------------- |
| left        | pos0          | **1.0**         | **0.0**         |
| center      | pos1          | **1.0**         | **0.0**         |
| right       | pos2          | **0.0**         | **1.0**         |


Now pair each value with the matching weight and multiply:


| Piece      | Weight (from filter) | ×   | Input (from table) | = product                   |
| ---------- | -------------------- | --- | ------------------ | --------------------------- |
| ch0 left   | 0.5                  | ×   | 1.0 (pos0, ch0)    | **0.5**                     |
| ch0 center | 0.5                  | ×   | 1.0 (pos1, ch0)    | **0.5**                     |
| ch0 right  | 0.0                  | ×   | 0.0 (pos2, ch0)    | **0.0**                     |
| ch1 left   | 0.0                  | ×   | 0.0 (pos0, ch1)    | **0.0**                     |
| ch1 center | 0.0                  | ×   | 0.0 (pos1, ch1)    | **0.0**                     |
| ch1 right  | 1.0                  | ×   | 1.0 (pos2, ch1)    | **1.0**                     |
| **Sum h**  |                      |     |                    | **0.5+0.5+0+0+0+1.0 = 2.0** |


```
ReLU(2.0) = 2.0     (positive → unchanged)
```

So the one-line formula

```
h = 0.5·1.0 + 0.5·1.0 + 0·0.0  +  0·0 + 0·0 + 1·1.0 = 2.0
```

is just those six table rows written in order:

- first three products = ch0 left / center / right  
- last three products = ch1 left / center / right

Nothing else is happening.

---

#### Step 4 — Same recipe at pos0 and pos2

**Centered on pos0** → window columns = (pad, pos0, pos1):


| Piece      | Weight | ×   | Input      | =                        |
| ---------- | ------ | --- | ---------- | ------------------------ |
| ch0 left   | 0.5    | ×   | 0 (pad)    | 0                        |
| ch0 center | 0.5    | ×   | 1.0 (pos0) | 0.5                      |
| ch0 right  | 0.0    | ×   | 1.0 (pos1) | 0                        |
| ch1 left   | 0.0    | ×   | 0 (pad)    | 0                        |
| ch1 center | 0.0    | ×   | 0.0 (pos0) | 0                        |
| ch1 right  | 1.0    | ×   | 0.0 (pos1) | 0                        |
| **Sum**    |        |     |            | **0.5** → ReLU → **0.5** |


**Centered on pos2** → window = (pos1, pos2, pad):


| Piece      | Weight | ×   | Input      | =                        |
| ---------- | ------ | --- | ---------- | ------------------------ |
| ch0 left   | 0.5    | ×   | 1.0 (pos1) | 0.5                      |
| ch0 center | 0.5    | ×   | 0.0 (pos2) | 0                        |
| ch0 right  | 0.0    | ×   | 0 (pad)    | 0                        |
| ch1 left   | 0.0    | ×   | 0.0 (pos1) | 0                        |
| ch1 center | 0.0    | ×   | 1.0 (pos2) | 0                        |
| ch1 right  | 1.0    | ×   | 0 (pad)    | 0                        |
| **Sum**    |        |     |            | **0.5** → ReLU → **0.5** |


Filter 0’s three scores: `[0.5, 2.0, 0.5]` — strongest in the middle where the real `C C O` sits.

---

#### Step 5 — Second filter (same windows, different weights)

Filter 1 weights (also chosen for the demo): “only look at ch0 on the left”

```
ch0: [1.0, 0.0, 0.0]    ch1: [0.0, 0.0, 0.0]
```

Only one product can be nonzero (ch0 × left). Results: `[0.0, 1.0, 1.0]`.

**After Conv1 we still have a table**, not the fingerprint yet:

```
              pos0   pos1   pos2
filter0        0.5    2.0    0.5
filter1        0.0    1.0    1.0
```

**Actual after Conv1:** table `[128 filters × 256 positions]`.

---

#### Step 6 — Conv2 (same idea, new input table)

Conv2 does **exactly the same window × weights × sum**, but its input is the Conv1 table above (not the characters).

Toy filter A weights on filter0’s row only: `[0.25, 0.50, 0.25]`  

At pos1, line up again:


| Piece   | Weight | ×   | Input from filter0 row | =                          |
| ------- | ------ | --- | ---------------------- | -------------------------- |
| left    | 0.25   | ×   | 0.5                    | 0.125                      |
| center  | 0.50   | ×   | 2.0                    | 1.0                        |
| right   | 0.25   | ×   | 0.5                    | 0.125                      |
| **Sum** |        |     |                        | **1.25** → ReLU → **1.25** |


After doing all positions the same way:

```
              pos0    pos1    pos2
filterA       0.625   1.25    0.625
filterB       0.0     1.0     1.0
```

**Actual after Conv2:** `[64 × 256]`.

---

#### Step 7 — Max-pool → fingerprint (output of this block)

For each filter row, keep the **largest** number across positions:

```
fingerprint[0] = max(0.625, 1.25, 0.625) = 1.25
fingerprint[1] = max(0.0,   1.0,  1.0)   = 1.0

Toy result:  [1.25, 1.0]
Actual:      64 such numbers → smiles_feats[drug] shape [64]
```

**Why max?** “Did this chemical pattern appear *anywhere* in the molecule?” — strength of the best hit, independent of string length.

```
Actual pipeline for one drug:
  256 char indices
    → embed          [256, 64]
    → rearrange      [64, 256]
    → conv1 + ReLU   [128, 256]
    → conv2 + ReLU   [64, 256]
    → max over 256   [64]          ← done
```

#### What the encoder output actually is

For **one drug**, the encoder returns **one list of 64 numbers** — a chemical ID card. No SMILES string remains.

```
ethanol  ≈ [0.2, 1.4, 0.0, 0.8, ..., 0.3]    // 64 numbers (example)
aspirin  ≈ [1.1, 0.3, 2.0, 0.1, ..., 0.9]    // different 64 numbers
```

Think of them as **64 pattern meters**: slot *j* ≈ “how strongly did learned detector *j* fire *anywhere* in this molecule?” (strength of best match — not a calibrated probability).  
(In the toy above, 2 detectors → `[1.25, 1.0]`.)

**Where** the pattern matched is **not** kept. Before max-pool, detector *j* still has a score at every character position; max-pool keeps only the largest one, so “fired at index 17” is discarded.

We usually **don’t need location** for this task: the model cares that a chemical motif is present, not whether it sat at the start or end of the SMILES string. SMILES is a linear writing of a molecule (order can vary), so string position is often noisy anyway. Max-pool also forces every drug into the same fixed **64**-dim card regardless of SMILES length.

What you give up: exact index, how many times the motif appeared (max ≠ count), and motif order along the string. The conv layers still used local neighbourhoods to *compute* the scores; only the final card drops “where.”

Later layers never read `CCO`; they only see this card. Similar SMILES tend to get similar cards. Block 2 then glues it to a **drug-ID** card (another 64 numbers: “how this drug behaved in the labels”):

```
SMILES card (structure)  ‖  ID card (dataset behaviour)  →  drug representation
```

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

Neither alone is sufficient. Two structurally similar drugs might behave very differently in combinations. Pure ID embeddings have no generalisation to unseen drugs. Combining both gives molecular + relational signals.

**Why not drop ID and let SMILES learn everything?** Both are updated by the **same loss**, but they are wired differently:

- **SMILES CNN = shared weights** for every drug. Updating it for Warfarin also moves every similar SMILES. So it *tends* to learn structure patterns that transfer across molecules (and that help the interaction labels) — not one drug’s private quirks.
- **ID embedding = one private row per drug.** Easy place for leftover drug-specific behaviour that doesn’t fit cleanly in a shared CNN.

This split is a **tendency from architecture**, not a hard rule the loss assigns. SMILES is not “pure chemistry for its own sake” — only chemistry useful for predicting the labels. Dropping ID does not cleanly move all ID info into SMILES.

**Why the linear after concat?** Concat only glues the two 64-lists side by side; they are not mixed yet. `Linear(128→128)` lets each output depend on **both** cards (weighted sum of all 128 inputs), then ReLU. Training learns how much chemistry vs ID to use.

### The Math

For each drug d:

```
drug_id_emb(d)  ∈ R^64     — learnable ID embedding (random init, trained)
smiles_feats(d) ∈ R^64     — chemical fingerprint from SmilesEncoder

concat_d = [drug_id_emb(d) ‖ smiles_feats(d)]  ∈ R^128
node_feat(d) = ReLU( W_proj · concat_d + b_proj )  ∈ R^128
```

W_proj ∈ R^(128×128) is a linear projection that lets the model learn how to weight the chemical vs relational information.

### Worked Example — Building One Drug Vector (full block)

**Actual sizes:** ID emb = **64**, SMILES = **64**, concat = **128**, Linear(**128→128**), output = **128**.

Toy shrinks each 64 → 2.

**Step A — Look up ID vector** (embedding table, no math)

```
Toy:    [0.8, -0.2]          Actual: 64 numbers
```

**Step B — SMILES fingerprint** (from Block 1)

```
Toy:    [0.1, 0.9]           Actual: 64 numbers
```

**Step C — Concatenate = glue the two lists end to end**

```
Toy:    [0.8, -0.2, 0.1, 0.9]     ← 4 numbers
Actual: 64 + 64 = 128 numbers
```

**Step D — Linear layer: each output = weighted sum of all inputs**

We invent a tiny weight matrix (2 output slots × 4 inputs). **Each weight is just a number the model would learn**; here we pick easy ones.

```
Output slot 0 uses weights [0.5, 0.0, 0.5, 0.0]:
  0.5·0.8 + 0.0·(-0.2) + 0.5·0.1 + 0.0·0.9
= 0.5·0.8 + 0.5·0.1
= 0.40 + 0.05
= 0.45

Output slot 1 uses weights [0.0, 0.5, 0.0, 0.5]:
  0.0·0.8 + 0.5·(-0.2) + 0.0·0.1 + 0.5·0.9
= 0.5·(-0.2) + 0.5·0.9
= -0.10 + 0.45
= 0.35

ReLU([0.45, 0.35]) = [0.45, 0.35]   (both already positive)
```

**Actual:** same idea with a **128×128** weight matrix → output `node_feats[d]` **has 128 numbers**.  
All drugs: shape `[num_drugs, 128]`.

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

**Intuition:** Drug A in prescription `{A, B, C}` and prescription `{A, D, E}`:

- Layer 1: A sees the average of B, C, D, E
- Layer 2: A sees drugs two hops away (e.g., what B co-occurs with in *other* prescriptions)

After 2 layers, each drug's feature vector is informed by the entire neighbourhood of prescriptions it appears in.

### Worked Example — One Hypergraph Conv Layer (full)

**Idea in plain English:**  

1. Each prescription averages its drugs’ numbers.
2. Each drug then averages the prescriptions it belongs to.

That’s one “round of gossip.” Do it twice in the real model.

**Actual sizes:** each drug has **128** numbers; both HypergraphConv layers keep **128**.  
Toy uses **1** number per drug so the arithmetic is visible. With 128 features you repeat the same steps independently on each feature.

Same hypergraph as Section 2:

```
Start values (toy channel 0):
  Warfarin=4, Aspirin=2, Ibuprofen=6, Metformin=8

Prescription e0 = {Warfarin, Aspirin, Ibuprofen}   (3 drugs)
Prescription e1 = {Warfarin, Metformin}            (2 drugs)
```

**① Average inside each prescription**

```
e0 average = (4+2+6)/3 = 12/3 = 4
e1 average = (4+8)/2   = 12/2 = 6
```

(Where do 12 and 3 come from? 12 = sum of member values; 3 = how many members.)

**② Send those averages back to each drug, then divide by how many prescriptions the drug is in**

```
Warfarin is in e0 and e1 → (4 + 6) / 2 = 5
Aspirin  only in e0      → 4 / 1 = 4
Ibuprofen only in e0     → 4 / 1 = 4
Metformin only in e1     → 6 / 1 = 6
```


| Drug      | Before | After | Why                        |
| --------- | ------ | ----- | -------------------------- |
| Warfarin  | 4      | 5     | Blended both prescriptions |
| Aspirin   | 2      | 4     | Became e0’s average        |
| Ibuprofen | 6      | 4     | Became e0’s average        |
| Metformin | 8      | 6     | Became e1’s average        |


**Actual:** same averaging, then a Linear(**128→128**) + ReLU, twice. Output shape still `[num_drugs, 128]`.

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

### Worked Example — Set Attention Pooling (full)

**Idea in one sentence:** give each drug in the combo a percentage of importance, then mix their vectors with those percentages.

**Actual sizes:** drug vectors ∈ ℝ¹²⁸; score MLP **128→64→1**; output combo ∈ ℝ¹²⁸.

Toy: each drug has **2** numbers (stand-in for 128).

```
Warfarin  x_W = [5.0, 1.0]
Aspirin   x_A = [4.0, 0.0]
Ibuprofen x_I = [4.0, 2.0]
```

**Step 1 — One importance score per drug**  
(In the real model a small neural net produces these. For the demo we just pick three scores:)

```
s_W = 2.0    s_A = 1.0    s_I = 0.5
```

Higher score = “this drug matters more for the interaction.”

**Step 2 — Softmax turns scores into percentages** (see Primer)


| Drug      | score s | e^s (≈)   | weight α = e^s / 11.76 |
| --------- | ------- | --------- | ---------------------- |
| Warfarin  | 2.0     | 7.39      | **0.63** (63%)         |
| Aspirin   | 1.0     | 2.72      | **0.23** (23%)         |
| Ibuprofen | 0.5     | 1.65      | **0.14** (14%)         |
| **Total** |         | **11.76** | **1.00**               |


**Step 3 — Mix the vectors with those percentages**

Do this **separately for each slot** of the vector:

```
slot0:  0.63·5.0 + 0.23·4.0 + 0.14·4.0 = 3.15 + 0.92 + 0.56 = 4.63
slot1:  0.63·1.0 + 0.23·0.0 + 0.14·2.0 = 0.63 + 0.00 + 0.28 = 0.91

Toy combo = [4.63, 0.91]
Actual:     same weighted average, but each vector has 128 slots → combo ∈ ℝ¹²⁸
```

**Done.** One vector per prescription. All of them: `[num_hyperedges, 128]`.

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

### Worked Example — Prediction Head (full: concat → MLP → prob → loss)

**Actual sizes:** `combo_vector ∈ ℝ¹²⁸`, `se_emb ∈ ℝ¹²⁸`, concat **256**, MLP Linear(**256→128**) → ReLU → Dropout(0.3) → Linear(**128→1**) logit.

Continue from the previous example’s combo `[4.63, 0.91]` (toy stand-in for ℝ¹²⁸).

```
Step 1 — Side-effect embedding
  Toy:    se_emb(GI bleed) = [0.50, 1.20]           shape [2]
  Actual: se_emb(s) ∈ ℝ¹²⁸                         shape [128]

Step 2 — Concatenate
  Toy:    input = [4.63, 0.91, 0.50, 1.20]          shape [4]
  Actual: input ∈ ℝ²⁵⁶                             shape [256]

Step 3 — MLP → single logit
  Toy:    suppose MLP ends at logit z = 1.5         shape [1]
  Actual: logit = MLP(input) ∈ ℝ                   shape scalar per sample
          (full batch: [num_hyperedges])

Step 4 — Probability (inference; training uses BCEWithLogits on raw logit)
  σ(1.5) = 1/(1+e^{-1.5}) ≈ 0.818

Step 5 — Loss
  If y=1:  L = -log(0.818) ≈ 0.201     (good)
  If y=0:  L = -log(0.182) ≈ 1.70      (bad — confident false positive)
  If y=1 but z=-2.0 (p≈0.119): L ≈ 2.13  (bad — missed interaction)
```

**Done for the forward pass.** Output of the model is the logit; after sigmoid you get a probability in (0,1). Training calls `loss.backward()` and updates all weights (SMILES CNN, ID emb, HypergraphConv, attn MLP, SE emb, final MLP).

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


| Column            | Content                                                       |
| ----------------- | ------------------------------------------------------------- |
| `DrugBankID`      | List of DrugBank IDs e.g. `['DB00001', 'DB00002', 'DB00003']` |
| `SE_label`        | UMLS CUI for the side effect e.g. `C0018681`                  |
| `hyperedge_label` | `1` = reported combination, `-1` = negative sample            |
| `time`            | Quarter e.g. `2019Q2`                                         |


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


| Metric        | Formula                           | What it measures                                                                                              |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **ROC-AUC**   | Area under ROC curve              | Ability to rank positives above negatives across all thresholds. 1.0 = perfect, 0.5 = random. Primary metric. |
| **PR-AUC**    | Area under Precision-Recall curve | Performance on the positive class specifically. More informative than ROC-AUC when classes are imbalanced.    |
| **F1-Score**  | 2·(P·R)/(P+R)                     | Harmonic mean of precision and recall at threshold 0.5.                                                       |
| **Accuracy**  | (TP+TN)/(TP+TN+FP+FN)             | Overall correct predictions.                                                                                  |
| **Precision** | TP/(TP+FP)                        | Of all predicted toxic combinations, how many actually are?                                                   |
| **Recall**    | TP/(TP+FN)                        | Of all actually toxic combinations, how many did we catch?                                                    |


In a safety-critical medical context, **recall is especially important** — a missed toxic combination (false negative) is more dangerous than a false alarm (false positive).

### Worked Example — Tiny Confusion Matrix

Suppose the test set has 10 predictions (threshold 0.5):


| Sample | True y | Prob | Pred | Result                  |
| ------ | ------ | ---- | ---- | ----------------------- |
| 1      | 1      | 0.90 | 1    | TP                      |
| 2      | 1      | 0.70 | 1    | TP                      |
| 3      | 1      | 0.40 | 0    | FN ← missed toxic combo |
| 4      | 0      | 0.20 | 0    | TN                      |
| 5      | 0      | 0.10 | 0    | TN                      |
| 6      | 0      | 0.60 | 1    | FP ← false alarm        |
| 7      | 1      | 0.85 | 1    | TP                      |
| 8      | 0      | 0.05 | 0    | TN                      |
| 9      | 1      | 0.55 | 1    | TP                      |
| 10     | 0      | 0.30 | 0    | TN                      |


```
TP=4  FP=1  TN=4  FN=1

Precision = 4/(4+1) = 0.80
Recall    = 4/(4+1) = 0.80
F1        = 2·0.80·0.80 / (0.80+0.80) = 0.80
Accuracy  = (4+4)/10 = 0.80
```

ROC-AUC / PR-AUC use the *raw probabilities* across all thresholds, not just 0.5 — so sample 3 (prob 0.40) still contributes as a ranking error even if we later lower the threshold to catch it.

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


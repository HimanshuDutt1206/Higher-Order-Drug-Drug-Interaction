# HODDI: How the Dataset Was Built

### A Complete Walkthrough from FAERS → Labels → Where This Project Continues

> **How to read this doc:** Same style as `explanationHGNN.md` and `explanationPolyformer.md`. Assumes ML *ideas*, not formulas. Worked examples label **where every number / decision comes from**. This file is about the **dataset paper**, not our model math.
>
> **Companion docs:** After you finish this file, the modeling story continues in:
>
> - `explanationHGNN.md` — hypergraph model we reimplemented / adapted
> - `explanationPolyformer.md` — inductive set-attention model we added
> - `explanationNegativeSampling.md` — schemes A–D for synthetic negatives (paper method + stress tests)



### Paper cheat-sheet (what HODDI claims)


| Piece                       | Paper number / choice                                                         |
| --------------------------- | ----------------------------------------------------------------------------- |
| Source                      | FDA FAERS quarterly XML                                                       |
| Time span                   | **2014Q3 → 2024Q3** (**41** quarters)                                         |
| Final merged records        | **109,744** (balanced pos + neg)                                              |
| Unique drugs (positives)    | **2,506**                                                                     |
| Unique side effects (CUIs+) | **3,950** (paper abstract also says 4,569 overall CUIs)                       |
| Positive SE confidence      | SapBERT ↔ MedDRA cosine **≥ 0.9**                                             |
| Label values                | `hyperedge_label = 1` (pos), `-1` (neg)                                       |
| Negative recipe             | Replace **1 drug** + **1 SE** from a positive; reject if key already positive |
| Paper eval subset (main)    | **2–8** drugs / record, SE frequency **5–50**                                 |
| Paper train/val/test        | Random **29 / 6 / 6** quarters (~70/15/15)                                    |




### Primer — only ideas used in this file

1. **FAERS report** = one adverse-event complaint filed to the FDA. It lists drugs the patient was on and side effects reported.
2. **Drug role** in FAERS:
  - Role **1** = primary suspect
  - Role **2** = secondary suspect
  - Role **3** = concomitant (taken at the same time)
3. **UMLS CUI** = a stable ID for a medical concept (e.g. a side-effect name mapped to one code like `C0010692`).
4. **Cosine similarity** = how aligned two embedding vectors are (1.0 = same direction). Here it measures “how close is this messy FAERS side-effect string to a clean MedDRA term?”
5. **Closed-world negative** = “we never saw this (combo, SE) reported, so treat it as negative for training.” That is an **assumption**, not a lab-proven “safe” label.
6. **Hyperedge** = one multi-drug prescription treated as a single multi-way link (see HGNN doc §2).

---



## Table of Contents

1. [The Problem the Paper Solves](#1-the-problem-the-paper-solves)
2. [Why Older Datasets Were Not Enough](#2-why-older-datasets-were-not-enough)
3. [Source Data: FAERS Quarters](#3-source-data-faers-quarters)
4. [Step 1 — Extract Key Fields from Each Report](#4-step-1--extract-key-fields-from-each-report)
5. [Step 2 — Conditional Filtering by Drug Roles](#5-step-2--conditional-filtering-by-drug-roles)
6. [Step 3 — Side Effect Standardization (SapBERT → MedDRA → CUI)](#6-step-3--side-effect-standardization-sapbert--meddra--cui)
7. [Step 4 — Drug Name Standardization (→ DrugBank IDs)](#7-step-4--drug-name-standardization--drugbank-ids)
8. [Step 5 — Positive Sample Construction](#8-step-5--positive-sample-construction)
9. [Step 6 — Negative Sample Construction](#9-step-6--negative-sample-construction)
10. [What the Finished Dataset Looks Like](#10-what-the-finished-dataset-looks-like)
11. [Evaluation Subsets the Paper Uses](#11-evaluation-subsets-the-paper-uses)
12. [What the Paper Benchmarked (Models + Results)](#12-what-the-paper-benchmarked-models--results)
13. [Important Limitations (Especially Negatives)](#13-important-limitations-especially-negatives)
14. [Where Our Capstone Continues](#14-where-our-capstone-continues)
15. [End-to-End Pipeline Summary](#15-end-to-end-pipeline-summary)

---



## 1. The Problem the Paper Solves

**Polypharmacy** = a patient takes several drugs at once. Interactions are not only “drug A vs drug B.” Real prescriptions are groups of 3, 4, 8+ drugs, and the adverse effect may depend on that **whole set**.

Older ML resources mostly answer:

- single drug → side effect, or
- **pair** of drugs → side effect

The HODDI paper’s claim:

> We still lack a large, standardized **higher-order** dataset: multi-drug combinations linked to specific side effects, built from real pharmacovigilance reports.

So they build **HODDI** (*Higher-Order Drug-Drug Interaction Dataset*) from FAERS and release it as a benchmark for computational pharmacovigilance.

**Prediction task implied by the labels** (same task our project uses):

> Given a **set of drugs** + one **side effect CUI**, is this a reported adverse association (**1**) or a constructed non-association (**-1**)?

---



## 2. Why Older Datasets Were Not Enough


| Dataset              | Typical scope                          | Gap for high-order polypharmacy                                       |
| -------------------- | -------------------------------------- | --------------------------------------------------------------------- |
| SIDER                | Single-drug effects from labels        | No multi-drug combos                                                  |
| OFFSIDES             | Single-drug / trial-oriented signals   | Still not true n-way combos                                           |
| TWOSIDES             | **Pairwise** drug interactions (FAERS) | Stops at 2 drugs                                                      |
| Decagon-style graphs | Often pairwise + PPI networks          | Pair edges cannot natively store an 8-drug prescription as one object |


Paper motivation in one sentence: **possible combinations grow exponentially; available labeled multi-drug data does not.** HODDI tries to fill that hole with FAERS multi-drug reports.

---



## 3. Source Data: FAERS Quarters

**FAERS** = FDA Adverse Event Reporting System. Anyone (clinicians, patients, manufacturers) can submit that “these drugs were involved when this adverse event happened.”

HODDI authors:

1. Download **quarterly** FAERS dumps from **2014Q3 through 2024Q3** → **41** quarters.
2. Each quarter is a zip of **XML** files.
3. Parse XML → pull the fields they need (next section).

```
FAERS zip (one quarter)
   └── many XML reports
         └── report_id, drugs(+roles), side-effect names, ...
```

This is **observational reporting data**, not a randomized trial. Reporting bias, incomplete drug lists, and confounding exist. The paper still uses it because it is one of the only large real-world sources of multi-drug adverse events.

---



## 4. Step 1 — Extract Key Fields from Each Report

From each FAERS record they keep roughly:

- **Report ID**
- **Side effect name(s)** (free / MedDRA-ish text as filed)
- **Drug names** + **drug roles** (1 / 2 / 3)



### Worked Example — Table 1 style from the paper

Report `24135951` (simplified):


| Field        | Value                                                        |
| ------------ | ------------------------------------------------------------ |
| Side effects | Psoriasis; Drug interaction                                  |
| Drugs        | Hydroxychloroquine (role 3), Prednisone (3), Terbinafine (3) |


Meaning: three **concomitant** meds were on board during the event. That is exactly the kind of multi-drug situation HODDI wants.

At this stage names are still messy strings. Standardization comes later.

---



## 5. Step 2 — Conditional Filtering by Drug Roles

Not every FAERS report is useful for “multi-drug interaction.” They keep a report only if it matches **one** of three role patterns:

### Condition 1 — Only concomitants (no primary/secondary suspects)

- Role 3 count ≥ 2
- Role 1 = 0
- Role 2 = 0



### Condition 2 — Primary suspect + concomitant (no secondary)

- Role 1 ≥ 1
- Role 3 ≥ 1
- Role 2 = 0



### Condition 3 — All three roles present

- Role 1 ≥ 1
- Role 2 ≥ 1
- Role 3 ≥ 1

```
Keep report?
  ├─ Cond 1: ≥2 concomitant, no suspects
  ├─ Cond 2: ≥1 primary + ≥1 concomitant, no secondary
  └─ Cond 3: primary + secondary + concomitant all present
```

**Why:** these patterns are their operational definition of “potential suspect multi-drug situations,” including cases where concomitant meds may matter.

**Temporal note from their analysis:** before late 2021, Conditions 2 and 3 dominate; from **2021Q4** onward Condition 1 becomes much more common. Reporting patterns drift over time — one reason chronological evaluation matters later.

---



## 6. Step 3 — Side Effect Standardization (SapBERT → MedDRA → CUI)

FAERS side-effect text is noisy (“same idea, different wording”). HODDI normalizes it:

1. Embed the FAERS side-effect **string** with **SapBERT** → 768-dim vector.
2. Embed standardized **MedDRA** terms the same way.
3. For each FAERS string, pick the MedDRA term with **highest cosine similarity**.
4. Map that MedDRA term → **UMLS CUI** via UMLS Metathesaurus.
5. Stratify confidence:
  - cosine **≥ 0.9** → high confidence (used for **positive** construction)
  - **0.8–0.9** and **< 0.8** tracked for analysis



### Worked Example — confidence gate

Suppose FAERS text embeds closest to MedDRA term *Gastrointestinal haemorrhage* with cosine **0.93**.

- Recommended CUI might be something like `C0017181` (illustrative).
- Because **0.93 ≥ 0.9**, this side effect is eligible for **positive** samples.

If cosine were **0.72**, the paper would **not** treat it as a high-confidence positive label.

In our project files, that CUI shows up as `SE_label` (or paper column name `SE_above_0.9` in their eval tables). You also have `data/SE_similarity_2014Q3_2024Q3.csv`, which is the kind of recommended-name / similarity auxiliary the paper’s SE pipeline produces.

**Paper model features:** they often use the **768-dim SapBERT vector of the recommended MedDRA term** as the side-effect feature.  
**Our project models:** use a **learnable SE embedding** indexed by CUI instead (see HGNN / PolyFormer docs). Same IDs, different feature source.

---



## 7. Step 4 — Drug Name Standardization (→ DrugBank IDs)

Drug strings are normalized then mapped into **DrugBank**:

1. Uppercase
2. Strip salt suffixes (e.g. hydrochloride / HCl)
3. Handle compound names
4. Lookup in DrugBank full database (synonyms / alternative names) → **DrugBank ID** like `DB00682`



### Worked Example

```
"WARFARIN SODIUM"  → normalize → map → DB00682
"aspirin"           → normalize → map → DB00945
```

Our project then joins those IDs to SMILES via:

- `data/Drugbank_ID_SMILE_all_structure links.csv` (primary in training scripts)
- `data/DrugBankID2SMILES.csv` (extra reference)

Missing SMILES are handled at train time with a fallback (e.g. `"C"`) in our code — the paper instead **excluded** drugs with missing SMILES in their SMILES2Vec feature setup.

---



## 8. Step 5 — Positive Sample Construction

**Positive** = a real FAERS-derived multi-drug record whose side effect cleared the **≥ 0.9** cosine gate.

Construction rules:

1. Keep high-confidence SE mappings only.
2. **Remove duplicate** records.
3. **Remove supersets** (very important):



### Superset removal (parsimony)

If a larger combo’s drugs **and** side effect already fully cover a smaller recorded association, drop the larger redundant one.

**Worked Example**

- Record X: `{A, B}` → SE₁ (keep — minimal)
- Record Y: `{A, B, C}` → SE₁ and Y’s drugs+SE completely overlap X’s association → **drop Y** as redundant higher-order duplicate of the same signal

Paper’s stated goal: keep the **most parsimonious** (minimal complete) representation of each association, reduce noise/redundancy, improve efficiency.

After this, positives are real reported associations in standardized form:

```
(report_id, SE_CUI, [DB..., DB...], hyperedge_label=1, time=YYYYQn)
```

---



## 9. Step 6 — Negative Sample Construction

FAERS does **not** reliably say “this combo is safe.” So HODDI **synthesizes** negatives.

### Paper recipe (exact idea)

For each positive sample:

1. Randomly replace **one drug** with another drug from the complement set (not creating an already-known positive key).
2. Randomly replace **the side effect** with another SE.
3. Accept only if the new `(drug set, SE)` does **not** appear among positives.
4. Build **equal counts** of positives and negatives (class balance).

```
POSITIVE:  drugs={A,B,C}   SE=S*
                │
                ├─ replace one drug  → {A,B,D}
                └─ replace SE        → S'
                │
                ▼
NEGATIVE:  drugs={A,B,D}   SE=S'    label=-1
           (only if ( {A,B,D}, S' ) not in positives)
```

They then write **41 quarterly pairs** of files (`Q_pos.csv`, `Q_neg.csv`) plus merged `pos.csv` / `neg.csv`.

### What this teaches a model

Roughly: “this **exact** corrupted (combo, SE) pair is not a known reported association.”

It does **not** prove pharmacological safety. See §13 for the subset false-negative issue you already spotted.

### Link to our earlier discussion

Your current mental model (“swap one drug + swap SE, ensure not already positive”) **is the HODDI paper’s official negative method** — not a random project hack. Our later design choice is whether to **keep**, **stress-test**, or **replace** that recipe.

---



## 10. What the Finished Dataset Looks Like



### Scale (2014Q3–2024Q3, paper Table 2)


| Quantity                  | Value   |
| ------------------------- | ------- |
| Total records (pos+neg)   | 109,744 |
| Unique drugs in positives | 2,506   |
| Unique drugs in negatives | 12,293  |
| Unique CUIs in positives  | 3,950   |
| Unique CUIs in negatives  | 4,581   |


Negatives have **more unique drugs/CUIs** because random replacement pulls in entities outside the positive-only vocabulary.

### Column schema (paper Table 8 / our `hoddi_merged.csv`)


| Column                      | Meaning                                                                      |
| --------------------------- | ---------------------------------------------------------------------------- |
| `report_id`                 | FAERS report id (negatives often tagged specially, e.g. suffix `n` in paper) |
| `SE_label` / `SE_above_0.9` | UMLS CUI for the side effect                                                 |
| `DrugBankID`                | list of DrugBank IDs in the combo                                            |
| `hyperedge_label`           | `1` positive, `-1` negative                                                  |
| `time`                      | quarter, e.g. `2014Q3`                                                       |


Our merged file also has `combo_str` (sorted IDs joined) for convenience.

### Distribution intuition (paper Figure 1 / appendix)

- Most records have about **2–10** drugs (long tail up to very large polypharmacy).
- Positive SE frequencies are **long-tailed** (many rare SEs).
- Negative SE frequencies look more **concentrated / normal-like** because SE resampling reshapes the SE histogram — a fingerprint of the negative generator.

---



## 11. Evaluation Subsets the Paper Uses

Full HODDI is large and heavy-tailed. For benchmarks they filter to denser, more “typical” higher-order rows:


| Subset        | Drugs / record | SE frequency | # sample pairs (paper) |
| ------------- | -------------- | ------------ | ---------------------- |
| Eval 1 (main) | 2–8            | 5–50         | 21,503                 |
| Eval 2        | 2–16           | 5–50         | 24,777                 |
| Eval 3        | 2–16           | 5–100        | 38,430                 |


They mainly report models on **Eval Subset 1**.

They also show a **clique-expansion** conversion (hyperedge → all pairwise edges) for classical GNNs — which **throws away** explicit higher-order structure. That contrast is why hypergraphs matter in their story.

---



## 12. What the Paper Benchmarked (Models + Results)



### Features in the paper

- Drugs: pretrained **SMILES2Vec** → 768-d (drop drugs with missing SMILES)
- Side effects: **SapBERT** MedDRA vectors → 768-d



### Architectures

1. **MLP** — mean/aggregated drug vector ‖ SE vector
2. **GCN / GAT** — pairwise heterogeneous graphs after conversion
3. **HyGNN** — hypergraph baseline
4. **HGNN-SA** — their hypergraph + SMILES + attention design (name we reuse)



### Split in the paper

Randomly choose quarters: **29 train / 6 val / 6 test** (~70/15/15).  
Note: that is **not** the same as “earliest 70% of time, then next 15%, then latest 15%.” Random quarter assignment can mix future and past more than a strict timeline split.

### Headline results (Eval Subset 1, paper Table 7)


| Model   | Precision | F1    | AUC   | PRAUC |
| ------- | --------- | ----- | ----- | ----- |
| HGNN-SA | 0.906     | 0.933 | 0.957 | 0.939 |
| HyGNN   | 0.903     | 0.932 | 0.954 | 0.935 |
| MLP     | 0.805     | 0.819 | 0.897 | 0.872 |
| GAT     | 0.743     | 0.809 | 0.851 | 0.789 |
| GCN     | 0.745     | 0.778 | 0.829 | 0.805 |


Paper takeaway:

1. Even **MLP** beats some graph models → higher-order *features/labels* are already valuable.
2. **Hypergraph** models do best → structure that matches multi-drug hyperedges helps.

---



## 13. Important Limitations (Especially Negatives)

These are the landmines for a viva / Review 2 discussion.

### 1) Negatives are synthetic, not proven safe

The paper itself says negatives are **artificially constructed** under the assumption that a random drug+SE corruption is unlikely to be a real association. That is a **closed-world training proxy**, not a clinical “safe combo” certificate.

### 2) Subset sufficiency can create false negatives

Your example:

- True mechanism: `{A,B}` causes SE  
- Observed positive: `{A,B,C}` + SE  
- Corrupt `C→D`, maybe keep or change SE depending on scheme  
- New row labeled negative even though `{A,B}` still present → **label may be wrong**

HODDI’s **superset removal on positives** reduces some redundancy, but it does **not** fully police synthetic negatives against “contains a known positive subset.”

### 3) Easy corruptions can inflate metrics

Replacing **both** drug and SE (scheme **A**, the paper / `hoddi_merged.csv` method) often makes negatives distributionally obvious (see the different SE frequency shapes for pos vs neg). High AUC partly measures “can we detect the corruption pattern?” not only “do we understand pharmacology?”

### 4) What we added for Review 2 (schemes B/C/D)

We **keep scheme A** as the primary baseline and regenerate alternate negatives from positives only:

- **B** — drug-corrupt only (same SE; harder membership test; subset risk exposed)
- **C** — SE-corrupt only (same combo; SE specificity test)
- **D** — drug-corrupt **plus subset-aware rejection** (blocks candidates that still contain a known positive subset for that SE)

PolyFormer is retrained on each regenerated table without overwriting the original A checkpoint/report. **Metric drops on harder schemes are expected** and should be read as evidence about **label difficulty**, not as proof the architecture “failed.”

Full algorithms, worked examples, and commands: [`explanations/explanationNegativeSampling.md`](explanationNegativeSampling.md).  
Comparison table: [`results/synthetic_negative_comparison.md`](../results/synthetic_negative_comparison.md).

### 5) FAERS confounding

Reported together ≠ causal. Indication bias, reporting bias, and incomplete regimens remain.

### 6) Feature / split choices affect comparability

Paper numbers (AUC ~0.96 on Eval 1 with SapBERT/SMILES2Vec features and random quarter picks) are **not automatically comparable** to our full-merge chronological runs with learnable embeddings.

---



## 14. Where Our Capstone Continues

This is the handoff line: **HODDI paper ends at dataset + their benchmarks; PolySignal / this repo starts here.**

### What we inherit from HODDI (unchanged starting point)


| Inherited piece                                | In our repo                               |
| ---------------------------------------------- | ----------------------------------------- |
| FAERS-derived multi-drug + SE associations     | `data/hoddi_merged.csv`                   |
| Labels `1` / `-1`                              | mapped to `target` / `y` in train scripts |
| DrugBank ID lists + quarter `time`             | used for graph/set construction + split   |
| Official negative philosophy (drug+SE corrupt) | already baked into merged labels          |
| SE similarity auxiliary                        | `data/SE_similarity_2014Q3_2024Q3.csv`    |
| Unique SE reference                            | `data/Side_effects_unique.csv`            |
| SMILES resources                               | DrugBank SMILES CSVs                      |


We did **not** re-download and re-parse 41 quarters of FAERS XML for the core label set; we consume the HODDI-style merged table and build models on top.

### What we change / add beyond the paper


| Topic           | HODDI paper                                   | Our project                                                                                                         |
| --------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Main data slice | Often Eval Subset 1 (2–8 drugs, SE freq 5–50) | Primarily **full merged** `hoddi_merged.csv`                                                                        |
| Time split      | Random 29/6/6 quarters                        | **Strict chronological** earliest 70% → next 15% → latest 15% (`2014Q3→2021Q2` / `2021Q3→2022Q4` / `2023Q1→2024Q3`) |
| Drug features   | Pretrained SMILES2Vec 768-d                   | **Learnable char-CNN** on SMILES (+ drug ID emb)                                                                    |
| SE features     | SapBERT 768-d                                 | **Learnable SE embedding** by CUI                                                                                   |
| Models          | MLP, GCN, GAT, HyGNN, HGNN-SA                 | **HGNN-SA** (adapted) + **PolyFormer** (our inductive baseline/SOTA in-repo)                                        |
| Inference story | Benchmark tables                              | + **Streamlit app** (`app.py`) for interactive combo+SE risk                                                        |
| Docs            | Paper                                         | Deep walkthroughs: this file + `explanationHGNN.md` + `explanationPolyformer.md`                                    |




### How our training scripts attach

```
hoddi_merged.csv
      │
      ├─ HGNN_train.py
      │     hyperedge_index from each row’s drugs
      │     chronological split
      │     → models/HGNN_model.pt , results/evaluate_HGNN.txt
      │
      └─ polyformer_train.py
            padded drug sets (MAX_DRUGS=16), drop 1-drug rows
            chronological split (same idea)
            → models/polyformer_model.pt , results/Evaluate_PolyFormer.txt
```

Detail of architectures: **stop reading this file** and open the matching model doc.

### Explicit “we continue from here” boundary

```
[HODDI paper territory]
FAERS → filter roles → standardize SE/drugs → positives
      → synthetic negatives (drug+SE swap) → quarterly/merged CSVs
      → paper eval subsets + paper model table
                │
                │  <<< handoff >>>
                ▼
[Our capstone territory]
Load hoddi_merged.csv
Strict future-quarter split
Re-feature drugs/SEs our way
Train HGNN-SA + PolyFormer
Ship app + explanation docs
( next decisions: negative-sampling experiments vs other additions )
```

Everything **before** the handoff is dataset construction explained above.  
Everything **after** is engineering and modeling we own — including any future change to negatives, hybrids, ablations, or explainability.

---



## 15. End-to-End Pipeline Summary

```
FAERS XML (2014Q3 … 2024Q3)
        │
        ▼ extract report_id, drugs+roles, SE text
        │
        ▼ keep Cond 1 / 2 / 3 multi-drug patterns
        │
        ▼ SapBERT align SE text → MedDRA → UMLS CUI
        │     (positives require cosine ≥ 0.9)
        │
        ▼ normalize drug names → DrugBank IDs
        │
        ▼ POSITIVES: dedupe + remove supersets
        │
        ▼ NEGATIVES: replace 1 drug + 1 SE;
        │            reject if (combo, SE) already positive;
        │            match positive count
        │
        ▼ quarterly pos/neg files + merged table
        │
        ▼ paper: filter eval subsets, benchmark MLP/GNN/HGNN
        │
═══════════════ our project starts ═══════════════
        │
        ▼ hoddi_merged.csv + SMILES tables
        │
        ▼ chronological 70/15/15 quarter split
        │
        ▼ HGNN-SA and/or PolyFormer training
        │
        ▼ metrics, checkpoints, Streamlit demo
```



### One-line memory hook

**HODDI turns messy FAERS multi-drug reports into balanced (combo, SE) classification data by standardizing entities and synthesizing negatives; our project takes that table and asks which architecture best predicts future-quarter risk.**

---



## Citation

Wang Z, Shi Y, Liu X, Chen C, Wen J, Wang R.  
**HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance.**  
[https://arxiv.org/pdf/2502.06274](https://arxiv.org/pdf/2502.06274)

Local copy in this repo: `hoddi paper.pdf`

---


# PolySignal: High-Order Polypharmacy Risk Prediction

PolySignal predicts whether a **combination of multiple drugs** is likely to cause a **specific adverse side effect**, using the [**HODDI**](https://arxiv.org/pdf/2502.06274) dataset (FAERS, 2014Q3–2024Q3).

Two models are implemented and compared under the **same chronological split**:

| Model | Idea |
| ----- | ---- |
| **HGNN-SA** | Hypergraph neural network + SMILES CNN + attention pooling |
| **PolyFormer** | Inductive set self-attention + side-effect cross-attention |

A **Streamlit app** (`app.py`) runs **PolyFormer** (scheme **C** checkpoint) on user-selected drugs and side effects.

---

## Problem

> Given a list of drugs (DrugBank IDs) and a target side effect (UMLS CUI), predict whether that combination will cause that side effect.

This is **high-order** polypharmacy: the input is a full prescription set, not a single drug or a pair.

---

## Dataset

**Main file:** `data/hoddi_merged.csv`

| Column | Meaning |
| ------ | ------- |
| `DrugBankID` | List of DrugBank IDs in the prescription |
| `SE_label` | UMLS CUI for the side effect |
| `hyperedge_label` | `1` = positive, `-1` = negative |
| `time` | Reporting quarter |

Labels are mapped to binary targets: `1 → positive`, `-1 → negative`.

Supporting files: DrugBank→SMILES mappings, unique side-effect names (`Side_effects_unique.csv`), and optional SE similarity (`SE_similarity_2014Q3_2024Q3.csv`).  
How HODDI was built: [`explanations/explanationHODDI.md`](explanations/explanationHODDI.md).

---

## Data split

**Strict chronological split by quarter** (not random):

| Split | Quarters | Range |
| ----- | -------- | ----- |
| Train | 70% | 2014Q3 → 2021Q2 |
| Validation | 15% | 2021Q3 → 2022Q4 |
| Test | 15% | 2023Q1 → 2024Q3 |

Test performance is on **future** quarters only (no leakage from later periods into training).

---

## Models

### HGNN-SA

- **Nodes** = drugs; **hyperedges** = prescription records.
- Drug features: learnable ID embedding + **SMILES CNN**.
- Two **HypergraphConv** layers, attention pooling per hyperedge, concat with learnable **SE embedding**, MLP → risk logit.
- Training and evaluation use a full hypergraph; the Streamlit app runs **PolyFormer only** (see `app.py`).

Details: [`explanations/explanationHGNN.md`](explanations/explanationHGNN.md).

### PolyFormer

- Drug set (padded, max 16 drugs): same ID + SMILES features.
- **Transformer encoder** over the set; **cross-attention** from a learnable SE embedding; MLP → logit.
- **Inductive**: predicts from an arbitrary drug set without building a full inference hypergraph.

Details: [`explanations/explanationPolyformer.md`](explanations/explanationPolyformer.md).

---

## Setup

From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric may need an install step matched to your CUDA/CPU build; see [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Train checkpoints are **not** always committed (large `.pt` files). After training, expect:

- `models/HGNN_model.pt`
- `models/polyformer_model.pt`

The repo may include PolyFormer weights for synthetic-negative runs: `models/polyformer_synth_{B,C,D}.pt`.

---

## Usage

Run all commands from the **project root** so `data/`, `models/`, and `results/` resolve correctly.

```bash
# Train scheme A (official HODDI in hoddi_merged.csv)
python src/HGNN_train.py
python src/polyformer_train.py

# Interactive demo (PolyFormer)
streamlit run app.py
```

---

## Repository structure

```text
Higher-Order-Drug-Drug-Interaction/
├── app.py                              # Streamlit: PolyFormer demo
├── README.md
├── requirements.txt
├── hoddi paper.pdf                     # HODDI reference
│
├── data/
│   ├── hoddi_merged.csv                # Scheme A (official negatives)
│   ├── hoddi_synth_B.csv               # Synthetic negatives (schemes B–D)
│   ├── hoddi_synth_C.csv
│   ├── hoddi_synth_D.csv
│   ├── Drugbank_ID_SMILE_all_structure links.csv
│   ├── DrugBankID2SMILES.csv
│   ├── Side_effects_unique.csv
│   └── SE_similarity_2014Q3_2024Q3.csv
│
├── src/
│   ├── HGNN_train.py
│   ├── polyformer_train.py             # CLI: --data, --out_model, --out_report, --tag
│   ├── generate_synthetic_negatives.py # Build B/C/D from positives
│   └── build_synth_comparison.py       # Aggregate A/B/C/D metrics
│
├── models/                             # Checkpoints (train or use committed synth_*)
├── results/                            # Evaluation reports and comparison tables
│
└── explanations/
    ├── explanationHODDI.md
    ├── explanationHGNN.md
    ├── explanationPolyformer.md
    ├── explanationNegativeSampling.md
    └── Polypharmacy Technical Review.pdf
```

---

## Results (scheme A — official HODDI negatives)

Same future test set for both models. Metrics from committed report files.

### HGNN-SA — `results/evaluate_HGNN.txt`

| Metric | Value |
| ------ | ----- |
| ROC-AUC | 0.8903 |
| PR-AUC | 0.8990 |
| F1 | 0.8054 |
| Accuracy | 0.8189 |
| Precision | 0.8698 |
| Recall | 0.7499 |

### PolyFormer — `results/Evaluate_PolyFormer.txt`

| Metric | Value |
| ------ | ----- |
| ROC-AUC | 0.9678 |
| PR-AUC | 0.9672 |
| F1 | 0.8940 |
| Accuracy | 0.8988 |
| Precision | 0.9384 |
| Recall | 0.8536 |

On scheme A, PolyFormer leads on every reported metric. HGNN-SA remains a strong hypergraph baseline; the gap is largest on ranking metrics (ROC-AUC / PR-AUC).

---

## Synthetic negative robustness

Official HODDI negatives (**scheme A**: replace one drug **and** one SE) live in `hoddi_merged.csv`. Schemes **B/C/D** are regenerated from **positives only**; PolyFormer is retrained per scheme without overwriting the scheme A checkpoint.

| Scheme | Idea | Data file |
| ------ | ---- | --------- |
| A | HODDI official (drug + SE corrupt) | `data/hoddi_merged.csv` |
| B | Replace one drug; keep SE | `data/hoddi_synth_B.csv` |
| C | Keep combo; replace SE | `data/hoddi_synth_C.csv` |
| D | Like B; reject if combo still contains a known positive subset for that SE | `data/hoddi_synth_D.csv` |

```bash
python src/generate_synthetic_negatives.py

python src/polyformer_train.py --data data/hoddi_synth_B.csv --out_model models/polyformer_synth_B.pt --out_report results/Evaluate_PolyFormer_synth_B.txt --tag synth_B_drug_corrupt
python src/polyformer_train.py --data data/hoddi_synth_C.csv --out_model models/polyformer_synth_C.pt --out_report results/Evaluate_PolyFormer_synth_C.txt --tag synth_C_se_corrupt
python src/polyformer_train.py --data data/hoddi_synth_D.csv --out_model models/polyformer_synth_D.pt --out_report results/Evaluate_PolyFormer_synth_D.txt --tag synth_D_drug_corrupt_subset_aware

python src/build_synth_comparison.py
```

- Schemes walkthrough: [`explanations/explanationNegativeSampling.md`](explanations/explanationNegativeSampling.md)
- Results and interpretation: [`results/synthetic_negative_comparison.md`](results/synthetic_negative_comparison.md)
- Generation stats: `results/synthetic_negative_generation_stats.csv`
- Written review: [`explanations/Polypharmacy Technical Review.pdf`](explanations/Polypharmacy%20Technical%20Review.pdf)

**PolyFormer test ROC-AUC by scheme:**

| A (official) | B (drug-corrupt) | C (SE-corrupt) | D (subset-aware) |
| ------------ | ---------------- | -------------- | ---------------- |
| 0.9678 | 0.7802 | 0.8905 | 0.7923 |

**Limitation:** synthetic negatives are closed-world training proxies, not proven-safe combinations. Lower AUC on harder schemes (B/D) is expected and shows that very high scores on scheme A partly reflect **easier** negative construction—not a solved clinical deployment problem.

---

## Summary

- **PolyFormer** (inductive set-attention) beats **HGNN-SA** (hypergraph) on the shared chronological test split for official HODDI labels.
- **Negative-sampling stress tests** show a large drop when only drugs are perturbed (B/D), which is the harder and more realistic membership question; scheme A metrics should be read with that context.
- **Docs + demo** tie the codebase to dataset construction, model design, and review-ready analysis.

---

## Citation

If you use or reference HODDI, please cite:

**Wang Z, Shi Y, Liu X, Chen C, Wen J, Wang R.** HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance.  
https://arxiv.org/pdf/2502.06274

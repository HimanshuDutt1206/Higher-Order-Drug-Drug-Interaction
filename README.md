# PolySignal: High-Order Polypharmacy Risk Prediction

PolySignal is a machine learning project for predicting whether a **combination of multiple drugs** is likely to cause a **specific adverse side effect**, using the **HODDI dataset**.

We implemented and compared two models:

- **HGNN-SA**: a hypergraph neural network with SMILES-based drug features
- **PolyFormer**: an inductive set-attention model for drug combinations

---

## Project Overview

The goal of this project is to solve the following problem:

> Given a list of drugs and a target side effect, predict whether that drug combination will cause that side effect.

This is a **high-order polypharmacy prediction task**, because the input is not just a single drug or a pair of drugs, but a group of drugs taken together.

We built two different models to solve this problem and compared them under the same data split.

---

## Dataset

We use the **HODDI** dataset (*Higher-Order Drug-Drug Interactions for Computational Pharmacovigilance*), derived from FAERS reports from:

- **2014Q3 to 2024Q3**
- **41 quarters total**

### Main dataset file
- `hoddi_merged.csv`

### Important columns
- `DrugBankID` → list of DrugBank IDs in the prescription
- `SE_label` → UMLS CUI for the side effect
- `hyperedge_label` → `1` for positive samples, `-1` for negative samples
- `time` → quarter of the record

We convert the labels as:
- `1 → positive`
- `-1 → negative`

---

## Data Split

We used a **strict chronological split by quarter**, not a random split.

- **Train:** earliest 70% of quarters  
  `2014Q3 → 2021Q2`

- **Validation:** next 15% of quarters  
  `2021Q3 → 2022Q4`

- **Test:** latest 15% of quarters  
  `2023Q1 → 2024Q3`

This means the models are always tested on **future data**, which makes the evaluation more realistic and avoids data leakage.

---

## Models

### 1) HGNN-SA

HGNN-SA treats each prescription record as a **hyperedge** in a hypergraph.

#### How it works
- **Nodes** = drugs
- **Hyperedges** = drug combinations from each record

Each drug is represented using:
- a learnable **drug ID embedding**
- a **SMILES CNN embedding** from its molecular structure

These features are passed through two **HypergraphConv** layers so that drugs can exchange information through the combinations they appear in.

Then:
- attention pooling is used to combine the drug features inside each hyperedge
- this pooled combination vector is concatenated with a learnable side-effect embedding
- an MLP outputs the final probability

#### Why this model is useful
It explicitly models higher-order relations between drugs using hypergraph message passing.

---

### 2) PolyFormer

PolyFormer treats the prescription as a **set of drugs**, rather than a hypergraph.

#### How it works
Each drug is represented using:
- a learnable **drug ID embedding**
- a **SMILES CNN embedding**

Then:
- the set of drugs is passed through **self-attention**
- the side effect embedding interacts with the drug set using **cross-attention**
- the resulting representation is passed through an MLP to produce the risk probability

#### Why this model is useful
PolyFormer is **inductive**, meaning it can make predictions directly from an arbitrary user-entered drug set without needing an inference-time hypergraph.

---

## Repository Structure

```text
Higher-Order-Drug-Drug-Interaction/
│
├── app.py
├── Drugbank_ID_SMILE_all_structure links.csv
├── DrugBankID2SMILES.csv
├── evaluate_HGNN.txt
├── Evaluate_PolyFormer.txt
├── HGNN_model.pt
├── HGNN_train.py
├── hoddi_merged.csv
├── polyformer_model.pt
├── polyformer_train.py
├── README.md
├── requirements.txt
├── SE_similarity_2014Q3_2024Q3.csv
└── Side_effects_unique.csv
```

---

## Results

Both models were evaluated on the same unseen future test set.

### HGNN-SA Performance
From `evaluate_HGNN.txt`:
- **ROC-AUC:** 0.8691
- **PR-AUC:** 0.8791
- **F1-score:** 0.7767
- **Accuracy:** 0.7954
- **Precision:** 0.8549
- **Recall:** 0.7116

### PolyFormer Performance
From `Evaluate_PolyFormer.txt`:
- **ROC-AUC:** 0.9651
- **PR-AUC:** 0.9658
- **F1-score:** 0.8897
- **Accuracy:** 0.8959
- **Precision:** 0.9457
- **Recall:** 0.8400

---

## Summary

PolyFormer significantly outperformed HGNN-SA on the same chronological future test set across all evaluation metrics.  
This suggests that an **inductive set-attention architecture** is highly effective for modeling **high-order drug interactions**, and can generalize better to unseen future prescriptions compared to hypergraph-based approaches.

---

## Citation

If you use or reference the HODDI dataset, please cite:

**Wang Z, Shi Y, Liu X, Chen C, Wen J, Wang R. 
HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance.https://arxiv.org/pdf/2502.06274**  

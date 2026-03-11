# PolySignal: High-Order Polypharmacy Risk Prediction (HGNN-SA vs PolyFormer)

PolySignal is a research prototype for predicting whether a **multi-drug (polypharmacy) combination** is likely to cause a **specific adverse side effect** (UMLS CUI), using the **HODDI dataset (2014Q3–2024Q3)**.  
The project implements and compares two modeling paradigms under a **strict quarter-wise chronological split**:

- **HGNN-SA**: Hypergraph Neural Network (HypergraphConv) + SMILES CNN + attention pooling  
- **PolyFormer** (novel): Inductive set-attention model + SMILES CNN + self-attention and SE→drug cross-attention

A **Streamlit dashboard** allows users to input arbitrary drug combinations and compare predictions from both models.

> Disclaimer: Research prototype; not for clinical use.

---

## Repository Contents

### Key files
- `hoddi_merged.csv`  
  Merged dataset (all quarters). Each row: drug set, side effect, label, time.
- `Drugbank_ID_SMILE_all_structure links.csv`  
  DrugBank ID → drug name + SMILES (used for chemistry features + UI mapping).
- `Side_effects_unique.csv`  
  UMLS CUI → side effect name (+ embeddings columns `0..767`, used mainly for mapping).
- `train.py`  
  Trains/evaluates HGNN-SA and writes `faculty_evaluation_report.txt`. Saves `sota_model.pt`.
- `polyformer_train.py` (or similarly named PolyFormer training script)  
  Trains/evaluates PolyFormer and writes `faculty_report_polyformer_*.txt`. Saves PolyFormer checkpoint.
- `app.py`  
  Streamlit app to compare **HGNN-SA vs PolyFormer** on the same user input.
- `faculty_evaluation_report.txt`  
  Final HGNN-SA test metrics.
- `faculty_report_polyformer_*.txt`  
  Final PolyFormer test metrics.
- Model checkpoints:
  - `sota_model.pt` (HGNN-SA)
  - `polyformer_model_learnable_se_nofilter.pt` (PolyFormer; filename may vary)

---

## Dataset Format (HODDI)

Each record contains:
- `DrugBankID`: list of DrugBank IDs (stored as a stringified list in CSV)
- `SE_label`: UMLS CUI for side effect (e.g., `C0013404`)
- `hyperedge_label`: `1` (positive) or `-1` (negative)
- `time`: quarter label (e.g., `2019Q2`)

We convert:
- `hyperedge_label` → `y ∈ {0,1}` by mapping `-1 → 0`, `1 → 1`.

---

## Split Strategy (Critical)
All experiments use **strict chronological splitting by quarter** (no shuffling):

- **Train**: earliest 70% of quarters (2014Q3 → 2021Q2)
- **Validation**: next 15% (2021Q3 → 2022Q4)
- **Test**: latest 15% (2023Q1 → 2024Q3)

This simulates real deployment: learn from past, predict future.

---

## Models

### 1) HGNN-SA (Hypergraph Neural Network + SMILES + Attention)
- **Nodes**: drugs  
- **Hyperedges**: each record/report (drug set)
- Drug features:
  - learnable Drug ID embedding
  - SMILES CNN embedding (char-level tokenization, max length 256)
- HypergraphConv message passing (2 layers)
- Attention pooling within each hyperedge
- Learnable side effect embedding
- MLP head for final prediction

**Notes on inference:**  
HGNN uses message passing over a hypergraph; stable inference is achieved by using a **train-only context hypergraph** plus the query hyperedge (implemented in the Streamlit app).

### 2) PolyFormer (Inductive Set-Attention, Novel)
PolyFormer treats each drug set as a **set of tokens**, not as part of a global hypergraph.

- Drug features:
  - learnable Drug ID embedding
  - SMILES CNN embedding
- **Self-attention** across drugs in the set (TransformerEncoder)
- **Cross-attention** from the side effect embedding (query) to the drug tokens
- Learnable side effect embedding (for fair comparison vs HGNN-SA)

**Combo size handling:**  
No filtering by size; to allow transformer batching, very large drug sets are deterministically truncated to `MAX_DRUGS=16` (reported during training).

---

## Results (Example)
Your exact numbers depend on training runs, but the final workflow produces text reports:

- `faculty_evaluation_report.txt` (HGNN-SA)
- `faculty_report_polyformer_*.txt` (PolyFormer)

Example (from a completed run):
- HGNN-SA ROC-AUC ≈ **0.869**
- PolyFormer ROC-AUC ≈ **0.965**

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -U pip
pip install pandas numpy scikit-learn plotly matplotlib streamlit
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter

## Citation

### HODDI dataset / paper
**Wang, Z., Shi, Y., Liu, X., Chen, C., Wen, J., & Wang, R.**  
*HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance.*  
arXiv preprint **arXiv:2502.06274** (2025).  
https://arxiv.org/abs/2502.06274

#### BibTeX
```bibtex
@article{wang2025hoddi,
  title={HODDI: A Dataset of High-Order Drug-Drug Interactions for Computational Pharmacovigilance},
  author={Wang, Zhaoying and Shi, Yingdan and Liu, Xiang and Chen, Can and Wen, Jun and Wang, Ren},
  journal={arXiv preprint arXiv:2502.06274},
  year={2025}
}
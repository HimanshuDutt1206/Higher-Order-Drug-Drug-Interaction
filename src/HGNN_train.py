"""
HGNN-SA hypergraph neural network training and evaluation script.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    HODDI_CSV,
    HGNN_DEFAULT_CKPT,
    RESULTS_DIR,
    DEFAULT_MAX_SMILES_LEN,
)
from src.data.dataset import parse_drug_ids, quarter_split
from src.data.smiles import load_drugbank_smiles, build_smiles_vocab, build_smiles_tensor
from src.models.hgnn import HGNN_SA
from src.utils.metrics import compute_classification_metrics, format_metrics_report
import pandas as pd


def prepare_sota_data(data_csv: str = str(HODDI_CSV)):
    print(f"Loading Dataset from {data_csv} and SMILES...")
    df = pd.read_csv(data_csv, low_memory=False)
    df["DrugBankID"] = df["DrugBankID"].apply(parse_drug_ids)
    df["target"] = df["hyperedge_label"].apply(lambda x: 0 if x == -1 else 1)

    all_drugs = sorted(list(set([d for lst in df["DrugBankID"] for d in lst])))
    drug2idx = {d: i for i, d in enumerate(all_drugs)}
    all_ses = df["SE_label"].unique()
    se2idx = {s: i for i, s in enumerate(all_ses)}

    drug2smiles = load_drugbank_smiles()
    char2idx, vocab_size = build_smiles_vocab(drug2smiles)
    all_smiles_tensor, _, _ = build_smiles_tensor(
        drug2idx, drug2smiles=drug2smiles, char2idx=char2idx, max_len=DEFAULT_MAX_SMILES_LEN
    )

    # STRICT CHRONOLOGICAL SPLIT (70 / 15 / 15)
    train_q, val_q, test_q = quarter_split(df, train_frac=0.70, val_frac=0.15)
    print(f"Split -> Train: {len(train_q)} quarters | Val: {len(val_q)} quarters | Test: {len(test_q)} quarters")

    splits = {
        "train": df[df["time"].isin(train_q)].reset_index(drop=True),
        "val": df[df["time"].isin(val_q)].reset_index(drop=True),
        "test": df[df["time"].isin(test_q)].reset_index(drop=True),
    }

    def build_tensors(split_df):
        d_idx, r_idx, s_idx, labels = [], [], [], []
        for i, row in split_df.iterrows():
            for d in row["DrugBankID"]:
                d_idx.append(drug2idx[d])
                r_idx.append(i)
            s_idx.append(se2idx[row["SE_label"]])
            labels.append(row["target"])
        return (
            torch.tensor([d_idx, r_idx], dtype=torch.long),
            torch.tensor(s_idx, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )

    data = {k: build_tensors(v) for k, v in splits.items()}
    data.update({
        "num_drugs": len(drug2idx),
        "num_ses": len(se2idx),
        "drug2idx": drug2idx,
        "se2idx": se2idx,
        "char2idx": char2idx,
        "smiles_tensor": all_smiles_tensor,
    })
    return data


def train_and_evaluate(max_epochs: int = 500, out_model: str = str(HGNN_DEFAULT_CKPT)):
    data = prepare_sota_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining SOTA HGNN-SA on {device}...")

    smiles_vocab_size = len(data["char2idx"]) + 1
    model = HGNN_SA(data["num_drugs"], data["num_ses"], smiles_vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

    train_h, train_s, train_y = [t.to(device) for t in data["train"]]
    val_h, val_s, val_y = [t.to(device) for t in data["val"]]
    smiles_t = data["smiles_tensor"].to(device)

    best_auc = 0.0
    patience_counter = 0
    early_stop_patience = 25

    # --- TRAINING PHASE ---
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_h, train_s, smiles_t)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_v = model(val_h, val_s, smiles_t)
            probs = torch.sigmoid(out_v).cpu().numpy()
            y_true = val_y.cpu().numpy()

            val_metrics = compute_classification_metrics(y_true, probs)
            auc = val_metrics["roc_auc"]
            scheduler.step(auc)

            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                Path(out_model).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "drug2idx": data["drug2idx"],
                    "se2idx": data["se2idx"],
                    "char2idx": data["char2idx"],
                    "smiles_tensor": data["smiles_tensor"],
                }, out_model)
            else:
                patience_counter += 1

        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val AUC: {auc:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_auc:.4f}")
            break

    print("\nTraining Complete. Moving to Final Test Set Evaluation...\n")

    # --- EVALUATION PHASE ---
    checkpoint = torch.load(out_model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_h, test_s, test_y = [t.to(device) for t in data["test"]]

    print("Running Inference on unseen Future Quarters...")
    with torch.no_grad():
        out_test = model(test_h, test_s, smiles_t)
        probs = torch.sigmoid(out_test).cpu().numpy()
        y_true = test_y.cpu().numpy()

    test_metrics = compute_classification_metrics(y_true, probs)

    details = {
        "Model Architecture": "Hypergraph Neural Network + SMILES CNN + Attention",
        "Data Split strategy": "Strict Chronological Quarter-Level (70/15/15)",
    }
    report = format_metrics_report(
        title="EVALUATION REPORT: POLYPHARMACY HGNN-SA",
        metrics=test_metrics,
        details=details,
    )
    print(report)

    out_report = RESULTS_DIR / "evaluate_HGNN.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved evaluation results to '{out_report}'.")


if __name__ == "__main__":
    train_and_evaluate(max_epochs=500)
"""
Training and benchmark script for the Inductive 3D-Aware PolyFormer.

Evaluates simultaneously on:
1. Validation Set (Model selection & LR scheduling)
2. Warm Temporal Test Set (Future calendar quarters on seen drugs)
3. Cold-Drug Holdout Test Set (Combinations with unseen novel drugs)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    HODDI_CSV,
    DRUGBANK_CSV,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEFAULT_MAX_DRUGS,
    DEFAULT_SEED,
)
from src.data.dataset import load_hoddi_data, batch_iter
from src.data.splits import create_cold_drug_split, save_split_manifest
from src.data.smiles import load_drugbank_smiles
from src.data.conformer_3d import (
    extract_and_cache_drug_features,
    compute_drug_features,
    TOTAL_DRUG_FEAT_DIM,
)
from src.models.inductive_polyformer import InductivePolyFormer
from src.utils.metrics import compute_classification_metrics, format_metrics_report


def seed_all(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pack_split_with_3d_features(
    split_df: pd.DataFrame,
    features_cache: dict,
    drug2smiles: dict,
    max_drugs: int = DEFAULT_MAX_DRUGS,
) -> dict:
    """
    Packs combinations into fixed [N, max_drugs, TOTAL_DRUG_FEAT_DIM] padded feature tensors.
    Uses cached 3D conformer + fingerprint features; falls back gracefully if drug is novel.
    """
    N = len(split_df)
    drug_feats_pad = np.zeros((N, max_drugs, TOTAL_DRUG_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((N, max_drugs), dtype=np.int64)

    zero_feat = np.zeros(TOTAL_DRUG_FEAT_DIM, dtype=np.float32)

    for i, lst in enumerate(split_df["DrugBankID"].tolist()):
        lst_sorted = sorted(set(lst))[:max_drugs]
        for k_idx, d in enumerate(lst_sorted):
            if d in features_cache:
                feat = features_cache[d]
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
            elif d in drug2smiles:
                feat = compute_drug_features(drug2smiles[d])
                features_cache[d] = torch.from_numpy(feat)
            else:
                feat = zero_feat

            drug_feats_pad[i, k_idx] = feat
            mask[i, k_idx] = 1

    return {
        "drug_features": torch.from_numpy(drug_feats_pad),
        "mask": torch.from_numpy(mask),
        "se_idx": torch.from_numpy(split_df["se_idx"].to_numpy(np.int64)),
        "y": torch.from_numpy(split_df["y"].to_numpy(np.int64)),
    }


def permute_drug_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set-invariance data augmentation: shuffles real drug slots within batch."""
    B, K, D = features.shape
    out = features.clone()
    for i in range(B):
        n = int(mask[i].sum().item())
        if n > 1:
            perm = torch.randperm(n, device=out.device)
            out[i, :n] = out[i, :n][perm]
    return out


def evaluate_inductive_model(
    model: nn.Module,
    split_tensors: dict,
    device: torch.device,
    batch_size: int = 2048,
) -> dict:
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for batch in batch_iter(split_tensors, batch_size, shuffle=False):
            drug_features = batch["drug_features"].to(device)
            mask = batch["mask"].to(device)
            se_idx = batch["se_idx"].to(device)
            y = batch["y"].to(device).float()

            logits = model(drug_features, mask, se_idx)
            probs = torch.sigmoid(logits)

            probs_all.append(probs.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    return compute_classification_metrics(y_all, probs_all)


def train_inductive(
    data_csv: str = str(HODDI_CSV),
    out_model: str = str(MODELS_DIR / "inductive_polyformer_3d.pt"),
    out_report: str = str(RESULTS_DIR / "Evaluate_Inductive_3D_PolyFormer.txt"),
    max_epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    cold_ratio: float = 0.15,
):
    seed_all(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Dataset and SMILES
    print(f"Loading HODDI data from '{data_csv}'...")
    df, drug2idx, se2idx = load_hoddi_data(data_csv, min_combo_size=2)
    drug2smiles = load_drugbank_smiles()

    # 2. Extract or Load 3D Conformer Features Cache
    features_cache = extract_and_cache_drug_features(drug2smiles)

    # 3. Create Cold-Drug Holdout Benchmark Splits
    splits = create_cold_drug_split(df, test_drug_ratio=cold_ratio, seed=DEFAULT_SEED)
    manifest_path = DATA_DIR / "splits_manifest.json"
    save_split_manifest(splits, manifest_path)

    # 4. Pack Tensors with 3D Features
    print("Packing split tensors with 3D molecular features...")
    train_t = pack_split_with_3d_features(splits["train"], features_cache, drug2smiles)
    val_t = pack_split_with_3d_features(splits["val"], features_cache, drug2smiles)
    warm_t = pack_split_with_3d_features(splits["test_warm"], features_cache, drug2smiles)
    cold_t = pack_split_with_3d_features(splits["test_cold"], features_cache, drug2smiles)

    # 5. Initialize Inductive 3D-Aware PolyFormer
    model = InductivePolyFormer(
        num_ses=len(se2idx),
        in_drug_dim=TOTAL_DRUG_FEAT_DIM,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = -1.0
    patience_ctr = 0

    print(f"\nStarting Inductive 3D-Aware PolyFormer Training (Max Epochs={max_epochs}, Batch={batch_size})...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []

        for batch in batch_iter(train_t, batch_size, shuffle=True):
            drug_features = batch["drug_features"].to(device)
            mask = batch["mask"].to(device)
            se_idx = batch["se_idx"].to(device)
            y = batch["y"].to(device).float()

            drug_features = permute_drug_features(drug_features, mask)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(drug_features, mask, se_idx)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        val_metrics = evaluate_inductive_model(model, val_t, device)
        scheduler.step(val_metrics["roc_auc"])

        mean_loss = float(np.mean(losses))
        print(
            f"Epoch {epoch:03d} | Train Loss: {mean_loss:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            patience_ctr = 0
            Path(out_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "se2idx": se2idx,
                "in_drug_dim": TOTAL_DRUG_FEAT_DIM,
                "d_model": 128,
                "max_drugs": DEFAULT_MAX_DRUGS,
            }, out_model)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Val AUC={best_val_auc:.4f}")
            break

    # 6. Final Evaluation Across All Splits
    print("\nLoading best model checkpoint for benchmark evaluation...")
    ckpt = torch.load(out_model, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    warm_metrics = evaluate_inductive_model(model, warm_t, device)
    cold_metrics = evaluate_inductive_model(model, cold_t, device)

    # 7. Formulate Comprehensive Benchmark Report
    report = f"""
======================================================================
🏥 BENCHMARK REPORT: INDUCTIVE 3D-AWARE POLYFORMER
======================================================================
Architecture: 3D Conformer + Pairwise Synergy + Set Transformer + SE Cross-Attention
Drug representation: 100% Inductive (1024-bit Morgan + 17 3D geometric descriptors)
Splits: Evaluated on Seen Drugs (Future Quarters) vs. Unseen Drugs (Cold-Start Holdout)

🏆 WARM TEST SET METRICS (Known Drugs, Future Quarters):
----------------------------------------------------------------------
ROC-AUC (Main Metric) : {warm_metrics['roc_auc']:.4f}
PR-AUC                : {warm_metrics['pr_auc']:.4f}
F1-Score              : {warm_metrics['f1']:.4f}
Accuracy              : {warm_metrics['acc']:.4f}
Precision             : {warm_metrics['precision']:.4f}
Recall                : {warm_metrics['recall']:.4f}

❄️ COLD TEST SET METRICS (Unseen Drugs Never in Training Set):
----------------------------------------------------------------------
ROC-AUC (Main Metric) : {cold_metrics['roc_auc']:.4f}
PR-AUC                : {cold_metrics['pr_auc']:.4f}
F1-Score              : {cold_metrics['f1']:.4f}
Accuracy              : {cold_metrics['acc']:.4f}
Precision             : {cold_metrics['precision']:.4f}
Recall                : {cold_metrics['recall']:.4f}

🧩 CONFUSION MATRICES:
Warm Test:
  Predicted Safe (0): {warm_metrics['cm'][0][0]:<8} | Predicted Toxic (1): {warm_metrics['cm'][0][1]}
  Actual Safe (0)   : {warm_metrics['cm'][1][0]:<8} | Actual Toxic (1)   : {warm_metrics['cm'][1][1]}

Cold Test (Unseen Drugs):
  Predicted Safe (0): {cold_metrics['cm'][0][0]:<8} | Predicted Toxic (1): {cold_metrics['cm'][0][1]}
  Actual Safe (0)   : {cold_metrics['cm'][1][0]:<8} | Actual Toxic (1)   : {cold_metrics['cm'][1][1]}
======================================================================
"""
    print(report)

    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved evaluation report to '{out_report}'.")
    print(f"Saved model checkpoint to '{out_model}'.")


def parse_args():
    p = argparse.ArgumentParser(description="Train Inductive 3D-Aware PolyFormer.")
    p.add_argument("--data", default=str(HODDI_CSV))
    p.add_argument("--out_model", default=str(MODELS_DIR / "inductive_polyformer_3d.pt"))
    p.add_argument("--out_report", default=str(RESULTS_DIR / "Evaluate_Inductive_3D_PolyFormer.txt"))
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_inductive(
        data_csv=args.data,
        out_model=args.out_model,
        out_report=args.out_report,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

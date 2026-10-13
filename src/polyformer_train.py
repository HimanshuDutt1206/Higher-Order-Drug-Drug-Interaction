"""
PolyFormer training and evaluation script.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    HODDI_CSV,
    POLYFORMER_DEFAULT_CKPT,
    RESULTS_DIR,
    DEFAULT_MAX_DRUGS,
    DEFAULT_SEED,
)
from src.data.dataset import (
    load_hoddi_data,
    quarter_split,
    pack_split,
    batch_iter,
    permute_drug_order,
)
from src.data.smiles import build_smiles_tensor, load_drugbank_smiles, build_smiles_vocab
from src.models.polyformer import PolyFormerLearnableSE
from src.utils.metrics import compute_classification_metrics, format_metrics_report


def seed_all(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_model(model, split_tensors, smiles_tok_t, device, batch_size=2048):
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for batch in batch_iter(split_tensors, batch_size, shuffle=False):
            drugs_pad = batch["drugs_pad"].to(device)
            mask = batch["mask"].to(device)
            se_idx = batch["se_idx"].to(device)
            y = batch["y"].to(device).float()

            logits = model(drugs_pad, mask, se_idx, smiles_tok_t)
            probs = torch.sigmoid(logits)

            probs_all.append(probs.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    return compute_classification_metrics(y_all, probs_all)


def train_polyformer(
    max_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    log_every_batches: int = 50,
    data_csv: str = None,
    out_model: str = None,
    out_report: str = None,
    tag: str = "",
    patience: int = 25,
):
    seed_all(DEFAULT_SEED)

    data_csv = str(data_csv or HODDI_CSV)
    out_model = str(out_model or POLYFORMER_DEFAULT_CKPT)
    out_report = str(out_report or (RESULTS_DIR / "Evaluate_PolyFormer.txt"))

    print(f"Loading HODDI data from {data_csv}...")
    df, drug2idx, se2idx = load_hoddi_data(data_csv, min_combo_size=2)

    # Load SMILES and tokenize
    print("Loading SMILES and building token tensor...")
    drug2smiles = load_drugbank_smiles()
    char2idx, smiles_vocab_size = build_smiles_vocab(drug2smiles)
    smiles_tok, _, _ = build_smiles_tensor(drug2idx, drug2smiles=drug2smiles, char2idx=char2idx)

    # Chronological quarter split
    train_q, val_q, test_q = quarter_split(df)
    train_df = df[df["time"].isin(train_q)].reset_index(drop=True)
    val_df = df[df["time"].isin(val_q)].reset_index(drop=True)
    test_df = df[df["time"].isin(test_q)].reset_index(drop=True)

    print(f"Quarters total: {len(sorted(df['time'].unique()))}")
    print(f"   Train: {len(train_q)} ({train_q[0]} -> {train_q[-1]}) | rows={len(train_df)}")
    print(f"   Val:   {len(val_q)} ({val_q[0]} -> {val_q[-1]}) | rows={len(val_df)}")
    print(f"   Test:  {len(test_q)} ({test_q[0]} -> {test_q[-1]}) | rows={len(test_df)}")
    print(f"   Num drugs={len(drug2idx)} | Num side effects={len(se2idx)} | MAX_DRUGS={DEFAULT_MAX_DRUGS}")

    train_t = pack_split(train_df, drug2idx, max_drugs=DEFAULT_MAX_DRUGS)
    val_t = pack_split(val_df, drug2idx, max_drugs=DEFAULT_MAX_DRUGS)
    test_t = pack_split(test_df, drug2idx, max_drugs=DEFAULT_MAX_DRUGS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Outputs -> model={out_model} | report={out_report}")

    smiles_tok_t = smiles_tok.to(device)

    model = PolyFormerLearnableSE(
        num_drugs=len(drug2idx),
        smiles_vocab_size=smiles_vocab_size,
        num_ses=len(se2idx),
        d_model=128,
        nhead=8,
        num_layers=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=10)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = -1.0
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        steps_per_epoch = int(np.ceil(train_t["y"].shape[0] / batch_size))
        batch_count = 0

        for batch in batch_iter(train_t, batch_size, shuffle=True):
            batch_count += 1
            drugs_pad = batch["drugs_pad"].to(device)
            mask = batch["mask"].to(device)
            se_idx = batch["se_idx"].to(device)
            y = batch["y"].to(device).float()

            drugs_pad = permute_drug_order(drugs_pad, mask)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(drugs_pad, mask, se_idx, smiles_tok_t)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())

            if (batch_count % log_every_batches) == 0:
                print(f"  epoch {epoch:03d} | batch {batch_count:04d}/{steps_per_epoch} | loss={loss.item():.4f}")

        val_metrics = eval_model(model, val_t, smiles_tok_t, device)
        sched.step(val_metrics["roc_auc"])

        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch:03d} | train_loss={mean_loss:.4f} | val_auc={val_metrics['roc_auc']:.4f} | val_f1={val_metrics['f1']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            patience_ctr = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "drug2idx": drug2idx,
                "se2idx": se2idx,
                "smiles_vocab_size": smiles_vocab_size,
                "max_drugs_model": DEFAULT_MAX_DRUGS,
                "data_csv": data_csv,
                "tag": tag,
            }, out_model)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early stopping. Best val_auc={best_val_auc:.4f}")
            break

    # Evaluate best model on test split
    ckpt = torch.load(out_model, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = eval_model(model, test_t, smiles_tok_t, device)

    details = {
        "Model": "SMILES CNN + Drug Set Self-Attention + SE Cross-Attention",
        "SE representation": "Learnable embedding (fair vs HGNN-SA)",
        "Split": "Strict quarter-wise chronological 70/15/15 (same as HGNN-SA)",
        "Combo size handling": f"No filtering; drug sets truncated to MAX_DRUGS={DEFAULT_MAX_DRUGS} (deterministic)",
    }
    if tag:
        details["Experiment tag"] = tag
    details["Data CSV"] = data_csv

    report = format_metrics_report(
        title="EVALUATION REPORT: POLYFORMER (NO SIZE FILTER, LEARNABLE SE)",
        metrics=test_metrics,
        details=details,
    )
    print(report)

    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved -> {out_report}")
    print(f"Saved -> {out_model}")


def parse_args():
    p = argparse.ArgumentParser(description="Train PolyFormer on a HODDI-style CSV.")
    p.add_argument("--data", default=str(HODDI_CSV), help="Path to dataset CSV")
    p.add_argument("--out_model", default=str(POLYFORMER_DEFAULT_CKPT), help="Checkpoint output path")
    p.add_argument("--out_report", default=str(RESULTS_DIR / "Evaluate_PolyFormer.txt"), help="Evaluation report output path")
    p.add_argument("--tag", default="", help="Optional tag written into the report header")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patience", type=int, default=25, help="Early-stop patience on val AUC")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_polyformer(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        log_every_batches=50,
        data_csv=args.data,
        out_model=args.out_model,
        out_report=args.out_report,
        tag=args.tag,
        patience=args.patience,
    )
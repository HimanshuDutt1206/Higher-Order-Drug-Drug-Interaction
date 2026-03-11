import ast
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)

# -----------------------
# Reproducibility
# -----------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(42)

HODDI_CSV = "hoddi_merged.csv"
DRUGBANK_CSV = "Drugbank_ID_SMILE_all_structure links.csv"

# We REMOVE the size filter. But the model still needs a max length.
# 16 covers the vast majority of HODDI records (most are <=10).
MAX_DRUGS = 16
MAX_SMILES_LEN = 256

OUT_MODEL = "polyformer_model_learnable_se_nofilter.pt"
OUT_REPORT = "faculty_report_polyformer_learnable_se_nofilter.txt"


# ==========================================================
# 1) SMILES encoder (CNN)
# ==========================================================
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)

    def forward(self, smiles_tokens):  # [B, L]
        x = self.embed(smiles_tokens).transpose(1, 2)  # [B, E, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)  # [B, out_dim]
        return x


# ==========================================================
# 2) PolyFormer with LEARNABLE SE embeddings
# ==========================================================
class PolyFormerLearnableSE(nn.Module):
    def __init__(self, num_drugs, smiles_vocab_size, num_ses,
                 d_model=128, nhead=8, num_layers=2):
        super().__init__()

        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.drug_proj = nn.Linear(128, d_model)

        # Learnable SE embedding (fair vs HGNN-SA)
        self.se_emb = nn.Embedding(num_ses, d_model)

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

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(self, drugs_pad, mask, se_idx, smiles_tok_all):
        """
        drugs_pad: [B, K] drug indices, -1 padded
        mask:      [B, K] 1 real, 0 pad
        se_idx:    [B]    side-effect integer index
        smiles_tok_all: [num_drugs, 256]
        """
        B, K = drugs_pad.shape

        drugs_safe = drugs_pad.clamp(min=0)

        # Drug ID embedding
        did = self.drug_id_emb(drugs_safe)  # [B,K,64]

        # FAST SMILES: unique drugs in batch only once
        flat = drugs_safe.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        smiles_tokens = smiles_tok_all[uniq]         # [U,256]
        smiles_uniq = self.smiles_enc(smiles_tokens) # [U,64]
        smiles_feat = smiles_uniq[inv].reshape(B, K, -1)  # [B,K,64]

        drug_feat = torch.cat([did, smiles_feat], dim=-1)  # [B,K,128]
        drug_feat = self.drug_proj(drug_feat)              # [B,K,d_model]

        key_padding = (mask == 0)  # True where padding

        # Self-attention
        drug_ctx = self.encoder(drug_feat, src_key_padding_mask=key_padding)  # [B,K,d_model]

        # Cross-attention: SE queries drugs
        q = self.se_emb(se_idx).unsqueeze(1)  # [B,1,d_model]
        attn_out, _ = self.cross_attn(query=q, key=drug_ctx, value=drug_ctx, key_padding_mask=key_padding)

        combo = attn_out.squeeze(1)  # [B,d_model]
        se_token = q.squeeze(1)      # [B,d_model]

        logits = self.mlp(torch.cat([combo, se_token], dim=-1)).squeeze(-1)
        return logits


# ==========================================================
# 3) Data prep: same quarter split logic as HGNN-SA
# ==========================================================
def quarter_split(df, train_frac=0.70, val_frac=0.15):
    quarters = sorted(df["time"].unique())
    n = len(quarters)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return quarters[:train_end], quarters[train_end:val_end], quarters[val_end:]


def load_and_prepare():
    print("📥 Loading HODDI (NO combo-size filter)...")
    df = pd.read_csv(HODDI_CSV)
    df["DrugBankID"] = df["DrugBankID"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["y"] = (df["hyperedge_label"] == 1).astype(np.int64)
    df["k"] = df["DrugBankID"].apply(len)

    # Drop degenerate 1-drug rows if any (PolyFormer needs >=2 drugs for interaction)
    df = df[df["k"] >= 2].reset_index(drop=True)

    # Drug vocab from df
    all_drugs = sorted({d for lst in df["DrugBankID"] for d in lst})
    drug2idx = {d: i for i, d in enumerate(all_drugs)}

    # Side effect vocab from df (learnable)
    df["SE_label"] = df["SE_label"].astype(str)
    all_ses = sorted(df["SE_label"].unique().tolist())
    se2idx = {s: i for i, s in enumerate(all_ses)}
    df["se_idx"] = df["SE_label"].map(se2idx).astype(np.int64)

    # Load DrugBank SMILES
    print("🧬 Loading DrugBank SMILES...")
    db = pd.read_csv(DRUGBANK_CSV, usecols=["DrugBank ID", "SMILES"])
    drug2smiles = dict(zip(db["DrugBank ID"], db["SMILES"]))

    # Build SMILES char vocab
    print("🔡 Building SMILES char vocab...")
    chars = set()
    for s in drug2smiles.values():
        if isinstance(s, str) and len(s) > 0:
            chars.update(list(s))
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 pad
    char2idx["<UNK>"] = len(char2idx) + 1
    smiles_vocab_size = len(char2idx) + 1
    unk = char2idx["<UNK>"]

    # Tokenize SMILES per drug
    print("🧪 Tokenizing SMILES (per drug)...")
    smiles_tok = np.zeros((len(drug2idx), MAX_SMILES_LEN), dtype=np.int64)
    for drug, di in drug2idx.items():
        s = drug2smiles.get(drug, None)
        if not isinstance(s, str) or len(s) == 0:
            s = "C"  # fallback
        toks = [char2idx.get(ch, unk) for ch in s[:MAX_SMILES_LEN]]
        smiles_tok[di, :len(toks)] = toks
    smiles_tok = torch.from_numpy(smiles_tok)

    # Quarter split
    train_q, val_q, test_q = quarter_split(df)
    train_df = df[df["time"].isin(train_q)].reset_index(drop=True)
    val_df = df[df["time"].isin(val_q)].reset_index(drop=True)
    test_df = df[df["time"].isin(test_q)].reset_index(drop=True)

    # Report truncation impact
    def trunc_count(frame):
        return int((frame["k"] > MAX_DRUGS).sum())

    print(f"📅 Quarters total: {len(sorted(df['time'].unique()))}")
    print(f"   Train: {len(train_q)} ({train_q[0]} -> {train_q[-1]}) | rows={len(train_df)} | >{MAX_DRUGS} drugs={trunc_count(train_df)}")
    print(f"   Val:   {len(val_q)} ({val_q[0]} -> {val_q[-1]}) | rows={len(val_df)} | >{MAX_DRUGS} drugs={trunc_count(val_df)}")
    print(f"   Test:  {len(test_q)} ({test_q[0]} -> {test_q[-1]}) | rows={len(test_df)} | >{MAX_DRUGS} drugs={trunc_count(test_df)}")
    print(f"   Num drugs={len(drug2idx)} | Num side effects={len(se2idx)} | MAX_DRUGS_MODEL={MAX_DRUGS}")

    return train_df, val_df, test_df, drug2idx, se2idx, smiles_tok, smiles_vocab_size


def pack_split(split_df, drug2idx):
    """
    Packs ANY-size combos into fixed [N, MAX_DRUGS] by truncation.
    Truncation is deterministic: sort drug IDs, take first MAX_DRUGS.
    (Stable across runs + fair evaluation.)
    """
    N = len(split_df)
    drugs_pad = np.full((N, MAX_DRUGS), -1, dtype=np.int64)
    mask = np.zeros((N, MAX_DRUGS), dtype=np.int64)

    for i, lst in enumerate(split_df["DrugBankID"].tolist()):
        # Deterministic truncation: sort for stability
        lst = sorted(set(lst))
        idxs = [drug2idx[d] for d in lst[:MAX_DRUGS]]

        drugs_pad[i, :len(idxs)] = idxs
        mask[i, :len(idxs)] = 1

    return {
        "drugs_pad": torch.from_numpy(drugs_pad),
        "mask": torch.from_numpy(mask),
        "se_idx": torch.from_numpy(split_df["se_idx"].to_numpy(np.int64)),
        "y": torch.from_numpy(split_df["y"].to_numpy(np.int64)),
    }


def batch_iter(tensors, batch_size, shuffle=True):
    N = tensors["y"].shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield {k: v[b] for k, v in tensors.items()}


def permute_drug_order(drugs_pad, mask):
    # set invariance: random shuffle real drug tokens within each row
    B, K = drugs_pad.shape
    out = drugs_pad.clone()
    for i in range(B):
        m = mask[i].bool()
        vals = out[i, m]
        if vals.numel() > 1:
            perm = torch.randperm(vals.numel(), device=vals.device)
            out[i, m] = vals[perm]
    return out


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
    y_pred = (probs_all > 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_all, probs_all),
        "pr_auc": average_precision_score(y_all, probs_all),
        "f1": f1_score(y_all, y_pred),
        "acc": accuracy_score(y_all, y_pred),
        "precision": precision_score(y_all, y_pred),
        "recall": recall_score(y_all, y_pred),
        "cm": confusion_matrix(y_all, y_pred),
    }


def train_polyformer(max_epochs=200, batch_size=512, lr=1e-3, weight_decay=1e-3, log_every_batches=50):
    train_df, val_df, test_df, drug2idx, se2idx, smiles_tok, smiles_vocab_size = load_and_prepare()

    train_t = pack_split(train_df, drug2idx)
    val_t = pack_split(val_df, drug2idx)
    test_t = pack_split(test_df, drug2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Device:", device)

    smiles_tok_t = smiles_tok.to(device)

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

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = -1.0
    patience = 25
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
                "max_drugs_model": MAX_DRUGS
            }, OUT_MODEL)
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"🛑 Early stopping. Best val_auc={best_val_auc:.4f}")
            break

    # Test eval
    ckpt = torch.load(OUT_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = eval_model(model, test_t, smiles_tok_t, device)

    cm = test_metrics["cm"]
    report = f"""
======================================================================
🏥 FACULTY EVALUATION REPORT: POLYFORMER (NO SIZE FILTER, LEARNABLE SE)
======================================================================
Model: SMILES CNN + Drug Set Self-Attention + SE Cross-Attention
SE representation: Learnable embedding (fair vs HGNN-SA)
Split: Strict quarter-wise chronological 70/15/15 (same as HGNN-SA)
Combo size handling: No filtering; drug sets truncated to MAX_DRUGS={MAX_DRUGS} (deterministic)

🏆 FINAL TEST SET METRICS (Unseen Future Data):
----------------------------------------------------------------------
ROC-AUC (Main Metric) : {test_metrics['roc_auc']:.4f}
PR-AUC                : {test_metrics['pr_auc']:.4f}
F1-Score              : {test_metrics['f1']:.4f}
Accuracy              : {test_metrics['acc']:.4f}
Precision             : {test_metrics['precision']:.4f}
Recall                : {test_metrics['recall']:.4f}

🧩 CONFUSION MATRIX:
                     Predicted Safe (0) | Predicted Toxic (1)
Actual Safe (0)    : {cm[0][0]:<18} | {cm[0][1]}
Actual Toxic (1)   : {cm[1][0]:<18} | {cm[1][1]}
======================================================================
"""
    print(report)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"💾 Saved -> {OUT_REPORT}")
    print(f"💾 Saved -> {OUT_MODEL}")


if __name__ == "__main__":
    train_polyformer(max_epochs=200, batch_size=512, log_every_batches=50)
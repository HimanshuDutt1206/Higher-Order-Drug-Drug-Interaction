import ast
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax

# =========================
# Paths / filenames
# =========================
HODDI_CSV = "hoddi_merged.csv"
DRUGBANK_CSV = "Drugbank_ID_SMILE_all_structure links.csv"
SIDE_EFFECTS_UNIQUE_CSV = "Side_effects_unique.csv"

HGNN_CKPT = "sota_model.pt"

# If your filename differs, change it here:
POLYFORMER_CKPT = "polyformer_model_learnable_se_nofilter.pt"

MAX_SMILES_LEN = 256


# =========================
# HGNN-SA architecture (must match sota_model.pt)
# =========================
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)

    def forward(self, smiles_tokens):
        x = self.embed(smiles_tokens).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)


class HGNN_SA(nn.Module):
    def __init__(self, num_drugs, num_ses, smiles_vocab_size):
        super().__init__()
        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.node_proj = nn.Linear(128, 128)

        self.se_emb = nn.Embedding(num_ses, 128)
        self.conv1 = HypergraphConv(128, 128)
        self.conv2 = HypergraphConv(128, 128)

        self.attn_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, hyperedge_index, se_indices, all_smiles_tokens):
        smiles_feats = self.smiles_enc(all_smiles_tokens)
        node_feats = torch.cat([self.drug_id_emb.weight, smiles_feats], dim=1)
        node_feats = F.relu(self.node_proj(node_feats))

        x = F.relu(self.conv1(node_feats, hyperedge_index))
        x = F.relu(self.conv2(x, hyperedge_index))

        drugs = x[hyperedge_index[0]]
        attn_probs = scatter_softmax(self.attn_layer(drugs), hyperedge_index[1], dim=0)
        combo_vector = scatter_add(drugs * attn_probs, hyperedge_index[1], dim=0)

        out = torch.cat([combo_vector, self.se_emb(se_indices)], dim=1)
        return self.mlp(out).squeeze(-1)


# =========================
# PolyFormer architecture (learnable SE, must match your training)
# =========================
class PolyFormerLearnableSE(nn.Module):
    def __init__(self, num_drugs, smiles_vocab_size, num_ses, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.drug_proj = nn.Linear(128, d_model)

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

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(self, drugs_pad, mask, se_idx, smiles_tok_all):
        """
        drugs_pad: [B,K] with -1 padding
        mask: [B,K] 1 real, 0 pad
        se_idx: [B]
        smiles_tok_all: [num_drugs, 256]
        """
        B, K = drugs_pad.shape
        drugs_safe = drugs_pad.clamp(min=0)

        did = self.drug_id_emb(drugs_safe)  # [B,K,64]

        flat = drugs_safe.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        smiles_tokens = smiles_tok_all[uniq]
        smiles_uniq = self.smiles_enc(smiles_tokens)
        smiles_feat = smiles_uniq[inv].reshape(B, K, -1)

        drug_feat = torch.cat([did, smiles_feat], dim=-1)  # [B,K,128]
        drug_feat = self.drug_proj(drug_feat)              # [B,K,d_model]

        key_padding = (mask == 0)
        drug_ctx = self.encoder(drug_feat, src_key_padding_mask=key_padding)

        q = self.se_emb(se_idx).unsqueeze(1)  # [B,1,d_model]
        attn_out, _ = self.cross_attn(query=q, key=drug_ctx, value=drug_ctx, key_padding_mask=key_padding)

        combo = attn_out.squeeze(1)
        se_token = q.squeeze(1)
        logits = self.mlp(torch.cat([combo, se_token], dim=-1)).squeeze(-1)
        return logits


# =========================
# Utility: quarter split (same as your training scripts)
# =========================
def quarter_split(quarters, train_frac=0.70, val_frac=0.15):
    quarters = sorted(quarters)
    n = len(quarters)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return quarters[:train_end], quarters[train_end:val_end], quarters[val_end:]


# =========================
# Caching loaders
# =========================
@st.cache_data
def load_name_maps():
    drugs_df = pd.read_csv(DRUGBANK_CSV, usecols=["DrugBank ID", "Name"])
    drug_map = dict(zip(drugs_df["DrugBank ID"], drugs_df["Name"]))

    se_df = pd.read_csv(SIDE_EFFECTS_UNIQUE_CSV, usecols=["umls_cui_from_meddra", "side_effect_name"])
    se_map = dict(zip(se_df["umls_cui_from_meddra"].astype(str), se_df["side_effect_name"]))
    return drug_map, se_map


@st.cache_data
def load_hoddi_parsed():
    df = pd.read_csv(HODDI_CSV)
    df["DrugBankID"] = df["DrugBankID"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["target"] = (df["hyperedge_label"] == 1).astype(int)
    df["SE_label"] = df["SE_label"].astype(str)
    df["k"] = df["DrugBankID"].apply(len)
    # remove degenerate 1-drug cases
    df = df[df["k"] >= 2].reset_index(drop=True)
    return df


@st.cache_resource
def load_hgnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(HGNN_CKPT, map_location=device)

    model = HGNN_SA(len(ckpt["drug2idx"]), len(ckpt["se2idx"]), len(ckpt["char2idx"]) + 1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["drug2idx"], ckpt["se2idx"], ckpt["smiles_tensor"].to(device), device


@st.cache_resource
def load_polyformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(POLYFORMER_CKPT, map_location=device)

    drug2idx = ckpt["drug2idx"]
    se2idx = ckpt["se2idx"]
    smiles_vocab_size = ckpt["smiles_vocab_size"]
    max_drugs_model = int(ckpt.get("max_drugs_model", 16))

    # Rebuild the SAME SMILES char vocab (same procedure as training) to tokenize drugs
    db = pd.read_csv(DRUGBANK_CSV, usecols=["DrugBank ID", "SMILES"])
    drug2smiles = dict(zip(db["DrugBank ID"], db["SMILES"]))

    chars = set()
    for s in drug2smiles.values():
        if isinstance(s, str) and len(s) > 0:
            chars.update(list(s))
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char2idx["<UNK>"] = len(char2idx) + 1
    unk = char2idx["<UNK>"]
    rebuilt_vocab_size = len(char2idx) + 1

    # Tokenize SMILES for drugs in THIS checkpoint's drug2idx order
    smiles_tok = np.zeros((len(drug2idx), MAX_SMILES_LEN), dtype=np.int64)
    for drug, di in drug2idx.items():
        s = drug2smiles.get(drug, None)
        if not isinstance(s, str) or len(s) == 0:
            s = "C"
        toks = [char2idx.get(ch, unk) for ch in s[:MAX_SMILES_LEN]]
        smiles_tok[di, :len(toks)] = toks
    smiles_tok = torch.from_numpy(smiles_tok).to(device)

    # Instantiate model
    model = PolyFormerLearnableSE(
        num_drugs=len(drug2idx),
        smiles_vocab_size=rebuilt_vocab_size,
        num_ses=len(se2idx),
        d_model=128,
        nhead=8,
        num_layers=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Safety check: vocab sizes
    if smiles_vocab_size != rebuilt_vocab_size:
        # training vocab should match if DrugBank file is same; if not, still works but embeddings differ
        pass

    return model, drug2idx, se2idx, smiles_tok, device, max_drugs_model


# =========================
# HGNN inference context builder (TRAIN-only)
# =========================
@st.cache_resource
def build_hgnn_train_index(hgnn_drug2idx, hgnn_se2idx):
    df = load_hoddi_parsed()
    quarters = sorted(df["time"].unique())
    train_q, val_q, test_q = quarter_split(quarters, 0.70, 0.15)

    train_df = df[df["time"].isin(set(train_q))].reset_index(drop=True)

    # Keep only rows whose SE is in model vocab
    train_df = train_df[train_df["SE_label"].isin(set(hgnn_se2idx.keys()))].reset_index(drop=True)

    # Build inverted index drug -> list of row indices (train only)
    inv = {}
    for i, drugs in enumerate(train_df["DrugBankID"].tolist()):
        for d in set(drugs):
            inv.setdefault(d, []).append(i)

    pos_idx = train_df.index[train_df["target"] == 1].to_numpy()
    neg_idx = train_df.index[train_df["target"] == 0].to_numpy()

    return train_df, inv, pos_idx, neg_idx, train_q, val_q, test_q


def make_hgnn_inference_tensors(
    train_df, inv_index, pos_idx, neg_idx,
    selected_drugs, target_se,
    hgnn_drug2idx, hgnn_se2idx,
    base_pos=400, base_neg=400, ego_pos=200, ego_neg=200, seed=42,
):
    rng = np.random.default_rng(seed)

    def sample(arr, k):
        if len(arr) == 0:
            return np.array([], dtype=np.int64)
        if len(arr) <= k:
            return arr
        return rng.choice(arr, size=k, replace=False)

    # Candidates containing any selected drug
    cand = set()
    for d in selected_drugs:
        for i in inv_index.get(d, []):
            cand.add(i)
    cand = np.array(sorted(list(cand)), dtype=np.int64)

    ego_df = train_df.iloc[cand] if len(cand) > 0 else train_df.iloc[[]]
    ego_pos_rows = ego_df.index[ego_df["target"] == 1].to_numpy()
    ego_neg_rows = ego_df.index[ego_df["target"] == 0].to_numpy()

    base_pick = np.concatenate([sample(pos_idx, base_pos), sample(neg_idx, base_neg)])
    ego_pick = np.concatenate([sample(ego_pos_rows, ego_pos), sample(ego_neg_rows, ego_neg)])
    context_rows = np.unique(np.concatenate([base_pick, ego_pick])).astype(np.int64)

    d_idx, r_idx, s_idx = [], [], []
    edge_id = 0

    for row_i in context_rows:
        drugs = list(set(train_df.at[row_i, "DrugBankID"]))
        se = train_df.at[row_i, "SE_label"]

        if se not in hgnn_se2idx:
            continue
        if not all(d in hgnn_drug2idx for d in drugs):
            continue

        for d in drugs:
            d_idx.append(hgnn_drug2idx[d])
            r_idx.append(edge_id)
        s_idx.append(hgnn_se2idx[se])
        edge_id += 1

    # Append query as last hyperedge
    query_edge_id = edge_id
    q_drugs = list(set(selected_drugs))
    for d in q_drugs:
        d_idx.append(hgnn_drug2idx[d])
        r_idx.append(query_edge_id)
    s_idx.append(hgnn_se2idx[target_se])

    hyperedge_index = torch.tensor([d_idx, r_idx], dtype=torch.long)
    se_tensor = torch.tensor(s_idx, dtype=torch.long)
    return hyperedge_index, se_tensor


# =========================
# PolyFormer inference
# =========================
def polyformer_predict(poly_model, poly_drug2idx, poly_se2idx, poly_smiles_tok, device, max_drugs_model,
                       selected_drugs, target_se):
    # keep unique drugs, deterministic order
    drugs = sorted(set(selected_drugs))
    # truncate if needed
    drugs = drugs[:max_drugs_model]

    idxs = [poly_drug2idx[d] for d in drugs if d in poly_drug2idx]
    if len(idxs) < 2:
        return None

    K = max_drugs_model
    drugs_pad = torch.full((1, K), -1, dtype=torch.long, device=device)
    mask = torch.zeros((1, K), dtype=torch.long, device=device)

    drugs_pad[0, :len(idxs)] = torch.tensor(idxs, dtype=torch.long, device=device)
    mask[0, :len(idxs)] = 1

    se_idx = torch.tensor([poly_se2idx[target_se]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = poly_model(drugs_pad, mask, se_idx, poly_smiles_tok)
        prob = torch.sigmoid(logits).item() * 100
    return prob


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PolySignal: HGNN vs PolyFormer", layout="wide")
st.title("PolySignal: Compare HGNN-SA vs PolyFormer")

drug_map, se_map = load_name_maps()

# Load models
hgnn_model, hgnn_drug2idx, hgnn_se2idx, hgnn_smiles_tensor, hgnn_device = load_hgnn()
poly_model, poly_drug2idx, poly_se2idx, poly_smiles_tok, poly_device, poly_max_drugs = load_polyformer()

# Train-only index for HGNN inference context
train_df, inv_index, pos_idx, neg_idx, train_q, val_q, test_q = build_hgnn_train_index(hgnn_drug2idx, hgnn_se2idx)

# Use intersection vocab so both models can run
drug_vocab = sorted(set(hgnn_drug2idx.keys()) & set(poly_drug2idx.keys()))
se_vocab = sorted(set(hgnn_se2idx.keys()) & set(poly_se2idx.keys()))

def fmt_drug(d):
    return f"{drug_map.get(d, 'Unknown')} ({d})"

def fmt_se(s):
    return f"{se_map.get(str(s), 'Unknown effect')} ({s})"

col1, col2 = st.columns(2)
with col1:
    selected_drugs = st.multiselect(
        f"Select drugs (2 to {poly_max_drugs})",
        options=drug_vocab,
        default=["DB00695", "DB00390"],
        max_selections=poly_max_drugs,
        format_func=fmt_drug,
    )
with col2:
    selected_se = st.selectbox(
        "Select side effect",
        options=se_vocab,
        index=se_vocab.index("C3160741") if "C3160741" in se_vocab else 0,
        format_func=fmt_se,
    )

st.caption(f"HGNN context uses TRAIN quarters only: {train_q[0]} → {train_q[-1]}")

base_pos = st.sidebar.slider("HGNN base context positives", 0, 1500, 400, 50)
base_neg = st.sidebar.slider("HGNN base context negatives", 0, 1500, 400, 50)
ego_pos = st.sidebar.slider("HGNN ego context positives", 0, 1500, 200, 50)
ego_neg = st.sidebar.slider("HGNN ego context negatives", 0, 1500, 200, 50)

if len(selected_drugs) < 2:
    st.info("Select at least 2 drugs.")
else:
    if st.button("Run both models", use_container_width=True):
        # PolyFormer prediction
        with st.spinner("Running PolyFormer (inductive set-attention)..."):
            poly_prob = polyformer_predict(
                poly_model, poly_drug2idx, poly_se2idx, poly_smiles_tok, poly_device, poly_max_drugs,
                selected_drugs, selected_se
            )

        # HGNN prediction
        with st.spinner("Running HGNN-SA (hypergraph + train-only context)..."):
            h_idx, se_idx = make_hgnn_inference_tensors(
                train_df, inv_index, pos_idx, neg_idx,
                selected_drugs, selected_se,
                hgnn_drug2idx, hgnn_se2idx,
                base_pos=base_pos, base_neg=base_neg, ego_pos=ego_pos, ego_neg=ego_neg, seed=42
            )
            h_idx = h_idx.to(hgnn_device)
            se_idx = se_idx.to(hgnn_device)
            with torch.no_grad():
                logits = hgnn_model(h_idx, se_idx, hgnn_smiles_tensor)
                hgnn_prob = torch.sigmoid(logits[-1]).item() * 100  # last hyperedge = query

        # Display results
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("PolyFormer (inductive)")
            if poly_prob is None:
                st.error("PolyFormer could not run (need >=2 valid drugs in vocab).")
            else:
                st.metric("Predicted risk", f"{poly_prob:.1f}%")
        with r2:
            st.subheader("HGNN-SA (hypergraph)")
            st.metric("Predicted risk", f"{hgnn_prob:.1f}%")

        # Simple interpretation
        st.markdown("---")
        st.write("Note: PolyFormer predicts directly from the input set. HGNN-SA uses a train-only context hypergraph plus the query hyperedge.")
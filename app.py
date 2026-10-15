import sys
from pathlib import Path

# Ensure root directory is on python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    DRUGBANK_CSV,
    SIDE_EFFECTS_CSV,
    POLYFORMER_SYNTH_C_CKPT,
    MODELS_DIR,
    DEFAULT_MAX_SMILES_LEN,
    DEFAULT_MAX_DRUGS,
)
from src.data.smiles import build_smiles_tensor, load_drugbank_smiles
from src.data.conformer_3d import compute_drug_features, TOTAL_DRUG_FEAT_DIM
from src.models.polyformer import PolyFormerLearnableSE
from src.models.inductive_polyformer import InductivePolyFormer

# Model Checkpoints
INDUCTIVE_CKPT = MODELS_DIR / "inductive_polyformer_3d.pt"
BASELINE_CKPT = POLYFORMER_SYNTH_C_CKPT
SIDE_EFFECTS_UNIQUE_CSV = str(SIDE_EFFECTS_CSV)
DRUGBANK_PATH = str(DRUGBANK_CSV)
MAX_SMILES_LEN = DEFAULT_MAX_SMILES_LEN


# =========================
# Caching loaders
# =========================
@st.cache_data
def load_name_maps():
    drugs_df = pd.read_csv(DRUGBANK_PATH, usecols=["DrugBank ID", "Name", "SMILES"])
    drug_map = dict(zip(drugs_df["DrugBank ID"], drugs_df["Name"]))
    drug2smiles = dict(zip(drugs_df["DrugBank ID"], drugs_df["SMILES"]))

    se_df = pd.read_csv(SIDE_EFFECTS_UNIQUE_CSV, usecols=["umls_cui_from_meddra", "side_effect_name"])
    se_map = dict(zip(se_df["umls_cui_from_meddra"].astype(str), se_df["side_effect_name"]))
    return drug_map, drug2smiles, se_map


@st.cache_resource
def _build_smiles_tensor(drug2idx, device):
    tensor, _, vocab_size = build_smiles_tensor(
        drug2idx, max_len=MAX_SMILES_LEN, device=device
    )
    return tensor, vocab_size


@st.cache_resource
def load_active_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if Inductive 3D Model exists
    if INDUCTIVE_CKPT.exists():
        ckpt = torch.load(INDUCTIVE_CKPT, map_location=device)
        se2idx = ckpt["se2idx"]
        model = InductivePolyFormer(
            num_ses=len(se2idx),
            in_drug_dim=ckpt.get("in_drug_dim", TOTAL_DRUG_FEAT_DIM),
            d_model=ckpt.get("d_model", 128),
            nhead=8,
            num_layers=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return {
            "type": "inductive_3d",
            "model": model,
            "se2idx": se2idx,
            "device": device,
            "max_drugs": ckpt.get("max_drugs", DEFAULT_MAX_DRUGS),
        }

    # Fallback to Baseline Scheme C PolyFormer
    ckpt = torch.load(BASELINE_CKPT, map_location=device)
    drug2idx = ckpt["drug2idx"]
    se2idx = ckpt["se2idx"]
    smiles_tok, rebuilt_vocab_size = _build_smiles_tensor(drug2idx, device)

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

    return {
        "type": "baseline",
        "model": model,
        "drug2idx": drug2idx,
        "se2idx": se2idx,
        "smiles_tok": smiles_tok,
        "device": device,
        "max_drugs": int(ckpt.get("max_drugs_model", 16)),
    }


def polyformer_predict(bundle, selected_drugs, target_se, drug2smiles):
    se2idx = bundle["se2idx"]
    if target_se not in se2idx:
        return None, "Side effect not in this model vocabulary"

    drugs = sorted(set(selected_drugs))[: bundle["max_drugs"]]
    device = bundle["device"]

    if bundle["type"] == "inductive_3d":
        # Pure Inductive 3D Inference: works for ANY drug in DrugBank!
        K = bundle["max_drugs"]
        feat_pad = np.zeros((1, K, TOTAL_DRUG_FEAT_DIM), dtype=np.float32)
        mask = torch.zeros((1, K), dtype=torch.long, device=device)

        for idx, d in enumerate(drugs):
            smiles = drug2smiles.get(d, "C")
            feat = compute_drug_features(smiles)
            feat_pad[0, idx] = feat
            mask[0, idx] = 1

        drug_feats = torch.from_numpy(feat_pad).to(device)
        se_idx = torch.tensor([se2idx[target_se]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = bundle["model"](drug_feats, mask, se_idx)
            prob = torch.sigmoid(logits).item() * 100
        return prob, None

    else:
        # Baseline transductive lookup fallback
        drug2idx = bundle["drug2idx"]
        idxs = [drug2idx[d] for d in drugs if d in drug2idx]
        if len(idxs) < 2:
            return None, "Need at least 2 drugs in this baseline vocabulary"

        K = bundle["max_drugs"]
        drugs_pad = torch.full((1, K), -1, dtype=torch.long, device=device)
        mask = torch.zeros((1, K), dtype=torch.long, device=device)
        drugs_pad[0, :len(idxs)] = torch.tensor(idxs, dtype=torch.long, device=device)
        mask[0, :len(idxs)] = 1
        se_idx = torch.tensor([se2idx[target_se]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = bundle["model"](drugs_pad, mask, se_idx, bundle["smiles_tok"])
            prob = torch.sigmoid(logits).item() * 100
        return prob, None



# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="PolyFormer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem; padding-bottom: 2.5rem;
        max-width: 1180px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 35%, #0f172a 100%);
    }
    .hero {
        background: linear-gradient(120deg, #0c4a6e 0%, #1d4ed8 55%, #2563eb 100%);
        border-radius: 18px;
        padding: 2rem 2.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.45);
    }
    .hero h1 {
        margin: 0; font-size: 2.75rem; font-weight: 800; letter-spacing: -0.03em;
        color: #f8fafc; line-height: 1.1;
    }
    .hero p {
        margin: 0.65rem 0 0; font-size: 1.125rem; line-height: 1.45;
        color: rgba(226, 232, 240, 0.92); max-width: 36rem;
    }
    .panel-title {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #94a3b8; margin: 0 0 1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        padding: 1.75rem 1.75rem 1.5rem !important;
        background: rgba(30, 41, 59, 0.55) !important;
        border-color: rgba(148, 163, 184, 0.22) !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important; font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stMultiSelect"] {
        margin-bottom: 1.85rem !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] {
        min-height: 7rem !important;
    }
    /* Scrollable value box; tags flow left-to-right at natural width */
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        min-height: 7rem !important;
        max-height: 20rem !important;
        height: auto !important;
        overflow-x: hidden !important;
        overflow-y: scroll !important;
        scrollbar-gutter: stable;
        padding: 0.85rem 2.75rem 0.85rem 1rem !important;
        font-size: 1rem !important;
        align-items: flex-start !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:first-child {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        align-content: flex-start !important;
        justify-content: flex-start !important;
        gap: 0.4rem 0.5rem !important;
        flex: 1 1 auto !important;
        min-width: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        grid-template-columns: unset !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div::-webkit-scrollbar {
        width: 10px;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        margin: 4px 0;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div::-webkit-scrollbar-thumb {
        background: #64748b;
        border-radius: 8px;
        border: 2px solid rgba(15, 23, 42, 0.5);
    }
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        scrollbar-width: thin;
        scrollbar-color: #64748b rgba(15, 23, 42, 0.5);
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        margin: 0 !important;
        padding: 0.35rem 0.5rem !important;
        font-size: 0.92rem !important;
        line-height: 1.35 !important;
        width: fit-content !important;
        max-width: 100% !important;
        flex: 0 0 auto !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    div[data-testid="stSelectbox"] {
        margin-bottom: 1.5rem !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 3.5rem !important;
        padding: 0.65rem 1rem !important;
        font-size: 1rem !important;
    }
    div[data-testid="column"] div[data-testid="stButton"] button {
        min-height: 3.25rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
        margin-top: 0.25rem;
    }
    .score-panel {
        min-height: 22rem;
        border-radius: 18px;
        padding: 2rem 1.5rem;
        text-align: center;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: linear-gradient(165deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(96, 165, 250, 0.35);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .score-label {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; color: #93c5fd; margin-bottom: 0.75rem;
    }
    .score-value {
        font-size: 5.25rem; font-weight: 800; line-height: 1;
        color: #f8fafc; letter-spacing: -0.04em;
    }
    .score-value span.pct {
        font-size: 2.25rem; font-weight: 600; color: #94a3b8; margin-left: 0.15rem;
    }
    .score-hint {
        font-size: 1.05rem; color: #94a3b8; margin-top: 0.75rem; max-width: 16rem; line-height: 1.4;
    }
    .score-meta {
        margin-top: 1.75rem; padding-top: 1.25rem;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        font-size: 0.9rem; color: #cbd5e1; width: 100%;
    }
    .score-meta strong { color: #e2e8f0; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

drug_map, drug2smiles, se_map = load_name_maps()

try:
    bundle = load_active_model()
except FileNotFoundError:
    st.error("Missing model checkpoint. Train inductive model with: python src/train_inductive.py")
    st.stop()

# Curated Master Drug Library: All 11,912 standardized DrugBank medications!
drug_vocab = sorted([d for d, s in drug2smiles.items() if isinstance(s, str) and len(s) > 0])
se_vocab = sorted(bundle["se2idx"].keys())
max_drugs = bundle["max_drugs"]


def fmt_drug(d):
    return f"{drug_map.get(d, 'Unknown')} ({d})"


def fmt_se(s):
    return f"{se_map.get(str(s), 'Unknown effect')} ({s})"


model_badge = (
    '<span style="background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.4); '
    'color: #4ade80; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;">'
    '✨ 3D-Aware Inductive Model Active</span>'
    if bundle["type"] == "inductive_3d"
    else '<span style="background: rgba(234, 179, 8, 0.2); border: 1px solid rgba(234, 179, 8, 0.4); '
    'color: #facc15; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;">'
    'ℹ️ Baseline Model Active</span>'
)

st.markdown(
    f"""
    <div class="hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1>PolyFormer</h1>
                <p>Higher-order drug–drug interaction risk prediction.</p>
            </div>
            <div>{model_badge}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

col_query, col_score = st.columns([11, 9], gap="large")

with col_query:
    with st.container(border=True):
        st.markdown('<p class="panel-title">Build query (11,912 Validated Drugs)</p>', unsafe_allow_html=True)
        selected_drugs = st.multiselect(
            "Drugs",
            options=drug_vocab,
            default=["DB00472", "DB00752"] if "DB00472" in drug_vocab else drug_vocab[:2],
            max_selections=max_drugs,
            format_func=fmt_drug,
            placeholder="Search all 11,912 drugs by name (e.g. Metformin, Warfarin)…",
        )
        selected_se = st.selectbox(
            "Side effect",
            options=se_vocab,
            index=se_vocab.index("C0699828") if "C0699828" in se_vocab else 0,
            format_func=fmt_se,
        )
        run_predict = st.button("Predict", type="primary", use_container_width=True)

if run_predict:
    if len(selected_drugs) < 2:
        st.session_state.last_prediction = {
            "prob": None,
            "err": "Select at least two drugs.",
            "n_drugs": len(selected_drugs),
            "se_name": se_map.get(str(selected_se), selected_se),
        }
    else:
        prob, err = polyformer_predict(bundle, selected_drugs, selected_se, drug2smiles)
        st.session_state.last_prediction = {
            "prob": prob,
            "err": err,
            "n_drugs": len(selected_drugs),
            "se_name": se_map.get(str(selected_se), selected_se),
        }


with col_score:
    pred = st.session_state.last_prediction
    if pred is None:
        score_html = (
            '<div class="score-hint">Choose drugs and a side effect, then click '
            '<strong>Predict</strong> to run PolyFormer.</div>'
        )
        meta = ""
    elif pred.get("err") and pred["prob"] is None:
        score_html = f'<div class="score-hint">{pred["err"]}</div>'
        meta = ""
    elif pred.get("err"):
        score_html = '<div class="score-value">—</div>'
        meta = (
            f'<div class="score-meta"><strong>{pred["n_drugs"]}</strong> drugs · '
            f'<strong>{pred["se_name"]}</strong></div>'
        )
    else:
        score_html = (
            f'<div class="score-value">{pred["prob"]:.1f}<span class="pct">%</span></div>'
        )
        meta = (
            f'<div class="score-meta"><strong>{pred["n_drugs"]}</strong> drugs · '
            f'<strong>{pred["se_name"]}</strong></div>'
        )

    st.markdown(
        f"""
        <div class="score-panel">
            <div class="score-label">PolyFormer score</div>
            {score_html}
            {meta}
        </div>
        """,
        unsafe_allow_html=True,
    )

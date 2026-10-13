"""
Global configuration and directory paths for Higher-Order Drug-Drug Interaction.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
EXPLANATIONS_DIR = ROOT_DIR / "explanations"

# Primary Data Files
HODDI_CSV = DATA_DIR / "hoddi_merged.csv"
DRUGBANK_CSV = DATA_DIR / "Drugbank_ID_SMILE_all_structure links.csv"
DRUGBANK_SMILES_CSV = DATA_DIR / "DrugBankID2SMILES.csv"
SIDE_EFFECTS_CSV = DATA_DIR / "Side_effects_unique.csv"

# Synthetic Datasets
HODDI_SYNTH_B = DATA_DIR / "hoddi_synth_B.csv"
HODDI_SYNTH_C = DATA_DIR / "hoddi_synth_C.csv"
HODDI_SYNTH_D = DATA_DIR / "hoddi_synth_D.csv"

# Model Checkpoints
POLYFORMER_DEFAULT_CKPT = MODELS_DIR / "polyformer_model.pt"
POLYFORMER_SYNTH_C_CKPT = MODELS_DIR / "polyformer_synth_C.pt"
HGNN_DEFAULT_CKPT = MODELS_DIR / "HGNN_model.pt"

# Model Hyperparameters / Defaults
DEFAULT_MAX_DRUGS = 16
DEFAULT_MAX_SMILES_LEN = 256
DEFAULT_SEED = 42
DEFAULT_D_MODEL = 128
DEFAULT_NHEAD = 8
DEFAULT_NUM_LAYERS = 2

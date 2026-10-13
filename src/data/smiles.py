"""
SMILES loading, character vocabulary extraction, and tokenization.
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config import DRUGBANK_CSV, DEFAULT_MAX_SMILES_LEN


def load_drugbank_smiles(
    drugbank_csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Loads DrugBank ID -> SMILES string mapping.
    """
    path = drugbank_csv_path or DRUGBANK_CSV
    db = pd.read_csv(path, usecols=["DrugBank ID", "SMILES"])
    return dict(zip(db["DrugBank ID"], db["SMILES"]))


def build_smiles_vocab(
    drug2smiles: Dict[str, str],
) -> Tuple[Dict[str, int], int]:
    """
    Builds character-to-index mapping for SMILES strings.
    Index 0 is reserved for padding, and an <UNK> token is appended.
    """
    chars = set()
    for s in drug2smiles.values():
        if isinstance(s, str) and len(s) > 0:
            chars.update(list(s))
            
    char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 is padding
    char2idx["<UNK>"] = len(char2idx) + 1
    vocab_size = len(char2idx) + 1  # include padding index 0
    return char2idx, vocab_size


def build_smiles_tensor(
    drug2idx: Dict[str, int],
    drug2smiles: Optional[Dict[str, str]] = None,
    char2idx: Optional[Dict[str, int]] = None,
    max_len: int = DEFAULT_MAX_SMILES_LEN,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, int], int]:
    """
    Builds [num_drugs, max_len] token tensor for all drugs in drug2idx.
    
    Returns:
        smiles_tok: Tensor of shape [len(drug2idx), max_len]
        char2idx: Tokenizer dictionary mapping chars to ints
        vocab_size: Total vocabulary size including padding
    """
    if drug2smiles is None:
        drug2smiles = load_drugbank_smiles()
        
    if char2idx is None:
        char2idx, vocab_size = build_smiles_vocab(drug2smiles)
    else:
        vocab_size = len(char2idx) + 1

    unk = char2idx.get("<UNK>", len(char2idx))
    smiles_tok = np.zeros((len(drug2idx), max_len), dtype=np.int64)

    for drug, di in drug2idx.items():
        s = drug2smiles.get(drug, None)
        if not isinstance(s, str) or len(s) == 0:
            s = "C"  # Fallback carbon atom
        toks = [char2idx.get(ch, unk) for ch in s[:max_len]]
        smiles_tok[di, :len(toks)] = toks

    smiles_tensor = torch.from_numpy(smiles_tok)
    if device is not None:
        smiles_tensor = smiles_tensor.to(device)

    return smiles_tensor, char2idx, vocab_size

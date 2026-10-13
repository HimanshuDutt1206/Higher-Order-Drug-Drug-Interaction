"""
Dataset loading, quarter-based chronological splitting, and batch packing for HODDI.
"""

from typing import List, Tuple, Dict, Any, Optional, Iterator, Union
from pathlib import Path
import ast
import random
import numpy as np
import pandas as pd
import torch

from src.config import HODDI_CSV, DEFAULT_MAX_DRUGS
from src.data.smiles import build_smiles_tensor, load_drugbank_smiles


def parse_drug_ids(raw_val: Any) -> List[str]:
    """
    Parses DrugBankID field into a deduplicated list of drug ID strings.
    Handles lists, strings representing python lists, or single strings.
    """
    if isinstance(raw_val, list):
        drugs = raw_val
    elif isinstance(raw_val, str):
        try:
            drugs = ast.literal_eval(raw_val)
        except (ValueError, SyntaxError):
            drugs = [raw_val.strip()]
    else:
        drugs = []
    
    # Preserve first-seen order while deduplicating
    seen = set()
    out = []
    for d in drugs:
        d_str = str(d).strip()
        if d_str and d_str not in seen:
            seen.add(d_str)
            out.append(d_str)
    return out


def quarter_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits unique chronological quarters into train, validation, and test splits.
    """
    quarters = sorted(df["time"].unique().tolist())
    n = len(quarters)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return quarters[:train_end], quarters[train_end:val_end], quarters[val_end:]


def load_hoddi_data(
    data_csv: Optional[Union[str, Path]] = None,
    min_combo_size: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Loads HODDI dataset, parses drug IDs, maps binary labels, and builds vocabularies.
    
    Returns:
        df: Prepared DataFrame with columns 'DrugBankID', 'y', 'k', 'se_idx', 'time'
        drug2idx: Drug string -> int index
        se2idx: Side effect string -> int index
    """
    path = data_csv or HODDI_CSV
    df = pd.read_csv(path, low_memory=False)
    
    df["DrugBankID"] = df["DrugBankID"].apply(parse_drug_ids)
    df["y"] = (df["hyperedge_label"] == 1).astype(np.int64)
    df["k"] = df["DrugBankID"].apply(len)

    if min_combo_size > 1:
        df = df[df["k"] >= min_combo_size].reset_index(drop=True)

    # Build drug vocabulary
    all_drugs = sorted({d for lst in df["DrugBankID"] for d in lst})
    drug2idx = {d: i for i, d in enumerate(all_drugs)}

    # Build side effect vocabulary
    df["SE_label"] = df["SE_label"].astype(str)
    all_ses = sorted(df["SE_label"].unique().tolist())
    se2idx = {s: i for i, s in enumerate(all_ses)}
    df["se_idx"] = df["SE_label"].map(se2idx).astype(np.int64)

    return df, drug2idx, se2idx


def pack_split(
    split_df: pd.DataFrame,
    drug2idx: Dict[str, int],
    max_drugs: int = DEFAULT_MAX_DRUGS,
) -> Dict[str, torch.Tensor]:
    """
    Packs any-size combinations into fixed [N, max_drugs] padded tensors.
    Combos are sorted and deterministically truncated to max_drugs.
    """
    N = len(split_df)
    drugs_pad = np.full((N, max_drugs), -1, dtype=np.int64)
    mask = np.zeros((N, max_drugs), dtype=np.int64)

    for i, lst in enumerate(split_df["DrugBankID"].tolist()):
        lst_sorted = sorted(set(lst))
        idxs = [drug2idx[d] for d in lst_sorted[:max_drugs] if d in drug2idx]

        drugs_pad[i, :len(idxs)] = idxs
        mask[i, :len(idxs)] = 1

    return {
        "drugs_pad": torch.from_numpy(drugs_pad),
        "mask": torch.from_numpy(mask),
        "se_idx": torch.from_numpy(split_df["se_idx"].to_numpy(np.int64)),
        "y": torch.from_numpy(split_df["y"].to_numpy(np.int64)),
    }


def batch_iter(
    tensors: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Yields mini-batches from packed tensors dictionary.
    """
    N = tensors["y"].shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield {k: v[b] for k, v in tensors.items()}


def permute_drug_order(drugs_pad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Data augmentation for set invariance: randomly shuffles real drug tokens within each row.
    """
    B, K = drugs_pad.shape
    out = drugs_pad.clone()
    for i in range(B):
        n = int(mask[i].sum().item())
        if n > 1:
            perm = torch.randperm(n, device=out.device)
            out[i, :n] = out[i, :n][perm]
    return out

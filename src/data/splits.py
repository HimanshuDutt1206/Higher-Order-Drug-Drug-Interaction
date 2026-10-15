"""
Benchmark split generators: Temporal split, Cold-Drug holdout split, and Disjoint split.
"""

from typing import Dict, List, Tuple, Set, Optional, Union, Any
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd

from src.config import HODDI_CSV, RESULTS_DIR, DEFAULT_SEED
from src.data.dataset import load_hoddi_data, quarter_split


def create_cold_drug_split(
    df: pd.DataFrame,
    test_drug_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Creates an inductive Cold-Drug Benchmark Split.
    
    1. Extracts all unique drugs present in the dataset.
    2. Randomly reserves `test_drug_ratio` (15%) of drugs as 'cold_test_drugs'.
    3. Categorizes records into:
       - 'train': Records where ALL drugs belong to seen train drugs.
       - 'val': Records containing only seen train drugs (for validation/early-stopping).
       - 'test_cold': Records containing AT LEAST ONE held-out unseen drug.
       - 'test_warm': Records containing ONLY seen train drugs from future quarters (for fair side-by-side comparison).
       
    Args:
        df: Prepared HODDI DataFrame.
        test_drug_ratio: Fraction of unique drugs to withhold for cold test set.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping split names ('train', 'val', 'test_cold', 'test_warm') to DataFrames,
        plus 'cold_drugs' and 'seen_drugs' lists.
    """
    rng = random.Random(seed)

    # 1. Gather all unique drugs
    all_drugs = sorted({d for lst in df["DrugBankID"] for d in lst})
    n_drugs = len(all_drugs)
    n_cold = int(n_drugs * test_drug_ratio)

    # Shuffle and pick cold drugs
    shuffled_drugs = list(all_drugs)
    rng.shuffle(shuffled_drugs)
    cold_drugs_set: Set[str] = set(shuffled_drugs[:n_cold])
    seen_drugs_set: Set[str] = set(shuffled_drugs[n_cold:])

    # 2. Partition records based on drug membership
    def has_cold_drug(drug_list: List[str]) -> bool:
        return any(d in cold_drugs_set for d in drug_list)

    df_cold = df[df["DrugBankID"].apply(has_cold_drug)].copy().reset_index(drop=True)
    df_seen = df[~df["DrugBankID"].apply(has_cold_drug)].copy().reset_index(drop=True)

    # 3. For seen-drug records, apply chronological quarter split to preserve temporal realism
    train_q, val_q, test_q = quarter_split(df_seen, train_frac=0.75, val_frac=0.10)
    
    train_df = df_seen[df_seen["time"].isin(train_q)].copy().reset_index(drop=True)
    val_df = df_seen[df_seen["time"].isin(val_q)].copy().reset_index(drop=True)
    test_warm_df = df_seen[df_seen["time"].isin(test_q)].copy().reset_index(drop=True)

    print(f"Cold-Drug Holdout Split Summary (seed={seed}):")
    print(f"  Total unique drugs: {n_drugs}")
    print(f"  Seen train drugs:   {len(seen_drugs_set)} ({len(seen_drugs_set)/n_drugs*100:.1f}%)")
    print(f"  Held-out cold drugs:{len(cold_drugs_set)} ({len(cold_drugs_set)/n_drugs*100:.1f}%)")
    print(f"  --- Record counts ---")
    print(f"  Train (seen drugs)  : {len(train_df)}")
    print(f"  Val (seen drugs)    : {len(val_df)}")
    print(f"  Test Warm (seen)    : {len(test_warm_df)}")
    print(f"  Test Cold (unseen)  : {len(df_cold)}")

    return {
        "train": train_df,
        "val": val_df,
        "test_cold": df_cold,
        "test_warm": test_warm_df,
        "cold_drugs": sorted(list(cold_drugs_set)),
        "seen_drugs": sorted(list(seen_drugs_set)),
    }


def save_split_manifest(
    splits: Dict[str, Any],
    manifest_path: Union[str, Path],
):
    """
    Saves split metadata and record indices to a JSON manifest for strict reproducibility.
    """
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test_cold": len(splits["test_cold"]),
        "n_test_warm": len(splits["test_warm"]),
        "cold_drugs_count": len(splits["cold_drugs"]),
        "seen_drugs_count": len(splits["seen_drugs"]),
        "cold_drugs": splits["cold_drugs"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved split manifest to '{path}'.")

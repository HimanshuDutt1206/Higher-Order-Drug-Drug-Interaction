"""
Generate synthetic-negative HODDI variants from positives only.

Schemes
-------
B: drug-corrupt (replace 1 drug, keep SE)
C: SE-corrupt (keep combo, replace SE)
D: drug-corrupt + subset-aware rejection

Does not modify data/hoddi_merged.csv (official scheme A).
"""

from __future__ import annotations

import ast
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "data" / "hoddi_merged.csv"
OUT_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42
MAX_ATTEMPTS = 50

Key = Tuple[FrozenSet[str], str]


def parse_drugs(x) -> List[str]:
    if isinstance(x, list):
        drugs = x
    elif isinstance(x, str):
        drugs = ast.literal_eval(x)
    else:
        raise TypeError(f"Unexpected DrugBankID type: {type(x)}")
    # unique, preserve first-seen order then we sort for canonical forms
    seen = set()
    out = []
    for d in drugs:
        d = str(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def combo_key(drugs: List[str]) -> FrozenSet[str]:
    return frozenset(drugs)


def combo_str(drugs: List[str]) -> str:
    return "_".join(sorted(drugs))


def drugs_list_str(drugs: List[str]) -> str:
    # Match hoddi_merged style: sorted unique DrugBank IDs as a Python list string
    return str(sorted(drugs))


def contains_positive_subset(
    candidate: FrozenSet[str],
    se: str,
    se_to_combos: Dict[str, List[FrozenSet[str]]],
) -> bool:
    """True if any known positive combo for this SE is a subset of candidate."""
    for pos in se_to_combos.get(se, []):
        if len(pos) <= len(candidate) and pos.issubset(candidate):
            return True
    return False


def load_positives(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DrugBankID"] = df["DrugBankID"].apply(parse_drugs)
    df["SE_label"] = df["SE_label"].astype(str)
    pos = df[df["hyperedge_label"] == 1].copy().reset_index(drop=True)
    pos["drugs_set"] = pos["DrugBankID"].apply(combo_key)
    return pos


def build_indexes(pos: pd.DataFrame):
    positive_keys: Set[Key] = set()
    se_to_combos: Dict[str, List[FrozenSet[str]]] = defaultdict(list)
    drug_pool: List[str] = sorted({d for drugs in pos["DrugBankID"] for d in drugs})
    se_pool: List[str] = sorted(pos["SE_label"].unique().tolist())

    for _, row in pos.iterrows():
        key = (row["drugs_set"], row["SE_label"])
        positive_keys.add(key)
        se_to_combos[row["SE_label"]].append(row["drugs_set"])

    # Deduplicate combo lists per SE
    for se in list(se_to_combos.keys()):
        se_to_combos[se] = list(set(se_to_combos[se]))

    return positive_keys, se_to_combos, drug_pool, se_pool


def make_positive_rows(pos: pd.DataFrame, synthetic_type: str) -> List[dict]:
    rows = []
    for _, row in pos.iterrows():
        drugs = list(row["DrugBankID"])
        rows.append(
            {
                "report_id": row["report_id"],
                "SE_label": row["SE_label"],
                "DrugBankID": drugs_list_str(drugs),
                "hyperedge_label": 1,
                "time": row["time"],
                "combo_str": combo_str(drugs),
                "synthetic_type": "positive",
                "source_report_id": "",
            }
        )
    return rows


def try_se_corrupt(
    drugs: List[str],
    se: str,
    se_pool: List[str],
    positive_keys: Set[Key],
    rng: random.Random,
) -> Optional[str]:
    drug_set = combo_key(drugs)
    others = [s for s in se_pool if s != se]
    if not others:
        return None
    for _ in range(MAX_ATTEMPTS):
        new_se = rng.choice(others)
        if (drug_set, new_se) not in positive_keys:
            return new_se
    return None


def generate_scheme(
    pos: pd.DataFrame,
    scheme: str,
    positive_keys: Set[Key],
    se_to_combos: Dict[str, List[FrozenSet[str]]],
    drug_pool: List[str],
    se_pool: List[str],
    rng: random.Random,
) -> Tuple[pd.DataFrame, dict]:
    assert scheme in {"B", "C", "D"}
    suffix = {"B": "nB", "C": "nC", "D": "nD"}[scheme]
    synthetic_type = {
        "B": "drug_corrupt",
        "C": "se_corrupt",
        "D": "drug_corrupt_subset_aware",
    }[scheme]

    rows = make_positive_rows(pos, synthetic_type)
    neg_emitted = 0
    failed = 0
    subset_rejects = 0  # D only: attempts rejected for subset
    # For B: among accepted negatives, how many still contain a positive subset
    contaminated_accepted = 0
    # For D: count how many times we skipped due to subset (approximate via attempts)
    subset_skip_events = 0

    for _, row in pos.iterrows():
        drugs = list(row["DrugBankID"])
        se = row["SE_label"]
        source_id = row["report_id"]

        if scheme == "C":
            new_se = try_se_corrupt(drugs, se, se_pool, positive_keys, rng)
            if new_se is None:
                failed += 1
                continue
            rows.append(
                {
                    "report_id": f"{source_id}_{suffix}",
                    "SE_label": new_se,
                    "DrugBankID": drugs_list_str(drugs),
                    "hyperedge_label": -1,
                    "time": row["time"],
                    "combo_str": combo_str(drugs),
                    "synthetic_type": synthetic_type,
                    "source_report_id": source_id,
                }
            )
            neg_emitted += 1
            continue

        # B or D
        # Measure contamination for stats: try until key-valid; for B accept even if contaminated
        found = None
        local_subset_skips = 0
        for _ in range(MAX_ATTEMPTS):
            idx = rng.randrange(len(drugs)) if drugs else 0
            current = set(drugs)
            candidates = [d for d in drug_pool if d not in current]
            if not candidates:
                break
            new_drug = rng.choice(candidates)
            new_drugs_list = list(drugs)
            new_drugs_list[idx] = new_drug
            new_set = combo_key(list(dict.fromkeys(new_drugs_list)))
            key = (new_set, se)
            if key in positive_keys:
                continue
            contaminated = contains_positive_subset(new_set, se, se_to_combos)
            if scheme == "D" and contaminated:
                local_subset_skips += 1
                continue
            found = (sorted(new_set), contaminated)
            break

        subset_skip_events += local_subset_skips
        if found is None:
            failed += 1
            if scheme == "D":
                subset_rejects += 1  # failed after subset-aware search
            continue

        new_drugs, contaminated = found
        if contaminated:
            contaminated_accepted += 1

        rows.append(
            {
                "report_id": f"{source_id}_{suffix}",
                "SE_label": se,
                "DrugBankID": drugs_list_str(new_drugs),
                "hyperedge_label": -1,
                "time": row["time"],
                "combo_str": combo_str(new_drugs),
                "synthetic_type": synthetic_type,
                "source_report_id": source_id,
            }
        )
        neg_emitted += 1

    n_pos = len(pos)
    stats = {
        "scheme": scheme,
        "description": synthetic_type,
        "n_positives": n_pos,
        "n_negatives": neg_emitted,
        "n_failed_negatives": failed,
        "subset_skip_events": subset_skip_events if scheme == "D" else 0,
        "subset_contamination_among_accepted_negatives": (
            contaminated_accepted / neg_emitted if neg_emitted else 0.0
        ),
        # For D, contamination among accepted should be ~0 by construction
        "subset_reject_rate_approx": (
            subset_skip_events / max(subset_skip_events + neg_emitted, 1)
            if scheme == "D"
            else 0.0
        ),
    }
    out_df = pd.DataFrame(rows)
    return out_df, stats


def verify_negatives(
    df: pd.DataFrame,
    positive_keys: Set[Key],
    se_to_combos: Dict[str, List[FrozenSet[str]]],
    subset_aware: bool,
) -> None:
    neg = df[df["hyperedge_label"] == -1]
    for _, row in neg.iterrows():
        drugs = parse_drugs(row["DrugBankID"])
        key = (combo_key(drugs), str(row["SE_label"]))
        assert key not in positive_keys, f"Negative collides with positive: {key}"
        if subset_aware:
            assert not contains_positive_subset(
                key[0], key[1], se_to_combos
            ), f"Subset-contaminated negative slipped through: {key}"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading positives from {SRC_CSV} ...")
    pos = load_positives(SRC_CSV)
    print(f"  positives: {len(pos)}")

    positive_keys, se_to_combos, drug_pool, se_pool = build_indexes(pos)
    print(f"  unique positive keys: {len(positive_keys)}")
    print(f"  drug pool: {len(drug_pool)} | SE pool: {len(se_pool)}")

    all_stats = []
    for scheme in ("B", "C", "D"):
        rng = random.Random(SEED)
        print(f"\nGenerating scheme {scheme} ...")
        df, stats = generate_scheme(
            pos, scheme, positive_keys, se_to_combos, drug_pool, se_pool, rng
        )
        out_path = OUT_DIR / f"hoddi_synth_{scheme}.csv"
        df.to_csv(out_path, index=False)
        print(
            f"  wrote {out_path} | rows={len(df)} "
            f"(pos={stats['n_positives']}, neg={stats['n_negatives']}, "
            f"failed={stats['n_failed_negatives']})"
        )
        print(
            f"  subset_contamination_among_accepted="
            f"{stats['subset_contamination_among_accepted_negatives']:.4f}"
        )
        verify_negatives(
            df,
            positive_keys,
            se_to_combos,
            subset_aware=(scheme == "D"),
        )
        # Spot-check sample of B contamination rate already in stats
        all_stats.append(stats)

    stats_path = RESULTS_DIR / "synthetic_negative_generation_stats.csv"
    pd.DataFrame(all_stats).to_csv(stats_path, index=False)
    print(f"\nWrote {stats_path}")


if __name__ == "__main__":
    main()

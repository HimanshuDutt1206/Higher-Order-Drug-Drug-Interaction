"""
Build PolyFormer comparison table across negative-sampling schemes A/B/C/D.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REPORTS = {
    "A": {
        "path": RESULTS / "Evaluate_PolyFormer.txt",
        "description": "HODDI official (drug+SE corrupt)",
    },
    "B": {
        "path": RESULTS / "Evaluate_PolyFormer_synth_B.txt",
        "description": "drug_corrupt (keep SE)",
    },
    "C": {
        "path": RESULTS / "Evaluate_PolyFormer_synth_C.txt",
        "description": "se_corrupt (keep combo)",
    },
    "D": {
        "path": RESULTS / "Evaluate_PolyFormer_synth_D.txt",
        "description": "drug_corrupt_subset_aware",
    },
}

METRIC_PATTERNS = {
    "roc_auc": r"ROC-AUC \(Main Metric\)\s*:\s*([0-9.]+)",
    "pr_auc": r"PR-AUC\s*:\s*([0-9.]+)",
    "f1": r"F1-Score\s*:\s*([0-9.]+)",
    "accuracy": r"Accuracy\s*:\s*([0-9.]+)",
    "precision": r"Precision\s*:\s*([0-9.]+)",
    "recall": r"Recall\s*:\s*([0-9.]+)",
}


def parse_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    out = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Could not parse {key} from {path}")
        out[key] = float(m.group(1))
    return out


def main():
    stats_path = RESULTS / "synthetic_negative_generation_stats.csv"
    stats = None
    if stats_path.exists():
        stats = pd.read_csv(stats_path).set_index("scheme")

    rows = []
    for scheme, meta in REPORTS.items():
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing report for scheme {scheme}: {path}")
        metrics = parse_report(path)
        row = {
            "scheme": scheme,
            "description": meta["description"],
            **metrics,
        }
        if stats is not None and scheme in stats.index:
            s = stats.loc[scheme]
            row["n_positives"] = int(s["n_positives"])
            row["n_negatives"] = int(s["n_negatives"])
            row["n_failed_negatives"] = int(s["n_failed_negatives"])
            row["subset_contamination_among_accepted_negatives"] = float(
                s["subset_contamination_among_accepted_negatives"]
            )
            row["subset_reject_rate_approx"] = float(s["subset_reject_rate_approx"])
        elif scheme == "A":
            row["n_positives"] = ""
            row["n_negatives"] = ""
            row["n_failed_negatives"] = ""
            row["subset_contamination_among_accepted_negatives"] = ""
            row["subset_reject_rate_approx"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS / "synthetic_negative_comparison.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# PolyFormer vs Synthetic Negative Schemes",
        "",
        "Strict chronological 70/15/15 quarter split. Scheme **A** is the official HODDI negatives already in `hoddi_merged.csv` (existing PolyFormer report). Schemes **B/C/D** were regenerated from positives only and PolyFormer was retrained.",
        "",
        "| Scheme | Description | ROC-AUC | PR-AUC | F1 | Accuracy | Precision | Recall |",
        "| ------ | ----------- | ------- | ------ | -- | -------- | --------- | ------ |",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['scheme']} | {r['description']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['f1']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Generation notes (B/C/D)",
            "",
            "See `results/synthetic_negative_generation_stats.csv` for failed-negative counts and subset-contamination rates.",
            "",
            "Full walkthrough: `explanations/explanationNegativeSampling.md`.",
            "",
        ]
    )
    md_path = RESULTS / "synthetic_negative_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

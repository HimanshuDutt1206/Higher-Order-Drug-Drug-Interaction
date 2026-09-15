# PolyFormer vs Synthetic Negative Schemes

Strict chronological 70/15/15 quarter split. Scheme **A** is the official HODDI negatives already in `hoddi_merged.csv` (existing PolyFormer report). Schemes **B/C/D** were regenerated from positives only and PolyFormer was retrained.

| Scheme | Description | ROC-AUC | PR-AUC | F1 | Accuracy | Precision | Recall |
| ------ | ----------- | ------- | ------ | -- | -------- | --------- | ------ |
| A | HODDI official (drug+SE corrupt) | 0.9678 | 0.9672 | 0.8940 | 0.8988 | 0.9384 | 0.8536 |
| B | drug_corrupt (keep SE) | 0.7802 | 0.7987 | 0.6559 | 0.7098 | 0.8057 | 0.5530 |
| C | se_corrupt (keep combo) | 0.8905 | 0.8925 | 0.8157 | 0.8157 | 0.8159 | 0.8155 |
| D | drug_corrupt_subset_aware | 0.7923 | 0.8102 | 0.6598 | 0.7181 | 0.8320 | 0.5467 |

## Why the results look like this

The ranking **A ≫ C > D ≈ B** matches how hard each negative scheme is — not a random score drop.

### A (0.97) — easiest task
Official HODDI negatives change **both** the drug set **and** the side effect. Positives and negatives differ on two axes at once. The model can lean on coarse cues (“this SE doesn’t belong with this combo pattern”) and look very strong. High AUC here is partly **label easiness**, not pure pharmacology skill.

### C (0.89) — medium
Same drugs, different SE. The model must learn **SE specificity**: “this combo goes with SE*, not random SE'.” Harder than A (no free drug+SE double corruption), but easier than B/D because the combo structure is unchanged and many random SE swaps are obviously unrelated. So C sits in the middle.

### B (0.78) — hard
Same SE, swap only one drug. The model must decide whether **membership of that one drug** matters for that SE. Negatives look a lot like positives, so ranking them apart is much harder → big AUC drop (~0.19 from A).

About **0.56%** of accepted B negatives still contained a **known positive subset** for that SE (`synthetic_negative_generation_stats.csv`). Those are noisy / false negatives and hurt learning a bit more.

### D (0.79) — similar to B, slightly cleaner
Same hard membership task as B, but rejects candidates that still contain a known positive subset. Contaminated negatives go to ~0%.

AUC is **almost the same as B** (even a touch higher). That means:
- the subset bug was real but **small in rate** (~0.56%)
- cleaning it does not magically make the task easy
- D’s value is **label quality / scientific honesty**, not a big metric win

### Why recall is low on B/D (~0.55)
On hard membership negatives, at threshold 0.5 the model becomes more conservative: higher precision, lower recall. It misses more true toxic cases because “almost-positive” negatives pull the decision boundary.

### Review takeaway
> Official HODDI negatives (A) inflate performance; when negatives keep the same SE and only perturb drugs (B/D), PolyFormer AUC falls to ~0.78–0.79, showing much of A’s score came from easier synthetic labels — not from a solved clinical problem.

## Generation notes (B/C/D)

See `results/synthetic_negative_generation_stats.csv` for failed-negative counts and subset-contamination rates.

Full walkthrough: `explanations/explanationNegativeSampling.md`.

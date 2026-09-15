# Negative Sampling for Higher-Order DDIs

### A Complete Walkthrough of Schemes A–D (Paper Method + Our Stress Tests)

> **How to read this doc:** Same style as `explanationHODDI.md` / `explanationHGNN.md`. Assumes ML *ideas*, not formulas. Worked examples label **where every decision comes from**.
>
> **Companions:**
>
> - `explanationHODDI.md` — how FAERS became HODDI positives + official negatives
> - `explanationPolyformer.md` / `explanationHGNN.md` — models that *consume* these labels
> - `results/synthetic_negative_comparison.md` — PolyFormer metrics under each scheme

### Scheme cheat-sheet


| Scheme | Name | What changes | What stays | In our repo |
| ------ | ---- | ------------ | ---------- | ----------- |
| **A** | HODDI / original | **1 drug** + **1 SE** | neither | `data/hoddi_merged.csv` (paper method; original train) |
| **B** | Drug-corrupt | **1 drug** | SE fixed | `data/hoddi_synth_B.csv` |
| **C** | SE-corrupt | **1 SE** | combo fixed | `data/hoddi_synth_C.csv` |
| **D** | Drug-corrupt + subset-aware | **1 drug** + reject subset-contaminated candidates | SE fixed | `data/hoddi_synth_D.csv` |


### Primer — only ideas used here

1. **Positive** = a real FAERS-derived `(drug set, side-effect CUI)` that HODDI kept after standardization.
2. **Positive key** = `(frozenset(unique DrugBank IDs), SE_label)`. Order does not matter; duplicate drugs inside a list are ignored for the key.
3. **Closed-world assumption** = if we never saw that key as positive, treat a constructed sample as **negative** for training. This is a **proxy**, not a lab-proven “safe” label.
4. **Subset false negative** = if `{A,B}` already causes SE, then `{A,B,D}` + SE may still be toxic, but a careless corrupt of `{A,B,C}` → `{A,B,D}` can still be labeled negative.
5. **1:1 balance** = about as many negatives as positives, so the classifier is not pushed toward always predicting “toxic.”

---

## Table of Contents

1. [Why We Need Synthetic Negatives at All](#1-why-we-need-synthetic-negatives-at-all)
2. [Shared Rules for Every Scheme](#2-shared-rules-for-every-scheme)
3. [Scheme A — HODDI Paper / Original Dataset](#3-scheme-a--hoddi-paper--original-dataset)
4. [Scheme B — Drug-Corrupt](#4-scheme-b--drug-corrupt)
5. [Scheme C — SE-Corrupt](#5-scheme-c--se-corrupt)
6. [Scheme D — Drug-Corrupt + Subset-Aware](#6-scheme-d--drug-corrupt--subset-aware)
7. [The Subset Bug (Worked Example)](#7-the-subset-bug-worked-example)
8. [What Each Scheme Forces the Model to Learn](#8-what-each-scheme-forces-the-model-to-learn)
9. [How We Generate B/C/D in Code](#9-how-we-generate-bcd-in-code)
10. [PolyFormer Comparison Protocol](#10-polyformer-comparison-protocol)
11. [Generation Stats Snapshot](#11-generation-stats-snapshot)
12. [Why the PolyFormer Results Look Like This](#12-why-the-polyformer-results-look-like-this)
13. [How to Talk About This in a Review](#13-how-to-talk-about-this-in-a-review)

---

## 1. Why We Need Synthetic Negatives at All

FAERS / HODDI give us **reported adverse associations** (positives). They almost never give trustworthy:

> “This exact combo was taken and this side effect definitely did **not** happen.”

So every supervised binary model needs a **negative construction policy**. The HODDI paper’s policy is Scheme **A**. Our Review-2 extension keeps A as the baseline and adds **B/C/D** as controlled stress tests of that labeling choice.

```
POSITIVES (real reports, standardized)
        │
        ▼ synthetic negative policy (A or B or C or D)
        │
BALANCED labeled table → PolyFormer / HGNN training
```

---

## 2. Shared Rules for Every Scheme

Starting from positives in `hoddi_merged.csv` (`hyperedge_label == 1`):

1. Build global `POSITIVE_KEYS = {(frozenset(drugs), SE), ...}`.
2. Build pools: all drugs / all SEs that appear in positives.
3. For each positive, try to emit **one** negative (1:1 target).
4. Reject any candidate whose key is already in `POSITIVE_KEYS`.
5. Keep the positive’s **`time`** quarter on its paired negative (so chronological splits stay meaningful).
6. Use unique sorted drugs for `combo_str` / keys.
7. Seed `42` for B/C/D regeneration (reproducible).

Scheme **A** already lives inside `hoddi_merged.csv`; we **do not regenerate or overwrite** it.

---

## 3. Scheme A — HODDI Paper / Original Dataset

**This is what the paper does and what our original PolyFormer / HGNN models trained on.**

### Algorithm

For each positive `(drugs, SE)`:

1. Replace **one random drug** with another from the complement set.
2. Replace **the side effect** with another SE.
3. Accept only if `(new_drugs, new_SE) ∉ POSITIVE_KEYS`.
4. Label `-1`, keep class balance ≈ 1:1.

```
POSITIVE:  {A, B, C} + SE*
                │
                ├─ replace one drug → {A, B, D}
                └─ replace SE       → SE'
                │
                ▼
NEGATIVE:  {A, B, D} + SE'     (if not already positive)
```

### What the model learns

Roughly: “this **exact** corrupted pair `(combo, SE)` is not a known report.”  
Changing **both** axes often makes negatives easy to spot (distributional cues on SE and membership).

### Failure modes

- Easy / optimistic metrics (corruption pattern leakage).
- Subset false negatives still possible if the new SE happens to still be caused by a remaining subset (less direct than B, but not impossible).
- Negatives are **not** clinically confirmed safe.

### When to cite in reviews

Baseline story: “We follow HODDI’s official negative construction for primary results.”

---

## 4. Scheme B — Drug-Corrupt

### Algorithm

For each positive `(drugs, SE)`:

1. Replace **one random drug** with a drug **not already in the combo**.
2. **Keep the same SE**.
3. Reject if `(new_drugs, SE) ∈ POSITIVE_KEYS`.

```
POSITIVE:  {A, B, C} + SE*
                │
                └─ replace one drug only → {A, B, D}
                │
                ▼
NEGATIVE:  {A, B, D} + SE*     (same side effect)
```

### What the model learns

Whether **membership of specific drugs** matters for a **fixed** side effect. Harder than A because SE identity is no longer a free cue from the corruption.

### Failure modes

- **Subset bug is front-and-center** (see §7): if `{A,B}` already explains `SE*`, labeling `{A,B,D}` negative is wrong.
- In our generation run, about **0.56%** of accepted B negatives still contained a known positive subset for that SE (`subset_contamination_among_accepted_negatives` in the stats CSV).

### Repo fields

- File: `data/hoddi_synth_B.csv`
- `synthetic_type = drug_corrupt` on negatives; positives marked `positive`
- Negative `report_id` like `{source}_nB`

---

## 5. Scheme C — SE-Corrupt

### Algorithm

For each positive `(drugs, SE)`:

1. **Keep the exact drug combo**.
2. Replace SE with another SE from the positive SE pool.
3. Reject if `(drugs, new_SE) ∈ POSITIVE_KEYS`.

```
POSITIVE:  {A, B, C} + SE*
                │
                └─ replace SE only → SE'
                │
                ▼
NEGATIVE:  {A, B, C} + SE'
```

### What the model learns

**Side-effect specificity** for the same polypharmacy set: “this combo is associated with SE*, not with a random other SE'.”

### Failure modes

- A real combo may cause **many** SEs; some SE' labels may be false negatives.
- Does **not** stress drug-membership the way B/D do.
- Subset-of-drugs issue is less relevant here (combo unchanged); the risk is multi-label SE truth.

### Repo fields

- File: `data/hoddi_synth_C.csv`
- `synthetic_type = se_corrupt`
- Negative `report_id` like `{source}_nC`

---

## 6. Scheme D — Drug-Corrupt + Subset-Aware

### Algorithm

Same as **B**, plus an extra reject rule:

> Reject candidate `(new_drugs, SE)` if **any known positive combo for that SE is a subset of `new_drugs`**.

```
POSITIVE:  {A, B, C} + SE*
                │
                └─ try replace C → D  → {A, B, D}
                │
                ├─ key already positive? → reject
                └─ exists pos P for SE* with P ⊆ {A,B,D}? → reject
                │
                ▼
NEGATIVE only if both checks pass
```

### What the model learns

Same membership stress as B, but with **fewer known-subset false negatives**. Still closed-world; still not “proven safe.”

### Failure modes

- Only blocks subsets that appear as **recorded positives**. Unknown minimal mechanisms still slip through.
- Slightly fewer negatives than positives if some rows fail after max attempts (we keep the positive anyway).

### Repo fields

- File: `data/hoddi_synth_D.csv`
- `synthetic_type = drug_corrupt_subset_aware`
- Negative `report_id` like `{source}_nD`

---

## 7. The Subset Bug (Worked Example)

Suppose the database contains:

- Positive₁: `{A, B}` + `SE*`  
- Positive₂: `{A, B, C}` + `SE*`

Now corrupt Positive₂ by replacing `C` with `D`:

| Scheme | Candidate | Labeled? | Correct? |
| ------ | --------- | -------- | -------- |
| A | `{A,B,D}` + some other `SE'` | often accepted | maybe OK if `SE'` unrelated; still synthetic |
| B | `{A,B,D}` + `SE*` | **accepted** (if key new) | **Wrong** if `{A,B}` already causes `SE*` |
| C | `{A,B,C}` + `SE'` | accepted if new | different failure mode |
| D | `{A,B,D}` + `SE*` | **rejected** because `{A,B} ⊆ {A,B,D}` for `SE*` | avoids this known false negative |

**Review sentence:**  
“Scheme D does not invent clinical negatives; it only refuses to call a sample negative when it still contains a **known** positive subset for that side effect.”

---

## 8. What Each Scheme Forces the Model to Learn


| Scheme | Main signal | Relative difficulty (typical) |
| ------ | ----------- | ----------------------------- |
| A | Joint `(combo, SE)` oddity after double corruption | Easiest / most optimistic |
| B | Drug membership for fixed SE | Harder; subset noise |
| C | SE identity for fixed combo | Medium; multi-SE truth risk |
| D | Drug membership for fixed SE, fewer known-subset FN | Hardest clean stress of B |

Do **not** expect D to beat A on AUC. A drop is often the scientifically honest outcome.

---

## 9. How We Generate B/C/D in Code

Script: [`src/generate_synthetic_negatives.py`](../src/generate_synthetic_negatives.py)

```bash
python src/generate_synthetic_negatives.py
```

Writes:

- `data/hoddi_synth_B.csv`
- `data/hoddi_synth_C.csv`
- `data/hoddi_synth_D.csv`
- `results/synthetic_negative_generation_stats.csv`

Extra columns vs original merge:

- `synthetic_type`
- `source_report_id` (empty on positives)

Verification inside the script:

- Every negative key ∉ `POSITIVE_KEYS`
- For D, no accepted negative contains a known positive subset for its SE

---

## 10. PolyFormer Comparison Protocol

Same model / split recipe as the original PolyFormer run:

- Chronological **70 / 15 / 15** quarters
- Learnable SE embeddings + SMILES CNN
- `MAX_DRUGS = 16`

| Scheme | Train data | Checkpoint | Report |
| ------ | ---------- | ---------- | ------ |
| A | `data/hoddi_merged.csv` | `models/polyformer_model.pt` (**not overwritten**) | `results/Evaluate_PolyFormer.txt` |
| B | `data/hoddi_synth_B.csv` | `models/polyformer_synth_B.pt` | `results/Evaluate_PolyFormer_synth_B.txt` |
| C | `data/hoddi_synth_C.csv` | `models/polyformer_synth_C.pt` | `results/Evaluate_PolyFormer_synth_C.txt` |
| D | `data/hoddi_synth_D.csv` | `models/polyformer_synth_D.pt` | `results/Evaluate_PolyFormer_synth_D.txt` |

```bash
python src/polyformer_train.py --data data/hoddi_synth_B.csv \
  --out_model models/polyformer_synth_B.pt \
  --out_report results/Evaluate_PolyFormer_synth_B.txt \
  --tag synth_B_drug_corrupt

# similarly for C and D, then:
python src/build_synth_comparison.py
```

Aggregated table: `results/synthetic_negative_comparison.md` (and `.csv`).

---

## 11. Generation Stats Snapshot

From `results/synthetic_negative_generation_stats.csv` (seed 42):


| Scheme | Positives | Negatives | Failed | Subset contamination among accepted neg | Approx subset-reject share (D) |
| ------ | --------- | --------- | ------ | --------------------------------------- | ------------------------------ |
| B | 111,072 | 111,072 | 0 | ~0.56% | — |
| C | 111,072 | 111,072 | 0 | 0% (N/A for SE-corrupt) | — |
| D | 111,072 | 111,061 | 11 | 0% (by construction) | ~1.3% of B-like attempts skipped for subset |

Interpretation: most B negatives are key-novel, but a non-zero slice is still subset-contaminated; D removes that slice at the cost of 11 positives without a paired negative.

PolyFormer metric numbers (test set, chronological split):


| Scheme | ROC-AUC | PR-AUC | F1 | Accuracy | Precision | Recall |
| ------ | ------- | ------ | -- | -------- | --------- | ------ |
| A (official HODDI) | **0.9678** | 0.9672 | 0.8940 | 0.8988 | 0.9384 | 0.8536 |
| B (drug-corrupt) | 0.7802 | 0.7987 | 0.6559 | 0.7098 | 0.8057 | 0.5530 |
| C (SE-corrupt) | 0.8905 | 0.8925 | 0.8157 | 0.8157 | 0.8159 | 0.8155 |
| D (subset-aware) | 0.7923 | 0.8102 | 0.6598 | 0.7181 | 0.8320 | 0.5467 |

**Summary:** A is easiest (optimistic). B/D are much harder membership tests (~0.78–0.79 AUC). C sits in between (SE specificity). D is similar to B overall, with cleaner negatives on the known-subset axis — not a free AUC win. Full table + narrative: `results/synthetic_negative_comparison.md`.

---

## 12. Why the PolyFormer Results Look Like This

The ranking **A ≫ C > D ≈ B** matches how hard each negative scheme is — not a random score drop.

### A (0.97) — easiest task

Official HODDI negatives change **both** the drug set **and** the side effect. Positives and negatives differ on two axes at once. The model can lean on coarse cues (“this SE doesn’t belong with this combo pattern”) and look very strong. High AUC here is partly **label easiness**, not pure pharmacology skill.

### C (0.89) — medium

Same drugs, different SE. The model must learn **SE specificity**: “this combo goes with SE*, not random SE'.” Harder than A (no free drug+SE double corruption), but easier than B/D because:

- the combo structure is unchanged
- many random SE swaps are obviously unrelated

So C sits in the middle.

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

---

## 13. How to Talk About This in a Review

**One-minute pitch**

> “HODDI only provides positives from FAERS, so negatives are synthetic. We keep the paper’s official scheme A for primary results, then regenerate harder schemes B/C/D from positives only and retrain PolyFormer to show how label assumptions move the metrics. Scheme D specifically reduces known subset false negatives.”

**Do say**

- Closed-world proxy labels
- Metric drops on harder negatives can be *good science*
- D reduces a *known* error mode; it does not certify safety

**Don’t say**

- “Our negatives are proven safe combinations”
- “Higher AUC on A means the model is clinically ready”

---

*Generator:* `src/generate_synthetic_negatives.py` · *Comparison:* `src/build_synth_comparison.py` · *Dataset story:* `explanations/explanationHODDI.md`

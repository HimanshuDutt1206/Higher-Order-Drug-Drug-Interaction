# Streamlit demo: sanity-check drug combinations

Use these in **`streamlit run app.py`** to see whether model scores line up with **well-known pharmacology**. They are **not** clinical recommendations—only quick checks that the HODDI-trained models behave sensibly on **held-out** `(drug set, side effect)` pairs.

## App availability (verified)

The dropdown lists **drugs and side effects that appear in both** `HGNN_model.pt` and `polyformer_model.pt` (intersection of training vocabs). A term can exist in `Side_effects_unique.csv` or MedDRA but still be **missing** from the app if it never made it into that intersection—for example **Bleeding tendency** (`C1458140`) and **Digoxin toxicity** (`C0274727`) are **not** selectable.

All cases below use IDs and side effects that **are** in the app. In the UI, pick by **display name** (name + CUI), e.g. `Bleeding time prolonged (C0151529)`.

| Side effect (pick this in app) | CUI | In HODDI | In Streamlit |
| ------------------------------ | --- | -------- | ------------ |
| Bleeding time prolonged | `C0151529` | Yes | Yes |
| Rhabdomyolysis | `C0035410` | Yes | Yes |
| Serotonin syndrome | `C0699828` | Yes | Yes |
| Blood potassium increased | `C0853760` | Yes | Yes |
| Bradyarrhythmia | `C0079035` | Yes | Yes |
| Hypotension | `C0020649` | Yes | Yes |

All listed **DrugBank IDs** appear somewhere in `hoddi_merged.csv` and in the app drug multiselect.

## How to use

1. Enter the **DrugBank IDs** (or pick from the multiselect by name).
2. Choose the **side effect** (UMLS CUI).
3. Click **Run both models**.
4. Compare **PolyFormer** and **HGNN-SA** scores (0–100% = sigmoid probability × 100).

### What “good” looks like (informal)

| Case type | Rough expectation |
| --------- | ----------------- |
| **Plausible interaction** (rows below marked *high*) | Both models often **> 50%**; they should usually **agree in direction** (both elevated vs both low). |
| **Negative control** (marked *low*) | Scores often **< 50%**; if both stay very high, the models are likely **overconfident** or using shortcuts (see README synthetic-negative section). |

Scores can still be wrong: training labels are **FAERS co-occurrence + synthetic negatives**, not proven causation. Two-drug queries are also **smaller** than typical training prescriptions.

## Dataset check (exact 2-drug hyperedge)

Each case below is **not** an **exact** row in `data/hoddi_merged.csv`: same **set** of two DrugBank IDs **and** the same `SE_label`. The model must **generalize**, not memorize that pair.

---

## Plausible interactions — expect relatively **high** scores

| Drugs (names) | DrugBank IDs | Side effect (app label) | CUI | Clinical note |
| ------------- | ------------ | ----------------------- | --- | ------------- |
| Warfarin + Acetylsalicylic acid (aspirin) | `DB00682`, `DB00945` | Bleeding time prolonged | `C0151529` | Anticoagulant + antiplatelet → bleeding risk |
| Warfarin + Ibuprofen | `DB00682`, `DB01050` | Bleeding time prolonged | `C0151529` | Anticoagulant + NSAID → bleeding risk |
| Simvastatin + Clarithromycin | `DB00641`, `DB01211` | Rhabdomyolysis | `C0035410` | CYP3A4 inhibition → statin myopathy risk |
| Atorvastatin + Clarithromycin | `DB01076`, `DB01211` | Rhabdomyolysis | `C0035410` | Same mechanism class |
| Fluoxetine + Phenelzine | `DB00472`, `DB00752` | Serotonin syndrome | `C0699828` | MAOI + SSRI — contraindicated |
| Lisinopril + Spironolactone | `DB00722`, `DB00421` | Blood potassium increased | `C0853760` | ACE inhibitor + K-sparing diuretic → hyperkalemia risk |
| Digoxin + Amiodarone | `DB00390`, `DB01118` | Bradyarrhythmia | `C0079035` | Combined cardiac therapy → rhythm/conduction risk (digoxin level/toxicity terms are not in the app vocab) |
| Sildenafil + Nitroglycerin | `DB00203`, `DB00862` | Hypotension | `C0020649` | PDE5 inhibitor + nitrate — severe hypotension risk |

---

## Negative controls — expect relatively **low** scores

| Drugs | DrugBank IDs | Side effect (app label) | CUI | Why it should be low |
| ----- | ------------ | ----------------------- | --- | -------------------- |
| Metformin + Acetaminophen | `DB00331`, `DB00316` | Serotonin syndrome | `C0699828` | No established serotonin syndrome link |
| Metformin + Acetaminophen | `DB00331`, `DB00316` | Rhabdomyolysis | `C0035410` | Unrelated to this pair |
| Ascorbic acid (vitamin C) + Metformin | `DB00126`, `DB00331` | Rhabdomyolysis | `C0035410` | Benign pair; unrelated AE |
| Warfarin + Aspirin | `DB00682`, `DB00945` | Serotonin syndrome | `C0699828` | **Wrong AE** for a bleeding pair (tests SE specificity) |
| Fluoxetine + Phenelzine | `DB00472`, `DB00752` | Bleeding time prolonged | `C0151529` | **Wrong AE** (dangerous pair, but not primarily bleeding) |

---

## Suggested workflow

1. Run **2–3 high** and **2–3 low** cases from the tables.
2. Note whether PolyFormer and HGNN **rank** highs above lows (more important than exact percentages).
3. If **low** cases still show ~80–90%, treat the demo as **not clinically calibrated**—consistent with strong scheme-A metrics but weaker B/D robustness (see `results/synthetic_negative_comparison.md`).

For HGNN-only behaviour (context sliders, train quarters), see [explanationHGNN.md §12](explanationHGNN.md#12-interactive-demo-inference-streamlit).

# Streamlit demo sanity checks (Scheme C)

Use with **`streamlit run app.py`** and checkpoint **`models/polyformer_synth_C.pt`**. Cases are chosen so **Scheme C** usually scores **plausible** rows higher than **negative** rows (verified locally; your % may differ slightly). Not clinical advice—scores are HODDI-style label probabilities, not causation.

Pick drugs and side effects by **display name (CUI)** in the app. Use every drug in a multi-drug row.

**Rough guide:** positives often **≈55–97%** on C; negatives often **≈8–38%** (wrong side effect or unrelated outcome for the same bag).

---

## Expected positives (relatively high scores)

| Drugs | DrugBank IDs | Side effect (app label) | CUI | ~Scheme C |
| ----- | ------------ | ----------------------- | --- | --------- |
| Fluoxetine + Phenelzine | `DB00472`, `DB00752` | Serotonin syndrome | `C0699828` | 97% |
| Sildenafil + Nitroglycerin | `DB00203`, `DB00862` | Hypotension | `C0020649` | 97% |
| Atorvastatin + Clarithromycin | `DB01076`, `DB01211` | Rhabdomyolysis | `C0035410` | 90% |
| Lisinopril + Spironolactone | `DB00722`, `DB00421` | Blood potassium increased | `C0853760` | 66% |
| Lisinopril + Spironolactone + Sildenafil + Torasemide + (DB12159) | `DB00722`, `DB00421`, `DB00203`, `DB00214`, `DB12159` | Blood potassium increased | `C0853760` | 66% |
| Digoxin + Amiodarone | `DB00390`, `DB01118` | Bradyarrhythmia | `C0079035` | 55% |

The five-drug hyperkalemia row is a **realistic polypharmacy bag** (same clinical pair as the two-drug row, plus co-meds); C scores similarly but better matches typical HODDI bag size.

---

## Expected negatives (relatively low scores)

| Drugs | DrugBank IDs | Side effect (app label) | CUI | ~Scheme C |
| ----- | ------------ | ----------------------- | --- | --------- |
| Fluoxetine + Phenelzine | `DB00472`, `DB00752` | Bleeding time prolonged | `C0151529` | 8% |
| Digoxin + Amiodarone | `DB00390`, `DB01118` | Bleeding time prolonged | `C0151529` | 9% |
| Warfarin + Acetylsalicylic acid (aspirin) | `DB00682`, `DB00945` | Serotonin syndrome | `C0699828` | 20% |
| Lisinopril + Spironolactone | `DB00722`, `DB00421` | Bleeding time prolonged | `C0151529` | 24% |
| Atorvastatin + Clarithromycin | `DB01076`, `DB01211` | Bleeding time prolonged | `C0151529` | 29% |
| Sildenafil + Nitroglycerin | `DB00203`, `DB00862` | Bleeding time prolonged | `C0151529` | 35% |
| Lisinopril + Spironolactone + Sildenafil + Torasemide + (DB12159) | `DB00722`, `DB00421`, `DB00203`, `DB00214`, `DB12159` | Bleeding time prolonged | `C0151529` | 39% |

Wrong-side-effect negatives use the **same drugs** as a positive row but an outcome that does not match textbook pharmacology for that pair.

**Avoid for this demo on C:** warfarin + aspirin + **bleeding** (often ~40%, weak positive); metformin / acetaminophen pairs (not in Scheme C drug vocab).

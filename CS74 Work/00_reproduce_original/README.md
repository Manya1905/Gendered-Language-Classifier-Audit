# 00 — Reproduce Original CS72 Results

Sanity checks that the original CS72 results can be recreated from the saved files,
before starting the audit experiments. Created 2026-11-15.

## Files

| File | What it is |
|---|---|
| `verify_checkpoint.py` | Rebuilds the original combined test set (same split logic and seed as CS72 `train_bert.py`) and runs the saved model on it. |
| `verify_checkpoint_output.txt` | Output of the 2026-11-15 run. |
| `original_test_predictions.csv` | Per-sentence results on the original test set (3,716 rows): text, job ID, industry, sample (`source`), flagged words, Claude's label (`suggested_label`), final label (`resolved_label`), the model's probabilities and prediction (`pred`: 0 = neutral, 1 = masculine, 2 = feminine). Reuse for the keyword baselines and E4. |
| `recompute_table4.py` | Recomputes paper Table 4 from `industry_analysis_sentences.csv` and compares it to the printed table. |
| `recompute_table4_output.txt` | Output of the 2026-11-15 run. |
| `check_duplicate_postings.py` | Measures repeated sentences/postings in the E5 data and checks whether removing duplicate postings changes the industry ranking. |
| `check_duplicate_postings_output.txt` | Output of the 2026-11-15 run. |

## How to re-run

Run from anywhere. The paths are relative to the project folder.

```
python verify_checkpoint.py          # ~10+ min on laptop CPU
python recompute_table4.py           # seconds
python check_duplicate_postings.py   # seconds
```

Inputs used:
- `From CS72/biased_multiclass/checkpoint-best/`: the original model
- `From CS72/Data/annotation_random.xlsx`, `annotation_biased.xlsx`: the annotation data
- `From CS72/From Remote Server/industry_analysis_sentences.csv`: the E5 predictions

## Results

1. **The checkpoint is the paper's model.** The rebuilt test set is 3,716 sentences from 120 postings, and the confusion matrix matches paper Table 3 exactly (macro-F1 0.8530, accuracy 0.9333).
2. **Paper Table 4 swaps the Healthcare and Education rows.** The correct ranking is Finance (77.0%) > Tech (67.4%) > Education (63.9%) > Healthcare (56.6%). The male-dominated vs female-dominated averages (72.2% vs 60.2%) are unaffected. The industry labels in the CSV were confirmed via job titles.
3. **Duplicate postings don't change the ranking.**
   - 36% of the E5 sentences are repeats, and 862 of the 10,000 postings are exact duplicates.
   - The ranking is the same with the duplicates removed.
   - A one-posting-per-company check is still to do in E5, using `company_id` from `From CS72/Data/postings.csv`.

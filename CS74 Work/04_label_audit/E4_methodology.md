# E4: Label audit — methodology

**Question:** does Claude's labeling match a human's? Every model in E1–E3 was trained and scored on Claude's labels, so if Claude's labels differ from a human reading, the scores measure agreement with Claude, not with people.

**Status:** done (2026-11-19), one labeler (me). Findings F25–F28.

## Sentences (`01_make_sheet.py`)

- Source: the **new evaluation set, unseen version** (`01_clean_reevaluation/data/new_eval.csv`, `flagged` and not `seen_in_annotated`). This replaces the original plan (clean test set): no model trained on these sentences, original included, so every model can be scored against my labels. E3 used the same set.
- **99 core sentences:** 33 per Claude label (masculine / feminine / neutral), random within each (seed 42).
- **20 extra:** Claude labeled neutral, BERT (clean) called masculine in at least 2 of 3 seeds (the F18 false alarms; 83 candidates). No overlap with the 99.
- At most 3 sentences per company across the sheet (119 sentences, 72 companies). 82 from enriched postings, 37 from random ones.
- All 119 shuffled together, so I couldn't tell which group a sentence came from.

## Labeling

- **Blind:** the sheet showed only the sentence and its masculine / feminine list words, which is exactly the information Claude got. No Claude label, reasoning, or model prediction. The key (`labeling/sheet_key.csv`) stayed closed until labeling was done.
- **Same definitions as Claude:** the Instructions tab copies the GENDERED / NEUTRAL rules and label definitions from the CS72 `prelabel.py` system prompt.
- 4 classes (masculine / feminine / both / neutral). "both" → masculine afterwards, as in training.
- Notes column for unclear cases. My labels: 97 neutral, 12 feminine, 9 masculine, 1 both.
- Long "sentences" (7 over 1,000 characters, F8) were labeled whole, as Claude did.

## Scoring (`02_score.py`)

- **A.** Me vs. Claude on the 99: agreement, Cohen's kappa, confusion matrix. Also agreement re-weighted to the natural mix of Claude labels among flagged unseen sentences (62% neutral / 19% / 19%), since 33/33/33 over-represents gendered sentences.
- **B.** My labels on the 20 extra sentences.
- **C.** Every model (original, BERT ×3, RoBERTa ×3) and keyword rule scored against my labels and against Claude's, on the same 99, using the saved E1 predictions (no models re-run). Accuracy and macro-F1, plus an "always neutral" reference.
- **D.** Disagreements by direction, by list word, and by industry, plus my notes.
- **E.** From my labeling note: how every labeler treats "analytical" across all unseen new-set sentences.
- Outputs: `results/e4_summary.md`, `results/e4_per_sentence.csv`.

## Results (short)

| | |
|---|---|
| Me vs. Claude (99) | 51.5% agreement, kappa 0.27; about 71% re-weighted |
| Direction | 47 of 50 disagreements are Claude gendered / me neutral |
| Models vs. Claude / vs. me | about 0.81–0.83 / 0.44–0.49 accuracy ("always neutral" gets 0.80 vs. me) |
| 20 false alarms | I said neutral for 18: the models over-call |
| "analytical" | models 31/34 masculine, Claude 24/34, me 1/8 |

## Limitations

- **One human labeler.** I can't tell whether Claude's bar is loose or mine is strict. `labeling/blank_labeling_sheet_for_others.xlsx` is a clean copy for more labelers; with 2+ people, human–human agreement becomes the yardstick for Claude.
- The stratified sample isn't the natural class mix, so kappa doesn't describe agreement on real data. The re-weighted agreement is an estimate.
- Small numbers of gendered labels on my side (8 masculine, 12 feminine), so per-class scores against my labels are noisy.
- Not done (optional): the Claude self-agreement check, and the 40 unlabeled test sentences.

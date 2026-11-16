# Findings Log

Running list of things discovered during the audit that should go in the CS74 report.
Each entry: what we found, the evidence, and where it goes in the report.
Newest entries at the bottom. Plans and to-dos stay in `Project_Proposal_Context.md`.

---

## F1. The original checkpoint reproduces the paper exactly
*2026-11-15 · Report: Methods (setup)*

- **Finding:** `From CS72/biased_multiclass/checkpoint-best/` is the model in the paper. Rebuilding the original test set gives 3,716 sentences from 120 postings, and the confusion matrix matches Table 3 exactly (macro-F1 0.8530, accuracy 0.9333).
- **Evidence:** `00_reproduce_original/verify_checkpoint_output.txt`
- **Why it matters:** confirms we're auditing the right model and can recreate the original split exactly.

## F2. The checkpoint was selected on the test set
*2026-11-14 · Report: E1, Corrections*

- **Finding:**
  - `train_bert.py` passes the test set as `eval_dataset` and uses `load_best_model_at_end=True`, so the best epoch was chosen by test F1.
  - The paper says "selected by validation F1".
- **Evidence:** `train_bert.py` (Trainer setup). Training log: epoch 4 test F1 0.8530 vs. epoch 3 0.8526.
- **Why it matters:** confirms gap 1, but the inflation from it is tiny (0.0004). Most of any E1 drop will come from leakage.

## F3. Labels are almost entirely Claude's, and "both" was never human-reviewed
*2026-11-14/15 · Report: E4, Limitations, Corrections*

- **Finding:**
  - Only 1 of 19,854 labels was changed by a human.
  - The CS72 `DESIGN_DECISIONS.md` says "the human makes all final labeling decisions".
  - `annotation_both_review.xlsx` (1,051 rows) was never filled in, so all 660 final "both" → masculine labels are Claude Sonnet's.
  - 165 flagged sentences were never labeled and defaulted to neutral.
- **Evidence:** the `Your Label (final)` column in both annotation files; the training log warning; the empty review workbook (in the Downloads CS72 folder).
- **Why it matters:** supports gap 3 and motivates E4.

## F4. The paper's random-vs-enriched comparison used different test sets
*2026-11-14 · Report: Corrections*

- **Finding:**
  - The paper compares random (0.83) with enriched (0.85).
  - The random model's 0.83 was measured on the random test set only (1,266 sentences).
  - On the same combined test set (3,716), the random model scores **0.79** (masculine F1 0.67).
- **Evidence:** `bert_rand_multi_8710080.out` (eval on 1,266); `bert_eval_8713971.out` (eval on 3,716).

## F5. Paper Table 4 swaps the Healthcare and Education rows
*2026-11-14 · Report: E5, Corrections*

- **Finding:**
  - The correct ranking is Finance (77.0%) > Tech (67.4%) > **Education (63.9%) > Healthcare (56.6%)**.
  - Education also has the most feminine language (73.3%), not Healthcare.
  - The male- vs female-dominated averages (72.2% vs 60.2%) are unchanged.
- **Evidence:** `00_reproduce_original/recompute_table4_output.txt`. Industry labels were confirmed via job titles.

## F6. The enriched sample is concentrated in a few companies' templates
*2026-11-15 · Report: E1 (dataset artifacts), Discussion, Future work*

- **Finding:** choosing the 75 most keyword-dense postings per industry repeatedly picked large employers whose long standard templates contain many list words.

  | | Random | Enriched |
  |---|---|---|
  | Postings | 300 | 300 |
  | Different companies | 269 | **166** |
  | Postings from companies with 2+ postings | 54 (18%) | **191 (64%)** |
  | Most postings from one company | 5 | **13 (BMO U.S., finance)** |
  | Sentences that appear more than once | 10.5% | **44.8%** |

- **Also:** the keyword score isn't adjusted for posting length, so it favors long postings. The enriched sample has 13,320 sentences vs. 6,534 for random, with the same 300 postings.
- **Why it matters:**
  - It explains the leakage (gap 2): repeat employers make train/test overlap almost unavoidable.
  - The model saw far fewer distinct examples than 13,320 rows suggest.
  - A model trained on repeated keyword-heavy templates has every reason to memorize keywords, which ties directly to the main research question.
- **Decision:** don't re-sample or build a new classifier. E1 keeps the same data and only changes the split. Report this finding plus a recommendation: cap 1–2 postings per company, adjust the keyword score for length, and remove duplicate sentences before choosing the sample.
- **E1 plan (revised 2026-11-15):** report (a) company split only, then (b) company split plus removing any val/test sentence that also appears in training. **Leave the training data as-is**, so step (b) changes only the evaluation side. Deduplicating within training would cut the enriched data from 13,320 rows to 9,284 unique sentences (masculine 1,353 → 1,053; feminine 1,170 → 814) and adds a second change, so it's an optional extra at most.
- **How much the test set shrinks:**
  - *Original split:* removing test sentences seen in the model's training data cuts 3,716 → 2,500 (−33%). Masculine 345 → 253; feminine 317 → 198. This confirms the proposal's "1/3 overlap".
  - *Company split (quick estimate, 5 seeds, 70/15/15):* only 0.4–3.4% of test sentences overlap with train. **The company split does most of the cleanup.**
  - *But* test-set size swung from 2,218 to 4,092 sentences across seeds, because a few companies have many postings. *E1 to do:* balance the company split by size and industry, not purely at random.
- **Evidence:** company counts from joining `job_id` to `From CS72/Data/postings.csv`. *To do: save this as a script in `00_reproduce_original/`.*

## F7. E5 data has heavy repetition, but it doesn't change the ranking
*2026-11-14 · Report: E5 (robustness check)*

- **Finding:**
  - 36% of the 213,779 E5 sentences are repeats, and 862 of the 10,000 postings are exact duplicates.
  - One finance employer (J. Galt) has 137 postings with the same masculine sentence.
  - Removing exact duplicates leaves the ranking unchanged.
- **Evidence:** `00_reproduce_original/check_duplicate_postings_output.txt`
- **To do in E5:** a one-posting-per-company check using `company_id`.

## F8. Sentence splitting failed on run-together text
*2026-11-14 · Report: Limitations, E3 method*

- **Finding:**
  - Postings whose line breaks were stripped produce "sentences" like `"…we serve.Position OverviewThe primary…"`.
  - 368 training sentences exceed 128 tokens (the longest is 598) and get cut off.
- **Decision:** keep the original segmentation. Only use sentences of 128 tokens or fewer in E3, and report the count of over-length test sentences.
- *To do: count over-length test sentences and how many of them are gendered.*

## F9. CS72 planned to tune the threshold on the test set (never used)
*2026-11-15 · Report: E5 background*

- **Finding:**
  - The CS72 `threshold_methodology.md` proposed tuning the threshold on the test set and claimed this caused "no information leakage". That plan wasn't used; the final pipeline uses argmax.
  - The same doc notes that Gaucher et al. tested whole ads, not single sentences, so the "any masculine sentence" posting rule has no direct support from the paper.
- **Why it matters:** E5 must tune on validation, if at all. The second point is good motivation for E5's threshold sweep.

## F10. Identical sentences got different labels
*2026-11-15 · Report: E4, Limitations*

- **Finding:** in the enriched sample, 151 sentences appear more than once with **different final labels** (after collapsing "both" → masculine), e.g. neutral in one posting and masculine in another.
- **Caveat:** some of these may come from the 165 failed sentences defaulting to neutral rather than true Claude inconsistency. *To do: separate the two cases and check the random sample too.*
- **Why it matters:** this is direct evidence about label consistency, for E4. It also means some "errors" on the test set could be the model disagreeing with one of two contradictory labels.
- **Likely cause (2026-11-16):** `prelabel.py` never sets `temperature`, so the API default (1.0) applied and Claude's labels were partly random. Identical sentences in different batches could get different labels. Also worth noting in Methods: the prompt shows Claude the flagged words, and it frames neutral as "ONLY" for purely structural usage, which leans borderline cases toward gendered.

## F11. Unflagged sentences pad the score, and the model rarely goes beyond the list
*2026-11-16 · Report: E1 (metrics), gap 4, main research question*

- **Finding (original test set):**
  - 61% of test sentences (2,253 / 3,716) are unflagged and auto-labeled neutral. The model predicts neutral for 2,239 of them (99.4%), so they're nearly free points.
  - Macro-F1 is 0.853 on all test sentences but **0.827 on flagged sentences only**.
  - The model calls only **14** unflagged sentences gendered, and all 14 count as errors. Several look genuinely gendered, e.g. "Cultivate a culture of high performance with emphasis on accountability, drive for results…" (masc 0.99) and "…passionate professionals who are dedicated wholeheartedly to our mission" (fem 0.97).
- **Why it matters:**
  - Gap 4 is visible in real data: going beyond the list is scored as a mistake.
  - The model goes beyond the list in only 14 of 2,253 cases (0.6%), which is early evidence of keyword reliance.
- **Decision:** don't remove unflagged sentences from training (that would be a new classifier). Instead, report flagged-only test scores as a headline metric in E1/E2.
- *To do: hand-label the 14 unflagged sentences predicted gendered, to quantify how many are real detections beyond the list.*
- **Evidence:** `00_reproduce_original/original_test_predictions.csv` (`flagged`, `pred` columns).

## F12. Two whole training postings are in the test set (split done per sample)
*2026-11-16 · Report: E1 (gap 2), Corrections*

- **Finding:**
  - `train_bert.py` splits the random and enriched samples *separately*. 4 postings are in both samples, and 3 of them landed in train on one side and test on the other.
  - Two affect the audited (enriched) model. "Program Manager II" (3902781031, 25 test sentences) and "Teacher Sp Ed – Special Education" (3904364647, 98 test sentences) are in its training data **and** in the combined test set (via the random sample).
  - That's **123 test sentences (3.3%)** the model saw, with the same labels, during training.
  - The third case ("Elementary School Teacher", 3884930741) was random-train / enriched-test, so it affects only the random model.
- **Why it matters:** this leakage is separate from boilerplate (F6). The combined test set's duplicate removal (`job_id` + `sentence_num`) only catches a posting that's in *both* test sets, not one that's in train on one side and test on the other.
- **E1 fix:** split both samples together as one pool, grouped by company.
- **Evidence:** a quick check reusing `verify_checkpoint.py`'s split functions. *To do: save it as a script in `00_reproduce_original/`.*

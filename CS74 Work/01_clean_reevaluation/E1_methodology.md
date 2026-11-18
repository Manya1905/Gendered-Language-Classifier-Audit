# E1 Methodology: Clean Re-evaluation

**Question:** How much of the original 0.85 macro-F1 is left once the data leakage is fixed?

**What E1 fixes:**
- **Gap 1:** the original model was chosen using the test set.
- **Gap 2:** some test sentences (and companies) also appeared in training.

The labels stay exactly the same. E1 only changes *how the data is split* and *how the model is chosen*.

*Status (2026-11-17): done. All steps ran; results are at the end of this file and in FINDINGS F15–F18.*

---

## Step 1: What gets split?

### We split companies, not sentences or postings

All of a company's postings go to the same group (train, val, or test). A company can never appear on both sides, so its reused boilerplate sentences can't leak from train into test.

Every annotated posting has a `company_id` (from the Kaggle `postings.csv`), so no posting is missing one.

| | Random sample | Enriched sample |
|---|---|---|
| Postings | 300 | 300 |
| Companies | 269 | 166 |
| Biggest company | 5 postings | 13 postings |

### We pool the random and enriched samples, then split once

We only *train* on enriched data, since that's the model being audited. But the random sample is still needed for **testing**:

1. **To compare with the original 0.85.** That score was measured on a test set that mixed random and enriched sentences, so the new test set needs both too.
2. **To prevent leakage across samples.** 23 companies have postings in *both* samples. If each sample were split separately, a company could land in train on the enriched side and in test on the random side. That's what happened in CS72 (FINDINGS F12).

So all 412 companies go into one list, and each is assigned to exactly one group:

| Group | Uses |
|---|---|
| Train | Only the **enriched** postings from train companies (the random ones are left out) |
| Val | Postings from val companies, used to choose the best epoch |
| Test | Postings from test companies, **both random and enriched** |

---

## Step 2: How companies are assigned to train / val / test

### The goal

Split the data about **70 / 15 / 15** (train / val / test).

### Problem 1: companies are different sizes

Example: 10 companies, and we want 2 of them in test.

| Company | Sentences |
|---|---|
| BMO | 500 |
| The other 9 | 50 each |

If we pick 2 companies at random:
- Picking BMO plus one other gives a test set of **550 sentences**. That's huge.
- Picking two small ones gives a test set of **100 sentences**. That's tiny.

So the test set's size depends on luck. On the real data, random tries gave test sets anywhere from **about 2,000 to 4,300 sentences**.

**Fix:** assign companies so each group ends up with the right *number of sentences*, not just the right *number of companies*.

### Problem 2: the test set should be a mini copy of the whole dataset

Getting the right size isn't enough; the test set also needs the right *mix*. The whole dataset is about 25% of each industry. With random picks, a test set could come out like this:

| Industry | Share of test set |
|---|---|
| Finance | 44% |
| Tech | 20% |
| Healthcare | 18% |
| Education | 18% |

Then the test score would mostly tell us how the model does on finance, not on job postings in general.

**Fix:** while assigning companies, keep every group close to the whole dataset's mix of:
- **industry** (tech / finance / healthcare / education)
- **label** (masculine / feminine / neutral), so the test set isn't short on masculine sentences
- **sample** (random / enriched), so both can be scored fairly

### How it's done

We use sklearn's `StratifiedGroupKFold`, a standard tool that does both fixes:
- Companies ("groups") are never split up.
- Each slice gets about the same size and the same mix.

It cuts the companies into 20 equal slices: **14 → train, 3 → val, 3 → test**. The split is made **once** (seed 42), saved to a file, and reused for everything, including all 3 training seeds.

*Staffing agencies (checked 2026-11-16): not a problem. The biggest companies are all real employers in one industry (BMO, CNA Insurance, UCSF, …). Staffing/recruiting firms have at most 3 postings each.*

*Note: balancing on labels uses Claude's labels only to arrange the split; it doesn't change any labels. CS72 did the same (it stratified by industry and "has gendered").*

### Result on the real data (seed 42)

| | Sentences | Companies | Tech | Finance | Healthcare | Education | Masculine | Enriched |
|---|---|---|---|---|---|---|---|---|
| **Whole dataset** | 19,854 | 412 | 26% | 22% | 27% | 25% | 8.5% | 67% |
| Train | 14,225 (72%) | 295 | 26% | 22% | 27% | 25% | 8.2% | 68% |
| Val | 2,822 (14%) | 55 | 24% | 23% | 28% | 24% | 10.5% | 66% |
| Test | 2,807 (14%) | 62 | 25% | 23% | 26% | 26% | 7.8% | 62% |

- **Size:** close to 70/15/15.
- **Industry:** within about 2 points everywhere.
- **Masculine and enriched:** a bit looser (within about 2–5 points). A company's sentences must stay together, so it can't be perfect.

*These numbers come from a quick check on the `.csv` copies of the data. The real split script will re-create and save them.*

---

## Step 3: Which sentences are removed from val and test
- Remove the **165 unlabeled sentences** from val and test (they stay in train, as in the original).
  - They are *not* removed before the split: they go with their company like every other sentence. Removing them first would also take them out of training, which would change the training data compared with CS72.
  - With the seed-42 split: **125 in train** (7 postings, kept as neutral), **0 in val**, **40 in test** (1 posting, dropped from both test versions).
  - The 40 test ones get hand-labeled blind during the E4 session, as a side note on how the models do on them (decided 2026-11-16). We decided against removing the 165 from training and retraining: it's ~1% of training data (too small to measure), comes from only 8 postings, and would mix a label question into E1.
  - In CS72 they were never removed. `train_bert.py` (`resolve_label`) defaulted them to neutral and only counted them. All 165 are in the enriched sample; 141 ended up in train and 24 in test.
- Two test versions, to show the effect of each fix: **(a)** company split only, and **(b)** company split plus removing val/test sentences that also appear in train.
- **"Same sentence" includes near-matches** (decided 2026-11-16): sentences are compared after lowercasing and collapsing extra spaces, since boilerplate often differs only in spacing or capitals.
- **Val is mixed** (random + enriched), like test, and there's only one version (decided 2026-11-16). It matches what test measures and what CS72 picked its epoch on. A second, enriched-only version would double the training runs for a choice that barely mattered in CS72 (F2).
- **Postings drawn into both samples** (4 postings): keep **one copy** (the enriched one) in val and test, so their sentences aren't graded twice. In train only the enriched copy is used anyway. With seed 42, 2 are in train and 2 are in val.
  - Keeping the enriched copy also means using its **labels**: Claude labeled each copy separately, and 7 of the 153 sentences got different labels (FINDINGS F10). The enriched copy matches what the model trains on.

## Step 4: Training

- Train on the **enriched** postings in the train group.
- Same hyperparameters as CS72: lr 2e-5, batch size 16, 4 epochs, warmup 0.1, weight decay 0.01, max length 128, inverse-frequency class weights, early stopping (patience 2).
- Choose the best epoch on **val**, not test.
- **3 seeds**, reported as mean ± SD. CS72 trained once (seed 42); with only ~220 masculine test sentences, one run can move macro-F1 by a point or two from luck alone.
- **RoBERTa (`roberta-base`) is trained the exact same way** (decided 2026-11-16): same `04_train.py`, same train/val files, same settings, same 3 seeds, epoch chosen on val. Only the model and its tokenizer change. Purpose: if RoBERTa, a stronger model, *also* relies on keywords, the shortcut likely comes from the data/labels rather than from BERT. Note for the report: RoBERTa's tokenizer splits text differently, so a slightly different number of sentences exceed 128 tokens.
- **Where it ran:** Discovery's GPU queue was backed up, so all runs use Dartmouth's no-queue CPU machines (8 cores per run, same `cs74` environment). This also matches CS72, whose model actually trained on CPU. Each run needs ~4 GB of memory and there's a per-user memory cap, so 6 runs didn't fit on one machine: **Polaris** ran BERT 42 and 44 and RoBERTa 44, and **Andes** ran BERT 43 and RoBERTa 42 and 43. (BERT 43 first crashed on Polaris with an out-of-memory error and was restarted from scratch on Andes.)
- **Full precision (no fp16).** CS72's job asked for fp16, but its GPU failed to load (a software error), so the original model actually trained on CPU in full precision. Ours do too.

## Step 5: Evaluation
- Run the test set **once** per model.
- Report macro-F1 and per-class F1 for: all sentences, random only, enriched only, and **flagged-only** (FINDINGS F11).
- **Original model:** it trained on 80% of the enriched postings, so most of the new test set is data it has already seen. So:
  - Compare the original model on its **own** test set (0.853) with the retrained models on the **new clean** test set.
  - For a fair head-to-head of all models on the same sentences, use the **new evaluation set** (step 6).

## Code plan

All scripts live in this folder. Each one saves its output to a file, so the next script just reads it.

| Script | Does | Saves | Runs on |
|---|---|---|---|
| `01_build_pool.py` | Loads both annotation workbooks with the same label logic as CS72 (copied from `00_reproduce_original/verify_checkpoint.py`). Adds `company_id`. Marks the 165 unlabeled sentences and the 4 postings in both samples. | `pool.csv` | Laptop |
| `02_make_split.py` | Assigns companies to train / val / test (step 2), seed 42. | `company_split.csv` | Laptop |
| `03_build_eval_sets.py` | Builds the sets from step 3: train (enriched only), val, test (a), test (b). | `train.csv`, `val.csv`, `test_a.csv`, `test_b.csv` | Laptop |
| `04_train.py` | CS72's `train_bert.py` with the same settings, but reading the files above and choosing the epoch on **val**. Run once per seed (3 seeds). | Model checkpoints | Remote server |
| `05_evaluate.py` | Runs each model on the test sets **once**, adds the keyword baselines, and builds the results tables (step 5). | `results/` | Remote server (Polaris) |
| `06_sample_new_eval.py` | Picks the new evaluation set (step 6). | `new_eval_sentences.csv` | Laptop |
| `07_label_new_eval.py` | Labels it with CS72's Claude setup. | `new_eval.csv` | Laptop |

| `08_check_set_gap.py` | Follow-up: compares the clean test set with the new set to explain the lower scores. | `results/set_gap_check.md` | Laptop |

Order actually run: 01 → 02 → 03 → 04 (server) → 06 → 07 → 05 (server).

## Step 6: New evaluation set (decided 2026-11-16)

A fresh set of postings that **no model has seen**, so the original model, the retrained models, RoBERTa, and the baselines can all be scored on the exact same sentences. Planned after the main E1 code, so it doesn't hold up the split and training.

To keep it fair:
- Only postings from **companies not in any of the 600 annotated postings**.
- Sample the **same way** as before (same 4 industries, same flagging), reusing `segment_and_sample.py`.
- Label with the **same Claude setup** as the training labels: the `prelabel.py` prompt, `claude-sonnet-4-6` (or note the substitute), default temperature.
- **Retry API failures**, so no sentences are left unlabeled (no new "165" problem).
### What was actually done (2026-11-17)

**Sampling (`06_sample_new_eval.py`).** CS72's sampling and sentence-splitting code was copied unchanged. As a check, it re-picks CS72's exact 300 + 300 postings and the same 13,320 + 6,534 sentences.
- **Size:** 15 enriched + 15 random postings per industry = **120 postings, 3,721 sentences**. Enriched postings were picked first; the random draw skipped them, so no posting is in both samples.
- **Avoiding repeats**, in 3 layers:
  1. Drop every posting from CS72's 412 companies, plus 127 postings with no company ID.
  2. Drop 8 postings whose text matches an annotated posting under a different company ID.
  3. Mark the **88 sentences** that match any of the 19,854 annotated sentences (`seen_in_annotated`). They're mostly boilerplate. They're kept, and scored both ways: **full** (3,721) and **unseen** (3,633, the headline).
- **Spread:** random 60 postings from 59 companies; enriched 60 from 50 (at most 3 from one company).
- **Limitation:** CS72 had already taken the most keyword-heavy postings, so the new enriched postings have slightly fewer list words (14.8 vs. 16.3 per posting).

**Labeling (`07_label_new_eval.py`).** Same Claude setup as CS72:
- The system prompt is read straight from `prelabel.py`.
- `claude-sonnet-4-6`, batches of 15, default temperature, no thinking.
- Failed batches were retried until **all 1,572 flagged sentences** had a label.
- The label mix matches CS72's (enriched: masculine 9.7% vs. 10.2%, feminine 8.8% vs. 8.8%).

## Step 5 in practice: who was scored on what

- **Clean test sets (test_a, test_b):** the 6 clean models and 2 keyword rules. The original model trained on most of these sentences, so it isn't scored here.
- **New evaluation set:** every model and every keyword rule.
- **CS72's own test set:** the original model (its saved predictions, re-checked on the server: 100% identical), plus 2 keyword rules.
- **Keyword baselines** (reported under E2):
  - **List rule:** the word-list label.
  - **Learned rule:** each list word's most common label in the training data. There are two versions: one learned from the clean `train.csv`, one from CS72's rebuilt training split (10,870 sentences; it matches CS72's 680 steps per epoch).
  - **Several list words in one sentence:** any masculine → masculine, else any feminine → feminine, else neutral.

## Results (macro-F1; clean models are mean ± SD over 3 seeds)

| Model | CS72's own test | Clean test (test_b) | New set, unseen (headline) |
|---|---|---|---|
| Original (CS72) | 0.853 | — | **0.798** |
| BERT (clean) | — | 0.817 ± 0.003 | **0.784 ± 0.003** |
| RoBERTa (clean) | — | 0.803 ± 0.005 | **0.777 ± 0.003** |
| Learned rule (clean train) | — | 0.698 | 0.685 |
| Learned rule (CS72 train) | 0.769 | — | 0.703 |
| List rule | 0.681 | 0.617 | 0.617 |

- **Answer to E1:** about **0.80 of the 0.853 survives**. The inflation was about 0.05 (FINDINGS F15).
- The original is slightly ahead of the clean models (0.798 vs. 0.784), all in feminine F1. It likely helps that it trained on 12% more data (F16).
- test_a ≈ test_b, and full ≈ unseen, both within 0.001 (F18).
- Training: 5 of 6 runs chose epoch 4 (the last), so the 4-epoch cap may slightly under-train all models equally.
- Full tables: `results/summary.md`. Training records: `results/run_info.csv`. Per-sentence predictions: `results/predictions/`.

## Follow-up: why do models score lower on the new set?

**The puzzle:** both the clean test set and the new set contain only companies the models never saw, yet every model scores lower on the new set (BERT 0.817 → 0.784). The list rule doesn't drop at all. Leakage can't explain it, so something about the new set is harder for the models.

**Why check it:** if the new set were unusual (different label mix, longer sentences, more "both" labels), the headline comparison would need a caveat. If the difference is in how particular sentences were labeled, that's a label question for E4, not a problem with E1's design.

**What `08_check_set_gap.py` does:** it reads the saved predictions (no models are re-run) and compares the two sets.
1. **Which class drops:** per-class precision, recall and F1.
2. **How the sets differ:** label mix, list words, sentence length, the share of masculine labels that came from "both", and which words appear in the masculine sentences and in the models' false "masculine" calls.

**What it found** (full output in `results/set_gap_check.md`; summary in FINDINGS F18):
- The sets are alike on every measure checked, so the new set isn't unusual.
- The drop is in **masculine precision**: the models call more sentences masculine that Claude labeled **neutral**, mostly leadership-type sentences.
- It can't tell apart "the models over-call these" from "Claude labeled these neutral more often this time". Reading the false positives would settle it; that fits the E4 hand-labeling session.

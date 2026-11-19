# E3 methodology: what the models look at (integrated gradients)

**Question:** when a model calls a sentence gendered, how much of that decision comes from the list word and how much from the rest of the sentence?

## Scripts

| Script | What it does | Output | Runs on |
|---|---|---|---|
| `01_sample.py` | Picks the 99 real sentences and adds the 114 minimal-pair sentences | `e3_sentences.csv` | Laptop |
| `02_run_ig.py` | Integrated gradients, one score per word | `results/ig_sentences_<model>.csv`, `results/ig_words_<model>.csv` | Polaris (about 10 min per model, CPU) |
| `03_analyze.py` | Keyword share, tables | `results/e3_summary.md`, `results/e3_per_sentence.csv`, `results/e3_context_pairs.csv` | Laptop |

## Models

The original (CS72) model, clean BERT seed 42, and clean RoBERTa seed 42 (secondary, see convergence). Each run checked that the model reproduces the predictions saved in E1/E2: 213/213 for both BERTs and 212/212 for RoBERTa.

## Sentences

- **Real (99):** flagged, unseen sentences from the new evaluation set (`flagged == True`, `seen_in_annotated == False`), at most 128 BERT tokens. 113 of 1,536 (7%) were over 128 tokens and excluded (F8).
  - 33 per class (Claude's label). Within each class, 17 the original got right and 16 it got wrong.
  - Random within each group (seed 42), at most 3 sentences per company (56 companies). Groups were filled smallest first so the cap couldn't starve them.
  - The same 99 sentences are used for every model, and right/wrong is recomputed for each model. Clean BERT got 53 right.
  - Keywords are the list words found by E1's matcher (`masculine_words` + `feminine_words`), 1.5 per sentence on average.
- **Minimal pairs (114 sentences, 57 pairs):** all E2 sentences. The keyword is the pair's `keyword`. Labels are the user's (the main E2 key).
- One sentence is over 128 RoBERTa tokens and was skipped for RoBERTa only.

## Integrated gradients

- Captum `LayerIntegratedGradients` on the **word-embedding layer**. It attributes the **logit** of the target class, not the probability, because most predictions are above 0.9 (F19) and the probability barely moves near 1.
- **Baseline:** same length, with `[CLS]`/`[SEP]` (`<s>`/`</s>`) kept and every other token replaced by `[PAD]`. Position ids are passed explicitly, so the input and baseline differ only in the words (RoBERTa would otherwise renumber `[PAD]` positions).
- **Steps:** 50. If the convergence delta was more than 5% of the logit gap, the sentence was redone with 300.
- **Target class:**
  - Real sentences: the model's **predicted** class. The question is what drove the decision the model made, including the wrong ones.
  - Pairs: the **keyword's own class** (`pair_type`), the same for both sentences. Both sentences then answer the same question ("how hard does *competitive* push toward masculine here?"). Labels couldn't define this: the user's and Claude's labels agree on a pair's gendered class in only 22 of 57 pairs, and many pairs have no single gendered class.
- **Words:** each whitespace word is tokenized on its own, and sub-token scores are summed into the word. This matches the whole-sentence tokenization for 213/213 sentences, for both tokenizers. `[CLS]`/`[SEP]` are dropped. Punctuation attached to a word counts with that word.
- **Keyword:** a word counts as a keyword if E1's word pattern (`[a-z]+(?:-[a-z]+)*`) finds a keyword in it. If a sentence has several list words, all of them count.

## Measures

- **Keyword share (main measure, absolute):** sum of |keyword scores| / sum of |all word scores|. Words that push *against* the prediction still matter, so absolute values are used.
- **Keyword share (positive-only):** reported as a check. It gives the same overall picture.
- **Keyword score (signed):** used for pairs. It shows how hard the keyword pushes toward its class in each sentence.
- **Context pairs:** pairs where the user labeled one sentence the keyword's class and the other neutral.

## Convergence

The scores should sum to the logit gap between the input and the baseline. The 5% check against the gap fails mainly for pair sentences where the gap is tiny (median about 0.2), so a small error becomes a large percentage. The **rule used** is: keep a sentence if |delta| ≤ 5% of the sum of |word scores|, the denominator of the share, so the error can move the share by at most about 5 points.

| Model | Kept | Dropped |
|---|---|---|
| Original | 198/213 | 15 pair sentences |
| BERT seed 42 | 210/213 | 3 pair sentences |
| RoBERTa seed 42 | 159/212 | 12 real, 41 pair |

RoBERTa converges much worse (some deltas above 3), so its results are secondary. Because of the drops, each model has a different number of usable context pairs (24 / 33 / 14).

## Limitations

- One seed per retrained model, and about 100 real sentences. Group sizes are 14–19, so differences of a few points are noise.
- IG depends on the baseline. A `[PAD]` baseline gives common words like "a" fairly large scores, which makes absolute shares look more context-driven. The positive-only shares are similar, so the conclusions hold.
- Right/wrong for the real sentences is against Claude's labels, which are partly random (F10).
- Results: `results/e3_summary.md`. Findings: F22–F24 in `FINDINGS.md`.

# Findings Log

Things found during the audit, sorted by **what they're for**. IDs (F1–F14) stay fixed because other notes refer to them.

| Section | Meaning |
|---|---|
| **A. Changes an experiment** | You have to do something differently because of this. |
| **B. Evidence for the write-up** | A result worth reporting. Doesn't change how you run anything. |
| **C. Corrections to the CS72 paper** | Something the CS72 paper got wrong. One line each in the report. |
| **D. Minor** | Small. At most a sentence in Limitations, or nothing. |

---

## A. Changes an experiment

**F6. Most enriched postings come from a few repeat employers.** · *E1*
- 64% of enriched postings come from companies with 2+ postings (BMO alone: 13), vs. 18% in the random sample.
- **Do:** split by company, and balance the splits by size and industry. Some companies are big enough to swing a split's size a lot: test size ranged from 2,218 to 4,092 sentences across random seeds.

**F12. Two postings the model trained on are also in the test set.** · *E1*
- The random and enriched samples were split separately. 4 postings were drawn into both samples; two of them and landed in train on one side and test on the other: 123 test sentences (3.3%).
- **Do:** pool both samples and split them together.

**F11. Unflagged sentences make the score look better than it is.** · *E1, E2*
- 61% of test sentences contain no list word and are almost always correctly called neutral (99.4%). Macro-F1 is 0.853 on everything but 0.827 on flagged sentences only.
- **Do:** report flagged-only scores as a headline number. Your research question is about sentences that contain a keyword, so this is the fairer measure.

**F8. Some "sentences" are whole paragraphs and get cut off.** · *E3*
- Sentence splitting failed on run-together text. 368 training sentences exceed BERT's 128-token limit, so their ends are never seen.
- **Do:** only use sentences of 128 tokens or fewer in E3, since attributions can't cover text the model never read.

**F3 (part). 165 flagged sentences were never labeled.** · *E1*
- API failures left them blank, so they default to neutral.
- In the original, they were **not** excluded: 141 were in train and 24 in test, all labeled neutral. The model called 11 of the 24 test ones gendered, and all 11 were counted as errors.
- **Do (decided 2026-11-16):** don't re-label them; keep them in train and exclude them from val/test. This is a change from the original, so apply the same exclusion to **every** model compared in E1. One line in Limitations.

---

## B. Evidence for the write-up

**F11 (part). The model almost never detects gendered language outside the word list.** · *Main research question*
- Out of 2,253 unflagged test sentences, it called only 14 gendered (0.6%). Several of those look genuinely gendered, e.g. "Cultivate a culture of high performance… drive for results" (masc 0.99). Under the current labels, all 14 count as errors.
- **Why it matters:** early evidence that the model leans on the list, and an example of gap 4 penalizing beyond-list detection. *To do: hand-check the 14 during your E4 labeling session.*

**F6 (part). Repeated templates help explain why the model might rely on keywords.** · *Discussion*
- The model trained on the same keyword-heavy templates many times over. In the enriched sample, 44.8% of sentences appear more than once.
- **Why it matters:** this gives a plausible reason for the keyword reliance you're testing for. Recommend for future work: cap each company at 1–2 postings and remove duplicates before sampling.

**F10. Claude gave different labels to identical sentences.** · *E4*
- 151 sentences that appear more than once have conflicting labels. A likely cause is that `prelabel.py` never set temperature, so the default (1.0) made labels partly random.
- **Why it matters:** some human–Claude disagreement in E4 will just be Claude's own randomness. The optional E4 self-agreement check measures how much.
- **Clean example (found 2026-11-16):** the 4 postings drawn into both samples were labeled twice by Claude, once per sample. 7 of their 153 sentences got different labels the second time (e.g. masculine vs. feminine, neutral vs. feminine). No API failures are involved here, so this is true inconsistency.
- *Caveat: some of the 151 may be failed sentences defaulting to neutral, not true inconsistency. Check before reporting the number.*

**F13. Splitting by company removes almost all the sentence overlap.** · *E1*
- In CS72, about 1/3 of test sentences also appeared in training. With the new company split, only **34 of 2,767 test sentences (1.2%)** still match a training sentence (near-match).
- **Why it matters:** test versions (a) and (b) will score almost the same, so any drop in E1 comes from the company split itself, not from removing the last duplicates. Say this when presenting the stepwise results.

**F14. Folding "both" into masculine makes some feminine words look masculine in the training data.** · *E2/E3, Discussion*
- "Interpersonal" is on the feminine list, but in the clean training set it appears in 127 flagged sentences: 72 that Claude called "both" (counted as masculine), 47 feminine, 4 masculine. So it sits in masculine-labeled sentences more often than feminine ones, and the learned keyword rule calls it masculine. Found 2026-11-17 while testing `05_evaluate.py`.
- **Why it matters:** any model trained on these labels sees the same pattern, so a "masculine" prediction may come from a feminine word that usually sits next to masculine ones ("analytical and interpersonal skills"). Worth one minimal pair in E2 and a check in E3.

**F2. Choosing the model on the test set barely mattered.** · *E1*
- Epoch 4 beat epoch 3 by only 0.0004 on the test set.
- **Why it matters:** if E1's score drops, you can say it's from leakage, not model selection.

**F1. The saved model reproduces the paper exactly.** · *Methods, 1 sentence*
- Same confusion matrix, macro-F1 0.853. Confirms you're auditing the right model.

---

## C. Corrections to the CS72 paper

| # | The paper says | Actually | Matters for |
|---|---|---|---|
| F5 | Industry ranking: Finance > Tech > **Healthcare > Education** (Table 4) | Finance > Tech > **Education > Healthcare**. The rows are swapped; the male/female group averages (72.2% vs 60.2%) are unchanged. | **E5 starting point**, so this one matters |
| F2 | Best checkpoint chosen by validation F1 | Chosen on the test set | E1 framing |
| F3 | Labels were reviewed by a human | 1 of 19,854 changed; the "both" review sheet was never filled in | E4 motivation |
| F4 | Enriched beat random 0.85 vs 0.83 | Different test sets. On the same test set it's 0.85 vs **0.79**. | Correction only |

---

## D. Minor

- **F7.** E5's 10,000 postings include repeats (862 exact duplicates), but removing them doesn't change the ranking. *At most one robustness sentence in E5.*
- **F9.** A CS72 planning doc proposed tuning the threshold on the test set; it was never used. Its one useful point: Gaucher et al. tested whole ads, not single sentences. *Optional motivation line for E5.*
- **F4.** See section C. It's a side claim from CS72, not central to the audit.

---

*Evidence and scripts: `00_reproduce_original/` (F1, F5, F7). Other numbers come from quick checks that haven't been saved as scripts yet (F6, F10, F11, F12).*

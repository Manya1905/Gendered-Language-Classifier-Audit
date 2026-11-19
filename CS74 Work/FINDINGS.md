# Findings Log

Things found during the audit, sorted by **what they're for**. IDs (F1–F24) stay fixed because other notes refer to them.

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

**F15. About 0.80 of the 0.853 survives (E1 headline).** · *E1, main result*
- The original model scores **0.853** on its own CS72 test set and **0.798** on the new evaluation set (unseen companies, unseen sentences, labeled the same way). That's a drop of 0.055.
- The clean models land in the same place: BERT **0.784 ± 0.003**, RoBERTa **0.777 ± 0.003** (3 seeds).
- **Why it matters:** this answers E1. The CS72 number was inflated by about 0.05 by leakage and test-set model selection. The real ability is about 0.78–0.80 macro-F1.
- Source: `01_clean_reevaluation/results/summary.md`.

**F16. The original model is slightly ahead of the clean models, but only on feminine.** · *E1*
- New set, unseen: original 0.798 vs. BERT 0.784 ± 0.003. That's more than the seed noise, so it's likely real.
- F1 masculine is tied (0.681 / 0.684 / 0.681). The gap is all in feminine (0.761 vs. 0.717 ± 0.004).
- **Likely reason (a limitation, not a finding about leakage):** the original trained on about 12% more sentences (10,870 vs. 9,715), because the clean split had to set aside 15% for validation. So the clean scores are, if anything, slightly conservative.
- RoBERTa is no better than BERT here (0.777 vs. 0.784). A stronger pretrained model doesn't help, which is consistent with the limit coming from the data and labels.

**F17. The models beat the keyword rules, but memorizing each word's usual label already gets most of the way.** · *Report under E2*
- New set, unseen, **flagged-only** (sentences with a list word): BERT 0.762, original 0.775, learned rule 0.632 (clean train) or 0.648 (CS72 train), list rule 0.345.
- Overall: learned rule 0.685–0.703 vs. models 0.78–0.80.
- On CS72's own test set the gap is similar: original 0.853 vs. learned rule (CS72 train) 0.769.
- **Why it matters:** the models add about 0.10–0.13 macro-F1 over pure word lookup on real sentences. That's real but modest. E2's minimal pairs test where the extra comes from.

**F18. Small checks that came out as expected.** · *E1, one sentence each*
- test_a vs. test_b: identical within 0.001 (BERT 0.818 vs. 0.817). Removing the 34 sentences shared with train changes nothing, as F13 predicted.
- New set, full vs. unseen: identical within 0.001. The 88 memorized boilerplate sentences give no advantage.
- All models score lower on the new set than on the clean test set (BERT 0.817 → 0.784). The list rule doesn't drop (0.617 vs. 0.617). Checked in `08_check_set_gap.py` (output: `results/set_gap_check.md`):
  - **The two sets look alike:** same masculine share (8.0% vs. 8.1%), flagged share (40% vs. 42%), list words per flagged sentence (1.64 vs. 1.60) and sentence length. The new set has *fewer* "both"→masculine labels (33% vs. 37%), so F14 doesn't explain it.
  - **The drop is masculine precision:** BERT 0.723 → 0.630, while recall barely moves (0.795 → 0.748). False "masculine" calls rise from 24.5 to 35.7 per 1,000 sentences, and 85% of them are sentences Claude labeled **neutral**.
  - They're spread across leadership-type words ("leadership", "leaders", "analysis", "decisions"), not one word.
  - **Two explanations remain:** the models over-call leadership-type words as masculine, or Claude labeled those sentences neutral more often this time (labels are partly random, F10). Weak hint toward the second: Claude labeled "leader" sentences gendered 57% of the time in CS72 (n=187) vs. 38% in the new set (n=40), and "lead" 51% (n=408) vs. 40% (n=52). But these are different sentences and small numbers, so this doesn't show inconsistency. The E4 self-agreement check (same sentences re-sent) is the real test. *Settling it means reading the false positives (could be added to the E4 session). For the report: "the drop is in masculine precision on leadership-type sentences Claude labeled neutral."*

**F19. The industry ranking doesn't depend on borderline predictions.** · *E5, threshold sweep*
- CS72 used argmax. Re-counting the original model's masculine sentences as P(masculine) ≥ t, for t = 0.50, 0.55, …, 0.90, the ranking stays **Finance > Tech > Education > Healthcare** at all 9 thresholds. Finance and tech stay above both female-dominated industries every time.
- All four industries drop together, so the gap between the male-dominated and female-dominated averages stays at about 12 points (12.0 at argmax, 11.7 at 0.90). Finance goes from 77.0% to 71.6%, healthcare from 56.6% to 51.4%.
- **Why:** most masculine predictions are confident. 85% of them (11,910 of 13,954) have P(masculine) ≥ 0.9.
- **Why it matters:** the CS72 ranking isn't an artifact of using argmax. Check passed first: argmax reproduces CS72's 77.0 / 67.4 / 63.9 / 56.6 exactly.
- Source: `05_downstream/results/threshold_sweep.md`.

**F20. Keyword rules alone reproduce the industry ranking, but with a much smaller gap.** · *E5, changing the labeler*
- The 3 keyword rules from E1 were run on the same 213,779 sentences (10,000 postings) and counted with CS72's metric:

  | Labeler | Finance | Tech | Education | Healthcare | Gap |
  |---|---|---|---|---|---|
  | Original BERT (argmax) | 77.0% | 67.4% | 63.9% | 56.6% | 12.0 pts |
  | List rule | 93.4% | 92.6% | 91.5% | 87.2% | 3.7 pts |
  | Learned rule (clean train) | 86.2% | 80.3% | 78.7% | 74.3% | 6.7 pts |
  | Learned rule (CS72 train) | 86.7% | 82.4% | 79.7% | 74.7% | 7.3 pts |

- All three rules give the same ranking as BERT. So the ranking is already in **which list words each industry uses**, without reading context.
- The rules flag far more postings (the list rule flags 87–93%, partly from prefix over-matching like "leading" and "analyst"). Almost everything is flagged, so the industries look alike and the gap shrinks.
- **Why it matters:** BERT's contribution to the CS72 result is making the gap bigger (12 vs. 4–7 points), not producing the ranking. Whether that comes from reading context or from a sharper version of the learned rule is the E2/E3 question.
- Checks passed first: CS72's matcher reproduces E1's list words on all 3,721 new-set sentences, and the rules reproduce E1's saved predictions exactly.
- Source: `05_downstream/results/baselines_industry.md`; full E5 table in `05_downstream/results/e5_summary.md`.

**F21. The retrained models give the same ranking, but tech vs. education depends on a few big employers.** · *E5, main result*
- Retrained on the clean split, all 6 models (3 seeds each) give **Finance > Tech > Education > Healthcare**, the same as the original:

  | Labeler | Finance | Tech | Education | Healthcare | Gap |
  |---|---|---|---|---|---|
  | Original (CS72) | 77.0% | 67.4% | 63.9% | 56.6% | 12.0 pts |
  | BERT (clean), 3 seeds | 76.9 ± 2.2% | 66.2 ± 1.9% | 61.8 ± 1.8% | 52.6 ± 0.8% | 14.3 ± 2.2 pts |
  | RoBERTa (clean), 3 seeds | 80.3 ± 0.6% | 70.8 ± 1.8% | 63.8 ± 3.3% | 55.4 ± 1.5% | 15.9 ± 2.0 pts |

- Seeds move individual industries by 2–4 points but never change the order. The gap is, if anything, slightly bigger without leakage (12.8–18.1 pts across seeds).
- Bootstrap over postings: the gap's 95% CI excludes 0 for every labeler, and the ranking survives in 92–100% of resamples for every model.
- Per-sentence masculine rate (not affected by posting length): finance and tech are above education and healthcare for every labeler.
- **But with one posting per company (3,765 postings), tech and education swap** for 9 of 10 labelers, original included (e.g. original: finance 75.4, education 69.2, tech 67.9, healthcare 59.5). Finance stays first and healthcare stays last for every labeler, and the models' gap stays clearly positive (7–9 pts). The learned rules' gap nearly disappears (1.3–1.6 pts).
- **Why it matters:** the claim "male-dominated industries use more masculine language" holds no matter which model labels the sentences, but the strict ordering "tech > education" partly comes from a few employers with many postings (see F7, F6). Safer claim for the report: finance highest, healthcare lowest, tech ≈ education.
- *Caveat: the bootstrap resamples postings, not companies, so its CIs are probably too narrow. Postings from the same company aren't independent, which is what the one-per-company check shows.*

**F22. When the model calls a sentence gendered, the list words carry about 40% of the decision, though they're only about 7% of the words.** · *E3, main result*
- Integrated gradients on 99 flagged, unseen new-set sentences (33 per class, half right / half wrong for the original). Keyword share = list words' |score| as % of all words' |score|, toward the model's predicted class.

  | Model | Predicted gendered: keyword share | Predicted neutral: keyword share | List words as % of words |
  |---|---|---|---|
  | Original | 41% (n=59) | 16% (n=40) | 6–7% |
  | BERT (clean, seed 42) | 40% (n=53) | 15% (n=46) | 5–7% |
  | RoBERTa (clean, seed 42) | 30% (n=56) | 12% (n=30) | 5–7% |

- In neutral calls, the keyword's score toward neutral is *negative* (original −0.77, BERT −1.24): the keyword still pushes toward gendered, and the rest of the sentence outweighs it.
- Positive-only shares give the same picture (overall 32% vs. 31% absolute for the original).
- **Why it matters:** the model leans heavily on the list words (about 6× their share of the sentence), but not only on them: about 60% of a gendered call comes from the other words, and context can override a keyword. Clean BERT looks the same as the original, so the reliance isn't caused by leakage; it comes from training on these labels.
- Source: `03_integrated_gradients/results/e3_summary.md`.

**F23. False "gendered" calls are less keyword-driven than correct ones.** · *E3*
- Correct masculine calls: keyword share 53% (original) / 49% (BERT). Neutral-labeled sentences the model called gendered: 33% / 27%. Missed masculine/feminine sentences (model said something else): 20–26%.
- **Why it matters:** the false alarms (the masculine-precision drop in F18) aren't just "saw a list word, said masculine". Other words in those sentences push too. Reading those sentences (E4) is how to tell whether that context is genuinely gendered or a pattern the labels taught (like F14).

**F24. In minimal pairs, context turns a list word's push down but not off, and off-list words get almost no push.** · *E3 + E2*
- Target = the keyword's own class, for both sentences of a pair. Context pairs = your labels give one sentence the keyword's class and the other neutral.
- **List words:** the keyword pushes toward its class harder in the gendered use in 10/14 pairs (original) and 14/21 (BERT). But it still pushes strongly in the neutral use: mean score 3.0 vs. 5.0 (original), 2.6 vs. 5.0 (BERT). Example: "competitive" in "competitive culture" vs. "competitive salary".
- **Off-list words** ("tough", "sweet", "driven"...): keyword score ≈ 0 in both sentences (original −0.55 vs. 0.10, BERT 0.28 vs. 0.26). The model doesn't treat them as gendered in either context.
- **Why it matters:** the model reads some context, but only around words that are on the list. Words outside the list get no gendered weight even when a person reads them as gendered. This is the vocabulary result at the word level.
- *Caveat:* the number of usable pairs differs by model (original 24, BERT 33, RoBERTa 14) because of the convergence filter (see E3 methodology). RoBERTa is secondary: only 159 of 212 sentences converged.

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

"""
E5 part 1b: label the 213,779 industry-analysis sentences with the 3 keyword rules from E1,
then count postings per industry exactly like CS72 (e5_common.pct_postings_masculine).

Rules (same as E1, 05_evaluate.py):
  List rule                   each list word gets its word-list label
  Learned rule (clean train)  each list word gets the label it most often had in E1's clean train split
  Learned rule (CS72 train)   same, from CS72's original training split
  Several words in a sentence: any masculine -> masculine, else any feminine -> feminine, else neutral.
  A word with no learned label (unseen or tied) falls back to its word-list label.
The learned word labels are E1's saved output of learn_word_labels():
01_clean_reevaluation/results/learned_rule_words.csv.

Reused, not rewritten:
  find_gendered_words()  imported from 01_clean_reevaluation/06_sample_new_eval.py (CS72's matcher)
  keyword logic          copied from 05_evaluate.py keyword_predict() (that function returns
                         E1-only columns, so it can't be called on these sentences)
Checks (on E1's new evaluation set, before touching the E5 sentences):
  1. find_gendered_words() reproduces the word columns saved in new_eval.csv
  2. the copied rules reproduce E1's saved predictions for all 3 rules

Usage (local, from CS74 Work/05_downstream):
    python e5_baselines.py
Saves: predictions/list_rule.csv, learned_clean.csv, learned_cs72.csv  (job_id, sentence_num,
       masculine_words, feminine_words, pred_id)
       results/baselines_industry.csv / .md
"""

import importlib
import sys

import pandas as pd

from e5_common import (HERE, INDUSTRIES, RESULTS, describe, pct_postings_masculine,
                       read_sentences)

E1 = HERE.parent / "01_clean_reevaluation"
sys.path.insert(0, str(E1))                                   # 06 imports E1's common.py
find_gendered_words = importlib.import_module("06_sample_new_eval").find_gendered_words

LABEL_ID = {"neutral": 0, "masculine": 1, "feminine": 2}
RULES = {"list_rule": ("List rule", None),
         "learned_clean": ("Learned rule (clean train)", "clean_train"),
         "learned_cs72": ("Learned rule (CS72 train)", "cs72_train")}


# ── Copied from 05_evaluate.py ──
def words(cell):
    return [w for w in str(cell).split(", ") if w] if isinstance(cell, str) else []


def keyword_predict(masc_col, fem_col, word_labels=None):
    """List rule (word_labels=None) or learned rule. Any masculine word wins, then feminine."""
    preds = []
    for m, f in zip(masc_col, fem_col):
        if word_labels is None:
            labs = [1] * len(words(m)) + [2] * len(words(f))
        else:
            labs = [word_labels.get(w, 1) for w in words(m)] + [word_labels.get(w, 2) for w in words(f)]
        preds.append(1 if 1 in labs else 2 if 2 in labs else 0)
    return preds


def load_word_labels():
    table = pd.read_csv(E1 / "results" / "learned_rule_words.csv")
    return {col: {w: LABEL_ID[lab] for w, lab in zip(table["word"], table[col]) if isinstance(lab, str)}
            for col in ["clean_train", "cs72_train"]}


def find_words(texts):
    found = [find_gendered_words(t) for t in texts]
    return [", ".join(m) for m, _ in found], [", ".join(f) for _, f in found]


def run_checks(word_labels):
    ne = pd.read_csv(E1 / "data" / "new_eval.csv", dtype={"job_id": str})
    masc, fem = find_words(ne["sentence_text"].astype(str))
    same_words = ((pd.Series(masc) == ne["masculine_words"].fillna("")).all()
                  and (pd.Series(fem) == ne["feminine_words"].fillna("")).all())
    assert same_words, "find_gendered_words() does not reproduce new_eval.csv's word columns"
    print(f"Check 1 passed: matcher reproduces the list words on all {len(ne):,} E1 sentences")

    for tag, (name, col) in RULES.items():
        saved = pd.read_csv(E1 / "results" / "predictions" / f"{tag}_new_eval.csv")
        ours = keyword_predict(ne["masculine_words"], ne["feminine_words"],
                               word_labels[col] if col else None)
        assert list(saved["pred_id"]) == ours, f"{name} does not reproduce E1's saved predictions"
    print("Check 2 passed: all 3 rules reproduce E1's saved predictions")


def main():
    word_labels = load_word_labels()
    run_checks(word_labels)

    df = read_sentences()
    print(f"\nFinding list words in {len(df):,} sentences...")
    df["masculine_words"], df["feminine_words"] = find_words(df["sentence_text"].astype(str))
    flagged = ((df["masculine_words"] != "") | (df["feminine_words"] != "")).mean()
    print(f"  {flagged:.1%} of sentences contain at least one list word")

    (HERE / "predictions").mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for tag, (name, col) in RULES.items():
        df["pred_id"] = keyword_predict(df["masculine_words"], df["feminine_words"],
                                        word_labels[col] if col else None)
        df[["job_id", "sentence_num", "masculine_words", "feminine_words", "pred_id"]].to_csv(
            HERE / "predictions" / f"{tag}.csv", index=False)
        pct = pct_postings_masculine(df, df["pred_id"] == 1)
        rows.append({"labeler": name, "masculine_sentences": int((df["pred_id"] == 1).sum()),
                     **{ind: pct[ind] for ind in INDUSTRIES}, **describe(pct)})

    table = pd.DataFrame(rows)
    table.round(2).to_csv(RESULTS / "baselines_industry.csv", index=False)
    lines = ["# E5: keyword rules on the 10,000 postings", "",
             "% of postings with at least one masculine sentence (CS72's metric). "
             "Original BERT (argmax), for comparison: finance 77.0, tech 67.4, education 63.9, healthcare 56.6.", "",
             "| Labeler | Masc. sentences | Finance | Tech | Education | Healthcare | Gap | Ranking | Holds? |",
             "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['labeler']} | {r['masculine_sentences']:,} | "
                     + " | ".join(f"{r[ind]:.1f}%" for ind in INDUSTRIES)
                     + f" | {r['gap']:.1f} pts | {r['ranking']} | {'yes' if r['holds'] else 'NO'} |")
    (RESULTS / "baselines_industry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:]))


if __name__ == "__main__":
    main()

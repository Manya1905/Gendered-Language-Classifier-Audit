"""
Check how much duplicated / repeated postings affect the CS72 industry ranking (E5).

E5 doesn't score predictions against labels, so duplicates aren't a leakage problem
there. The question is weighting: can a few employers that post the same job many
times move an industry's "% of postings with a masculine sentence"?

Finding (2026-11-14): 36% of sentences are repeats and 862/10,000 postings are exact
duplicates, but removing exact-duplicate postings leaves the ranking unchanged
(Finance > Tech > Education > Healthcare). A stricter one-posting-per-company check
needs company_id from the Kaggle postings.csv (to do in E5).

Usage:
    python check_duplicate_postings.py
Runtime: a few seconds (no model needed).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SENTENCES = PROJECT / "From CS72" / "From Remote Server" / "industry_analysis_sentences.csv"


def main():
    d = pd.read_csv(SENTENCES, dtype={"job_id": str})

    # ── Sentence-level repetition ──
    dup_rate = d.sentence_text.duplicated().mean() * 100
    print(f"Sentences: {len(d):,}   unique: {d.sentence_text.nunique():,}   repeated rows: {dup_rate:.1f}%")

    masc = d[d.predicted_class == 1]
    print(f"Masculine-predicted sentences: {len(masc):,}   unique: {masc.sentence_text.nunique():,}\n")
    print("Most repeated masculine sentences:")
    for text, n in masc.sentence_text.value_counts().head(5).items():
        print(f"  {n:>4}x  {text[:110]}{'...' if len(text) > 110 else ''}")
    print()

    # ── Posting-level duplicates ──
    postings = d.groupby("job_id").agg(
        industry=("industry", "first"),
        text=("sentence_text", " ".join),
        has_masc=("predicted_class", lambda x: (x == 1).any()),
    )
    deduped = postings.drop_duplicates("text")
    print(f"Postings: {len(postings):,}   exact-duplicate postings: {len(postings) - len(deduped):,}\n")

    # ── Does the ranking change? ──
    out = pd.DataFrame({
        "orig_%masc": postings.groupby("industry").has_masc.mean() * 100,
        "dedup_%masc": deduped.groupby("industry").has_masc.mean() * 100,
        "postings_after_dedup": deduped.groupby("industry").size(),
    }).round(1).sort_values("orig_%masc", ascending=False)
    print(out.to_string())
    same = list(out.sort_values("orig_%masc").index) == list(out.sort_values("dedup_%masc").index)
    print("\nRanking unchanged after removing exact duplicates:", same)


if __name__ == "__main__":
    main()

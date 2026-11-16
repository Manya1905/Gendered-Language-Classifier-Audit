"""
Recompute CS72 paper Table 4 (per-industry gendered-language prevalence) from the
saved sentence-level predictions, and compare it to what the paper printed.

Finding (2026-11-14): the paper's Table 4 swaps the Healthcare and Education rows.
The correct ranking by % of postings with >=1 masculine sentence is
Finance > Tech > Education > Healthcare.

Also checks that the industry labels are right by printing the most common job
titles in education vs healthcare.

Usage:
    python recompute_table4.py
Runtime: a few seconds (no model needed).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SENTENCES = PROJECT / "From CS72" / "From Remote Server" / "industry_analysis_sentences.csv"

# As printed in the CS72 paper (Table 4)
PAPER = pd.DataFrame(
    {
        "pct_masc": [77.0, 67.4, 63.9, 56.6],
        "avg_masc": [1.60, 1.43, 1.43, 1.11],
        "pct_fem": [53.4, 49.2, 73.3, 70.6],
        "avg_fem": [1.18, 1.01, 1.87, 1.66],
    },
    index=["finance", "tech", "healthcare", "education"],
)


def main():
    d = pd.read_csv(SENTENCES, dtype={"job_id": str})
    print(f"Sentences: {len(d):,}   Postings: {d.job_id.nunique():,}")
    print(d.groupby("industry").job_id.nunique().rename("postings").to_string(), "\n")

    # predicted_class: 0 = neutral, 1 = masculine, 2 = feminine (argmax, no threshold)
    per_posting = d.groupby(["industry", "job_id"]).agg(
        masc=("predicted_class", lambda x: (x == 1).sum()),
        fem=("predicted_class", lambda x: (x == 2).sum()),
    ).reset_index()

    recomputed = per_posting.groupby("industry").agg(
        pct_masc=("masc", lambda x: (x > 0).mean() * 100),
        avg_masc=("masc", "mean"),
        pct_fem=("fem", lambda x: (x > 0).mean() * 100),
        avg_fem=("fem", "mean"),
    ).round(2).sort_values("pct_masc", ascending=False)

    print("RECOMPUTED from industry_analysis_sentences.csv:")
    print(recomputed.to_string(), "\n")
    print("AS PRINTED in the CS72 paper (Table 4):")
    print(PAPER.to_string(), "\n")

    swapped = (
        abs(recomputed.loc["education", "pct_masc"] - PAPER.loc["healthcare", "pct_masc"]) < 0.1
        and abs(recomputed.loc["healthcare", "pct_masc"] - PAPER.loc["education", "pct_masc"]) < 0.1
    )
    print("Healthcare/Education rows swapped in paper:", swapped)

    male = recomputed.loc[["finance", "tech"], "pct_masc"].mean()
    female = recomputed.loc[["healthcare", "education"], "pct_masc"].mean()
    print(f"Male-dominated avg % masc: {male:.1f}   Female-dominated: {female:.1f}  (paper: 72.2 vs 60.2)\n")

    # Sanity check that industry labels in the CSV are correct
    titles = d.drop_duplicates("job_id")
    for ind in ["education", "healthcare"]:
        print(f"Most common titles in '{ind}':")
        print(titles[titles.industry == ind].title.value_counts().head(8).to_string(), "\n")


if __name__ == "__main__":
    main()

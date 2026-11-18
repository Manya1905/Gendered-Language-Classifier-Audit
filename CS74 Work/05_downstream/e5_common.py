"""
E5 shared pieces: file paths and CS72's posting-level metric.

The metric is copied from CS72's industry_analysis.py (lines 281-320), not imported,
because importing that file would parse its command-line args and load models.
Every E5 script uses pct_postings_masculine(), so all models are counted the same way.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                # .../CS74 Work/05_downstream
PROJECT = HERE.parents[1]                              # .../CS74 Final Project
SENTENCES_CSV = PROJECT / "From CS72" / "From Remote Server" / "industry_analysis_sentences.csv"
RESULTS = HERE / "results"

INDUSTRIES = ["finance", "tech", "education", "healthcare"]
MALE_DOM = ["finance", "tech"]
FEMALE_DOM = ["education", "healthcare"]

# CS72's numbers, recomputed from the CSV (F5: the paper's Table 4 swaps education/healthcare)
CS72_PCT = {"finance": 77.0, "tech": 67.4, "education": 63.9, "healthcare": 56.6}


def read_sentences():
    df = pd.read_csv(SENTENCES_CSV, dtype={"job_id": str})
    assert len(df) == 213_779, len(df)
    assert df["job_id"].nunique() == 10_000
    return df


def pct_postings_masculine(df, is_masc):
    """CS72's metric: % of postings per industry with at least one masculine sentence.

    Same steps as industry_analysis.py:
      group sentences by job_id -> masculine_count = number of masculine sentences
      has_masculine = masculine_count > 0
      pct_with_masculine = postings_with_masculine / total_postings * 100
    is_masc: True/False per sentence (row-aligned with df).
    Returns unrounded percentages; CS72 rounded to 1 decimal only for display.
    """
    posting = (df.assign(is_masc=np.asarray(is_masc, dtype=bool))
                 .groupby("job_id")
                 .agg(industry=("industry", "first"), masculine_count=("is_masc", "sum")))
    has_masculine = posting["masculine_count"] > 0
    pct = has_masculine.groupby(posting["industry"]).sum() / posting["industry"].value_counts() * 100
    return pct.reindex(INDUSTRIES)


def describe(pct):
    """Ranking and the main check: do both male-dominated industries stay above both female-dominated ones?"""
    ranking = " > ".join(pct.sort_values(ascending=False).index)
    male_avg, female_avg = pct[MALE_DOM].mean(), pct[FEMALE_DOM].mean()
    return {
        "ranking": ranking,
        "holds": bool(pct[MALE_DOM].min() > pct[FEMALE_DOM].max()),
        "male_dom_avg": male_avg,      # CS72's "72.2%"
        "female_dom_avg": female_avg,  # CS72's "60.2%"
        "gap": male_avg - female_avg,
    }

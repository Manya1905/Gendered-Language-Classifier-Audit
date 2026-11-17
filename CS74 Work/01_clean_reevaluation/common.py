"""
Shared pieces for the E1 scripts: file paths, and CS72's exact loading + labeling logic.

The loading and labeling functions are copied from CS72 train_bert.py (via
00_reproduce_original/verify_checkpoint.py, which reproduces the paper's confusion
matrix exactly), so labels are identical to the ones the original model trained on.
"""
from pathlib import Path

import openpyxl
import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
HERE = Path(__file__).resolve().parent             # .../CS74 Work/01_clean_reevaluation
PROJECT = HERE.parents[1]                          # .../CS74 Final Project
DATA = PROJECT / "From CS72" / "Data"
RANDOM_XLSX = DATA / "annotation_random.xlsx"
ENRICHED_XLSX = DATA / "annotation_biased.xlsx"    # "biased" = the enriched sample
POSTINGS_CSV = DATA / "postings.csv"               # raw Kaggle file, has company_id
ORIGINAL_MODEL = PROJECT / "From CS72" / "biased_multiclass" / "checkpoint-best"

OUT = HERE / "data"                                # outputs of scripts 01-03
POOL_CSV = OUT / "pool.csv"
SPLIT_CSV = OUT / "company_split.csv"

SPLIT_SEED = 42
N_SLICES = 20                                      # 14 train / 3 val / 3 test = 70/15/15
TRAIN_SLICES, VAL_SLICES = 14, 3

VALID_LABELS = {"masculine", "feminine", "both", "neutral"}
LABEL_TO_ID = {"neutral": 0, "masculine": 1, "feminine": 2}


# ──────────────────────────────────────────────
# Copied from CS72 train_bert.py
# ──────────────────────────────────────────────
def load_workbook_data(path, source_label):
    """Read one annotation workbook (one sheet per industry) into a DataFrame."""
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        industry = sheet_name.lower()
        for row in ws.iter_rows(min_row=2, values_only=True):
            cell_a = row[0]
            if cell_a is not None and str(cell_a).startswith("--- Posting:"):
                continue
            if cell_a is None or str(cell_a).strip() == "":
                continue
            sentence_text = row[3]
            if sentence_text is None or str(sentence_text).strip() == "":
                continue
            job_id = row[0]
            job_id = str(int(job_id)) if isinstance(job_id, float) else str(job_id)
            rows.append({
                "job_id": job_id,
                "title": row[1],
                "sentence_num": row[2],
                "sentence_text": str(sentence_text),
                "flagged": str(row[4]).strip().upper() == "YES" if row[4] else False,
                "masculine_words": row[5] or "",
                "feminine_words": row[6] or "",
                "suggested_label": str(row[7]).strip().lower() if row[7] else "",
                "your_label": str(row[9]).strip().lower() if row[9] else "",
                "industry": industry,
                "source": source_label,
            })
    return pd.DataFrame(rows)


def resolve_label(row):
    """your_label > suggested_label (Claude) > neutral."""
    if row["your_label"] in VALID_LABELS:
        return row["your_label"]
    if row["suggested_label"] in VALID_LABELS:
        return row["suggested_label"]
    return "neutral"


def prepare_labels(df):
    """Resolve labels, collapse 'both' into masculine, add numeric ids."""
    df["label"] = df.apply(resolve_label, axis=1).replace({"both": "masculine"})
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df["has_gendered"] = (df["label"] != "neutral").astype(int)
    return df


# ──────────────────────────────────────────────
# New for E1
# ──────────────────────────────────────────────
def read_pool():
    return pd.read_csv(POOL_CSV, dtype={"job_id": str})


def summarize(df):
    """One row of the 'is this set a mini copy of the whole data?' table."""
    ind = df["industry"].value_counts(normalize=True)
    return {
        "sentences": len(df),
        "postings": df["job_id"].nunique(),
        "companies": df["company_id"].nunique(),
        **{i: f"{100 * ind.get(i, 0):.0f}%" for i in ["tech", "finance", "healthcare", "education"]},
        "masculine": int((df["label"] == "masculine").sum()),
        "feminine": int((df["label"] == "feminine").sum()),
        "enriched": f"{100 * (df['source'] == 'enriched').mean():.0f}%",
    }
def normalize(text):
    """For near-match duplicate checks: lowercase and collapse all whitespace."""
    return " ".join(str(text).lower().split())

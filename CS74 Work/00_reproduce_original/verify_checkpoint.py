"""
Verify the original CS72 checkpoint (biased_multiclass) reproduces the paper's results.

Recreates the original combined test set exactly as train_bert.py did (posting-level
80/20 split, seed=42, done separately for the random and enriched samples, then
concatenated), runs the saved model on it, and checks the confusion matrix against
the CS72 paper (Table 3).

Loading / labeling / splitting logic is copied from CS72 train_bert.py so the split
is identical.

Usage (from anywhere):
    python verify_checkpoint.py

Outputs (next to this script):
    original_test_predictions.csv   per-sentence labels, predictions, probabilities

Runtime: ~10+ min on a laptop CPU (3,716 sentences).
"""
import os
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer

# ──────────────────────────────────────────────
# Paths (relative to the project folder)
# ──────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]                         # .../CS74 Final Project
CS72 = PROJECT / "From CS72"
DATA = CS72 / "Data"
MODEL = CS72 / "biased_multiclass" / "checkpoint-best"
OUT = HERE / "original_test_predictions.csv"

VALID_LABELS = {"masculine", "feminine", "both", "neutral"}
LABEL_TO_ID_MULTI = {"neutral": 0, "masculine": 1, "feminine": 2}

# Paper Table 3 (rows = true N/M/F, cols = predicted N/M/F)
PAPER_CM = np.array([[2922, 77, 55], [50, 284, 11], [35, 20, 262]])


# ──────────────────────────────────────────────
# Same logic as CS72 train_bert.py
# ──────────────────────────────────────────────
def load_workbook_data(path, source_label):
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
    df["resolved_label"] = df.apply(resolve_label, axis=1).replace({"both": "masculine"})
    df["binary_id"] = (df["resolved_label"] != "neutral").astype(int)
    df["multi_id"] = df["resolved_label"].map(LABEL_TO_ID_MULTI)
    return df


def split_by_posting(df, test_size=0.2, seed=42):
    info = df.groupby("job_id").agg(
        industry=("industry", "first"), has_gendered=("binary_id", "max")
    ).reset_index()
    info["strat_key"] = info["industry"] + "_" + info["has_gendered"].astype(str)
    if (info["strat_key"].value_counts() < 2).any():
        info["strat_key"] = info["industry"]
    train_ids, test_ids = train_test_split(
        info["job_id"], test_size=test_size, random_state=seed, stratify=info["strat_key"]
    )
    return df[df.job_id.isin(set(train_ids))].copy(), df[df.job_id.isin(set(test_ids))].copy()


def main():
    # ── Rebuild the original test set ──
    r = prepare_labels(load_workbook_data(DATA / "annotation_random.xlsx", "random"))
    b = prepare_labels(load_workbook_data(DATA / "annotation_biased.xlsx", "biased"))
    r_tr, r_te = split_by_posting(r)
    b_tr, b_te = split_by_posting(b)
    test = pd.concat([r_te, b_te], ignore_index=True).drop_duplicates(
        subset=["job_id", "sentence_num"], keep="first"
    )
    print(f"Random train/test postings: {r_tr.job_id.nunique()}/{r_te.job_id.nunique()}  "
          f"sentences {len(r_tr)}/{len(r_te)}")
    print(f"Biased train/test postings: {b_tr.job_id.nunique()}/{b_te.job_id.nunique()}  "
          f"sentences {len(b_tr)}/{len(b_te)}")
    print(f"Combined test: {len(test)} sentences, {test.job_id.nunique()} postings")

    # ── Run the saved model ──
    tok = BertTokenizer.from_pretrained(MODEL)
    model = BertForSequenceClassification.from_pretrained(MODEL).eval()
    torch.set_num_threads(os.cpu_count())
    texts = test.sentence_text.tolist()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], padding="max_length", truncation=True,
                      max_length=128, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1).numpy())
    probs = np.concatenate(probs)
    preds = probs.argmax(1)
    y = test.multi_id.values

    # ── Compare to the paper ──
    print(classification_report(y, preds, target_names=["neutral", "masculine", "feminine"], digits=4))
    cm = confusion_matrix(y, preds)
    print("Confusion matrix (rows=true N/M/F, cols=pred N/M/F):\n", cm)
    print("MATCHES PAPER:", np.array_equal(cm, PAPER_CM))
    print("macro-F1 %.4f  acc %.4f" % (f1_score(y, preds, average="macro"), accuracy_score(y, preds)))

    test = test.assign(prob_neutral=probs[:, 0], prob_masculine=probs[:, 1],
                       prob_feminine=probs[:, 2], pred=preds)
    test.to_csv(OUT, index=False)
    print("saved", OUT.name)


if __name__ == "__main__":
    main()

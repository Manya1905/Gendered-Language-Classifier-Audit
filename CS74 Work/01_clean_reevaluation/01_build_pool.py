"""
E1 step 1: build the pooled dataset.

- Loads the random and enriched annotation workbooks with CS72's exact label logic.
- Pools them into one table (one row per sentence per sample).
- Adds company_id / company_name from the raw Kaggle postings.csv.
- Marks:
    unlabeled       the 165 flagged sentences that never got a label (default neutral)
    in_both_samples postings drawn into both the random and the enriched sample
- Adds text_norm (lowercased, whitespace collapsed) for near-match duplicate checks.

Prints checks against known numbers, then saves data/pool.csv.

Usage:
    python 01_build_pool.py
"""
import pandas as pd

from common import (ENRICHED_XLSX, OUT, POOL_CSV, POSTINGS_CSV, RANDOM_XLSX, VALID_LABELS,
                    load_workbook_data, normalize, prepare_labels)


def check(name, got, expected):
    status = "OK " if got == expected else "MISMATCH"
    print(f"  [{status}] {name}: {got} (expected {expected})")
    return got == expected


def main():
    # ── Load both samples ──
    random_df = prepare_labels(load_workbook_data(RANDOM_XLSX, "random"))
    enriched_df = prepare_labels(load_workbook_data(ENRICHED_XLSX, "enriched"))
    pool = pd.concat([random_df, enriched_df], ignore_index=True)

    # ── Mark the unlabeled sentences: flagged, but no label from you or Claude ──
    pool["unlabeled"] = (pool["flagged"]
                         & ~pool["your_label"].isin(VALID_LABELS)
                         & ~pool["suggested_label"].isin(VALID_LABELS))

    # ── Mark postings that were drawn into both samples ──
    n_sources = pool.groupby("job_id")["source"].nunique()
    pool["in_both_samples"] = pool["job_id"].isin(n_sources[n_sources > 1].index)

    # ── Add company from the raw Kaggle postings ──
    postings = pd.read_csv(POSTINGS_CSV, usecols=["job_id", "company_id", "company_name"])
    postings["job_id"] = postings["job_id"].astype(str)
    postings["company_id"] = postings["company_id"].astype("Int64")
    pool = pool.merge(postings, on="job_id", how="left", validate="many_to_one")

    pool["text_norm"] = pool["sentence_text"].map(normalize)

    # ── Checks against numbers we already know ──
    print("Checks:")
    ok = all([
        check("total sentences", len(pool), 19854),
        check("masculine", int((pool.label == "masculine").sum()), 1686),
        check("feminine", int((pool.label == "feminine").sum()), 1596),
        check("neutral", int((pool.label == "neutral").sum()), 16572),
        check("postings (random)", pool[pool.source == "random"].job_id.nunique(), 300),
        check("postings (enriched)", pool[pool.source == "enriched"].job_id.nunique(), 300),
        check("unlabeled sentences", int(pool.unlabeled.sum()), 165),
        check("unlabeled in enriched", int(pool[pool.source == "enriched"].unlabeled.sum()), 165),
        check("postings in both samples", pool[pool.in_both_samples].job_id.nunique(), 4),
        check("sentences missing company_id", int(pool.company_id.isna().sum()), 0),
        check("companies", pool.company_id.nunique(), 412),
    ])
    print("All checks passed." if ok else "SOME CHECKS FAILED - look before continuing.")

    print("\nSentences by industry and sample:")
    print(pool.groupby(["industry", "source"]).size().unstack())

    # ── Save ──
    OUT.mkdir(exist_ok=True)
    cols = ["source", "industry", "company_id", "company_name", "job_id", "title",
            "sentence_num", "sentence_text", "text_norm", "flagged", "masculine_words",
            "feminine_words", "suggested_label", "your_label", "label", "label_id",
            "has_gendered", "unlabeled", "in_both_samples"]
    pool[cols].to_csv(POOL_CSV, index=False)
    print(f"\nSaved {POOL_CSV.relative_to(OUT.parent.parent)} ({len(pool)} rows)")


if __name__ == "__main__":
    main()

"""
E1 step 2: assign every company to train, val, or test.

How (see E1_methodology.md, step 2):
- The unit is the COMPANY: all of a company's postings land in the same group.
- Companies are cut into 20 slices with sklearn's StratifiedGroupKFold, so each slice
  has about the same number of sentences and the same mix of
  industry x sample (random/enriched) x label.
- Slices 0-13 -> train, 14-16 -> val, 17-19 -> test  (70 / 15 / 15).
- Made once with seed 42 and saved, so every later script and all 3 training seeds
  use the same split.

Usage:
    python 02_make_split.py
Saves:
    data/company_split.csv   one row per company: company_id, company_name, slice, split
"""
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (N_SLICES, SPLIT_CSV, SPLIT_SEED, TRAIN_SLICES, VAL_SLICES, read_pool,
                    summarize)


def slice_to_split(s):
    if s < TRAIN_SLICES:
        return "train"
    if s < TRAIN_SLICES + VAL_SLICES:
        return "val"
    return "test"


def main():
    pool = read_pool()

    # The "kind" of each sentence that the slices should all have the same mix of
    kind = pool["industry"] + "_" + pool["source"] + "_" + pool["label"]

    sgkf = StratifiedGroupKFold(n_splits=N_SLICES, shuffle=True, random_state=SPLIT_SEED)
    slice_of = {}
    with warnings.catch_warnings():
        # sklearn warns that some kinds have fewer members than 20 slices; that's expected
        warnings.filterwarnings("ignore", message="The least populated class")
        for s, (_, idx) in enumerate(sgkf.split(pool, kind, groups=pool["company_id"])):
            for company in pool["company_id"].iloc[idx].unique():
                assert company not in slice_of, f"company {company} landed in two slices"
                slice_of[company] = s

    companies = (pool.groupby("company_id")
                 .agg(company_name=("company_name", "first"),
                      postings=("job_id", "nunique"),
                      sentences=("job_id", "size"))
                 .reset_index())
    companies["slice"] = companies["company_id"].map(slice_of)
    companies["split"] = companies["slice"].map(slice_to_split)

    # ── Checks ──
    assert companies["slice"].notna().all(), "some company got no slice"
    assert companies["company_id"].is_unique

    # ── Is each group a mini copy of the whole dataset? ──
    pool = pool.merge(companies[["company_id", "split"]], on="company_id", validate="many_to_one")
    table = pd.DataFrame({"ALL": summarize(pool),
                          **{g: summarize(pool[pool.split == g]) for g in ["train", "val", "test"]}}).T
    table.insert(1, "share", [f"{100 * n / len(pool):.0f}%" for n in table["sentences"]])
    print("Before removals (step 3 removes the 165 and duplicates from val/test):\n")
    print(table.to_string())

    companies.to_csv(SPLIT_CSV, index=False)
    print(f"\nSaved {SPLIT_CSV.name} ({len(companies)} companies: "
          + ", ".join(f"{g} {(companies.split == g).sum()}" for g in ["train", "val", "test"]) + ")")


if __name__ == "__main__":
    main()

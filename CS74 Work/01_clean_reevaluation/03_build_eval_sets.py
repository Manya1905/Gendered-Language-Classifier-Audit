"""
E1 step 3: build the train / val / test files.

Rules (see E1_methodology.md, step 3):
- TRAIN: enriched sentences from train companies. Kept exactly as-is, including the
  unlabeled sentences (default neutral) and repeated sentences, same as CS72.
- VAL and TEST, in this order:
    1. drop the unlabeled sentences (their "neutral" is a default, not a label)
    2. postings drawn into both samples: keep only the enriched copy
    3. drop sentences that also appear in TRAIN (near-match: lowercase, collapse spaces)
  Test is saved twice to show the effect of step 3:
    test_a = steps 1-2 only (company split only)
    test_b = steps 1-3      (company split + cross-split duplicates removed)
  Val always gets steps 1-3.

Usage:
    python 03_build_eval_sets.py
Saves (in data/):
    train.csv, val.csv, test_a.csv, test_b.csv
"""
import pandas as pd

from common import OUT, SPLIT_CSV, read_pool, summarize


def clean_eval(df, train_texts, drop_train_duplicates):
    """Apply the val/test removals. Returns the cleaned set and a log of what was removed."""
    log = {"start": len(df)}
    df = df[~df["unlabeled"]]
    log["- unlabeled"] = log["start"] - len(df)
    n = len(df)
    df = df[~(df["in_both_samples"] & (df["source"] == "random"))]
    log["- 2nd copy of posting"] = n - len(df)
    if drop_train_duplicates:
        n = len(df)
        df = df[~df["text_norm"].isin(train_texts)]
        log["- also in train"] = n - len(df)
    log["end"] = len(df)
    return df.copy(), log


def main():
    pool = read_pool()
    split = pd.read_csv(SPLIT_CSV)
    pool = pool.merge(split[["company_id", "split"]], on="company_id", validate="many_to_one")

    # ── Train: what the new model learns from ──
    train = pool[(pool.split == "train") & (pool.source == "enriched")].copy()
    train_texts = set(train["text_norm"])

    # ── Val and test ──
    val, val_log = clean_eval(pool[pool.split == "val"], train_texts, True)
    test_a, a_log = clean_eval(pool[pool.split == "test"], train_texts, False)
    test_b, b_log = clean_eval(pool[pool.split == "test"], train_texts, True)

    # ── Checks ──
    groups = {g: set(pool[pool.split == g].company_id) for g in ["train", "val", "test"]}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"]
                or groups["val"] & groups["test"]), "a company is in two groups"
    for name, df in [("val", val), ("test_a", test_a), ("test_b", test_b)]:
        assert not df["unlabeled"].any(), f"unlabeled sentence left in {name}"
        assert not df.duplicated(["job_id", "sentence_num"]).any(), f"double-counted sentence in {name}"
    assert not val["text_norm"].isin(train_texts).any()
    assert not test_b["text_norm"].isin(train_texts).any()
    assert train["source"].eq("enriched").all()
    print("All checks passed.\n")

    # ── Report ──
    print("Removals from val / test:")
    print(pd.DataFrame({"val": val_log, "test_a": a_log, "test_b": b_log}).fillna("").to_string())

    sets = {"train": train, "val": val, "test_a": test_a, "test_b": test_b}
    table = pd.DataFrame({k: summarize(v) for k, v in sets.items()}).T
    table["flagged"] = [int(v["flagged"].sum()) for v in sets.values()]
    table["unlabeled"] = [int(v["unlabeled"].sum()) for v in sets.values()]
    print("\n" + table.to_string())

    for name, df in sets.items():
        df.drop(columns=["split"]).to_csv(OUT / f"{name}.csv", index=False)
    print("\nSaved " + ", ".join(f"{k}.csv" for k in sets))


if __name__ == "__main__":
    main()

"""
E1 step 5: score every model ONCE on every test set and build the results tables.

Models
  - BERT (clean)      retrained on the company split, 3 seeds   output_cpu/seed_42, 43, 44
  - RoBERTa (clean)   same training, 3 seeds                    output_cpu/roberta-base_seed_*
  - Original (CS72)   the audited model                         original_model/
  - 3 keyword baselines (no context at all; they read the list words already found in
    each sentence by CS72's matching code: Gaucher et al. + the 12 curated expansions)
      List rule                    each list word gets its word-list label
      Learned rule (clean train)   each list word gets the label it most often had in
                                   train.csv (what the new models trained on)
      Learned rule (CS72 train)    same, learned from CS72's original training split
                                   (what the original model trained on)
    Several list words in one sentence: any masculine -> masculine, else any feminine ->
    feminine, else neutral (the same rule as the labels, where "both" counts as masculine).
    A word never seen in training, or tied, falls back to its word-list label.

Test sets (each model is run once per set)
  new_unseen   new evaluation set minus the 88 sentences found in the annotated data  <- headline
  new_full     new evaluation set, all 3,721 sentences
  test_b       clean test set minus sentences also in train
  test_a       clean test set
  cs72_*       CS72's own test set. The original model is NOT compared on the clean test
               sets (it trained on most of them). Its saved CS72 predictions are re-scored
               "as published" (0.853) and "without unlabeled" (minus the 24 unlabeled).
Who is scored where:
  new_unseen, new_full   every model and every baseline
  test_a, test_b         new models, list rule, learned rule (clean train)
  cs72_*                 original model, list rule, learned rule (CS72 train)

Every score is also split into subsets: all, random, enriched, flagged-only (F11).

Check: before trusting the original model on the server, it is re-run on its own CS72
test set and must reproduce its saved predictions (and 0.853).

Usage (on Polaris, from ~/cs74/01_clean_reevaluation):
    python 05_evaluate.py
Needs: output_cpu/ (6 models), original_model/ (From CS72/biased_multiclass/checkpoint-best),
       data/{test_a,test_b,new_eval,train,pool}.csv, original_test_predictions.csv
Saves to results/:
    summary.md                 headline tables (mean +/- SD over 3 seeds)
    all_scores.csv             every model x seed x test set x subset
    run_info.csv               the 6 training records in one table
    predictions/<model>_<set>.csv   per-sentence predictions (and probabilities)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["neutral", "masculine", "feminine"]          # ids 0 / 1 / 2
HERE = Path(__file__).resolve().parent
SUBSETS = {
    "all": lambda d: d,
    "random": lambda d: d[d.source == "random"],
    "enriched": lambda d: d[d.source == "enriched"],
    "flagged_only": lambda d: d[d.flagged],
}
KEEP_COLS = ["source", "industry", "company_id", "job_id", "sentence_num", "sentence_text",
             "flagged", "masculine_words", "feminine_words", "label", "label_id"]
MODEL_ORDER = ["Original (CS72)", "BERT (clean)", "RoBERTa (clean)", "List rule",
               "Learned rule (clean train)", "Learned rule (CS72 train)"]
SET_TITLES = {
    "new_unseen": "New evaluation set, unseen sentences (HEADLINE: every model on equal footing)",
    "new_full": "New evaluation set, all sentences",
    "test_b": "Clean test set, minus sentences also in train (test_b)",
    "test_a": "Clean test set (test_a)",
    "cs72_as_published": "CS72's own test set, as published",
    "cs72_without_unlabeled": "CS72's own test set, without the 24 unlabeled sentences",
}


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────
def score(y_true, y_pred):
    """Macro scores + per-class scores (the audit's main metrics)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "n": len(y_true),
        "n_masculine": int((y_true == 1).sum()),
        "n_feminine": int((y_true == 2).sum()),
        "macro_f1": f1.mean(),
        "macro_precision": p.mean(),
        "macro_recall": r.mean(),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{name}": f1[i] for i, name in enumerate(LABELS)},
        **{f"precision_{name}": p[i] for i, name in enumerate(LABELS)},
        **{f"recall_{name}": r[i] for i, name in enumerate(LABELS)},
    }


def score_rows(model, seed, test_set, df):
    """One row per subset. df needs label_id and pred_id."""
    return [{"model": model, "seed": seed, "test_set": test_set, "subset": sub,
             **score(pick(df)["label_id"].values, pick(df)["pred_id"].values)}
            for sub, pick in SUBSETS.items()]


# ──────────────────────────────────────────────
# Neural models
# ──────────────────────────────────────────────
def predict(model_dir, texts, device, batch_size=64, max_length=128):
    """Same input processing as training: pad/truncate to 128 tokens."""
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding="max_length", truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            probs.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
    return np.concatenate(probs)


def with_preds(test, probs):
    out = test[KEEP_COLS + (["seen_in_annotated"] if "seen_in_annotated" in test else [])].copy()
    out["pred_id"] = probs.argmax(1)
    out["pred"] = [LABELS[i] for i in out["pred_id"]]
    for i, name in enumerate(LABELS):
        out[f"prob_{name}"] = probs[:, i]
    return out


# ──────────────────────────────────────────────
# Keyword baselines
# ──────────────────────────────────────────────
def words(cell):
    return [w for w in str(cell).split(", ") if w] if isinstance(cell, str) else []


def learn_word_labels(train):
    """Each list word -> the label it most often had in the training sentences containing it.
    Ties are left out (they fall back to the word-list label)."""
    rows = [(w, r.label_id) for r in train[train.flagged].itertuples()
            for w in words(r.masculine_words) + words(r.feminine_words)]
    counts = pd.DataFrame(rows, columns=["word", "label_id"]).value_counts().unstack(fill_value=0)
    top = counts.max(axis=1)
    unique_top = (counts.eq(top, axis=0).sum(axis=1) == 1)
    return counts[unique_top].idxmax(axis=1).to_dict()


def keyword_predict(df, word_labels=None):
    """List rule (word_labels=None) or learned rule. Any masculine word wins, then feminine."""
    preds = []
    for m, f in zip(df["masculine_words"], df["feminine_words"]):
        if word_labels is None:
            labs = [1] * len(words(m)) + [2] * len(words(f))
        else:
            labs = [word_labels.get(w, 1) for w in words(m)] + [word_labels.get(w, 2) for w in words(f)]
        preds.append(1 if 1 in labs else 2 if 2 in labs else 0)
    out = df[KEEP_COLS].copy()
    out["pred_id"] = preds
    out["pred"] = [LABELS[i] for i in preds]
    return out


def cs72_train_split(pool):
    """Rebuild CS72's enriched training split: CS72 train_bert.py split_by_posting(), seed 42.
    Check: CS72's log shows 680 steps/epoch at batch 16 -> 10,865-10,880 sentences."""
    e = pool[pool.source == "enriched"]
    info = e.groupby("job_id").agg(industry=("industry", "first"),
                                   has_gendered=("has_gendered", "max")).reset_index()
    key = info["industry"] + "_" + info["has_gendered"].astype(str)
    assert key.value_counts().min() >= 2
    train_ids, test_ids = train_test_split(info["job_id"], test_size=0.2, random_state=42, stratify=key)
    return e[e.job_id.isin(set(train_ids))], set(test_ids)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="output_cpu")
    ap.add_argument("--original-model", default="original_model")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--original-preds", default="original_test_predictions.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    out = Path(args.out_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    read = lambda name: pd.read_csv(data / f"{name}.csv", dtype={"job_id": str, "company_id": str})

    tests = {name: read(name) for name in ["test_a", "test_b", "new_eval"]}
    train, pool = read("train"), read("pool")
    assert len(tests["new_eval"]) == 3721 and tests["new_eval"].seen_in_annotated.sum() == 88

    # ── CS72's own test set (saved predictions of the original model) ──
    orig = pd.read_csv(args.original_preds, dtype={"job_id": str})
    orig["source"] = orig["source"].replace({"biased": "enriched"})
    orig = orig.merge(pool[["source", "job_id", "sentence_num", "unlabeled", "company_id",
                            "masculine_words", "feminine_words", "label"]],
                      on=["source", "job_id", "sentence_num"], how="left", validate="one_to_one")
    assert orig["unlabeled"].notna().all()
    orig = orig.rename(columns={"multi_id": "label_id", "pred": "pred_id"})
    cs72_sets = {"cs72_as_published": orig, "cs72_without_unlabeled": orig[~orig.unlabeled.astype(bool)]}

    # ── CS72's training split, rebuilt (for the CS72-trained learned rule) ──
    cs72_train, cs72_test_ids = cs72_train_split(pool)
    n_steps = -(-len(cs72_train) // 16)
    print(f"CS72 training split rebuilt: {len(cs72_train):,} sentences, {n_steps} steps/epoch")
    assert n_steps == 680, "rebuilt CS72 split does not match CS72's log (680 steps/epoch)"
    assert set(orig.loc[orig.source == "enriched", "job_id"]) <= cs72_test_ids

    rows = []

    def add(model, seed, set_name, preds):
        """Score one prediction table; for the new set, also score the unseen version."""
        if set_name == "new_eval":
            rows.extend(score_rows(model, seed, "new_full", preds))
            rows.extend(score_rows(model, seed, "new_unseen", preds[~preds.seen_in_annotated]))
        else:
            rows.extend(score_rows(model, seed, set_name, preds))

    def save(tag, set_name, preds):
        preds.to_csv(out / "predictions" / f"{tag}_{set_name}.csv", index=False)

    # ── 1. Original model: check it on its own test set, then run it on the new set ──
    print("\nOriginal model: re-running on its own CS72 test set (check)...")
    probs = predict(args.original_model, orig["sentence_text"].tolist(), device)
    agree = (probs.argmax(1) == orig["pred_id"].values).mean()
    redo_f1 = score(orig["label_id"].values, probs.argmax(1))["macro_f1"]
    print(f"  agrees with the saved predictions on {agree:.2%} of sentences, macro-F1 {redo_f1:.4f}")
    assert agree > 0.999 and round(redo_f1, 3) == 0.853, "original model on the server does not reproduce CS72"
    for name, d in cs72_sets.items():
        rows.extend(score_rows("Original (CS72)", None, name, d))
    paper = [r for r in rows if r["test_set"] == "cs72_as_published" and r["subset"] == "all"][0]
    assert round(paper["macro_f1"], 3) == 0.853

    preds = with_preds(tests["new_eval"], predict(args.original_model,
                                                  tests["new_eval"]["sentence_text"].tolist(), device))
    save("original", "new_eval", preds)
    add("Original (CS72)", None, "new_eval", preds)
    print(f"  Original on new set: macro-F1 {rows[-4]['macro_f1']:.4f} (unseen, all sentences)")

    # ── 2. New models: each one on each set, once ──
    models = []
    for d in sorted(Path(args.models_dir).iterdir()):
        if not (d / "checkpoint-best").is_dir() or d.name.startswith("quick"):
            continue
        family = "RoBERTa (clean)" if d.name.startswith("roberta") else "BERT (clean)"
        models.append((family, int(d.name.split("seed_")[1]), d))
    print("\nModels found:", [(f, s) for f, s, _ in models])
    assert len(models) == 6, "expected 3 BERT + 3 RoBERTa models"

    for family, seed, d in models:
        tag = f"{'roberta' if 'RoBERTa' in family else 'bert'}_seed{seed}"
        for set_name, test in tests.items():
            preds = with_preds(test, predict(d / "checkpoint-best", test["sentence_text"].tolist(), device))
            save(tag, set_name, preds)
            add(family, seed, set_name, preds)
        print(f"  {family} seed {seed}: macro-F1 test_b {[r for r in rows if r['model'] == family and r['seed'] == seed and r['test_set'] == 'test_b' and r['subset'] == 'all'][0]['macro_f1']:.4f}, "
              f"new (unseen) {[r for r in rows if r['model'] == family and r['seed'] == seed and r['test_set'] == 'new_unseen' and r['subset'] == 'all'][0]['macro_f1']:.4f}")

    # ── 3. Keyword baselines ──
    learned_clean, learned_cs72 = learn_word_labels(train), learn_word_labels(cs72_train)
    print(f"\nLearned rules: {len(learned_clean)} words (clean train), {len(learned_cs72)} words (CS72 train)")
    baselines = [("List rule", "list_rule", None, ["test_a", "test_b", "new_eval", "cs72"]),
                 ("Learned rule (clean train)", "learned_clean", learned_clean, ["test_a", "test_b", "new_eval"]),
                 ("Learned rule (CS72 train)", "learned_cs72", learned_cs72, ["new_eval", "cs72"])]
    for name, tag, word_labels, where in baselines:
        for set_name in where:
            if set_name == "cs72":
                for cs_name, d in cs72_sets.items():
                    rows.extend(score_rows(name, None, cs_name, keyword_predict(d, word_labels)))
                continue
            preds = keyword_predict(tests[set_name], word_labels)
            if set_name == "new_eval":
                preds["seen_in_annotated"] = tests["new_eval"]["seen_in_annotated"].values
            save(tag, set_name, preds)
            add(name, None, set_name, preds)
    # The learned word labels themselves, for the write-up
    pd.DataFrame({"word": sorted(set(learned_clean) | set(learned_cs72))}).assign(
        clean_train=lambda t: t.word.map(learned_clean).map(lambda i: LABELS[int(i)] if pd.notna(i) else ""),
        cs72_train=lambda t: t.word.map(learned_cs72).map(lambda i: LABELS[int(i)] if pd.notna(i) else ""),
    ).to_csv(out / "learned_rule_words.csv", index=False)

    scores = pd.DataFrame(rows)
    scores.to_csv(out / "all_scores.csv", index=False)

    # ── Training records ──
    info = [json.loads((d / "run_info.json").read_text()) | {"model": f} for f, _, d in models]
    pd.json_normalize(info).to_csv(out / "run_info.csv", index=False)

    # ── Summary: one table per test set ──
    def fmt(vals):
        return f"{vals.mean():.3f}" if len(vals) == 1 else f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}"

    lines = ["# E1 results", "",
             "Macro-F1 unless noted. New models: mean ± SD over 3 seeds. Others: one run.",
             "Per-class F1 columns are on all sentences of the set.", ""]
    for set_name, title in SET_TITLES.items():
        s = scores[scores.test_set == set_name]
        if s.empty:
            continue
        lines += [f"## {title}", "",
                  "| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for model in [m for m in MODEL_ORDER if m in set(s.model)]:
            g = s[s.model == model]
            sub = lambda k, col="macro_f1": fmt(g[g.subset == k][col])
            lines.append(f"| {model} | {g[g.subset == 'all']['n'].iloc[0]:,} | {sub('all')} | {sub('random')} | "
                         f"{sub('enriched')} | {sub('flagged_only')} | {sub('all', 'f1_masculine')} | "
                         f"{sub('all', 'f1_feminine')} | {sub('all', 'f1_neutral')} |")
        lines.append("")

    # Limitation note: how many sentences are cut off at 128 tokens
    lines += ["## Sentences longer than 128 tokens (cut off)", ""]
    for name, model_name in [("BERT", "bert-base-uncased"), ("RoBERTa", "roberta-base")]:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tok.model_max_length = 10**6        # we only count lengths here; silence the warning
        for set_name, test in tests.items():
            n_long = sum(len(tok.encode(t)) > 128 for t in test["sentence_text"])
            lines.append(f"- {name}, {set_name}: {n_long} of {len(test)}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {out}/: summary.md, all_scores.csv, run_info.csv, learned_rule_words.csv, predictions/")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()

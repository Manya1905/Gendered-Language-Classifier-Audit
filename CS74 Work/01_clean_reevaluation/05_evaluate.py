"""
E1 step 5: score every trained model on the test set ONCE and build the results tables.

Models scored:
  - BERT, retrained on the clean split   (output_cpu/seed_42, seed_43, seed_44)
  - RoBERTa, same training               (output_cpu/roberta-base_seed_42, ...)
  - Original CS72 model: NOT re-run. Its predictions on its own (leaky) test set already
    exist in 00_reproduce_original/original_test_predictions.csv (they reproduce the
    paper exactly), so we re-score those. Reported twice:
       "as published"         all 3,716 sentences (the paper's 0.853)
       "without unlabeled"    minus the 24 unlabeled sentences, the same exclusion
                              every new model gets (FINDINGS F3)

Test sets (from 03_build_eval_sets.py):
  test_a = company split only;  test_b = + sentences also in train removed.

Each score is reported for these subsets of the test set:
  all, random sample, enriched sample, flagged-only (sentences with a list word, F11)

Usage (on Polaris, after all training has finished):
    python 05_evaluate.py
Saves to results/:
    predictions/<model>_<test>.csv   per-sentence label, prediction, probabilities
    all_scores.csv                   every model x seed x test set x subset
    summary.md                       the headline tables (mean +/- SD over seeds)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["neutral", "masculine", "feminine"]          # ids 0 / 1 / 2
HERE = Path(__file__).resolve().parent
SUBSETS = {
    "all": lambda d: d,
    "random": lambda d: d[d.source == "random"],
    "enriched": lambda d: d[d.source == "enriched"],
    "flagged_only": lambda d: d[d.flagged],
}


def score(y_true, y_pred):
    """Macro scores + per-class F1 (the audit's main metrics)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="output_cpu")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--original-preds",
                    default=str(HERE.parent / "00_reproduce_original" / "original_test_predictions.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    out = Path(args.out_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    tests = {name: pd.read_csv(data / f"{name}.csv", dtype={"job_id": str})
             for name in ["test_a", "test_b"]}
    pool = pd.read_csv(data / "pool.csv", dtype={"job_id": str})

    # ── Find the trained models ──
    models = []
    for d in sorted(Path(args.models_dir).iterdir()):
        if not (d / "checkpoint-best").is_dir() or d.name.startswith("quick"):
            continue
        family = "RoBERTa (clean)" if d.name.startswith("roberta") else "BERT (clean)"
        models.append((family, int(d.name.split("seed_")[1]), d / "checkpoint-best"))
    print("Models found:", [(f, s) for f, s, _ in models])
    assert len(models) == 6, "expected 3 BERT + 3 RoBERTa models - is training finished?"

    rows = []

    # ── New models: run each one on each test set, once ──
    for family, seed, path in models:
        for test_name, test in tests.items():
            probs = predict(path, test["sentence_text"].tolist(), device)
            pred = probs.argmax(1)
            preds = test[["source", "industry", "company_id", "job_id", "sentence_num",
                          "sentence_text", "flagged", "label", "label_id"]].copy()
            preds["pred_id"] = pred
            preds["pred"] = [LABELS[i] for i in pred]
            for i, name in enumerate(LABELS):
                preds[f"prob_{name}"] = probs[:, i]
            tag = f"{'roberta' if 'RoBERTa' in family else 'bert'}_seed{seed}_{test_name}"
            preds.to_csv(out / "predictions" / f"{tag}.csv", index=False)

            for sub, pick in SUBSETS.items():
                s = pick(preds)
                rows.append({"model": family, "seed": seed, "test_set": test_name, "subset": sub,
                             **score(s["label_id"].values, s["pred_id"].values)})
            print(f"  {family} seed {seed} {test_name}: macro-F1 "
                  f"{rows[-4]['macro_f1']:.4f} (all), {rows[-1]['macro_f1']:.4f} (flagged-only)")

    # ── Original CS72 model on its own test set (re-scored, not re-run) ──
    orig = pd.read_csv(args.original_preds, dtype={"job_id": str})
    orig["source"] = orig["source"].replace({"biased": "enriched"})
    orig = orig.merge(pool[["source", "job_id", "sentence_num", "unlabeled"]],
                      on=["source", "job_id", "sentence_num"], how="left", validate="one_to_one")
    assert orig["unlabeled"].notna().all()
    orig = orig.rename(columns={"multi_id": "label_id", "pred": "pred_id"})
    for version, d in [("as published", orig), ("without unlabeled", orig[~orig.unlabeled])]:
        for sub, pick in SUBSETS.items():
            s = pick(d)
            rows.append({"model": "Original (CS72)", "seed": 42, "test_set": f"original, {version}",
                         "subset": sub, **score(s["label_id"].values, s["pred_id"].values)})
    paper = [r for r in rows if r["model"] == "Original (CS72)" and r["subset"] == "all"][0]
    assert round(paper["macro_f1"], 3) == 0.853, "original predictions no longer match the paper"

    scores = pd.DataFrame(rows)
    scores.to_csv(out / "all_scores.csv", index=False)

    # ── Summary: mean +/- SD over seeds ──
    def fmt(g, col):
        m, sd = g[col].mean(), g[col].std(ddof=1) if len(g) > 1 else 0.0
        return f"{m:.3f}" if len(g) == 1 else f"{m:.3f} ± {sd:.3f}"

    cols = ["macro_f1", "f1_neutral", "f1_masculine", "f1_feminine", "accuracy"]
    lines = ["# E1 results", "",
             "Mean ± SD over 3 seeds for the new models. The original model is one run.", ""]
    for sub in SUBSETS:
        lines += [f"## Subset: {sub}", "",
                  "| Model | Test set | n | " + " | ".join(cols) + " |",
                  "|---|---|---|" + "---|" * len(cols)]
        for (model, test_set), g in scores[scores.subset == sub].groupby(["model", "test_set"], sort=False):
            lines.append(f"| {model} | {test_set} | {g['n'].iloc[0]} | "
                         + " | ".join(fmt(g, c) for c in cols) + " |")
        lines.append("")

    # Limitation note: how many test sentences are cut off at 128 tokens
    lines += ["## Test sentences longer than 128 tokens (cut off)", ""]
    for name, model_name in [("BERT", "bert-base-uncased"), ("RoBERTa", "roberta-base")]:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tok.model_max_length = 10**6        # we only count lengths here; silence the warning
        for test_name, test in tests.items():
            n_long = sum(len(tok.encode(t)) > 128 for t in test["sentence_text"])
            lines.append(f"- {name}, {test_name}: {n_long} of {len(test)}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {out / 'summary.md'}, {out / 'all_scores.csv'}, predictions/")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()

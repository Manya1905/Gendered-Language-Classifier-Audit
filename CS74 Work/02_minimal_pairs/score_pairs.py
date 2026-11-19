"""
E2: does each model read context, or just the keyword?  Minimal pairs.

A minimal pair = 2 job-posting sentences that share one word; only how that word is used changes.
    We have a competitive salary.    neutral    (the word describes pay)
    We have a competitive culture.   masculine  (the word describes the culture)
A model that only looks at the word gives both sentences the same label, so it can't get the pair right.

Sets (column "set" in minimal_pairs.csv)
  list      the shared word is on the gendered word list (the main test)
  off_list  the shared word is gendered but not on the list
  mixed     F14: a feminine word alone vs the same sentence plus one masculine word

Models: the original CS72 model, and E1's retrained BERT and RoBERTa (seeds 42, 43, 44 each).

Steps
  1. Check the pairs file (laptop):    python score_pairs.py --check-only
  2. Run the models (Polaris):         python -u score_pairs.py --predict
       -> results/model_predictions.csv   (don't open it until the labels are final)
  3. Score (laptop, labels final):     python score_pairs.py
       Scores against every answer key available, each in its own folder:
         results/you/    the labels in minimal_pairs.csv (the user's own judgment)
         results/cs72/   CS72's rules, labeled by Claude (labeling/claude_cs72_labels.csv)
       Each folder: summary.md (sentence and pair accuracy, per set), per_sentence.csv
       (every model's answer for every sentence), scores.csv.
       Pairs where a key gives both sentences the same label are left out of that key's
       pair score (they no longer contrast); sentence accuracy uses every sentence.
Labels: neutral / masculine / feminine / both (or neu / masc / fem).
"both" counts as masculine, the same rule as CS72's training labels.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = ["neutral", "masculine", "feminine"]          # model output ids 0 / 1 / 2, as in training
SHORT = {"neu": "neutral", "masc": "masculine", "fem": "feminine", "both": "masculine"}
SETS = ["list", "off_list", "mixed"]
MODELS = ["Original"] + [f"{family} {seed}" for family in ["BERT", "RoBERTa"] for seed in (42, 43, 44)]


def read_pairs(path):
    pairs = pd.read_csv(path, dtype={"label": str})
    pairs["label"] = pairs["label"].str.strip().str.lower().replace(SHORT)
    return pairs


def check(pairs):
    """Everything that must be true before scoring. Returns a list of problems."""
    problems = []
    n_bad = (~pairs.label.isin(LABELS)).sum()
    if n_bad:
        problems.append(f"{n_bad} sentences have no valid label")
    if not pairs.set.isin(SETS).all():
        problems.append(f"every set must be one of {SETS}")
    for pid, g in pairs.groupby("pair_id"):
        if len(g) != 2:
            problems.append(f"pair {pid}: has {len(g)} sentences, needs 2")
    return problems


def same_label_pairs(pairs):
    return sorted(pid for pid, g in pairs.groupby("pair_id") if g.label.nunique() == 1)


# ──────────────────────────────────────────────
# Step 2: run the models (Polaris)
# ──────────────────────────────────────────────
def run_models(pairs, e1, out_file):
    """Run every model on every sentence, with E1's predict() (same input processing as E1)."""
    import torch
    spec = importlib.util.spec_from_file_location("evaluate_e1", e1 / "05_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    paths = {"Original": e1 / "original_model"}
    for seed in (42, 43, 44):
        paths[f"BERT {seed}"] = e1 / "output_cpu" / f"seed_{seed}" / "checkpoint-best"
        paths[f"RoBERTa {seed}"] = e1 / "output_cpu" / f"roberta-base_seed_{seed}" / "checkpoint-best"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for name in MODELS:
        print(f"Running {name}...")
        probs = ev.predict(paths[name], pairs["sentence_text"].tolist(), device)
        d = pairs[["pair_id", "sentence_text"]].assign(model=name)
        d["pred"] = [LABELS[i] for i in probs.argmax(1)]
        for i, label in enumerate(LABELS):
            d[f"prob_{label}"] = probs[:, i]
        rows.append(d)
    pd.concat(rows).to_csv(out_file, index=False)
    print(f"Saved {out_file}")


# ──────────────────────────────────────────────
# Step 3: score (laptop)
# ──────────────────────────────────────────────
def score(pairs, pred_file, out, key_name):
    out.mkdir(parents=True, exist_ok=True)
    same = same_label_pairs(pairs)
    preds = pd.read_csv(pred_file).set_index(["pair_id", "sentence_text"])

    # Per-sentence table: one column per model
    per = pairs[["pair_id", "set", "keyword", "sentence_text", "label"]].copy()
    keys = list(zip(per.pair_id, per.sentence_text))
    for name in MODELS:
        answers = preds[preds.model == name]["pred"]
        per[name] = [answers.get(k) for k in keys]
    if per[MODELS].isna().any().any():
        sys.exit("Some sentences have no model predictions (edited since --predict?). Re-run --predict.")
    for family in ["BERT", "RoBERTa"]:
        per[f"{family} right (of 3)"] = sum(per[m] == per.label for m in MODELS if m.startswith(family + " "))
    per.to_csv(out / "per_sentence.csv", index=False)

    # Counts per model and set: sentences right, and pairs with BOTH sentences right
    rows = []
    for name in MODELS:
        right = per[name] == per.label
        contrasting = ~per.pair_id.isin(same)
        pair_right = right[contrasting].groupby([per.set[contrasting], per.pair_id[contrasting]]).all()
        for s in SETS:
            pr = pair_right[s] if s in pair_right.index.get_level_values(0) else pair_right.iloc[:0]
            rows.append({"model": name, "set": s,
                         "sentences_right": int(right[per.set == s].sum()), "n_sentences": int((per.set == s).sum()),
                         "pairs_right": int(pr.sum()), "n_pairs": len(pr)})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)

    # Summary: the original, then mean ± SD over the 3 seeds for BERT and RoBERTa
    def cell(g, col, n):
        vals = g[col]
        if n == 0:
            return "-"
        if len(vals) == 1:
            return f"{vals.iloc[0]}/{n}"
        return f"{vals.mean():.1f} ± {vals.std():.1f} / {n} ({', '.join(map(str, vals))})"

    dropped = pairs[pairs.pair_id.isin(same)].drop_duplicates("pair_id")
    lines = [f"# E2 results: minimal pairs, answer key: {key_name}", "",
             "Pairs right = both sentences of the pair right. BERT and RoBERTa: mean ± SD over seeds 42, 43, 44 "
             "(each seed in brackets).",
             f"Left out of the pair score (this key gives both sentences the same label): {len(same)} pairs: "
             + ", ".join(f"{r.pair_id} {r.keyword}" for r in dropped.itertuples()), ""]
    for s in SETS:
        sc = scores[scores.set == s]
        n_sent, n_pairs = sc.n_sentences.iloc[0], sc.n_pairs.iloc[0]
        lines += [f"## {s} ({n_sent} sentences, {n_pairs} contrasting pairs)", "", "| Model | Sentences right | Pairs right |", "|---|---|---|"]
        for label, prefix in [("Original (CS72)", "Original"), ("BERT (clean)", "BERT "), ("RoBERTa (clean)", "RoBERTa ")]:
            g = sc[sc.model.str.startswith(prefix)]
            lines.append(f"| {label} | {cell(g, 'sentences_right', n_sent)} | {cell(g, 'pairs_right', n_pairs)} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=HERE / "minimal_pairs.csv", type=Path)
    ap.add_argument("--e1-dir", default=HERE.parent / "01_clean_reevaluation", type=Path)
    ap.add_argument("--out-dir", default=HERE / "results", type=Path)
    ap.add_argument("--check-only", action="store_true", help="check the pairs file only")
    ap.add_argument("--predict", action="store_true", help="run the models (no labels needed)")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    pairs = read_pairs(args.pairs)

    if args.predict:
        run_models(pairs, args.e1_dir, args.out_dir / "model_predictions.csv")
        return

    keys = {"you": ("your own judgment", pairs)}
    claude_file = HERE / "labeling" / "claude_cs72_labels.csv"
    if claude_file.exists():
        claude = pd.read_csv(claude_file, keep_default_na=False)[["pair_id", "sentence_text", "cs72_label"]]
        cs72 = pairs.drop(columns="label").merge(claude, on=["pair_id", "sentence_text"], validate="one_to_one")
        keys["cs72"] = ("CS72's rules (labeled by Claude)", cs72.rename(columns={"cs72_label": "label"}))

    print(f"{pairs.pair_id.nunique()} pairs, {len(pairs)} sentences:",
          pairs.drop_duplicates("pair_id").set.value_counts().to_dict())
    for key_name, labeled in keys.values():
        problems = check(labeled)
        if problems:
            sys.exit(f"Answer key '{key_name}' not ready:\n  " + "\n  ".join(problems))
        print(f"Key '{key_name}': OK, {len(same_label_pairs(labeled))} pairs with the same label on both sentences")
    if not args.check_only:
        for folder, (key_name, labeled) in keys.items():
            score(labeled, args.out_dir / "model_predictions.csv", args.out_dir / folder, key_name)


if __name__ == "__main__":
    main()

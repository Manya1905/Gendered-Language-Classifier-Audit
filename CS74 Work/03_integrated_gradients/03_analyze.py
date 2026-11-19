"""
E3 step 3: keyword share from the IG word scores.  Runs on the laptop (reads results/*.csv).

Per sentence:
  keyword share (abs)  = sum |keyword word scores| / sum |all word scores|      <- main measure
  keyword share (pos)  = sum of positive keyword scores / sum of positive scores  <- check
  keyword score        = signed sum of keyword word scores (push toward the target class)

Convergence rule: keep a sentence if |delta| <= 5% of sum |word scores| (the share's denominator),
so the IG error can move the share by at most ~5 points.  Dropped counts are reported.

Real sentences: share by model x class (Claude label) x right/wrong (that model's own prediction).
Pairs: target = keyword's class for both sentences. Using YOUR labels, a "context pair" is one
where one sentence is labeled the keyword's class (gendered use) and the other neutral (neutral use).
Question: does the keyword push harder toward its class in the gendered-use sentence?

Usage:  python 03_analyze.py   ->  results/e3_summary.md, results/e3_per_sentence.csv
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODELS = {"original": "Original", "bert42": "BERT (clean, seed 42)", "roberta42": "RoBERTa (clean, seed 42)"}
TOL = 0.05


def per_sentence(m):
    s = pd.read_csv(RES / f"ig_sentences_{m}.csv")
    s = s[s.skipped.isna() | (s.skipped == "")].drop(columns=["skipped"])
    w = pd.read_csv(RES / f"ig_words_{m}.csv")
    w["abs"] = w.score.abs()
    w["pos"] = w.score.clip(lower=0)
    g = w.groupby("e3_id")
    kw = w[w.is_keyword].groupby("e3_id")
    agg = pd.DataFrame({
        "total_abs": g["abs"].sum(), "total_pos": g["pos"].sum(),
        "kw_abs": kw["abs"].sum(), "kw_pos": kw["pos"].sum(), "kw_score": kw["score"].sum(),
        "n_words": g.size(),
    }).fillna(0)
    agg["share_abs"] = agg.kw_abs / agg.total_abs
    agg["share_pos"] = agg.kw_pos / agg.total_pos.where(agg.total_pos > 0)
    s = s.merge(agg, left_on="e3_id", right_index=True)
    s["converged"] = s.delta.abs() <= TOL * s.total_abs
    return s


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    sents = pd.read_csv(HERE / "e3_sentences.csv")
    lines = ["# E3 results: integrated gradients, keyword share", ""]
    allrows = []
    real_tables, pair_tables, conv = [], [], []
    for m, nice in MODELS.items():
        if not (RES / f"ig_sentences_{m}.csv").exists():
            continue
        s = per_sentence(m).merge(sents[["e3_id", "label", "keywords", "pair_id", "set", "pair_type",
                                         "keyword_class", "sentence_text"]], on="e3_id", suffixes=("", "_s"))
        s["model"] = m
        allrows.append(s)
        conv.append(f"| {nice} | {len(s)} | {int(s.converged.sum())} | "
                    f"{int((~s.converged & (s.part == 'real')).sum())} real, "
                    f"{int((~s.converged & (s.part == 'pair')).sum())} pair |")
        s = s[s.converged]

        # ---- real sentences
        r = s[s.part == "real"].copy()
        r["correct"] = r.label == r.pred
        for (label, correct), d in r.groupby(["label", "correct"]):
            real_tables.append((nice, label, "right" if correct else "wrong", len(d),
                                d.share_abs.mean(), d.share_abs.median(), d.share_pos.mean()))
        for correct, d in r.groupby("correct"):
            real_tables.append((nice, "all", "right" if correct else "wrong", len(d),
                                d.share_abs.mean(), d.share_abs.median(), d.share_pos.mean()))
        real_tables.append((nice, "all", "all", len(r), r.share_abs.mean(), r.share_abs.median(), r.share_pos.mean()))

        # ---- pairs
        p = s[s.part == "pair"]
        rows = []
        for pid, d in p.groupby("pair_id"):
            if len(d) != 2:
                continue
            kc = d.keyword_class.iloc[0]
            g_use = d[d.label == kc]
            n_use = d[d.label == "neutral"]
            if len(g_use) == 1 and len(n_use) == 1:
                g_use, n_use = g_use.iloc[0], n_use.iloc[0]
                rows.append({"pair_id": pid, "set": g_use.set, "keyword": g_use.keywords,
                             "kw_score_gendered": g_use.kw_score, "kw_score_neutral": n_use.kw_score,
                             "share_gendered": g_use.share_abs, "share_neutral": n_use.share_abs,
                             "pred_gendered": g_use.pred, "pred_neutral": n_use.pred, "kc": kc})
        pr = pd.DataFrame(rows)
        if len(pr):
            pr["push_higher_in_gendered"] = pr.kw_score_gendered > pr.kw_score_neutral
            pr["both_right"] = (pr.pred_gendered == pr.kc) & (pr.pred_neutral == "neutral")
            pr["model"] = nice
            pair_tables.append(pr)
        # all pairs: how much does the keyword's push change between the two contexts?
        d2 = p.groupby("pair_id").filter(lambda d: len(d) == 2).groupby("pair_id").kw_score
        spread = (d2.max() - d2.min())
        lines_pairs_all = (nice, len(spread), spread.median(), p.kw_score.abs().median())
        allrows[-1].attrs["pairs_all"] = lines_pairs_all

    lines += ["## Convergence (kept if |delta| <= 5% of the sum of |word scores|)", "",
              "| Model | Sentences | Kept | Dropped |", "|---|---|---|---|", *conv, ""]

    lines += ["## Real sentences: keyword share (target = model's predicted class)", "",
              "Right/wrong = that model's prediction vs. Claude's label. Share = keyword words' |score| as % of all words.", "",
              "| Model | Class | Right/wrong | n | Mean share (abs) | Median (abs) | Mean share (positive-only) |",
              "|---|---|---|---|---|---|---|"]
    for t in real_tables:
        lines.append(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {pct(t[4])} | {pct(t[5])} | {pct(t[6])} |")

    lines += ["", "## Minimal pairs: does the keyword push harder when it's used in a gendered way?", "",
              "Target = the keyword's own class for both sentences. Context pairs = your labels give one sentence "
              "the keyword's class and the other neutral.", "",
              "| Model | Context pairs | Keyword pushes harder in gendered-use sentence | Mean keyword score: gendered use | neutral use | Mean share: gendered use | neutral use | Model gets both right |",
              "|---|---|---|---|---|---|---|---|"]
    for pr in pair_tables:
        lines.append(f"| {pr.model.iloc[0]} | {len(pr)} | {int(pr.push_higher_in_gendered.sum())}/{len(pr)} | "
                     f"{pr.kw_score_gendered.mean():.2f} | {pr.kw_score_neutral.mean():.2f} | "
                     f"{pct(pr.share_gendered.mean())} | {pct(pr.share_neutral.mean())} | {int(pr.both_right.sum())}/{len(pr)} |")
    lines += ["", "By set (list / off_list / mixed):", "",
              "| Model | Set | Pairs | Pushes harder in gendered use | Mean score gendered | Mean score neutral |",
              "|---|---|---|---|---|---|"]
    for pr in pair_tables:
        for st, d in pr.groupby("set"):
            lines.append(f"| {pr.model.iloc[0]} | {st} | {len(d)} | {int(d.push_higher_in_gendered.sum())}/{len(d)} | "
                         f"{d.kw_score_gendered.mean():.2f} | {d.kw_score_neutral.mean():.2f} |")

    lines += ["", "All pairs (any labels): how much the keyword's push changes between the two sentences", "",
              "| Model | Pairs | Median change in keyword score | Median |keyword score| |", "|---|---|---|---|"]
    for a in allrows:
        n, k, med, base = a.attrs["pairs_all"]
        lines.append(f"| {n} | {k} | {med:.2f} | {base:.2f} |")

    out = pd.concat(allrows, ignore_index=True)
    out.to_csv(RES / "e3_per_sentence.csv", index=False)
    if pair_tables:
        pd.concat(pair_tables).to_csv(RES / "e3_context_pairs.csv", index=False)
    (RES / "e3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

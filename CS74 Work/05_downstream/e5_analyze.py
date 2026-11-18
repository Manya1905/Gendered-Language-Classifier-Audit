"""
E5 final step: does CS72's industry ranking depend on which labeler labels the sentences?

Labelers (10 runs): Original (CS72) | BERT (clean) seeds 42/43/44 | RoBERTa (clean) seeds 42/43/44 |
                    List rule | Learned rule (clean train) | Learned rule (CS72 train)
All use argmax (or the rule's label); a sentence counts as masculine if its label is masculine.

For each run:
  1. MAIN  % of postings per industry with >= 1 masculine sentence (CS72's metric, e5_common),
           ranking, and "holds" = finance and tech both above education and healthcare
  2. Per-sentence rate: % of sentences that are masculine, per industry
     (the main metric depends on posting length: longer postings have more chances)
  3. Bootstrap (1,000 resamples of postings within each industry, seed 42):
     95% CI for the gap (male-dominated avg - female-dominated avg), and how often
     the ranking / "holds" survive resampling
  4. One posting per company (F7, stricter than dropping exact duplicates): one posting
     picked at random (seed 42) per company_id; the 28 postings with no company are kept

Usage (local, from CS74 Work/05_downstream), after copying the 6 Polaris files into predictions/:
    python e5_analyze.py
    python e5_analyze.py --allow-missing      (test run without the Polaris files)
Needs: predictions/{bert,roberta}_seed{42,43,44}.csv   (e5_predict.py, Polaris)
       predictions/{list_rule,learned_clean,learned_cs72}.csv   (e5_baselines.py)
       From CS72/Data/postings.csv   (for company_id)
Saves: results/e5_summary.md, results/e5_all_runs.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from e5_common import (FEMALE_DOM, HERE, INDUSTRIES, MALE_DOM, PROJECT, RESULTS, describe,
                       pct_postings_masculine, read_sentences)

PREDS = HERE / "predictions"
POSTINGS_CSV = PROJECT / "From CS72" / "Data" / "postings.csv"
N_BOOT = 1000
SEED = 42
RUNS = ([("Original (CS72)", None, None)]
        + [("BERT (clean)", s, f"bert_seed{s}") for s in (42, 43, 44)]
        + [("RoBERTa (clean)", s, f"roberta_seed{s}") for s in (42, 43, 44)]
        + [("List rule", None, "list_rule"),
           ("Learned rule (clean train)", None, "learned_clean"),
           ("Learned rule (CS72 train)", None, "learned_cs72")])
CS72_ORDER = "finance > tech > education > healthcare"


def load_labels(df, allow_missing):
    """Masculine yes/no per sentence for each run, row-aligned with df."""
    labels = {}
    for name, seed, tag in RUNS:
        if tag is None:
            labels[(name, seed)] = (df["predicted_class"] == 1).values
            continue
        path = PREDS / f"{tag}.csv"
        if not path.exists():
            if allow_missing:
                print(f"  missing {path.name}, skipped")
                continue
            raise FileNotFoundError(f"{path} (copy it from Polaris, or use --allow-missing)")
        p = pd.read_csv(path, dtype={"job_id": str}, usecols=["job_id", "sentence_num", "pred_id"])
        assert len(p) == len(df) and (p["job_id"].values == df["job_id"].values).all() \
            and (p["sentence_num"].values == df["sentence_num"].values).all(), f"{tag}: rows don't line up"
        labels[(name, seed)] = (p["pred_id"] == 1).values
    return labels


def posting_table(df, is_masc):
    """One row per posting: industry, has_masculine (same steps as pct_postings_masculine)."""
    return (df.assign(is_masc=is_masc).groupby("job_id")
              .agg(industry=("industry", "first"), has_masc=("is_masc", "max")))


def bootstrap(posting):
    """Resample postings within each industry; return gap CI and how often the result survives."""
    rng = np.random.default_rng(SEED)
    groups = {ind: posting.loc[posting["industry"] == ind, "has_masc"].values.astype(float)
              for ind in INDUSTRIES}
    pct = np.column_stack([
        rng.choice(g, size=(N_BOOT, len(g)), replace=True).mean(axis=1) * 100
        for g in groups.values()])                                        # N_BOOT x 4
    col = {ind: i for i, ind in enumerate(INDUSTRIES)}
    male = pct[:, [col[i] for i in MALE_DOM]]
    female = pct[:, [col[i] for i in FEMALE_DOM]]
    gap = male.mean(axis=1) - female.mean(axis=1)
    same_order = np.all(np.diff(pct, axis=1) < 0, axis=1)               # INDUSTRIES is CS72's order
    return {"gap_ci_low": np.percentile(gap, 2.5), "gap_ci_high": np.percentile(gap, 97.5),
            "boot_holds": (male.min(axis=1) > female.max(axis=1)).mean() * 100,
            "boot_same_ranking": same_order.mean() * 100}


def one_per_company(df):
    """job_ids kept: one random posting per company; postings with no company_id all kept."""
    comp = pd.read_csv(POSTINGS_CSV, usecols=["job_id", "company_id"], dtype=str)
    jobs = df[["job_id"]].drop_duplicates().merge(comp, on="job_id", how="left")
    no_company = jobs[jobs["company_id"].isna()]
    picked = jobs.dropna(subset=["company_id"]).sample(frac=1, random_state=SEED) \
                 .drop_duplicates("company_id")
    keep = set(picked["job_id"]) | set(no_company["job_id"])
    print(f"  one per company: {len(keep):,} of {len(jobs):,} postings kept "
          f"({len(no_company)} with no company_id kept as-is)")
    return keep


def fmt_mean_sd(values):
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}" if len(values) > 1 else f"{values[0]:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_sentences()
    labels = load_labels(df, args.allow_missing)
    keep = one_per_company(df)
    in_keep = df["job_id"].isin(keep).values

    rows = []
    for (name, seed), is_masc in labels.items():
        pct = pct_postings_masculine(df, is_masc)
        sent_rate = pd.Series(is_masc).groupby(df["industry"].values).mean().reindex(INDUSTRIES) * 100
        pct_company = pct_postings_masculine(df[in_keep], is_masc[in_keep])
        rows.append({
            "labeler": name, "seed": seed, "masculine_sentences": int(is_masc.sum()),
            **{f"pct_{i}": pct[i] for i in INDUSTRIES}, **describe(pct),
            **{f"sent_rate_{i}": sent_rate[i] for i in INDUSTRIES},
            "sent_rate_holds": bool(sent_rate[MALE_DOM].min() > sent_rate[FEMALE_DOM].max()),
            **bootstrap(posting_table(df, is_masc)),
            **{f"company_{i}": pct_company[i] for i in INDUSTRIES},
            **{f"company_{k}": v for k, v in describe(pct_company).items()},
        })
        print(f"  {name}{f' seed {seed}' if seed else ''}: {pct.round(1).to_dict()}")
    runs = pd.DataFrame(rows)
    runs.round(3).to_csv(out / "e5_all_runs.csv", index=False)

    # ── Markdown ──
    run_name = lambda r: r["labeler"] + (f" (seed {int(r['seed'])})" if pd.notna(r["seed"]) else "")
    yes = lambda b: "yes" if b else "NO"
    L = ["# E5 results: does the industry ranking depend on the labeler?", "",
         "% of postings with at least one masculine sentence (CS72's metric), 2,500 postings per industry. "
         "Gap = (finance + tech)/2 − (education + healthcare)/2. "
         "Holds = finance and tech both above education and healthcare. "
         f"CS72's ranking: {CS72_ORDER}.", ""]

    L += ["## 1. Main result (retrained models: mean ± SD over 3 seeds)", "",
          "| Labeler | Finance | Tech | Education | Healthcare | Gap | Same ranking as CS72 | Holds |",
          "|" + "---|" * 8]
    for name, g in runs.groupby("labeler", sort=False):
        L.append(f"| {name} | " + " | ".join(fmt_mean_sd(g[f"pct_{i}"].values) + "%" for i in INDUSTRIES)
                 + f" | {fmt_mean_sd(g['gap'].values)} pts"
                 + f" | {(g['ranking'] == CS72_ORDER).sum()}/{len(g)} | {g['holds'].sum()}/{len(g)} |")

    L += ["", "## 2. Every run, with bootstrap", "",
          f"Bootstrap: {N_BOOT:,} resamples of postings within each industry. "
          "\"% same ranking\" / \"% holds\" = share of resamples where it still holds.", "",
          "| Run | Masc. sentences | Gap | Gap 95% CI | Ranking | % same ranking | % holds |",
          "|" + "---|" * 7]
    for _, r in runs.iterrows():
        L.append(f"| {run_name(r)} | {r['masculine_sentences']:,} | {r['gap']:.1f} | "
                 f"[{r['gap_ci_low']:.1f}, {r['gap_ci_high']:.1f}] | {r['ranking']} | "
                 f"{r['boot_same_ranking']:.0f}% | {r['boot_holds']:.0f}% |")

    L += ["", "## 3. Per-sentence masculine rate (% of sentences)", "",
          "Not affected by posting length.", "",
          "| Run | Finance | Tech | Education | Healthcare | Holds |", "|" + "---|" * 6]
    for _, r in runs.iterrows():
        L.append(f"| {run_name(r)} | " + " | ".join(f"{r[f'sent_rate_{i}']:.2f}%" for i in INDUSTRIES)
                 + f" | {yes(r['sent_rate_holds'])} |")

    L += ["", f"## 4. One posting per company ({len(keep):,} postings)", "",
          "| Run | Finance | Tech | Education | Healthcare | Gap | Ranking | Holds |", "|" + "---|" * 8]
    for _, r in runs.iterrows():
        L.append(f"| {run_name(r)} | " + " | ".join(f"{r[f'company_{i}']:.1f}%" for i in INDUSTRIES)
                 + f" | {r['company_gap']:.1f} | {r['company_ranking']} | {yes(r['company_holds'])} |")

    (out / "e5_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nSaved {out / 'e5_summary.md'} and e5_all_runs.csv")


if __name__ == "__main__":
    main()

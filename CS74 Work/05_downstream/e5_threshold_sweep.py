okay, """
E5 part 2: threshold sweep on the ORIGINAL model's saved predictions (no model run needed).

CS72 used argmax: a sentence was masculine if masculine was its most likely class.
Here a sentence is masculine at threshold t if P(masculine) >= t, for t = 0.50 ... 0.90.
(At t = 0.50 this is slightly stricter than argmax: a sentence can be argmax-masculine
with P(masculine) below 0.5, e.g. 0.45 / 0.35 / 0.20.)

At each t: % of postings per industry with >= 1 masculine sentence (CS72's metric),
the ranking, and whether finance and tech both stay above education and healthcare.
Nothing is tuned or chosen; every threshold is reported.

Check first: argmax must reproduce CS72's 77.0 / 67.4 / 63.9 / 56.6, or the script stops.

Usage (local, from CS74 Work/05_downstream):
    python e5_threshold_sweep.py
Saves: results/threshold_sweep.csv, results/threshold_sweep.md
"""

import numpy as np
import pandas as pd

from e5_common import (CS72_PCT, INDUSTRIES, RESULTS, describe,
                       pct_postings_masculine, read_sentences)

THRESHOLDS = np.round(np.arange(0.50, 0.901, 0.05), 2)


def one_row(df, label, is_masc):
    pct = pct_postings_masculine(df, is_masc)
    return {"threshold": label,
            "masculine_sentences": int(np.sum(is_masc)),
            **{ind: pct[ind] for ind in INDUSTRIES},
            **describe(pct)}


def main():
    df = read_sentences()
    print(f"Loaded {len(df):,} sentences, {df.job_id.nunique():,} postings")

    # Check: argmax reproduces CS72
    argmax_masc = df["predicted_class"] == 1
    got = pct_postings_masculine(df, argmax_masc).round(1)
    for ind, want in CS72_PCT.items():
        assert np.isclose(got[ind], want), f"{ind}: got {got[ind]}, CS72 had {want}"
    print("Check passed: argmax reproduces CS72 "
          + ", ".join(f"{ind} {got[ind]:.1f}%" for ind in INDUSTRIES))

    rows = [one_row(df, "argmax (CS72)", argmax_masc)]
    rows += [one_row(df, f"{t:.2f}", df["masculine_prob"] >= t) for t in THRESHOLDS]
    table = pd.DataFrame(rows)

    RESULTS.mkdir(exist_ok=True)
    table.round(2).to_csv(RESULTS / "threshold_sweep.csv", index=False)

    # Markdown table
    header = ("| Threshold | Masc. sentences | Finance | Tech | Education | Healthcare "
              "| Male-dom avg | Female-dom avg | Gap | Ranking | Holds? |")
    lines = ["# E5 part 2: threshold sweep (original model)", "",
             "% of postings with at least one masculine sentence (2,500 postings per industry).",
             "Masculine at threshold t = P(masculine) >= t. \"Holds\" = finance and tech both "
             "above education and healthcare.", "",
             header, "|" + "---|" * 11]
    for r in rows:
        lines.append(
            f"| {r['threshold']} | {r['masculine_sentences']:,} | "
            + " | ".join(f"{r[ind]:.1f}%" for ind in INDUSTRIES)
            + f" | {r['male_dom_avg']:.1f}% | {r['female_dom_avg']:.1f}% | {r['gap']:.1f} pts"
            + f" | {r['ranking']} | {'yes' if r['holds'] else 'NO'} |")
    n_hold = sum(r["holds"] for r in rows[1:])
    lines += ["", f"Holds at {n_hold} of {len(THRESHOLDS)} thresholds."]
    (RESULTS / "threshold_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[5:]))
    print(f"\nSaved {RESULTS / 'threshold_sweep.csv'} and threshold_sweep.md")


if __name__ == "__main__":
    main()

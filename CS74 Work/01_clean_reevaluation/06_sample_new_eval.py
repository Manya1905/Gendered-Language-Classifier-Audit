"""
E1 step 6a: pick a NEW evaluation set of postings that no model has seen.

Same sampling as CS72's segment_and_sample.py, just smaller and only from new postings:
  15 enriched + 15 random postings per industry = 120 postings.

Avoiding repeats (see E1_methodology.md, step 6):
  Layer 1  drop every posting from the 412 companies in CS72's 600 annotated postings
           (and postings with no company id, which can't be checked)
  Layer 2  drop postings whose full text matches one of the 600 (same job, new id)
  Layer 3  mark sentences that match ANY of the 19,854 annotated sentences
           (column seen_in_annotated), so results can be reported with and without them
  Within the new set: enriched is picked first, and random skips those postings.

Before picking, the copied CS72 code is run on all postings to check it re-picks
CS72's exact 300 enriched / 300 random postings and their sentences.

Usage:  python 06_sample_new_eval.py
Saves:  data/new_eval_postings.csv, data/new_eval_sentences.csv
"""
import re

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

from common import DATA, OUT, POOL_CSV, POSTINGS_CSV, normalize, read_pool

nltk.download("punkt_tab", quiet=True)

CLEANED_CSV = DATA / "cleaned_postings.csv"
NEW_POSTINGS_CSV = OUT / "new_eval_postings.csv"
NEW_SENTENCES_CSV = OUT / "new_eval_sentences.csv"
PER_INDUSTRY = 15            # per sample per industry (CS72 used 75)
RANDOM_SEED = 42             # same as CS72

# ──────────────────────────────────────────────
# Copied from CS72 segment_and_sample.py (word lists and matching, unchanged)
# ──────────────────────────────────────────────
MASCULINE_ENTRIES = [
    "active", "adventurous", "aggress*", "ambition", "ambitious", "analy*",
    "assert*", "athlet*", "autonom*", "boast*", "challenge*",
    "compet*", "confident", "courag*", "decide", "decisive",
    "decision*", "determin*", "dominant", "dominance", "force*",
    "greedy", "headstrong", "hierarch*", "hostil*", "impulsive",
    "independent", "independence", "individual*", "intellect*", "lead*", "logic",
    "masculine", "objective", "opinion", "outspoken", "persist",
    "principle*", "reckless", "stubborn", "superior",
    "self-confident", "self-confidence", "self-sufficience", "self-sufficient", "self-reliance", "self-reliant",
]

FEMININE_ENTRIES = [
    "affectionate", "child*", "cheer*", "commit*", "communal",
    "compassion*", "connect*", "considerate", "cooperat*", "depend*",
    "emotional", "empath*", "feminine", "flatterable", "gentle",
    "honest", "interpersonal", "interdependence", "interdependent", "interpersonal",
    "kind", "kinship", "loyal*", "modesty", "nag", "nurtur*",
    "pleasant*", "polite", "quiet*", "respon*", "sensitiv*",
    "submissive", "support*", "sympath*", "tender*", "together*",
    "trust*", "understand*", "warm*", "whin*", "yield*",
]

MASCULINE_EXPANDED = ["adventuresome", "strong-willed", "manly"]
FEMININE_EXPANDED = [
    "loving", "endearing", "chatty", "perky", "jovial",
    "kindness", "caring", "subservient", "reliable",
]


def build_word_matcher(entries, expanded):
    stems = []
    exact = set()
    for entry in entries:
        if entry.endswith("*"):
            stems.append(entry[:-1].lower())
        else:
            exact.add(entry.lower())
    for word in expanded:
        exact.add(word.lower())
    return stems, exact


masc_stems, masc_exact = build_word_matcher(MASCULINE_ENTRIES, MASCULINE_EXPANDED)
fem_stems, fem_exact = build_word_matcher(FEMININE_ENTRIES, FEMININE_EXPANDED)


def count_gendered_hits(text):
    """Count how many gendered stems/words match in the text (CS72's enrichment score)."""
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
    count = 0
    all_stems = masc_stems + fem_stems
    all_exact = masc_exact | fem_exact
    matched = set()
    for w in words:
        if w in matched:
            continue
        if w in all_exact:
            matched.add(w)
            count += 1
            continue
        for stem in all_stems:
            if w.startswith(stem) and stem not in matched:
                matched.add(stem)
                count += 1
                break
    return count


def find_gendered_words(sentence):
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", sentence.lower())
    masc_found = []
    fem_found = []
    seen = set()
    for w in words:
        if w in seen:
            continue
        if w in masc_exact:
            masc_found.append(w)
            seen.add(w)
            continue
        masc_match = False
        for stem in masc_stems:
            if w.startswith(stem):
                masc_found.append(w)
                seen.add(w)
                masc_match = True
                break
        if masc_match:
            continue
        if w in fem_exact:
            fem_found.append(w)
            seen.add(w)
            continue
        for stem in fem_stems:
            if w.startswith(stem):
                fem_found.append(w)
                seen.add(w)
                break
    return masc_found, fem_found


def segment_postings(sample_df):
    rows = []
    for _, posting in sample_df.iterrows():
        sentences = sent_tokenize(posting["description_clean"])
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            masc_found, fem_found = find_gendered_words(sentence)
            all_found = masc_found + fem_found
            rows.append({
                "job_id": posting["job_id"],
                "title": posting["title"],
                "industry_group": posting["industry_group"],
                "sentence_index": i,
                "sentence_text": sentence,
                "flagged": len(all_found) > 0,
                "masculine_words": ", ".join(masc_found) if masc_found else "",
                "feminine_words": ", ".join(fem_found) if fem_found else "",
                "all_flagged_words": ", ".join(all_found) if all_found else "",
            })
    return pd.DataFrame(rows)


def pick_enriched(df, n):
    """CS72 rule: the n postings with the most list words, per industry."""
    ids = (df.groupby("industry_group", group_keys=False)
           .apply(lambda g: g.nlargest(min(n, len(g)), "gendered_hits")).index)
    return df.loc[ids]


def pick_random(df, n):
    """CS72 rule: n random postings per industry, seed 42."""
    ids = (df.groupby("industry_group", group_keys=False)
           .apply(lambda g: g.sample(n=min(n, len(g)), random_state=RANDOM_SEED)).index)
    return df.loc[ids]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    df = pd.read_csv(CLEANED_CSV, dtype={"job_id": str})
    print(f"Cleaned postings: {len(df):,}")
    df["gendered_hits"] = df["description_clean"].apply(count_gendered_hits)

    pool = read_pool()
    annotated_ids = {s: set(pool.loc[pool.source == s, "job_id"]) for s in ["random", "enriched"]}

    # ── Check: the copied code re-picks CS72's exact postings and sentences ──
    print("\nCheck: copied CS72 code reproduces the original samples")
    for name, picker, n in [("enriched", pick_enriched, 75), ("random", pick_random, 75)]:
        redo = picker(df, n)
        same = set(redo["job_id"]) == annotated_ids[name]
        print(f"  {name}: re-picked the same 300 postings: {same}")
        assert same, f"copied sampling code does not reproduce CS72's {name} sample"
        redo_sents = segment_postings(redo)
        n_pool = (pool.source == name).sum()
        print(f"  {name}: re-split into {len(redo_sents):,} sentences (annotated: {n_pool:,})")
        assert len(redo_sents) == n_pool, "copied sentence splitting does not reproduce CS72"

    # ── Layer 1: drop CS72's companies, and postings with no company id ──
    companies = pd.read_csv(POSTINGS_CSV, dtype={"job_id": str}, usecols=["job_id", "company_id"])
    df = df.merge(companies, on="job_id", how="left", validate="one_to_one")
    used_companies = set(pool["company_id"])
    n0 = len(df)
    no_company = df["company_id"].isna()
    df = df[~no_company & ~df["company_id"].isin(used_companies)].copy()
    print(f"\nLayer 1: dropped {no_company.sum()} postings with no company id and "
          f"{n0 - no_company.sum() - len(df):,} from CS72's {len(used_companies)} companies")

    # ── Layer 2: drop postings whose full text matches an annotated posting ──
    all_cleaned = pd.read_csv(CLEANED_CSV, dtype={"job_id": str})
    all_annotated = annotated_ids["random"] | annotated_ids["enriched"]
    annotated_texts = set(all_cleaned.loc[all_cleaned.job_id.isin(all_annotated),
                                          "description_clean"].map(normalize))
    same_text = df["description_clean"].map(normalize).isin(annotated_texts)
    df = df[~same_text]
    print(f"Layer 2: dropped {same_text.sum()} postings with the same text as an annotated one")
    print(f"Postings left to pick from: {len(df):,}")
    print(df["industry_group"].value_counts().sort_index().to_string())

    # ── Pick: enriched first, then random from what's left ──
    enriched = pick_enriched(df, PER_INDUSTRY).assign(source="enriched")
    random_ = pick_random(df[~df.job_id.isin(enriched.job_id)], PER_INDUSTRY).assign(source="random")
    picked = pd.concat([random_, enriched], ignore_index=True)

    # Keyword gap vs CS72 (the top postings were already used by CS72's enriched sample)
    orig = all_cleaned[all_cleaned.job_id.isin(all_annotated)].copy()
    orig["gendered_hits"] = orig["description_clean"].apply(count_gendered_hits)
    print("\nAverage list words per posting:")
    for s in ["random", "enriched"]:
        print(f"  {s:9s} CS72 {orig[orig.job_id.isin(annotated_ids[s])].gendered_hits.mean():5.1f}"
              f"   new {picked[picked.source == s].gendered_hits.mean():5.1f}")

    # ── Split into sentences (CS72 code) and mark already-annotated sentences (layer 3) ──
    sents = []
    for s in ["random", "enriched"]:
        part = picked[picked.source == s]
        seg = segment_postings(part).merge(part[["job_id", "company_id"]], on="job_id")
        sents.append(seg.assign(source=s))
    sents = pd.concat(sents, ignore_index=True)
    sents["text_norm"] = sents["sentence_text"].map(normalize)
    sents["seen_in_annotated"] = sents["text_norm"].isin(set(pool["text_norm"]))

    # ── Checks ──
    assert len(picked) == 120 and picked.job_id.is_unique, "expected 120 distinct postings"
    assert (picked.groupby(["source", "industry_group"]).size() == PER_INDUSTRY).all()
    assert not picked.company_id.isin(used_companies).any(), "a CS72 company slipped in"
    assert not picked.job_id.isin(all_annotated).any(), "an annotated posting slipped in"
    assert sents.job_id.nunique() == 120

    # ── Report ──
    print("\nNew evaluation set:")
    for s in ["random", "enriched"]:
        p, t = picked[picked.source == s], sents[sents.source == s]
        top = p.company_id.value_counts()
        print(f"  {s:9s} {len(p)} postings from {p.company_id.nunique()} companies "
              f"(most from one company: {top.iloc[0]}) | {len(t):,} sentences, "
              f"{t.flagged.sum():,} flagged, {t.seen_in_annotated.sum()} already in annotated data")
    print(f"  total     {len(sents):,} sentences, {sents.flagged.sum():,} to send to Claude")

    OUT.mkdir(exist_ok=True)
    picked.drop(columns=["description_clean"]).to_csv(NEW_POSTINGS_CSV, index=False)
    sents.to_csv(NEW_SENTENCES_CSV, index=False)
    print(f"\nSaved {NEW_POSTINGS_CSV.name}, {NEW_SENTENCES_CSV.name}")


if __name__ == "__main__":
    main()

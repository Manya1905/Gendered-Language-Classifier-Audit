"""
E3 step 1: pick the sentences for integrated gradients.  Runs on the laptop.

Part "real" (99 sentences): flagged, unseen sentences from the new evaluation set.
  - 33 per class (Claude's label): neutral / masculine / feminine.
  - Within each class: 17 the ORIGINAL model got right, 16 it got wrong.
  - Random within each group (seed 42); at most 3 sentences per company overall.
  - Only sentences of <= 128 BERT tokens (F8), so attributions cover the whole sentence.
  - Keywords = the list words E1's matcher found (masculine_words + feminine_words).

Part "pair" (114 sentences): all E2 minimal-pair sentences.
  - Keyword = the pair's `keyword`; keyword class = from `pair_type` (the keyword's own gender).

Output: e3_sentences.csv (one row per sentence), pushed to Polaris for 02_run_ig.py.

Usage:  python 01_sample.py
"""
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
E1 = WORK / "01_clean_reevaluation"
PRED = E1 / "results" / "predictions"
PAIRS = WORK / "02_minimal_pairs" / "minimal_pairs.csv"
PAIR_PREDS = WORK / "02_minimal_pairs" / "results" / "model_predictions.csv"
TOKENIZER = WORK.parent / "From CS72" / "biased_multiclass" / "checkpoint-best"  # bert-base-uncased vocab
OUT = HERE / "e3_sentences.csv"

SEED = 42
PER_CLASS = 33
N_WRONG = 16           # per class; the rest (17) are right
COMPANY_CAP = 3
MAX_TOKENS = 128

WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")   # same word pattern as E1's list matcher


def split_list(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [w.strip().lower() for w in str(s).split(",") if w.strip()]


def keyword_found(sentence, keywords):
    words = set(WORD_RE.findall(sentence.lower()))
    return all(k in words for k in keywords)


def sample_real(tok):
    orig = pd.read_csv(PRED / "original_new_eval.csv")
    bert = pd.read_csv(PRED / "bert_seed42_new_eval.csv")
    rob = pd.read_csv(PRED / "roberta_seed42_new_eval.csv")
    assert (orig.sentence_text.values == bert.sentence_text.values).all()
    assert (orig.sentence_text.values == rob.sentence_text.values).all()
    df = orig.copy()
    df["row_in_new_eval"] = df.index
    df["pred_original"] = orig.pred
    df["pred_bert42"] = bert.pred
    df["pred_roberta42"] = rob.pred

    df = df[df.flagged & ~df.seen_in_annotated].copy()
    df["n_tokens"] = [len(tok.tokenize(s)) + 2 for s in df.sentence_text]
    too_long = (df.n_tokens > MAX_TOKENS).sum()
    df = df[df.n_tokens <= MAX_TOKENS].copy()
    df["correct_original"] = df.label == df.pred_original
    print(f"Real: {len(df) + too_long} flagged unseen; {too_long} dropped as > {MAX_TOKENS} tokens; {len(df)} left")

    # Smallest groups first, so the company cap never starves them.
    groups = []
    for label in ["neutral", "masculine", "feminine"]:
        for correct, n in [(False, N_WRONG), (True, PER_CLASS - N_WRONG)]:
            g = df[(df.label == label) & (df.correct_original == correct)]
            groups.append((len(g), label, correct, n, g))
    groups.sort(key=lambda x: x[0])

    per_company = {}
    picked = []
    for size, label, correct, n, g in groups:
        g = g.sample(frac=1, random_state=SEED)
        take = []
        for idx, row in g.iterrows():
            if len(take) == n:
                break
            if per_company.get(row.company_id, 0) >= COMPANY_CAP:
                continue
            per_company[row.company_id] = per_company.get(row.company_id, 0) + 1
            take.append(idx)
        assert len(take) == n, f"only {len(take)} for {label}/{correct}"
        picked += take
        print(f"  {label:9s} {'right' if correct else 'wrong'}: picked {n} of {size}")

    s = df.loc[picked].copy()
    s["keywords"] = [
        ";".join(dict.fromkeys(split_list(m) + split_list(f)))
        for m, f in zip(s.masculine_words, s.feminine_words)
    ]
    out = pd.DataFrame({
        "part": "real",
        "sentence_text": s.sentence_text,
        "keywords": s.keywords,
        "keyword_class": "",                  # real sentences: target = each model's predicted class
        "label": s.label,                     # Claude's label (CS72 setup)
        "pred_original": s.pred_original,
        "pred_bert42": s.pred_bert42,
        "pred_roberta42": s.pred_roberta42,
        "correct_original": s.correct_original,
        "company_id": s.company_id,
        "industry": s.industry,
        "row_in_new_eval": s.row_in_new_eval,
        "pair_id": pd.NA, "set": "", "pair_type": "",
        "n_tokens": s.n_tokens,
    })
    print(f"  companies: {s.company_id.nunique()}, max per company: {s.company_id.value_counts().max()}")
    return out


def load_pairs(tok):
    p = pd.read_csv(PAIRS)
    preds = pd.read_csv(PAIR_PREDS)
    key = {"Original": "pred_original", "BERT 42": "pred_bert42", "RoBERTa 42": "pred_roberta42"}
    wide = (preds[preds.model.isin(key)]
            .pivot_table(index=["pair_id", "sentence_text"], columns="model", values="pred", aggfunc="first")
            .rename(columns=key).reset_index())
    p = p.merge(wide, on=["pair_id", "sentence_text"], how="left", validate="one_to_one")
    assert p[list(key.values())].notna().all().all(), "missing E2 predictions for some pair sentences"
    label_map = {"neu": "neutral", "masc": "masculine", "fem": "feminine"}
    out = pd.DataFrame({
        "part": "pair",
        "sentence_text": p.sentence_text,
        "keywords": p.keyword.str.lower(),
        "keyword_class": p.pair_type.str.split("_").str[0],   # masculine_word -> masculine; feminine_word_alone... -> feminine
        "label": p.label.map(label_map).fillna(p.label),      # YOUR label (main E2 key)
        "pred_original": p.pred_original,
        "pred_bert42": p.pred_bert42,
        "pred_roberta42": p.pred_roberta42,
        "correct_original": pd.NA,
        "company_id": pd.NA, "industry": "", "row_in_new_eval": pd.NA,
        "pair_id": p.pair_id, "set": p.set, "pair_type": p.pair_type,
        "n_tokens": [len(tok.tokenize(s)) + 2 for s in p.sentence_text],
    })
    print(f"Pairs: {len(out)} sentences, {out.pair_id.nunique()} pairs, max tokens {out.n_tokens.max()}")
    return out


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False)
    real = sample_real(tok)
    pairs = load_pairs(tok)
    df = pd.concat([real, pairs], ignore_index=True)
    df.insert(0, "e3_id", range(len(df)))

    missing = [(r.e3_id, r.keywords) for r in df.itertuples()
               if not keyword_found(r.sentence_text, r.keywords.split(";"))]
    print(f"Keyword check: {len(missing)} sentences where a keyword isn't a whole word in the text")
    for m in missing:
        print("   ", m, df.loc[m[0], "sentence_text"])

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name}: {len(df)} rows")


if __name__ == "__main__":
    main()

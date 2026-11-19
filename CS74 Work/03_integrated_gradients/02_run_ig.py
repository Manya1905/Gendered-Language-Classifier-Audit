"""
E3 step 2: integrated gradients.  Runs on Polaris (needs the model weights + captum).

For each sentence in e3_sentences.csv and each model:
  - Target class:  "real" sentences -> the model's own PREDICTED class
                   "pair" sentences -> the keyword's class (pair_type), same for both sentences of a pair
  - Captum LayerIntegratedGradients on the WORD embedding layer, attributing the target LOGIT.
  - Baseline: same length, every token replaced by [PAD] except [CLS]/[SEP] (<s>/</s> for RoBERTa).
  - 50 steps; if the convergence delta is > 5% of (logit(input) - logit(baseline)), redo with 300 steps.
  - Token scores = sum over the embedding dimension; sub-tokens summed back into whitespace words.
    [CLS]/[SEP] are dropped.  Scores are kept SIGNED here; 03_analyze.py decides absolute vs positive.

Checks printed before anything is trusted:
  - word-by-word tokenization == whole-sentence tokenization (so word grouping is exact)
  - the model's prediction == the prediction saved in E1/E2 (so the right weights are loaded)

Outputs (results/):
  ig_sentences_<model>.csv   one row per sentence: target, prediction, probs, delta, checks
  ig_words_<model>.csv       one row per word: signed score, keyword flag

Usage (from ~/cs74/03_integrated_gradients):
  quick check:  python -u 02_run_ig.py --models original bert42 --limit 5
  full run:     python -u 02_run_ig.py --models original bert42
  RoBERTa:      python -u 02_run_ig.py --models roberta42
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "01_clean_reevaluation"
MODELS = {
    "original": E1 / "original_model",
    "bert42": E1 / "output_cpu" / "seed_42" / "checkpoint-best",
    "roberta42": E1 / "output_cpu" / "roberta-base_seed_42" / "checkpoint-best",
}
SENTENCES = HERE / "e3_sentences.csv"
OUT = HERE / "results"
CLASSES = ["neutral", "masculine", "feminine"]     # label ids 0, 1, 2 (E1's LABEL_TO_ID)
WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")        # E1's list-matcher word pattern
MAX_TOKENS = 128
DELTA_TOL = 0.05
RETRY_STEPS = 300


def word_tokens(tok, sentence, is_roberta):
    """Tokenize word by word so every token knows which whitespace word it came from."""
    words = sentence.split()
    toks, owner = [], []
    for i, w in enumerate(words):
        piece = tok.tokenize((" " + w) if (is_roberta and i > 0) else w)
        toks += piece
        owner += [i] * len(piece)
    ids = [tok.cls_token_id] + tok.convert_tokens_to_ids(toks) + [tok.sep_token_id]
    return words, ids, owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["original", "bert42"], choices=list(MODELS))
    ap.add_argument("--model-dir", help="override the path (only with one --models)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--limit", type=int, help="first N real + first N pair sentences (quick check)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(SENTENCES)
    if args.limit:
        df = pd.concat([df[df.part == "real"].head(args.limit), df[df.part == "pair"].head(args.limit)])
        print(f"QUICK CHECK: {len(df)} sentences, results NOT saved")

    for name in args.models:
        model_dir = Path(args.model_dir) if args.model_dir else MODELS[name]
        is_roberta = "roberta" in name
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)   # same as E1/E2
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        emb = model.base_model.embeddings.word_embeddings
        pad = tok.pad_token_id

        def forward(input_ids, position_ids):
            mask = torch.ones_like(input_ids)
            return model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids).logits

        lig = LayerIntegratedGradients(forward, emb)
        print(f"\n[{name}] {model_dir} | {len(df)} sentences | steps {args.steps}", flush=True)

        sent_rows, word_rows = [], []
        t0 = time.time()
        for k, r in enumerate(df.itertuples(), 1):
            words, ids, owner = word_tokens(tok, r.sentence_text, is_roberta)
            whole = tok(r.sentence_text)["input_ids"]
            tokens_match = ids == whole
            if len(ids) > MAX_TOKENS:
                sent_rows.append({"e3_id": r.e3_id, "skipped": f"{len(ids)} tokens"})
                continue

            input_ids = torch.tensor([ids])
            base_ids = torch.tensor([[ids[0]] + [pad] * (len(ids) - 2) + [ids[-1]]])
            # Explicit positions: RoBERTa would otherwise give [PAD] tokens a different position.
            start = (pad + 1) if is_roberta else 0
            pos = torch.arange(start, start + len(ids)).unsqueeze(0)

            with torch.no_grad():
                logits = forward(input_ids, pos)[0]
                base_logits = forward(base_ids, pos)[0]
            probs = torch.softmax(logits, -1).numpy()
            pred = CLASSES[int(logits.argmax())]
            target_name = pred if r.part == "real" else r.keyword_class
            target = CLASSES.index(target_name)
            gap = float(logits[target] - base_logits[target])

            steps = args.steps
            while True:
                attr, delta = lig.attribute(
                    input_ids, baselines=base_ids, additional_forward_args=(pos,), target=target,
                    n_steps=steps, internal_batch_size=steps, return_convergence_delta=True)
                rel = abs(float(delta)) / max(abs(gap), 1e-6)
                if rel <= DELTA_TOL or steps >= RETRY_STEPS:
                    break
                steps = RETRY_STEPS

            tok_scores = attr.sum(-1)[0].detach().numpy()[1:-1]      # drop [CLS]/[SEP]
            word_scores = np.zeros(len(words))
            n_sub = np.zeros(len(words), dtype=int)
            for s, o in zip(tok_scores, owner):
                word_scores[o] += s
                n_sub[o] += 1

            keywords = set(str(r.keywords).split(";"))
            saved = getattr(r, f"pred_{name}", None)
            sent_rows.append({
                "e3_id": r.e3_id, "part": r.part, "model": name, "target": target_name, "pred": pred,
                "prob_neutral": probs[0], "prob_masculine": probs[1], "prob_feminine": probs[2],
                "logit_target": float(logits[target]), "baseline_logit_target": float(base_logits[target]),
                "attr_sum": float(tok_scores.sum() + attr.sum(-1)[0, [0, -1]].sum()),
                "delta": float(delta), "rel_delta": rel, "steps": steps,
                "n_tokens": len(ids), "tokens_match": tokens_match,
                "saved_pred": saved, "pred_matches_saved": (saved == pred) if isinstance(saved, str) else None,
                "skipped": "",
            })
            for i, w in enumerate(words):
                word_rows.append({
                    "e3_id": r.e3_id, "model": name, "word_idx": i, "word": w, "score": word_scores[i],
                    "n_subtokens": n_sub[i],
                    "is_keyword": bool(set(WORD_RE.findall(w.lower())) & keywords),
                })
            if k % 25 == 0 or k == len(df):
                print(f"  {k}/{len(df)}  {time.time() - t0:.0f}s", flush=True)

        s = pd.DataFrame(sent_rows)
        w = pd.DataFrame(word_rows)
        done = s[s.skipped == ""]
        print(f"[{name}] checks:")
        print(f"  skipped (> {MAX_TOKENS} tokens): {(s.skipped != '').sum()}")
        print(f"  word-by-word tokens == whole-sentence tokens: {done.tokens_match.sum()}/{len(done)}")
        m = done.pred_matches_saved.dropna()
        print(f"  prediction == saved E1/E2 prediction: {int(m.sum())}/{len(m)}")
        print(f"  convergence: rel delta <= {DELTA_TOL}: {(done.rel_delta <= DELTA_TOL).sum()}/{len(done)} "
              f"(median {done.rel_delta.median():.4f}, max {done.rel_delta.max():.4f}; "
              f"{(done.steps > args.steps).sum()} needed {RETRY_STEPS} steps)")
        kw = w.groupby("e3_id").is_keyword.any()
        print(f"  sentences with >= 1 keyword located: {int(kw.sum())}/{len(kw)}")
        if not args.limit:
            s.to_csv(OUT / f"ig_sentences_{name}.csv", index=False)
            w.to_csv(OUT / f"ig_words_{name}.csv", index=False)
            print(f"  wrote results/ig_sentences_{name}.csv, results/ig_words_{name}.csv")
        else:
            print("  example:", " ".join(f"{a}({b:+.2f})" for a, b in
                                         zip(w[w.e3_id == w.e3_id.iloc[-1]].word, w[w.e3_id == w.e3_id.iloc[-1]].score)))


if __name__ == "__main__":
    main()

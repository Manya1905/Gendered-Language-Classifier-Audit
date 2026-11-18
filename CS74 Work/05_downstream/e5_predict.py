"""
E5 part 1: run one model on the 213,779 industry-analysis sentences (on Polaris).

Reads sentence_text straight from CS72's industry_analysis_sentences.csv, so every model
sees exactly the sentences CS72's model saw (no re-segmenting).
Input processing matches E1's predict(): same tokenizer, truncation at 128 tokens, softmax.

Speed: sentences are sorted by length and each batch is padded only to its longest
sentence (instead of always to 128). With the attention mask this changes probabilities
only by float rounding. --check verifies that on the original model, against the
probabilities CS72 saved in the CSV. --pad-128 turns it off (pads exactly like E1).

Usage (on Polaris, from ~/cs74/05_downstream):
  check:  python -u e5_predict.py --model-dir ../01_clean_reevaluation/original_model \
              --tag original_check --limit 2000 --check
  run:    python -u e5_predict.py --model-dir ../01_clean_reevaluation/output_cpu/seed_42/checkpoint-best \
              --tag bert_seed42
Saves: predictions/<tag>.csv  (job_id, sentence_num, prob_neutral, prob_masculine,
       prob_feminine, pred_id), in the CSV's original row order.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["neutral", "masculine", "feminine"]          # ids 0 / 1 / 2, same as CS72 and E1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True, help="output name, e.g. bert_seed42")
    ap.add_argument("--sentences", default="industry_analysis_sentences.csv")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (for the check)")
    ap.add_argument("--check", action="store_true",
                    help="compare to the original model's probabilities saved in the CSV")
    ap.add_argument("--pad-128", action="store_true", help="pad every sentence to 128 tokens, like E1")
    args = ap.parse_args()

    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
    torch.set_num_threads(threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.sentences, dtype={"job_id": str})
    if args.limit:
        df = df.head(args.limit)
    else:
        assert len(df) == 213_779, len(df)
    texts = df["sentence_text"].astype(str).tolist()
    print(f"[{args.tag}] {len(texts):,} sentences | model {args.model_dir} | "
          f"device {device}, {threads} threads | padding {'128' if args.pad_128 else 'per batch'}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)   # same as E1
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # Longest first, so a memory problem shows up in the first minute, not at the end
    order = np.argsort([-len(t) for t in texts], kind="stable")
    probs = np.zeros((len(texts), 3), dtype=np.float32)
    start = time.time()
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for b in range(n_batches):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=128, return_tensors="pt",
                      padding="max_length" if args.pad_128 else True).to(device)
            probs[idx] = torch.softmax(model(**enc).logits, -1).cpu().numpy()
            if b % 100 == 0 or b == n_batches - 1:
                done = min((b + 1) * args.batch_size, len(texts))
                elapsed = time.time() - start
                # Early batches are the longest sentences, so this ETA starts pessimistic
                eta = elapsed / done * (len(texts) - done)
                print(f"[{args.tag}] {done:,}/{len(texts):,}  {done / elapsed:.0f} sent/s  "
                      f"elapsed {elapsed / 60:.1f} min  ETA <= {eta / 60:.0f} min", flush=True)

    out = df[["job_id", "sentence_num"]].copy()
    for i, name in enumerate(LABELS):
        out[f"prob_{name}"] = probs[:, i]
    out["pred_id"] = probs.argmax(1)

    Path(args.out_dir).mkdir(exist_ok=True)
    path = Path(args.out_dir) / f"{args.tag}.csv"
    out.to_csv(path, index=False)
    print(f"[{args.tag}] saved {path} | {len(out):,} rows | "
          f"masculine {int((out.pred_id == 1).sum()):,}, feminine {int((out.pred_id == 2).sum()):,} | "
          f"total {(time.time() - start) / 60:.1f} min")

    if args.check:
        diff = max(np.abs(out["prob_masculine"] - df["masculine_prob"]).max(),
                   np.abs(out["prob_feminine"] - df["feminine_prob"]).max())
        agree = (out["pred_id"] == df["predicted_class"]).mean()
        print(f"[{args.tag}] CHECK vs CS72's saved predictions: max prob difference {diff:.2e}, "
              f"same label on {agree:.2%} of sentences")
        print("  PASS" if agree == 1.0 and diff < 1e-3 else "  LOOK AT THIS before running the 6 models")


if __name__ == "__main__":
    main()

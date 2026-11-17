"""
E1 step 4: train BERT on the clean split, choosing the best epoch on VALIDATION.

This is CS72's train_bert.py (train_model) with only these changes:
  1. Reads train.csv / val.csv made by 03_build_eval_sets.py (company split).
  2. eval_dataset = VALIDATION set (CS72 used the test set here -> gap 1).
  3. Full precision (fp16 off). CS72 asked for fp16, but its GPU failed to load,
     so the original model actually trained on CPU in full precision.
  4. No test-set evaluation here. The test set is scored ONCE, in 05_evaluate.py.
Everything else is identical: bert-base-uncased, lr 2e-5, batch 16, 4 epochs,
warmup 0.1, weight decay 0.01, max length 128, inverse-frequency ("balanced")
class weights, early stopping (patience 2) on macro-F1, labels 0/1/2 = N/M/F.

RoBERTa: the same script with --model-name roberta-base trains RoBERTa in exactly the
same way (same data, settings, seeds, and epoch choice on val). Only the model and its
tokenizer change. The model is loaded with the Auto* classes, which give the identical
BertTokenizer / BertForSequenceClassification for bert-base-uncased.

Usage (one seed per run):
    python 04_train.py --seed 42
    python 04_train.py --seed 42 --model-name roberta-base
    python 04_train.py --seed 42 --quick      # 1-minute smoke test: 200 sentences, 2 epochs

Needs transformers 4.x (pip install "transformers<5"). transformers 5.x skips the
LayerNorm weights when reloading the best epoch; the check at the end catches this.

Saves to output/seed_<seed>/ for BERT, output/roberta-base_seed_<seed>/ for RoBERTa:
    checkpoint-best/      the chosen model + tokenizer
    training_log.json     loss and validation scores per epoch
    run_info.json         seed, chosen epoch, data sizes, library versions, device
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

LABEL_NAMES = ["neutral", "masculine", "feminine"]   # ids 0 / 1 / 2, same as CS72


# ──────────────────────────────────────────────
# Copied from CS72 train_bert.py
# ──────────────────────────────────────────────
class GenderedLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(texts, padding="max_length", truncation=True,
                     max_length=max_length, return_tensors="pt")


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights, so missing a rare gendered sentence costs more."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": precision,
            "recall": recall, "f1": f1}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 200 train / 100 val sentences, 1 epoch")
    # CS72 hyperparameters (do not change)
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    args = p.parse_args()

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed {args.seed} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ── Data ──
    data = Path(args.data_dir)
    train_df = pd.read_csv(data / "train.csv")
    val_df = pd.read_csv(data / "val.csv")
    assert (train_df["source"] == "enriched").all(), "train.csv should be enriched only"
    epochs = args.epochs
    # BERT keeps the plain "seed_42" folder name; other models get their name in front
    tag = "" if args.model_name == "bert-base-uncased" else f"{args.model_name.split('/')[-1]}_"
    out = Path(args.out_dir) / f"{tag}seed_{args.seed}"
    if args.quick:
        train_df = train_df.sample(200, random_state=0)
        val_df = val_df.sample(100, random_state=0)
        epochs = 2          # 2 epochs, so "pick the best epoch" is actually exercised
        out = Path(args.out_dir) / f"quick_{tag}seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    for name, y in [("train", y_train), ("val", y_val)]:
        counts = {LABEL_NAMES[i]: int((y == i).sum()) for i in range(3)}
        print(f"  {name}: {len(y):,} sentences {counts}")

    # Inverse-frequency class weights, computed on the training labels (same as CS72)
    present = np.unique(y_train)
    w = compute_class_weight("balanced", classes=present, y=y_train)
    full_w = np.ones(3)
    full_w[present] = w
    class_weights = torch.tensor(full_w, dtype=torch.float32)
    print(f"  class weights: {dict(zip(LABEL_NAMES, full_w.round(3)))}")

    # ── Tokenize ──
    # use_fast=False -> the same (slow) tokenizer class CS72 used for BERT
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    print(f"  model {args.model_name}, tokenizer {type(tokenizer).__name__}")
    n_trunc = sum(len(tokenizer.encode(t)) > args.max_length for t in train_df["sentence_text"])
    print(f"  {n_trunc} train sentences are longer than {args.max_length} tokens (truncated, as in CS72)")
    train_ds = GenderedLanguageDataset(
        tokenize_texts(train_df["sentence_text"].tolist(), tokenizer, args.max_length),
        torch.tensor(y_train, dtype=torch.long))
    val_ds = GenderedLanguageDataset(
        tokenize_texts(val_df["sentence_text"].tolist(), tokenizer, args.max_length),
        torch.tensor(y_val, dtype=torch.long))

    # ── Model + training (CS72 settings) ──
    # Warmup: newer transformers removed `warmup_ratio`, so compute the same number of
    # steps it used: ceil(ratio x total steps). (CS72 log: 272 warmup steps of 2,720.)
    total_steps = epochs * math.ceil(len(train_ds) / args.batch_size)
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    print(f"  {total_steps} training steps, {warmup_steps} warmup steps")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",            # macro-F1
        greater_is_better=True,
        fp16=False,                            # full precision, to match the original
        seed=args.seed,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,                   # VALIDATION, not test (the gap-1 fix)
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    result = trainer.train()

    # ── Save the chosen model and a record of the run ──
    best_dir = out / "checkpoint-best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    log = trainer.state.log_history
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    val_by_epoch = {round(e["epoch"]): round(e["eval_f1"], 4) for e in log if "eval_f1" in e}
    best_epoch = max(val_by_epoch, key=val_by_epoch.get)

    # ── Safety check: reload the SAVED model from disk and re-score it on val ──
    # It must reproduce the chosen epoch's val macro-F1. This catches a bug in some
    # transformers versions where reloading the best epoch silently skips weights.
    saved = AutoModelForSequenceClassification.from_pretrained(str(best_dir)).to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_ds, batch_size=64):
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            preds.append(saved(**batch).logits.argmax(-1).cpu().numpy())
    reloaded_f1 = precision_recall_fscore_support(
        y_val, np.concatenate(preds), average="macro", zero_division=0)[2]
    print(f"  Check: saved model re-scores val macro-F1 {reloaded_f1:.4f} "
          f"(chosen epoch had {trainer.state.best_metric:.4f})")
    assert abs(reloaded_f1 - trainer.state.best_metric) < 1e-3, \
        "SAVED MODEL DOES NOT MATCH THE CHOSEN EPOCH - do not use this run"

    info = {
        "seed": args.seed,
        "quick": args.quick,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "train_sentences": len(train_df),
        "val_sentences": len(val_df),
        "class_weights": full_w.round(4).tolist(),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "val_macro_f1_by_epoch": val_by_epoch,
        "chosen_epoch": best_epoch,
        "chosen_val_macro_f1": val_by_epoch[best_epoch],
        "reloaded_val_macro_f1": round(float(reloaded_f1), 4),
        "training_loss": round(result.training_loss, 4),
        "minutes": round((time.time() - start) / 60, 1),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hyperparameters": {k: v for k, v in vars(args).items()
                            if k not in ("data_dir", "out_dir", "quick")} | {"epochs": epochs},
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))

    # Per-epoch checkpoints are ~1.3 GB each with optimizer state; checkpoint-best is all we need.
    shutil.rmtree(out / "checkpoints", ignore_errors=True)

    print(f"\nVal macro-F1 by epoch: {val_by_epoch}")
    print(f"Chosen epoch: {best_epoch} (val macro-F1 {val_by_epoch[best_epoch]})")
    print(f"Saved to {out}  ({info['minutes']} min)")


if __name__ == "__main__":
    main()

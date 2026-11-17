"""
E1 step 6b: label the new evaluation set with the SAME Claude setup as the CS72 labels.

Identical to CS72's prelabel.py:
  - the system prompt is read straight out of From CS72/Code/prelabel.py (not retyped),
  - same model (claude-sonnet-4-6), max_tokens 3000, batches of 15, same batch prompt,
  - no temperature set (CS72 used the default), no thinking,
  - only flagged sentences go to Claude; unflagged ones are neutral,
  - batches never mix the random and enriched samples, and sentences are numbered
    within each sample, as in CS72.
One change: failed batches are RETRIED until every flagged sentence has a label
(CS72 skipped failed batches, which is where its 165 unlabeled sentences came from).

Every answer is saved as it arrives (data/new_eval_claude_raw.jsonl), so if the script
stops for any reason, running it again picks up where it left off and nothing is paid twice.

Setup (PowerShell, in this terminal only; the key is never written to a file):
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
Usage:
    python 07_label_new_eval.py --test     # 2 batches only, to check it works
    python 07_label_new_eval.py            # everything (about 105 batches)
Saves: data/new_eval.csv   (same columns as test_a.csv, plus seen_in_annotated)
"""
import argparse
import ast
import hashlib
import json
import time

import anthropic
import pandas as pd

from common import LABEL_TO_ID, OUT, PROJECT, VALID_LABELS, prepare_labels

PRELABEL_PY = PROJECT / "From CS72" / "Code" / "prelabel.py"
SENTENCES_CSV = OUT / "new_eval_sentences.csv"
RAW_JSONL = OUT / "new_eval_claude_raw.jsonl"
FINAL_CSV = OUT / "new_eval.csv"


def read_cs72_settings():
    """Read SYSTEM_PROMPT, MODEL and BATCH_SIZE from CS72's prelabel.py without running it."""
    tree = ast.parse(PRELABEL_PY.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SYSTEM_PROMPT", "MODEL", "BATCH_SIZE"):
                found[name] = ast.literal_eval(node.value)
    return found["SYSTEM_PROMPT"], found["MODEL"], found["BATCH_SIZE"]


SYSTEM_PROMPT, MODEL, BATCH_SIZE = read_cs72_settings()
assert MODEL == "claude-sonnet-4-6" and BATCH_SIZE == 15


# ── Copied from CS72 prelabel.py ──
def build_batch_prompt(sentences):
    """Build a prompt for a batch of sentences."""
    lines = []
    for s in sentences:
        lines.append(
            f"[{s['idx']}] Sentence: \"{s['text']}\"\n"
            f"    Flagged words: {s['flagged_words']}\n"
            f"    (masculine: {s['masc_words']}, feminine: {s['fem_words']})"
        )
    return "Label each sentence below:\n\n" + "\n\n".join(lines)


def call_api(client, batch):
    """One request, same settings as CS72. Returns the parsed list, or None on failure."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_batch_prompt(batch)}],
        )
    except anthropic.RateLimitError:
        print("rate limited, waiting 30s", end=" ")
        time.sleep(30)
        return None
    except anthropic.APIError as e:
        print(f"API error ({e.__class__.__name__}), waiting 5s", end=" ")
        time.sleep(5)
        return None
    if response.stop_reason not in ("end_turn", "stop_sequence"):
        print(f"stopped early ({response.stop_reason})", end=" ")
        return None
    text = "".join(b.text for b in response.content if b.type == "text")
    start, end = text.find("["), text.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end]), response.model   # the model that actually answered
    except json.JSONDecodeError:
        return None


def make_batches(sents, done):
    """CS72 batching: flagged sentences in file order, 15 at a time, one sample at a time."""
    batches = []
    for source in ["random", "enriched"]:
        todo = sents[(sents.source == source) & sents.flagged & ~sents.key.isin(done)]
        items = [{"key": r.key, "source": source, "idx": int(r.idx), "text": r.sentence_text,
                  "flagged_words": r.all_flagged_words,
                  "masc_words": r.masculine_words if pd.notna(r.masculine_words) else "",
                  "fem_words": r.feminine_words if pd.notna(r.feminine_words) else ""}
                 for r in todo.itertuples()]
        batches += [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    return batches


def load_done():
    done = {}
    if RAW_JSONL.exists():
        for line in RAW_JSONL.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="label 2 batches only")
    ap.add_argument("--max-rounds", type=int, default=10)
    args = ap.parse_args()

    sents = pd.read_csv(SENTENCES_CSV, dtype={"job_id": str, "company_id": str})
    # Number sentences within each sample, as CS72 did (it labeled each sample file separately)
    sents["idx"] = sents.groupby("source").cumcount()
    sents["key"] = sents["source"] + ":" + sents["idx"].astype(str)
    n_flagged = int(sents.flagged.sum())

    client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY
    done = load_done()
    print(f"Model {MODEL} | {n_flagged:,} flagged sentences | already labeled: {len(done):,}")
    print(f"System prompt from {PRELABEL_PY.name}: {len(SYSTEM_PROMPT):,} characters, "
          f"md5 {hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()}")
    print(f"  starts: {SYSTEM_PROMPT[:70]}...")

    for round_ in range(1, args.max_rounds + 1):
        batches = make_batches(sents, done)
        if not batches:
            break
        if args.test:
            batches = batches[:2]
        print(f"\nRound {round_}: {len(batches)} batches to send")
        for i, batch in enumerate(batches):
            print(f"  Batch {i + 1}/{len(batches)} ({batch[0]['source']})...", end=" ", flush=True)
            answer = call_api(client, batch)
            if answer is None:
                print("FAILED, will retry next round")
                continue
            results, answered_by = answer
            assert answered_by.startswith(MODEL), f"answered by {answered_by}, not {MODEL}"
            by_idx = {s["idx"]: s for s in batch}
            n_ok = 0
            with RAW_JSONL.open("a", encoding="utf-8") as f:
                for r in results:
                    s = by_idx.get(r.get("index"))
                    label = str(r.get("label", "")).strip().lower()
                    if s is None or label not in VALID_LABELS or s["key"] in done:
                        continue
                    rec = {"key": s["key"], "label": label, "reasoning": r.get("reasoning", ""),
                           "model": answered_by}
                    f.write(json.dumps(rec) + "\n")
                    done[s["key"]] = rec
                    n_ok += 1
            print(f"OK ({n_ok}/{len(batch)} labeled, answered by {answered_by})")
            time.sleep(1)                 # same pause as CS72
        if args.test:
            print(f"\nTest done: {len(done):,} of {n_flagged:,} labeled so far. "
                  "Run without --test to label the rest.")
            return

    missing = n_flagged - sum(k in done for k in sents.loc[sents.flagged, "key"])
    assert missing == 0, f"{missing} flagged sentences still unlabeled - run the script again"

    # ── Build the final file, with the same label rules as the training data ──
    sents["suggested_label"] = sents["key"].map(lambda k: done[k]["label"] if k in done else "")
    sents["reasoning"] = sents["key"].map(lambda k: done[k]["reasoning"] if k in done else "")
    sents["your_label"] = ""
    sents = prepare_labels(sents)         # unflagged -> neutral, both -> masculine
    sents["unlabeled"] = False
    final = sents.rename(columns={"industry_group": "industry", "sentence_index": "sentence_num"})[
        ["source", "industry", "company_id", "job_id", "title", "sentence_num", "sentence_text",
         "text_norm", "flagged", "masculine_words", "feminine_words", "suggested_label",
         "reasoning", "label", "label_id", "has_gendered", "unlabeled", "seen_in_annotated"]]
    assert final.label_id.isin(LABEL_TO_ID.values()).all()
    final.to_csv(FINAL_CSV, index=False)

    print(f"\nAll {n_flagged:,} flagged sentences labeled. Saved {FINAL_CSV.name}")
    print("Claude's labels (flagged only):",
          sents.loc[sents.flagged, "suggested_label"].value_counts().to_dict())
    for name, d in [("full", final), ("unseen", final[~final.seen_in_annotated])]:
        print(f"  {name:6s} {len(d):,} sentences: {d.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

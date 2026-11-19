"""
E2: the CS72 answer key for the minimal pairs.

Labels every sentence the way CS72's training data was labeled, so the models can be scored
against the definition they were trained on (secondary answer key; the main key is human judgment).

Same setup as CS72 (reuses E1's 07_label_new_eval.py, which copies CS72's prelabel.py):
  - CS72's own system prompt, model (claude-sonnet-4-6), batch prompt, batches of 15,
    default temperature, no thinking
  - only sentences with a list word go to Claude; the rest are neutral (so the off-list
    pairs are neutral, exactly what CS72's pipeline would have given them)
  - "both" counts as masculine
Sentences are sent in the blind sheet's shuffled order, so a pair's two sentences are
never next to each other in a batch. Failed batches are retried.

Setup (cmd, this terminal only; the key is never written to a file):
    set ANTHROPIC_API_KEY=sk-ant-...
Usage:
    python label_with_claude.py --dry-run    # show the batches, no API calls
    python label_with_claude.py              # label (about 6 requests)
Saves: labeling/claude_cs72_raw.jsonl   every answer as it arrives (re-running resumes)
       labeling/claude_cs72_labels.csv  pair_id, sentence_text, claude_label, cs72_label, reasoning
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
E1_DIR = HERE.parent / "01_clean_reevaluation"
RAW = HERE / "labeling" / "claude_cs72_raw.jsonl"
OUT = HERE / "labeling" / "claude_cs72_labels.csv"


def load(name, file):
    sys.path.insert(0, str(E1_DIR))                 # so the E1 scripts find common.py
    spec = importlib.util.spec_from_file_location(name, E1_DIR / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    e1_label = load("label_e1", "07_label_new_eval.py")        # CS72 prompt, model, call_api
    find_words = load("sample_e1", "06_sample_new_eval.py").find_gendered_words

    pairs = pd.read_csv(HERE / "minimal_pairs.csv")
    order = pd.read_csv(HERE / "labeling" / "sheet_key.csv")   # the blind sheet's shuffled order
    sents = order.merge(pairs[["pair_id", "sentence_text"]], on=["pair_id", "sentence_text"], validate="one_to_one")
    assert len(sents) == len(pairs), "sheet_key.csv does not match minimal_pairs.csv"
    found = sents.sentence_text.map(find_words)
    sents["masc_words"] = found.map(lambda mf: ", ".join(mf[0]))
    sents["fem_words"] = found.map(lambda mf: ", ".join(mf[1]))
    sents["flagged"] = (sents.masc_words != "") | (sents.fem_words != "")
    sents["idx"] = range(len(sents))

    done = {}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["idx"]] = r
    print(f"Model {e1_label.MODEL} | {int(sents.flagged.sum())} sentences with a list word go to Claude, "
          f"{int((~sents.flagged).sum())} without are neutral | already labeled: {len(done)}")

    client = None
    for round_ in range(1, 11):
        todo = [{"idx": int(r.idx), "text": r.sentence_text,
                 "flagged_words": ", ".join(w for w in [r.masc_words, r.fem_words] if w),
                 "masc_words": r.masc_words, "fem_words": r.fem_words}
                for r in sents[sents.flagged & ~sents.idx.isin(done)].itertuples()]
        if not todo:
            break
        batches = [todo[i:i + e1_label.BATCH_SIZE] for i in range(0, len(todo), e1_label.BATCH_SIZE)]
        if args.dry_run:
            print(f"{len(batches)} batches. First batch prompt:\n")
            print(e1_label.build_batch_prompt(batches[0]))
            return
        client = client or e1_label.anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        print(f"Round {round_}: {len(batches)} batches")
        for i, batch in enumerate(batches, 1):
            print(f"  Batch {i}/{len(batches)}...", end=" ", flush=True)
            answer = e1_label.call_api(client, batch)
            if answer is None:
                print("FAILED, will retry")
                continue
            results, answered_by = answer
            assert answered_by.startswith(e1_label.MODEL), f"answered by {answered_by}"
            ids = {s["idx"] for s in batch}
            n_ok = 0
            with RAW.open("a", encoding="utf-8") as f:
                for r in results:
                    label = str(r.get("label", "")).strip().lower()
                    if r.get("index") in ids and label in e1_label.VALID_LABELS and r["index"] not in done:
                        rec = {"idx": r["index"], "label": label, "reasoning": r.get("reasoning", ""),
                               "model": answered_by}
                        f.write(json.dumps(rec) + "\n")
                        done[r["index"]] = rec
                        n_ok += 1
            print(f"OK ({n_ok}/{len(batch)})")
            time.sleep(1)

    missing = sents.flagged & ~sents.idx.isin(done)
    assert not missing.any(), f"{missing.sum()} sentences still unlabeled - run again"

    sents["claude_label"] = sents.idx.map(lambda i: done[i]["label"] if i in done else "")
    sents["reasoning"] = sents.idx.map(lambda i: done[i]["reasoning"] if i in done else "")
    sents["cs72_label"] = sents.claude_label.replace({"both": "masculine", "": "neutral"})
    out = pairs[["pair_id", "sentence_text"]].merge(
        sents[["pair_id", "sentence_text", "masc_words", "fem_words", "claude_label", "cs72_label", "reasoning"]],
        on=["pair_id", "sentence_text"])
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT.name}: {out.cs72_label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

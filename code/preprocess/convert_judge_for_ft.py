#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path

SYSTEM = (
  "You are a careful evaluator. Given a multiple-choice question, options, and "
  "several candidate answers, assign a quality score to EACH candidate so the "
  "scores sum to 1. Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
)

def main(in_path, out_train, out_val, val_ratio, seed):
    random.seed(seed)

    # Load and shuffle all judge examples
    lines = [json.loads(x) for x in Path(in_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    random.shuffle(lines)

    # Split
    n_val = max(1, int(len(lines) * val_ratio))
    val = lines[:n_val]
    train = lines[n_val:]

    def to_messages(rec):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["completion"].strip()}
            ]
        }

    # Ensure parent directories exist
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_val).parent.mkdir(parents=True, exist_ok=True)

    # Write files
    for path, subset in [(out_train, train), (out_val, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(train)} train and {len(val)} val examples "
          f"to {out_train} and {out_val} (from {len(lines)} total; val_ratio={val_ratio}).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="data/judge_mmlu_preselection.jsonl")
    ap.add_argument("--out-train", required=True, help="output train jsonl")
    ap.add_argument("--out-val", required=True, help="output val jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    main(args.inp, args.out_train, args.out_val, args.val_ratio, args.seed)

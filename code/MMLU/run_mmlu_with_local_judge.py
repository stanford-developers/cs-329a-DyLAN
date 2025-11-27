# code/MMLU/run_mmlu_with_local_judge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, importlib
from typing import List
from local_judge import LocalJudge

def patch_utils_for_local_judge(ckpt_dir: str):
    """Replace utils.judge_importance_weights so LLMLP.backward() uses the local judge."""
    import utils  # your existing utils.py

    judge = LocalJudge(ckpt_dir)

    def judge_importance_weights_local(replies: List[str], question: str, qtype: str, mtype: str):
        try:
            weights = judge.score_replies(question, replies)
        except Exception:
            # robust fallback
            k = max(1, len(replies))
            weights = [1.0 / k] * k
        # Keep the original signature: (weights, prompt_tokens, completion_tokens)
        return weights, 0, 0

    utils.judge_importance_weights = judge_importance_weights_local

def main():
    ap = argparse.ArgumentParser(
        description="Run llmlp_listwise_mmlu.py but route judge weights to a local LoRA checkpoint."
    )
    ap.add_argument("csv")        # e.g., data/MMLU/evaluation/<subject>_val.csv
    ap.add_argument("exp")        # experiment name prefix
    ap.add_argument("model")      # base generator model string (e.g., openai/gpt-oss-20b)
    ap.add_argument("out_dir")    # where to write outputs
    ap.add_argument("roles")      # "['Economist','Doctor','Lawyer','Mathematician']"
    ap.add_argument("--judge-ckpt", required=True, help="Path to merged local checkpoint directory")
    args = ap.parse_args()

    # ensure DyLAN uses judge weights
    os.environ["AIP_JUDGE_WEIGHTS"] = "1"

    # install our judge
    patch_utils_for_local_judge(args.judge_ckpt)

    # Call the *existing* script with the same CLI it expects.
    sys.argv = [
        "llmlp_listwise_mmlu.py",
        args.csv, args.exp, args.model, args.out_dir, args.roles
    ]
    ll = importlib.import_module("llmlp_listwise_mmlu")
    ll.main()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start a LoRA fine-tuning job on Together for the DyLAN judge.

Example:
python code/MMLU/finetune/start_lora_ft.py \
  --train data/ft/judge_train.jsonl \
  --val   data/ft/judge_val.jsonl \
  --model "${MODEL}" \
  --epochs 2 \
  --batch-size 8 \
  --lr 1e-5 \
  --suffix mmlu-judge-v1 \
  --wandb-project dylan-lora \
  --wait
"""

import argparse, json, os, sys, time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from together import Together
try:
    # available in modern SDKs; if absent, we just skip the check step
    from together.utils import check_file as _check_file
except Exception:
    _check_file = None


def eprint(*a, **k): print(*a, file=sys.stderr, **k)

def _make_client() -> Together:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set. Put it in your .env or shell environment.")
    base_url = os.getenv("TOGETHER_BASE_URL")
    if base_url:
        try:
            return Together(api_key=api_key, base_url=base_url)
        except TypeError:
            eprint("[WARN] together.Together() does not accept base_url in this SDK; using default.")
    return Together(api_key=api_key)

def _check(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if _check_file:
        try:
            rep = _check_file(path)
            eprint(f"[check] {path}: {json.dumps(rep, indent=2)}")
            assert rep.get("is_check_passed", True)
        except Exception as e:
            eprint(f"[check] Skipping structured checks for {path}: {e}")

def _upload(client: Together, path: str) -> str:
    _check(path)
    eprint(f"[upload] {path}")
    resp = client.files.upload(file=path)  # returns obj with .id
    file_id = getattr(resp, "id", None) or (resp.get("id") if isinstance(resp, dict) else None)
    if not file_id:
        raise RuntimeError(f"Failed to get file id from upload response: {resp}")
    eprint(f"[upload] -> {file_id}")
    return file_id

def _poll_until_done(client: Together, job_id: str, poll_secs: int = 15) -> Dict[str, Any]:
    term = {"completed", "error", "cancelled"}
    while True:
        info = client.fine_tuning.retrieve(id=job_id)
        status = info.get("status") if isinstance(info, dict) else getattr(info, "status", None)
        model_output_name = info.get("model_output_name") if isinstance(info, dict) else getattr(info, "model_output_name", None)
        eprint(f"[{time.strftime('%H:%M:%S')}] job={job_id} status={status}"
               + (f" model_output_name={model_output_name}" if model_output_name else ""))
        if status in term:
            try: eprint(json.dumps(info, indent=2, default=str))
            except Exception: eprint(info)
            return info if isinstance(info, dict) else info.__dict__
        time.sleep(poll_secs)

def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Train JSONL (Together conversational format).")
    p.add_argument("--val",   default=None, help="Validation JSONL (optional).")
    p.add_argument("--model", default=os.getenv("MODEL", "openai/gpt-oss-20b"),
                   help="Base model to fine-tune (e.g., openai/gpt-oss-20b).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5, dest="learning_rate")
    p.add_argument("--checkpoints", type=int, default=1, help="n_checkpoints")
    p.add_argument("--n-evals", type=int, default=None,
                   help="#eval passes during training (default: 1 if --val is provided else 0).")
    p.add_argument("--suffix", default="mmlu-judge", help="Suffix for output model name.")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", None))
    p.add_argument("--wandb-api-key", default=os.getenv("WANDB_API_KEY", None))
    p.add_argument("--wait", action="store_true", help="Poll until the job finishes.")
    p.add_argument("--poll", type=int, default=15, help="Poll interval seconds when --wait.")
    args = p.parse_args()

    client = _make_client()

    train_id = _upload(client, args.train)
    val_id   = _upload(client, args.val) if args.val else None

    n_evals = args.n_evals if args.n_evals is not None else (1 if val_id else 0)

    payload = dict(
        model=args.model,
        training_file=train_id,
        validation_file=val_id,
        n_epochs=args.epochs,
        n_checkpoints=args.checkpoints,
        n_evals=n_evals,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        suffix=args.suffix,
        wandb_project_name=args.wandb_project,
        wandb_api_key=args.wandb_api_key,
    )
    payload = {k: v for k, v in payload.items() if v is not None}

    eprint("[start] creating fine-tune job with payload:")
    eprint(json.dumps({**payload, "training_file": train_id, "validation_file": val_id}, indent=2))
    resp = client.fine_tuning.create(**payload)
    job_id = resp.get("id") if isinstance(resp, dict) else getattr(resp, "id", None)
    print(json.dumps(resp, indent=2, default=str))
    if not job_id:
        eprint("[ERROR] No job id in response."); sys.exit(2)

    eprint(f"[job] id={job_id}")
    if args.wait:
        info = _poll_until_done(client, job_id, poll_secs=args.poll)
        out_name = info.get("model_output_name")
        if out_name:
            print(f"\n[RESULT] output model name: {out_name}")
        else:
            eprint("[WARN] model_output_name missing; check the dashboard.")
    else:
        eprint("Tip: run with --wait or monitor the Together dashboard (Jobs tab).")

if __name__ == "__main__":
    main()

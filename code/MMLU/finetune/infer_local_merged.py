#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, re, ast
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

SYSTEM = (
  "You are a careful evaluator. Given a multiple-choice question, options, and "
  "several candidate answers, assign a quality score to EACH candidate so the "
  "scores sum to 1. Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
)

USER = """Question:
What is 2+2?

Options:
(A) 3
(B) 4
(C) 5
(D) 22

Candidate 1: The answer is (B).
Candidate 2: It looks like (D).
Candidate 3: Maybe (A).

Scores:"""

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def _as_str(x):
    if isinstance(x, str): return x
    if isinstance(x, dict) and "content" in x: return x["content"]
    return None

def load_tokenizer(ckpt_dir: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ckpt_dir, "tokenizer.json"))

    # Attach special tokens (handle both string and object formats)
    st_path = os.path.join(ckpt_dir, "special_tokens_map.json")
    specials = {}
    if os.path.exists(st_path):
        raw = json.load(open(st_path))
        for key in ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"):
            s = _as_str(raw.get(key))
            if s: specials[key] = s
        if "additional_special_tokens" in raw:
            add = []
            for item in raw["additional_special_tokens"]:
                s = _as_str(item) if not isinstance(item, str) else item
                if s: add.append(s)
            if add: specials["additional_special_tokens"] = add
    if specials:
        tok.add_special_tokens(specials)

    # Chat template
    tmpl_path = os.path.join(ckpt_dir, "chat_template.jinja")
    if os.path.exists(tmpl_path):
        tok.chat_template = open(tmpl_path, "r").read()

    # Fallbacks for EOS/PAD if still missing
    if tok.eos_token is None:
        for cand in ("<|end|>", "<|endoftext|>", "</s>"):
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0:
                tok.eos_token = cand
                break
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"

    return tok

def count_candidates(text: str) -> int:
    return max(1, len(re.findall(r"^Candidate\s+\d+:", text, flags=re.MULTILINE)))

def parse_list(text: str, k: int):
    # Prefer the last [...] block
    blocks = list(re.finditer(r"\[[^\[\]]+\]", text))
    if blocks:
        try:
            arr = ast.literal_eval(blocks[-1].group(0))
            if isinstance(arr, list): return arr[:k]
        except Exception:
            pass
    # Fallback: grab floats anywhere and cut to k
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return nums[:k] if nums else None

def renorm(ws, k):
    if not ws: return [1.0/k]*k
    w = [max(0.0, float(x)) for x in ws[:k]]
    s = sum(w)
    return [x/s for x in w] if s > 0 else [1.0/k]*k

def main(ckpt_dir: str):
    device = pick_device()
    # bf16 is stable on CUDA/MPS for this model; CPU uses fp32
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    tok = load_tokenizer(ckpt_dir)

    # 100% local model load
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER}]

    # Build prompt from chat template, then nudge to final channel with a constraint
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    K = count_candidates(USER)
    schema = f'{{"type":"array","items":{{"type":"number"}}, "minItems":{K}, "maxItems":{K}}}'
    prompt += f"\n<|constrain|>{schema}\n<|return|>\n["  # pre-seed '['

    # Tokenize; drop token_type_ids for decoder-only models
    enc = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    # Stop on any of these tokens
    eos_ids = []
    for cand in (tok.eos_token, "<|end|>", "<|endoftext|>"):
        if not cand: continue
        tid = tok.convert_tokens_to_ids(cand)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    eos_ids = list(dict.fromkeys(eos_ids))  # unique

    # (Optional) discourage drifting into analysis
    bad_words_ids = []
    bw = tok.encode("analysis", add_special_tokens=False)
    if bw: bad_words_ids.append(bw)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6*K + 4,          # enough for "[0.1, 0.2, ...]"
            do_sample=False,                  # greedy
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids if eos_ids else None,
            bad_words_ids=bad_words_ids or None,
            use_cache=True,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()

    # Because we pre-seeded '[', ensure it’s present for parsing.
    if not text.startswith("["):
        text = "[" + text
    if "]" not in text:
        text = text.splitlines()[0] + "]"

    print("\n--- RAW MODEL OUTPUT ---\n", text)
    scores = parse_list(text, K)
    if scores is None:
        print("\n[WARN] Could not parse a list from the output above.")
    else:
        scores = renorm(scores, K)
        print("\nParsed scores (renormalized):", scores)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="Path to merged checkpoint directory")
    args = ap.parse_args()
    main(args.ckpt_dir)

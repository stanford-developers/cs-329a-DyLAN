# code/MMLU/local_judge.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, re, ast
from typing import List, Tuple
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

SYSTEM = (
  "You are a careful evaluator. Given a multiple-choice question, options, and "
  "several candidate answers, assign a quality score to EACH candidate so the "
  "scores sum to 1. Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
)

def _as_str(x):
    if isinstance(x, str): return x
    if isinstance(x, dict) and "content" in x: return x["content"]
    return None

def _load_tokenizer(ckpt_dir: str) -> PreTrainedTokenizerFast:
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

    # Chat template (Together’s GPT‑OSS uses this)
    tmpl_path = os.path.join(ckpt_dir, "chat_template.jinja")
    if os.path.exists(tmpl_path):
        tok.chat_template = open(tmpl_path, "r").read()

    if tok.eos_token is None:
        for cand in ("<|end|>", "<|endoftext|>", "</s>"):
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0:
                tok.eos_token = cand
                break
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    return tok

def _pick_device() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

class LocalJudge:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.device = _pick_device()
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        self.tok = _load_tokenizer(ckpt_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, local_files_only=True, dtype=self.dtype,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_replies(self, question_with_options: str, replies: List[str]) -> List[float]:
        # Build a single prompt from the question + numbered candidates
        lines = [ "Question:\n" + question_with_options.strip(), "", "Candidate responses:" ]
        for i, r in enumerate(replies, 1):
            lines.append(f"Candidate {i}: {r.strip()}")
        lines.append("")
        lines.append("Scores:")

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)}
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        enc = self.tok(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        eos_ids = []
        for tok_str in (self.tok.eos_token, "<|end|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)
        # unique & stable order
        eos_ids = list(dict.fromkeys(eos_ids))

        out = self.model.generate(
            **enc, max_new_tokens=6 * len(replies) + 6, do_sample=False,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=eos_ids or None
        )
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()

        # parse [a, b, c] (fallback: any floats)
        m = list(re.finditer(r"\[[^\[\]]+\]", text))
        if m:
            try:
                arr = ast.literal_eval(m[-1].group(0))
                if isinstance(arr, list): return self._renorm(arr, len(replies))
            except Exception:
                pass
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        return self._renorm(nums, len(replies))

    @staticmethod
    def _renorm(ws, k):
        if not ws: return [1.0/k]*k
        w = [max(0.0, float(x)) for x in ws[:k]]
        s = sum(w)
        return [x/s for x in w] if s > 0 else [1.0/k]*k

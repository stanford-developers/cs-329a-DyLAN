#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalJudge: score multiple candidate answers for a single MCQ using a *local* merged checkpoint.

Usage (from Python):
    from local_judge import LocalJudge
    j = LocalJudge("code/MMLU/finetune/ckpts/merged")
    scores, raw = j.score(prompt_text, k=3)

- Loads tokenizer+model strictly from the given directory (no HF downloads).
- Robust special-token handling (EOS/PAD).
- Drops token_type_ids (decoder-only).
- Returns renormalized [w1..wk] that sum to 1 and the raw decoded text.
"""
from __future__ import annotations
import os, json, re, ast
from typing import List, Optional, Tuple
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM


SYSTEM = (
    "You are a careful evaluator. Given a multiple-choice question, options, and "
    "several candidate answers, assign a quality score to EACH candidate so the "
    "scores sum to 1. Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _as_str(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and "content" in x:
        return x["content"]
    return None


class LocalJudge:
    def __init__(self, ckpt_dir: str, device: Optional[str] = None):
        self.ckpt_dir = ckpt_dir
        self.device = _pick_device() if device is None else device
        # bf16 on GPU/MPS, fp32 on CPU
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32

        # --- tokenizer 100% local ---
        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"tokenizer.json not found in {ckpt_dir}")
        self.tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)

        # Attach special tokens (accepts both string and {"content": "..."} formats)
        st_path = os.path.join(ckpt_dir, "special_tokens_map.json")
        specials = {}
        if os.path.exists(st_path):
            raw = json.load(open(st_path))
            for key in ("bos_token", "eos_token", "pad_token", "unk_token",
                        "sep_token", "cls_token", "mask_token"):
                s = _as_str(raw.get(key))
                if s:
                    specials[key] = s
            if "additional_special_tokens" in raw:
                add = []
                for item in raw["additional_special_tokens"]:
                    s = _as_str(item) if not isinstance(item, str) else item
                    if s:
                        add.append(s)
                if add:
                    specials["additional_special_tokens"] = add
        if specials:
            self.tok.add_special_tokens(specials)

        # Chat template
        tmpl_path = os.path.join(ckpt_dir, "chat_template.jinja")
        if os.path.exists(tmpl_path):
            self.tok.chat_template = open(tmpl_path, "r").read()

        # Fallbacks for EOS/PAD if still missing
        if self.tok.eos_token is None:
            for cand in ("<|end|>", "<|endoftext|>", "</s>"):
                tid = self.tok.convert_tokens_to_ids(cand)
                if isinstance(tid, int) and tid >= 0:
                    self.tok.eos_token = cand
                    break
        if self.tok.pad_token is None:
            # pad with eos if needed
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"

        # --- model 100% local ---
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            local_files_only=True,
            dtype=self.dtype,             # transformers>=4.44: prefer 'dtype' over 'torch_dtype'
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Pre-compute EOS ids we may want to stop on
        self.eos_ids = []
        for cand in (self.tok.eos_token, "<|end|>", "<|endoftext|>"):
            if not cand:
                continue
            tid = self.tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0:
                self.eos_ids.append(tid)
        # Deduplicate
        self.eos_ids = list(dict.fromkeys(self.eos_ids))

        # Avoid drifting into analysis
        self.bad_words_ids = []
        bw = self.tok.encode("analysis", add_special_tokens=False)
        if bw:
            self.bad_words_ids.append(bw)

    @staticmethod
    def _count_candidates(prompt_text: str) -> int:
        return max(1, len(re.findall(r"^Candidate\s+\d+:", prompt_text, flags=re.MULTILINE)))

    @staticmethod
    def _parse_list(text: str, k: int):
        # Try the last [...] block
        blocks = list(re.finditer(r"\[[^\[\]]+\]", text))
        if blocks:
            try:
                arr = ast.literal_eval(blocks[-1].group(0))
                if isinstance(arr, list):
                    return arr[:k]
            except Exception:
                pass
        # Fallback: extract floats and cut to k
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        return nums[:k] if nums else None

    @staticmethod
    def _renorm(ws, k):
        if not ws:
            return [1.0 / k] * k
        w = [max(0.0, float(x)) for x in ws[:k]]
        s = sum(w)
        return [x / s for x in w] if s > 0 else [1.0 / k] * k

    def score(self, user_prompt: str, k: Optional[int] = None) -> Tuple[list, str]:
        """
        :param user_prompt: the full "question + options + Candidate i: ..." text, ending with "Scores:"
        :param k: number of candidates; if None we infer from text
        :return: (scores_list, raw_generated_text)
        """
        K = self._count_candidates(user_prompt) if k is None else int(k)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt}]

        # Build the chat-formatted prompt
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Lightweight structural hint + force switch to final answer; pre-seed the bracket
        schema = f'{{"type":"array","items":{{"type":"number"}}, "minItems":{K}, "maxItems":{K}}}'
        prompt += f"\n<|constrain|>{schema}\n<|return|>\n["

        # Tokenize (drop token_type_ids)
        enc = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=6 * K + 4,     # enough for "[0.1, 0.2, ...]"
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.eos_ids if self.eos_ids else None,
                bad_words_ids=self.bad_words_ids or None,
                use_cache=True,
            )

        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()

        # Because we pre-seeded “[”, ensure it’s present for parsing
        if not text.startswith("["):
            text = "[" + text
        if "]" not in text:
            text = text.splitlines()[0] + "]"

        scores = self._parse_list(text, K)
        scores = self._renorm(scores, K)
        return scores, text

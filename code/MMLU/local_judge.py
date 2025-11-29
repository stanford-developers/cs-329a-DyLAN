# code/MMLU/local_judge.py
# Robust local LLM-as-judge with strict parsing and stable outputs.

from __future__ import annotations
import os, re, ast, json, time
from typing import List, Sequence, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

SYSTEM = (
  "You are a careful evaluator. Given a multiple-choice question, options, and "
  "several candidate answers, assign a quality score to EACH candidate so the "
  "scores sum to 1. Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
)

def _as_str(x):
    if isinstance(x, str): return x
    if isinstance(x, dict) and "content" in x: return x["content"]
    return None

def load_tokenizer(ckpt_dir: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ckpt_dir, "tokenizer.json"))

    # Special tokens (tolerant to object/string formats)
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

    # Fallbacks for EOS/PAD if missing
    if tok.eos_token is None:
        for cand in ("<|end|>", "<|endoftext|>", "</s>"):
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0:
                tok.eos_token = cand
                break
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    return tok

def _eos_ids(tok: PreTrainedTokenizerFast) -> Optional[List[int]]:
    ids: List[int] = []
    for cand in (tok.eos_token, "<|end|>", "<|endoftext|>", "</s>"):
        if not cand: continue
        tid = tok.convert_tokens_to_ids(cand)
        if isinstance(tid, int) and tid >= 0 and tid not in ids:
            ids.append(tid)
    return ids or None

def _parse_bracket_list(text: str, k: int) -> Optional[List[float]]:
    """
    Strict: only accept the last [...] block; no "scan-any-number" fallback.
    If parsing fails or length mismatches, return None -> caller will uniformize.
    """
    blocks = list(re.finditer(r"\[[^\[\]]*\]", text))
    if not blocks:
        return None
    last = blocks[-1].group(0)
    try:
        arr = ast.literal_eval(last)
        if not isinstance(arr, (list, tuple)):
            return None
        vals = [float(x) for x in arr]
        if len(vals) != k:
            return None
        return vals
    except Exception:
        return None

def _renorm_or_uniform(ws: Optional[List[float]], k: int) -> List[float]:
    if not ws or len(ws) != k:
        return [1.0 / k] * k
    v = [max(0.0, float(x)) for x in ws]
    s = sum(v)
    if s <= 0.0:
        return [1.0 / k] * k
    return [x / s for x in v]

class LocalJudge:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32

        self.tok = load_tokenizer(ckpt_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            local_files_only=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

    def score(self,
              question: str,
              options: Sequence[str],      # e.g., ["0","4","2","6"]
              candidate_labels: Sequence[str],  # e.g., ["(A)","(B)",...], same length as roles
              roles: Sequence[str],        # role names in the exact order you want
              system: str = SYSTEM) -> Tuple[List[float], str]:
        """
        Returns (weights, raw_text). 'weights' is a length-K vector that sums to 1.
        """
        assert len(candidate_labels) == len(roles), "labels and roles must align"
        K = len(roles)

        # Build the judge prompt body (same across scripts)
        opt_lines = [f"({ch}) {txt}" for ch, txt in zip("ABCD", options)]
        lines = [
            "Question:",
            question.strip(),
            "",
            "Options:",
            *opt_lines,
            "",
        ]
        for i, (r, lab) in enumerate(zip(roles, candidate_labels), 1):
            lab = lab.strip()
            # tolerate inputs like "B" or "(B)"
            if len(lab) == 1 and lab in "ABCD": lab = f"({lab})"
            lines.append(f"Candidate {i}: [{r}] predicts {lab}.")
        lines += ["", "Scores:"]
        user = "\n".join(lines)

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        schema = f'{{"type":"array","items":{{"type":"number"}}, "minItems":{K}, "maxItems":{K}}}'
        prompt += f"\n<|constrain|>{schema}\n<|return|>\n["  # pre-seed '['

        enc = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        eos_ids = _eos_ids(self.tok)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=6*K + 4,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )

        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        if not text.startswith("["):
            text = "[" + text

        scores = _parse_bracket_list(text, K)
        weights = _renorm_or_uniform(scores, K)
        return weights, text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a JSONL dataset for *listwise LLM‑judge* fine‑tuning from DyLAN pre‑selection results.

What it consumes
----------------
Per subject in your DyLAN pre‑selection run:
  <stem>_73.txt   # small 6-line summary (not strictly required)
  <stem>_73.json  # concatenated JSON arrays; one top-level object per question
  <stem>_73.log   # verbose run log; used only as a fallback for question/options

And the per‑subject MMLU CSVs (headered or headerless) under --mmlu_eval_dir
(e.g. data/MMLU/small_team_selection or medium_team_selection).

What it produces
----------------
A JSONL where each line looks like:
{
  "prompt": "<judge instruction with question, choices, and K candidate answers>",
  "completion": "[w1, w2, ..., wK]",
  "meta": {...},
  "question": "...",
  "choices": {"A":"...","B":"...","C":"...","D":"..."},
  "agent_responses": ["...", "...", ...]
}

Key fixes
---------
- Correctly handles headerless CSVs (common in small/medium_team_selection) by
  using csv.reader and detecting headers.
- Tries CSV first; falls back to parsing the .log only if needed.
- Labels are derived from which candidate chose the gold letter; otherwise a
  length-based heuristic is used and renormalized.

Usage
-----
python code/preprocess/build_judge_dataset.py \
  --results_dir code/MMLU/standard_dylan/mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian \
  --mmlu_eval_dir data/MMLU/small_team_selection \
  --outfile data/judge_mmlu_preselection.jsonl
"""

from __future__ import annotations
import argparse, csv, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CHOICE_KEYS = ["A", "B", "C", "D"]
DEFAULT_ROLES = ["Economist","Doctor","Lawyer","Mathematician","Psychologist","Programmer","Historian"]

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def parse_concat_json(json_path: str) -> List[Any]:
    """The subject JSON is a sequence of top-level JSON arrays concatenated."""
    data = Path(json_path).read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()
    i = 0
    out: List[Any] = []
    while True:
        while i < len(data) and data[i].isspace():
            i += 1
        if i >= len(data):
            break
        obj, end = dec.raw_decode(data[i:])
        out.append(obj)
        i += end
    return out

def first_non_null(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None

def extract_choice_letter(text: str) -> Optional[str]:
    """Match a trailing '(A)'/'(B)' style letter at the end of the candidate text."""
    m = re.search(r"\(([A-D])\)\s*[^\n]*\Z", (text or "").strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None

def normalize(x: List[float]) -> List[float]:
    s = float(sum(x))
    if s > 0:
        return [t/s for t in x]
    return [1.0/len(x)]*len(x) if x else []

def build_labels(cands: List[str], gold: Optional[str]) -> List[float]:
    if gold is None:
        # Unknown gold -> weak heuristic: shorter responses get slightly higher weight
        return normalize([max(0.0, 1.0 - 0.001*len(t or "")) for t in cands])
    flags = [1.0 if extract_choice_letter(t or "") == gold else 0.0 for t in cands]
    if sum(flags) == 0:
        # No candidate picked the gold; fall back to weak heuristic
        return normalize([max(0.0, 1.0 - 0.001*len(t or "")) for t in cands])
    return normalize(flags)

# -------- Robust CSV loader (handles headered *and* headerless) --------
def load_mmlu_row(subject_csv: str, q_index: int) -> Dict[str, Any]:
    """
    Returns {"question": str, "choices": {A,B,C,D}, "correct": "A"/...}.
    Works with:
      - headerless rows: <q>, <A>, <B>, <C>, <D>, <ans>
      - headered rows with columns like: question/prompt/text/q, A,B,C,D, answer/correct/label/...
    """
    with open(subject_csv, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if any((c or "").strip() for c in r)]
    if not rows:
        raise ValueError(f"Empty CSV: {subject_csv}")

    header_lower = [c.strip().lower() for c in rows[0]]
    has_header = (
        ("answer" in header_lower or "correct" in header_lower or "label" in header_lower or "gold" in header_lower or "target" in header_lower)
        and ("question" in header_lower or "prompt" in header_lower or "text" in header_lower or "q" in header_lower)
    )

    body = rows[1:] if has_header else rows
    if q_index < 0 or q_index >= len(body):
        raise IndexError(f"q_index {q_index} out of range for {subject_csv} (n={len(body)})")
    row = body[q_index]

    if has_header:
        hdr = [c.strip() for c in rows[0]]
        d = {hdr[i]: (row[i] if i < len(row) else "") for i in range(len(hdr))}
        def get_any(keys: Iterable[str], default: Optional[str]=None):
            for k in keys:
                if k in d and d[k] != "":
                    return d[k]
            return default
        question = get_any(["question","Question","prompt","text","q"]) or ""
        choices  = {k: (get_any([k, k.lower()]) or "") for k in CHOICE_KEYS}
        correct  = (get_any(["answer","correct","label","gold","target"]) or "").strip().upper()
    else:
        # headerless: expect at least 6 columns
        if len(row) < 6:
            row = row + [""]*(6 - len(row))
        question = (row[0] or "").strip()
        choices  = {"A": (row[1] or "").strip(),
                    "B": (row[2] or "").strip(),
                    "C": (row[3] or "").strip(),
                    "D": (row[4] or "").strip()}
        correct  = (row[5] or "").strip().upper()

    # Normalize correct
    if correct not in CHOICE_KEYS:
        m = re.search(r"([A-D])", correct, re.IGNORECASE) or re.search(r"([1-4])", correct)
        if not m:
            raise KeyError(f"Missing gold answer column in {subject_csv}")
        correct = m.group(1).upper() if m.group(1) in "ABCD" else {"1":"A","2":"B","3":"C","4":"D"}[m.group(1)]
    return {"question": question, "choices": choices, "correct": correct}

def parse_question_from_log(log_path: str, q_index: int) -> Optional[Dict[str, Any]]:
    """
    Very rough fallback: looks for "Here is the question:" blocks and tries to parse A..D options.
    """
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    idxs = [m.start() for m in re.finditer(r"Here is the question:", txt)]
    if q_index >= len(idxs):
        return None
    window = txt[idxs[q_index]: idxs[q_index]+3000]
    m = re.search(
        r"Here is the question:\s*(.*?)\s*A\)\s*(.*?)[,;\n]\s*B\)\s*(.*?)[,;\n]\s*C\)\s*(.*?)[,;\n]\s*D\)\s*(.*?)(?:\n|$)",
        window, re.DOTALL
    )
    if not m:
        return None
    return {
        "question": m.group(1).strip(),
        "choices": {"A": m.group(2).strip(), "B": m.group(3).strip(),
                    "C": m.group(4).strip(), "D": m.group(5).strip()},
        "correct": None,
    }

def to_judge_prompt(question: str, choices: Dict[str,str], cands: List[str]) -> str:
    opts = "\n".join([f"({k}) {choices.get(k,'')}" for k in CHOICE_KEYS])
    cfmt = "\n".join([f"Candidate {i+1}: {t}" for i,t in enumerate(cands)])
    instr = (
        "You are a precise evaluator. Given a multiple-choice question, its options, and several candidate answers,\n"
        "assign a quality score to EACH candidate so the scores sum to 1.\n"
        "Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10]."
    )
    return f"{instr}\n\nQuestion:\n{question}\n\nOptions:\n{opts}\n\n{cfmt}\n\nScores:"

def find_subject_csv(mmlu_eval_dir: str, stem: str) -> Optional[str]:
    """
    Map '<stem>' (e.g., 'abstract_algebra_test_73') to a CSV path in mmlu_eval_dir.
    Handles '<stem>_test/_val' and strips trailing '_<digits>'.
    """
    s = re.sub(r"_\d+$", "", stem)  # drop trailing _73
    if not (s.endswith("_test") or s.endswith("_val")):
        s += "_test"
    cand = os.path.join(mmlu_eval_dir, f"{s}.csv")
    if os.path.exists(cand):
        return cand
    alt = s[:-5] + "_val" if s.endswith("_test") else s[:-4] + "_test"
    cand2 = os.path.join(mmlu_eval_dir, f"{alt}.csv")
    return cand2 if os.path.exists(cand2) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Dir with *_73.json/txt/log from your DyLAN pre-selection run.")
    ap.add_argument("--mmlu_eval_dir", default="data/MMLU/small_team_selection",
                    help="Where the per-subject CSVs live (small_team_selection/ or medium_team_selection/).")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--roles", default=None,
                    help="JSON list of roles if you used a non-default order.")
    ap.add_argument("--min_candidates", type=int, default=2,
                    help="Skip questions with < this many non-empty candidates.")
    ap.add_argument("--max_candidates", type=int, default=None,
                    help="Cap the number of candidates per question (keep first K).")
    args = ap.parse_args()

    roles = DEFAULT_ROLES if args.roles is None else json.loads(args.roles)
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    txt_files = sorted([p for p in os.listdir(args.results_dir) if p.endswith("_73.txt")])
    if not txt_files:
        warn(f"No *_73.txt files found under {args.results_dir}")

    n_written = n_skipped = n_used_csv = n_used_log = 0

    with open(args.outfile, "w", encoding="utf-8") as out_f:
        for txt_name in txt_files:
            stem = txt_name[:-4]                     # drop ".txt"
            json_path = os.path.join(args.results_dir, stem + ".json")
            log_path  = os.path.join(args.results_dir, stem + ".log")

            if not os.path.exists(json_path):
                warn(f"{stem}: missing JSON alongside .txt; skipping.")
                n_skipped += 1
                continue

            subject_csv = find_subject_csv(args.mmlu_eval_dir, stem)
            if subject_csv is None:
                warn(f"{stem}: no <subject>.csv found in {args.mmlu_eval_dir}; will try log fallback if present.")

            # Parse per-question role triples from the concatenated JSON
            try:
                questions = parse_concat_json(json_path)
            except Exception as e:
                warn(f"{stem}: failed to parse JSON ({e}); skipping.")
                n_skipped += 1
                continue

            for q_idx, per_role_triples in enumerate(questions):
                # Gather first non-empty response per role across up to 3 rounds
                candidates: List[str] = []
                for triple in per_role_triples:
                    if not isinstance(triple, list):
                        continue
                    txt = None
                    if len(triple) >= 1:
                        txt = first_non_null(triple[0])
                    if txt is None and len(triple) >= 2:
                        txt = first_non_null(triple[1])
                    if txt is None and len(triple) >= 3:
                        txt = first_non_null(triple[2])
                    if txt:
                        candidates.append(txt)

                candidates = [t for t in candidates if isinstance(t, str) and t.strip()]
                if len(candidates) < args.min_candidates:
                    n_skipped += 1
                    continue
                if args.max_candidates is not None:
                    candidates = candidates[:args.max_candidates]

                # Load (question, options, correct) from CSV; else try log fallback (no gold)
                mmlu = None
                if subject_csv is not None:
                    try:
                        mmlu = load_mmlu_row(subject_csv, q_idx)
                        n_used_csv += 1
                    except Exception as e:
                        warn(f"{stem}: failed to load MMLU row {q_idx}: {e}; trying log fallback.")

                if mmlu is None and os.path.exists(log_path):
                    mmlu = parse_question_from_log(log_path, q_idx)
                    if mmlu is not None:
                        n_used_log += 1

                if mmlu is None:
                    warn(f"{stem}: could not load question/options for q={q_idx}; skipping.")
                    n_skipped += 1
                    continue

                prompt = to_judge_prompt(mmlu["question"], mmlu["choices"], candidates)
                labels = build_labels(candidates, mmlu.get("correct"))

                out_f.write(json.dumps({
                    "prompt": prompt,
                    "completion": json.dumps(labels),
                    "meta": {
                        "results_stem": stem,
                        "q_index": q_idx,
                        "subject_csv": os.path.basename(subject_csv) if subject_csv else None,
                        "correct": mmlu.get("correct"),
                        "n_candidates": len(candidates),
                        "source": "csv" if mmlu.get("correct") else "log",
                        "roles": roles,
                    },
                    "question": mmlu["question"],
                    "choices": mmlu["choices"],
                    "agent_responses": candidates,
                }, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[DONE] wrote {n_written} examples to {args.outfile} "
          f"(skipped: {n_skipped}; used_csv: {n_used_csv}; used_log_fallback: {n_used_log}).")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run DyLAN pre-selection style generation, but score candidates with a *local* fine-tuned judge.

This file fixes:
  - Roles parsing: accepts JSON (["Economist", ...]) or Python-style list (['Economist', ...]).
  - CLI: accepts either *positional* or *flagged* arguments (so your shell script keeps working).

Minimal I/O:
  - Reads a single MMLU subject CSV (like the ones under data/MMLU/small_team_selection).
  - Prints progress to stdout (your exp script can tee to a log if desired).
  - You can extend this to write *_73.{json,txt,log} if you want the classic DyLAN artifacts.

Example (works with your current bash wrapper):
  python code/MMLU/run_mmlu_with_local_judge.py \
      "$csv" "$subject" "$MODEL" "$OUT_DIR" "['Economist','Doctor',...]" \
      --judge-ckpt "$JUDGE_CKPT"

Or the flagged form:
  python code/MMLU/run_mmlu_with_local_judge.py \
      --csv path/to/abstract_algebra_test.csv \
      --subject abstract_algebra_test \
      --model openai/gpt-oss-20b \
      --out-dir code/MMLU/standard_dylan/mmlu_with_local_judge \
      --roles-json "['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']" \
      --judge-ckpt code/MMLU/finetune/ckpts/merged
"""
from __future__ import annotations
import argparse, csv, json, os, sys, ast, re
from typing import List, Tuple
from dotenv import load_dotenv
from together import Together

from local_judge import LocalJudge

DEFAULT_ROLES_7 = [
    "Economist", "Doctor", "Lawyer", "Mathematician", "Psychologist", "Programmer", "Historian"
]

# Very light role system prompts (you can swap in your repo's prompt_lib if you prefer)
ROLE_SYSTEM = {
    "Economist":     "You are an economist. You are good at economics, finance, and business.",
    "Doctor":        "You are a doctor. Provide factual, concise medical reasoning.",
    "Lawyer":        "You are a lawyer. You reason precisely about rules, definitions, and evidence.",
    "Mathematician": "You are a mathematician. You reason rigorously with proofs and calculations.",
    "Psychologist":  "You are a psychologist. You explain motivations and cognitive pitfalls clearly.",
    "Programmer":    "You are a programmer. You reason step by step and test edge cases mentally.",
    "Historian":     "You are a historian. You focus on factual accuracy and chronology."
}

def eprint(*a, **k): print(*a, file=sys.stderr, **k)


def parse_roles(s: str | None) -> List[str]:
    """
    Accept either JSON or Python-literal style lists, e.g.:
      '["Economist","Doctor"]'    or    "['Economist','Doctor']"
    """
    if not s:
        return DEFAULT_ROLES_7[:]
    try:
        return json.loads(s)
    except Exception:
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
        except Exception:
            pass
    eprint(f"[WARN] Could not parse roles from: {s!r}; falling back to default 7 roles.")
    return DEFAULT_ROLES_7[:]


def build_user_prompt(question: str, choices: dict) -> str:
    opts = "\n".join([f"({k}) {choices.get(k,'')}" for k in ("A","B","C","D")])
    return (
        "Here is the question:\n"
        f"{question}: A) {choices.get('A','')}, B) {choices.get('B','')}, "
        f"C) {choices.get('C','')}, D) {choices.get('D','')}\n\n"
        "Put your answer in the form (X) at the end of your response. "
        "(X) represents choice (A), (B), (C), or (D)."
    )


def extract_letter(text: str) -> str | None:
    m = re.search(r"\(([A-D])\)\s*\Z", text.strip(), re.I)
    return m.group(1).upper() if m else None


def generate_role_answer(client: Together, model: str, role: str, user_prompt: str) -> str:
    sys_prompt = ROLE_SYSTEM.get(role, f"You are a helpful {role.lower()}.")
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2, max_tokens=512)
    return resp.choices[0].message.content.strip()


def judge_prompt_from_candidates(question: str, choices: dict, cands: List[str]) -> str:
    opts = "\n".join([f"(A) {choices.get('A','')}",
                      f"(B) {choices.get('B','')}",
                      f"(C) {choices.get('C','')}",
                      f"(D) {choices.get('D','')}"])
    cfmt = "\n".join([f"Candidate {i+1}: {t}" for i, t in enumerate(cands)])
    instr = ("You are a precise evaluator. Given a multiple-choice question, its options, and several "
             "candidate answers, assign a quality score to EACH candidate so the scores sum to 1.\n"
             "Return ONLY a Python-style list of floats, e.g., [0.55, 0.35, 0.10].")
    return f"{instr}\n\nQuestion:\n{question}\n\nOptions:\n{opts}\n\n{cfmt}\n\nScores:"


def read_mmlu_csv(path: str) -> List[Tuple[str, dict, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.reader(f):
            if not any(c.strip() for c in r):
                continue
            # Expect either headerless:  Q, A, B, C, D, letter   or with header in first row
            if rows == []:
                hdr_lower = [c.strip().lower() for c in r]
                has_header = ("answer" in hdr_lower or "correct" in hdr_lower) and \
                             ("question" in hdr_lower or "prompt" in hdr_lower or "text" in hdr_lower or "q" in hdr_lower)
                if has_header:
                    header = [c.strip() for c in r]
                    continue
                else:
                    # headerless → treat this row as data
                    pass
            if 'header' in locals():
                d = {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
                q = d.get("question") or d.get("prompt") or d.get("text") or d.get("q") or ""
                choices = {"A": d.get("A",""), "B": d.get("B",""), "C": d.get("C",""), "D": d.get("D","")}
                gold = (d.get("answer") or d.get("correct") or "").strip().upper()
            else:
                # headerless
                r = r + [""] * (6 - len(r)) if len(r) < 6 else r
                q, A, B, C, D, gold = r[0], r[1], r[2], r[3], r[4], r[5].strip().upper()
                choices = {"A": A, "B": B, "C": C, "D": D}
            if gold not in ("A","B","C","D"):
                m = re.search(r"([A-D])", gold, re.I) or re.search(r"([1-4])", gold)
                if m:
                    gold = m.group(1).upper() if m.group(1) in "ABCD" else {"1":"A","2":"B","3":"C","4":"D"}[m.group(1)]
                else:
                    gold = ""
            rows.append((q, choices, gold))
    return rows


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    # flagged args (optional)
    ap.add_argument("--csv", dest="csv_flag")
    ap.add_argument("--subject", dest="subject_flag")
    ap.add_argument("--model", dest="model_flag")
    ap.add_argument("--out-dir", dest="outdir_flag")
    ap.add_argument("--roles-json", dest="roles_flag",
                    help="JSON or Python list string of roles")
    ap.add_argument("--judge-ckpt", required=True, help="Path to local merged checkpoint for the judge")

    # positional fallback (so your current bash wrapper keeps working)
    ap.add_argument("pos_csv", nargs="?", help="CSV path")
    ap.add_argument("pos_subject", nargs="?", help="subject/stem")
    ap.add_argument("pos_model", nargs="?", help="base model, e.g. openai/gpt-oss-20b")
    ap.add_argument("pos_outdir", nargs="?", help="output directory")
    ap.add_argument("pos_roles", nargs="?", help="roles as JSON or Python list string")
    args = ap.parse_args()

    csv_path   = args.csv_flag     or args.pos_csv
    subject    = args.subject_flag or args.pos_subject
    model_name = args.model_flag   or args.pos_model or os.getenv("MODEL", "openai/gpt-oss-20b")
    out_dir    = args.outdir_flag  or args.pos_outdir or "code/MMLU/standard_dylan/mmlu_with_local_judge"
    roles_list = parse_roles(args.roles_flag or args.pos_roles)

    if not csv_path or not subject:
        ap.error("You must provide CSV path and subject (either as flags or positionals).")

    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] MODEL={model_name}")
    print(f"[info] JUDGE_CKPT={args.judge_ckpt}")
    print(f"[info] CSV={csv_path}")
    print(f"[info] OUT_DIR={out_dir}")
    print(f"[info] ROLES={roles_list}")

    # load dataset and clients
    rows = read_mmlu_csv(csv_path)
    if not rows:
        print(f"[WARN] No rows read from {csv_path}")
        return

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    judge = LocalJudge(args.judge_ckpt)

    subj_dir = os.path.join(out_dir, subject)
    os.makedirs(subj_dir, exist_ok=True)

    # Iterate questions
    all_answers = []      # per-question: list of role answers
    all_scores  = []      # per-question: judge scores (len = len(roles))
    all_preds   = []      # per-question: each role's predicted letter

    for qi, (q, choices, gold) in enumerate(rows):
        print(f">>> Q{qi}: {q[:80]}{'...' if len(q)>80 else ''}")

        # Build role answers
        user_prompt = build_user_prompt(q, choices)
        role_texts, role_letters = [], []
        for rname in roles_list:
            ans = generate_role_answer(client, model_name, rname, user_prompt)
            role_texts.append(ans)
            role_letters.append(extract_letter(ans) or "?")
            print(f"  - {rname}: {role_letters[-1]}")

        # Judge
        j_prompt = judge_prompt_from_candidates(q, choices, role_texts)
        scores, raw = judge.score(j_prompt, k=len(role_texts))
        print(f"  judge scores: {scores}")

        all_answers.append(role_texts)
        all_scores.append(scores)
        all_preds.append(role_letters)

    # (Optional) Write a compact JSON with everything for later analysis
    out_json = os.path.join(subj_dir, f"{subject}_local_judge.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "subject": subject,
            "roles": roles_list,
            "items": [
                {
                    "question": rows[i][0],
                    "choices": rows[i][1],
                    "gold": rows[i][2],
                    "role_answers": all_answers[i],
                    "role_letters": all_preds[i],
                    "judge_scores": all_scores[i],
                }
                for i in range(len(rows))
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()

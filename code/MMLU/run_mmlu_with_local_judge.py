# code/MMLU/run_mmlu_with_local_judge.py
# Produce Agent Importance (AIP) with a local LLM-as-judge and baseline model predictions.

from __future__ import annotations
import os, sys, csv, re, json, time, argparse, random
from typing import List, Dict, Tuple, Sequence

# Optional Together client (used if TOGETHER_API_KEY is available)
try:
    from together import Together
except Exception:
    Together = None  # handled gracefully

from local_judge import LocalJudge

ROLE_SYSTEM = {
    "Economist":      "You are an economist. You are good at economics, finance, and business.",
    "Doctor":         "You are a doctor. You are good at medicine and clinical reasoning.",
    "Lawyer":         "You are a lawyer. You are good at law, policy, and legal reasoning.",
    "Mathematician":  "You are a mathematician. You are good at abstract algebra and mathematics.",
    "Psychologist":   "You are a psychologist. You are good at psychology and reasoning about people.",
    "Programmer":     "You are a programmer. You are good at computer science and engineering.",
    "Historian":      "You are a historian. You are good at history and social analysis.",
}

def read_small_selection_csv(path: str) -> List[Dict[str, str]]:
    """
    Tolerant reader for the 'small_team_selection' CSVs with columns:
      Q, A, B, C, D, GOLD_LETTER
    (No header row; your example confirmed this.)
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        for line in r:
            if not line: continue
            # Skip lines that are all blanks
            if all(not (c or "").strip() for c in line): continue
            if len(line) < 6:
                # pad if needed (very rare)
                line = (line + [""] * 6)[:6]
            q, a, b, c, d, gold = [c.strip() for c in line[:6]]
            gold = (gold[:1].upper() if gold else "")
            if gold not in ("A","B","C","D"):
                # if somehow missing gold, keep empty; we still can run judge
                gold = ""
            rows.append({"q": q, "opts": [a, b, c, d], "gold": gold})
    return rows

def parse_letter(text: str) -> str:
    """
    Extract a single choice letter from model text. Try (X) then bare X.
    Return 'A' if nothing found (fail-safe, deterministic).
    """
    m = re.search(r"\(([A-D])\)", text)
    if m: return m.group(1)
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else "A"

def call_together_choice(client, model: str, system: str, question: str, opts: Sequence[str]) -> str:
    user = [
        "Here is the question:",
        "Can you answer the following question as accurately as possible?",
        question.strip(),
        "",
        "Options:",
        f"(A) {opts[0]}",
        f"(B) {opts[1]}",
        f"(C) {opts[2]}",
        f"(D) {opts[3]}",
        "",
        "Put your answer in the form (X) at the end of your response.",
        "(X) is one of (A), (B), (C), or (D).",
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )
    text = resp.choices[0].message.content
    return parse_letter(text or "")

def choose_for_all_roles(
    model: str,
    rows: List[Dict[str, str]],
    roles: Sequence[str],
) -> List[Dict[str, str]]:
    """
    For each row and role, call the base model once to get a letter (A..D).
    Returns a list of per-row dicts: {"labels": ["A","C",... len=K], "gold": "B"}
    """
    client = None
    if Together is not None and os.environ.get("TOGETHER_API_KEY"):
        client = Together()  # uses env

    out = []
    for r in rows:
        labels = []
        for role in roles:
            sys_prompt = ROLE_SYSTEM.get(role, f"You are a {role}. Think step by step.")
            if client is None:
                # Deterministic offline fallback: pretend the model says (A) for every role.
                # This keeps the run alive without an API key; the judge will then return uniform weights.
                letter = "A"
            else:
                letter = call_together_choice(client, model, sys_prompt, r["q"], r["opts"])
            labels.append(letter)
        out.append({"labels": labels, "gold": r["gold"]})
    return out

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--roles-json", required=True,
                    help="JSON list (e.g., \"['Economist',...]\") or path to a .json file")
    ap.add_argument("--judge-ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # roles can be JSON string or a path to .json that contains a list
    if os.path.isfile(args.roles_json):
        roles: List[str] = json.load(open(args.roles_json))
    else:
        roles = json.loads(args.roles_json)
    assert isinstance(roles, list) and len(roles) >= 1

    rows = read_small_selection_csv(args.csv)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    N = len(rows)

    base = f"{args.subject}_{N}"
    os.makedirs(args.outdir, exist_ok=True)
    p_json = os.path.join(args.outdir, f"{base}.json")
    p_txt  = os.path.join(args.outdir, f"{base}.txt")
    p_log  = os.path.join(args.outdir, f"{base}.log")

    if (not args.overwrite) and all(os.path.exists(p) for p in (p_json, p_txt, p_log)):
        print(f"[skip] {base} already exists; use --overwrite to recompute.")
        return

    # LOG header
    with open(p_log, "w") as lg:
        lg.write(f"[START] {now()} file={args.csv} subject={args.subject} model={args.model}\n")
        lg.write(f"[cfg] roles={roles}\n")
        lg.write(f"[cfg] judge_ckpt={args.judge_ckpt}\n")

    # Get labels per role; then score with local judge
    per_row = choose_for_all_roles(args.model, rows, roles)
    judge = LocalJudge(args.judge_ckpt)

    sum_weights = [0.0] * len(roles)
    for i, (r, pr) in enumerate(zip(rows, per_row), 1):
        weights, raw = judge.score(
            question=r["q"], options=r["opts"],
            candidate_labels=[f"({x})" for x in pr["labels"]],
            roles=roles,
        )
        # aggregate
        for j, w in enumerate(weights):
            sum_weights[j] += w

        if args.verbose:
            with open(p_log, "a") as lg:
                lg.write("\n")
                lg.write(f"[row {i}] gold={r['gold']} labels={pr['labels']}\n")
                lg.write(f"[judge] weights={weights}\n")
                lg.write(f"[judge raw]\n{raw}\n")

    # Normalize across roles
    s = sum(sum_weights)
    if s <= 0:
        final = [1.0 / len(roles)] * len(roles)
    else:
        final = [x / s for x in sum_weights]

    # Write JSON
    obj = {
        "subject": args.subject,
        "n_examples": N,
        "roles": roles,                       # stable order as given
        "aip": {r: float(w) for r, w in zip(roles, final)}
    }
    with open(p_json, "w") as f:
        json.dump(obj, f, indent=2)

    # Write TXT
    with open(p_txt, "w") as f:
        f.write(f"Subject: {args.subject}\n")
        f.write(f"N: {N}\n\n")
        f.write("Agent Importance (normalized):\n")
        for r, w in zip(roles, final):
            f.write(f"- {r}: {w:0.4f}\n")

    with open(p_log, "a") as lg:
        lg.write(f"\n[END] {now()} wrote {p_json}, {p_txt}, {p_log}\n")

    print(f"[ok] wrote {p_json}, {p_txt}, {p_log}")

if __name__ == "__main__":
    main()

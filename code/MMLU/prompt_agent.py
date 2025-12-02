#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PromptAgent for DyLAN (MMLU)

- Proposes prompt edits for roles and interaction/ranker prompts.
- Evaluates candidates on a small/medium training split via your existing runner.
- Optimizes an objective: score = accuracy - α * (api_calls_per_q) - β * (k_tokens_per_q)

Usage (examples):

  # Train on small_team_selection
  python code/MMLU/prompt_agent.py \
    --train_dir "$(pwd)/data/MMLU/small_team_selection" \
    --scripts_dir "$(pwd)/code/MMLU" \
    --model "${MODEL:-openai/gpt-oss-20b}" \
    --base_profile "$(pwd)/code/MMLU/profiles/base_profile.json" \
    --out_profile "$(pwd)/code/MMLU/fine_tuned_prompts/best_profile.json" \
    --exp_name "fine_tuned_prompts"

  # Train on medium_team_selection
  # python code/MMLU/prompt_agent.py --train_dir "$(pwd)/data/MMLU/medium_team_selection" ...

Notes:
- We rely on PROMPT_PROFILE in prompt_lib.py to inject role/interaction/ranker prompts.
- We do black-box eval by calling llmlp_listwise_mmlu.py per CSV (same args as exp_mmlu.sh).

"""

import argparse, os, sys, json, time, random, subprocess, shlex, tempfile, re, ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# -------------------------------
# Utility: objective function
# -------------------------------

def objective(
    acc: float,
    n_q: int,
    api_calls_total: int,
    prompt_toks: int,
    compl_toks: int,
    alpha_api: float,
    beta_tok: float,
) -> float:
    """
    Compute score with accuracy dominating and small penalties for API calls / tokens.
    - api_calls_per_q: api_calls_total / n_q
    - k_tokens_per_q: (prompt+completion)/n_q / 1000.0
    """
    if n_q <= 0:
        return -1e9
    api_per_q = api_calls_total / float(n_q)
    k_tok_per_q = (prompt_toks + compl_toks) / float(n_q) / 1000.0
    return float(acc) - alpha_api * api_per_q - beta_tok * k_tok_per_q


# -------------------------------
# Robust log parser
# -------------------------------

BOOL_LIST_RE = re.compile(r"^\s*\[(?:\s*(?:True|False)\s*(?:,\s*)?)+\]\s*$")
INT_LINE_RE = re.compile(r"^\s*\d+\s*$")
TOKENS_PAIR_RE = re.compile(
    r"(?:prompt[_\s-]*tokens?\s*[:=]\s*(\d+).*)?(?:completion[_\s-]*tokens?\s*[:=]\s*(\d+))?",
    re.IGNORECASE,
)

def _parse_result_log(log_path: Path) -> Tuple[float, int, int, int, int, int]:
    """
    Parse an llmlp_listwise_mmlu.py streaming .log like the one the user posted.
    We extract:
      - accuracy from the last boolean list line (e.g., [True, False, ...])
      - api_calls_total from the last integer line near the end (e.g., "10")
      - tokens (prompt/completion) if present; otherwise 0.

    Returns: (accuracy, n_questions, api_calls_total, tokens_total, prompt_tokens, completion_tokens)
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Missing result log: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    n_questions = 0
    accuracy = 0.0
    api_calls_total = 0
    prompt_tokens = 0
    completion_tokens = 0

    # 1) find the last boolean list
    bool_line_idx = None
    for i in range(len(text) - 1, -1, -1):
        line = text[i].strip()
        if BOOL_LIST_RE.match(line):
            bool_line_idx = i
            break

    if bool_line_idx is not None:
        try:
            arr = ast.literal_eval(text[bool_line_idx].strip())
            if isinstance(arr, list) and all(isinstance(x, bool) for x in arr) and len(arr) > 0:
                n_questions = len(arr)
                accuracy = sum(1 for x in arr if x) / float(n_questions)
        except Exception:
            pass

    # 2) try to get api_calls_total
    #    heuristic: pick the last pure integer line that is NOT part of a vector/matrix block
    #    (the weight matrices follow after; they start with "[" not pure digits)
    for i in range(len(text) - 1, -1, -1):
        line = text[i].strip()
        if INT_LINE_RE.match(line):
            try:
                api_calls_total = int(line)
                break
            except Exception:
                continue

    # Fallback if still zero: estimate by counting invocations
    if api_calls_total <= 0:
        api_calls_total = sum(1 for line in text if "question context:" in line)

    # 3) optional tokens (look for "prompt tokens:" ... "completion tokens:")
    #    allow either on one line or on two lines
    # Pass 1: single line with both
    for i in range(len(text) - 1, -1, -1):
        m = TOKENS_PAIR_RE.search(text[i])
        if m:
            p, c = m.group(1), m.group(2)
            if p: prompt_tokens = int(p)
            if c: completion_tokens = int(c)
            break
    # Pass 2: separate lines
    if prompt_tokens == 0 or completion_tokens == 0:
        for i in range(len(text) - 1, -1, -1):
            s = text[i].lower()
            if "prompt" in s and "token" in s:
                nums = re.findall(r"\d+", s)
                if nums:
                    prompt_tokens = int(nums[-1])
                    break
        for i in range(len(text) - 1, -1, -1):
            s = text[i].lower()
            if "completion" in s and "token" in s:
                nums = re.findall(r"\d+", s)
                if nums:
                    completion_tokens = int(nums[-1])
                    break

    return accuracy, n_questions, api_calls_total, (prompt_tokens + completion_tokens), prompt_tokens, completion_tokens


# -------------------------------
# Running one CSV / subject
# -------------------------------

def _eval_one_csv(
    scripts_dir: Path,
    csv_path: Path,
    model: str,
    exp_name: str,
    roles: List[str],
    out_root: Path,
    env: Dict[str, str],
) -> Tuple[float, int, int, int, int, int, Path]:
    """
    Run one subject CSV with llmlp_listwise_mmlu.py and parse its .log.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    filename = csv_path.stem
    roles_str = "[" + ",".join(f"'{r}'" for r in roles) + "]"

    log_path = out_root / f"{filename}_73.log"  # keep _73 like exp_mmlu.sh for consistency

    # If already finished and parseable, skip execution
    if log_path.exists():
        try:
            acc, n_q, api, tok_total, p_tok, c_tok = _parse_result_log(log_path)
            if n_q > 0:
                return acc, n_q, api, tok_total, p_tok, c_tok, log_path
        except Exception:
            pass  # re-run if parsing fails

    # Run llmlp_listwise_mmlu.py the same way exp_mmlu.sh does
    runner = scripts_dir / "llmlp_listwise_mmlu.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    cmd = [
        sys.executable,
        str(runner),
        str(csv_path),
        filename,
        model,
        exp_name,
        roles_str,
    ]

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[START] file={csv_path} model={model}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        logf.write(f"\n[END] status={proc.returncode}\n")

    if proc.returncode != 0:
        raise RuntimeError(f"Runner failed for {csv_path}. See log: {log_path}")

    acc, n_q, api, tok_total, p_tok, c_tok = _parse_result_log(log_path)
    if n_q <= 0:
        raise RuntimeError(f"Could not parse accuracy from {log_path}")
    return acc, n_q, api, tok_total, p_tok, c_tok, log_path


@dataclass
class EvalSummary:
    acc: float
    n_questions: int
    api_calls_total: int
    prompt_tokens: int
    completion_tokens: int
    logs: List[Path]

def evaluate_profile(
    train_dir: Path,
    scripts_dir: Path,
    model: str,
    roles: List[str],
    out_root: Path,
    exp_name: str,
    seed: int,
    alpha_api: float,
    beta_tok: float,
    env: Dict[str, str],
) -> Tuple[EvalSummary, float]:
    """
    Evaluate the given profile on all CSVs in train_dir. Returns the summary and the objective.
    """
    csvs = sorted(train_dir.glob("*.csv"))
    total_correct = 0.0
    total_q = 0
    total_api_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    logs: List[Path] = []

    for csv_path in csvs:
        acc, n_q, api, tok_total, p_tok, c_tok, log_path = _eval_one_csv(
            scripts_dir, csv_path, model, exp_name, roles, out_root, env
        )
        logs.append(log_path)
        total_q += n_q
        total_correct += acc * n_q
        total_api_calls += api
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

    acc_overall = (total_correct / float(total_q)) if total_q > 0 else 0.0
    score = objective(
        acc_overall,
        total_q,
        total_api_calls,
        total_prompt_tokens,
        total_completion_tokens,
        alpha_api,
        beta_tok,
    )

    return (
        EvalSummary(
            acc=acc_overall,
            n_questions=total_q,
            api_calls_total=total_api_calls,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            logs=logs,
        ),
        score,
    )


# -------------------------------
# Profile mutation (very small, safe edits)
# -------------------------------

def _load_profile(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_profile(p: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

MUTATION_POOL = [
    # strengthen ranker instruction (listwise)
    ("RANKER_INSTRUCTION_SINGLE_CHOICE",
     "\n\nPlease choose the best 2 solutions and justify briefly. Return only a list like [i,j] at the end."),
    ("RANKER_INSTRUCTION_MATH",
     "\n\nPlease choose the best 2 solutions and justify briefly. Return only a list like [i,j] at the end."),
    # agent interaction additions
    ("AGENT_INTERACTION_SINGLE_CHOICE",
     "\n\nDouble-check arithmetic and confirm the final choice explicitly as (A|B|C|D)."),
    ("AGENT_INTERACTION_MATH",
     "\n\nVerify each step; avoid leaps; show the final numeric answer clearly."),
]

def _mutate_profile(base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    p = json.loads(json.dumps(base))  # deep copy
    # apply 1–2 lightweight mutations
    k = rng.randint(1, 2)
    keys = rng.sample(MUTATION_POOL, k)
    for key, addition in keys:
        old = p.get(key, "")
        if addition not in old:
            p[key] = (old + addition).strip()
    return p


# -------------------------------
# Optimization loop
# -------------------------------

def optimize(
    train_dir: Path,
    scripts_dir: Path,
    model: str,
    roles: List[str],
    base_profile_path: Path,
    out_profile_path: Path,
    out_runs_dir: Path,
    exp_name: str,
    iters: int,
    seed: int,
    alpha_api: float,
    beta_tok: float,
    env: Dict[str, str],
) -> Dict[str, Any]:
    rng = random.Random(seed)

    base_profile = _load_profile(base_profile_path)
    # Ensure PROMPT_PROFILE points to the current candidate before each eval
    env = dict(env)  # copy
    env["PROMPT_PROFILE"] = str(base_profile_path)

    # 1) Evaluate base
    base_out_dir = out_runs_dir / f"{exp_name}_{'_'.join(roles)}"
    base_summary, base_score = evaluate_profile(
        train_dir, scripts_dir, model, roles, base_out_dir, exp_name, seed, alpha_api, beta_tok, env
    )
    best = {
        "profile": base_profile,
        "score": base_score,
        "summary": base_summary.__dict__,
        "profile_path": str(base_profile_path),
    }

    # 2) Iterate prompt mutations
    tmp_dir = out_profile_path.parent / ".promptagent_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for t in range(1, iters + 1):
        candidate = _mutate_profile(base_profile, rng)
        cand_path = tmp_dir / f"candidate_{t:03d}.json"
        _save_profile(candidate, cand_path)

        env["PROMPT_PROFILE"] = str(cand_path)
        cand_out_dir = out_runs_dir / f"{exp_name}_{'_'.join(roles)}_iter{t:03d}"
        summary, score = evaluate_profile(
            train_dir, scripts_dir, model, roles, cand_out_dir, exp_name, seed, alpha_api, beta_tok, env
        )

        if score > best["score"]:
            best = {
                "profile": candidate,
                "score": score,
                "summary": summary.__dict__,
                "profile_path": str(cand_path),
            }

    # 3) Write the winner
    _save_profile(best["profile"], out_profile_path)
    return best


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--scripts_dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base_profile", type=str, required=True)
    ap.add_argument("--out_profile", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default="promptagent")
    ap.add_argument("--iters", type=int, default=6, help="Number of mutation rounds.")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--roles", type=str, default="Economist,Doctor,Lawyer,Mathematician,Psychologist,Programmer,Historian")
    # Accuracy-dominant defaults (small cost penalties)
    ap.add_argument("--alpha_api", type=float, default=float(os.getenv("PA_ALPHA_API", "0.02")))
    ap.add_argument("--beta_tok", type=float, default=float(os.getenv("PA_BETA_TOK", "0.000002")))
    args = ap.parse_args()

    repo_root = Path.cwd()
    train_dir = Path(args.train_dir).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve()
    base_profile_path = Path(args.base_profile).expanduser().resolve()
    out_profile_path = Path(args.out_profile).expanduser().resolve()

    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    model = args.model
    exp_name = args.exp_name

    # Logs and per-run outputs go here
    out_runs_dir = scripts_dir / "promptagent_runs"

    # Inherit the repo-root .env if present (like exp_mmlu.sh does)
    env = os.environ.copy()
    dotenv = repo_root / ".env"
    if dotenv.exists():
        # naive .env loader: KEY=VAL per line (no quotes)
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    result = optimize(
        train_dir=train_dir,
        scripts_dir=scripts_dir,
        model=model,
        roles=roles,
        base_profile_path=base_profile_path,
        out_profile_path=out_profile_path,
        out_runs_dir=out_runs_dir,
        exp_name=exp_name,
        iters=args.iters,
        seed=args.seed,
        alpha_api=args.alpha_api,
        beta_tok=args.beta_tok,
        env=env,
    )

    # Pretty print
    print("\n=== PromptAgent result ===")
    print(json.dumps({
        "best_score": result["score"],
        "best_profile_path": str(out_profile_path),
        "eval_summary": result["summary"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PromptAgent-style hill-climbing for DyLAN (MMLU).

- Uses a meta-LLM to *rewrite* the JSON prompt profile (roles, system prompts,
  interaction templates, ranker instructions).
- Evaluates each candidate profile on a small training split via your existing
  runner (llmlp_listwise_mmlu.py).
- Optimizes an objective: score = accuracy - α * (api_calls_per_q)
                            - β * (k_tokens_per_q),
  with accuracy dominating.

Search algorithm: simple hill-climbing
------------------------------------------------
1. Start from base_profile.json (state P).
2. Evaluate P on train_dir (small_team_selection or medium_team_selection):
   - Run DyLAN MMLU for each CSV in train_dir.
   - Parse per-question correctness from logs.
   - Collect a small set of *error examples* (question, gold, predicted).
3. Ask a meta-LLM to propose K new profiles P^(1..K) in JSON, based on:
   - The current profile P, and
   - The error examples.
4. Evaluate each candidate P^(k) on train_dir.
5. If any candidate beats the current best score, move to the best candidate
   and repeat; otherwise stop early.
6. Write the best profile to --out_profile.

Usage (from repo root):

  python code/MMLU/prompt_agent.py \
    --train_dir "$(pwd)/data/MMLU/small_team_selection" \
    --scripts_dir "$(pwd)/code/MMLU" \
    --model "${MODEL:-openai/gpt-oss-20b}" \
    --base_profile "$(pwd)/code/MMLU/profiles/base_profile.json" \
    --out_profile "$(pwd)/code/MMLU/fine_tuned_prompts/best_profile.json" \
    --exp_name "fine_tuned_prompts" \
    --iters 3

Then:

  export PROMPT_PROFILE="$(pwd)/code/MMLU/fine_tuned_prompts/best_profile.json"
  cd code/MMLU
  bash exp_mmlu.sh
  bash exp_mmlu_evaluation.sh

Meta-LLM configuration
----------------------
We use the `together` Python client. The client is created with:

  api_key = PROMPTAGENT_API_KEY or TOGETHER_API_KEY
  base_url = PROMPTAGENT_API_BASE or TOGETHER_BASE_URL (optional)

Meta model:

  PROMPTAGENT_META_MODEL (if set) else --model argument.

So for Together:

  export TOGETHER_API_KEY=...
  export TOGETHER_BASE_URL=https://api.together.xyz/v1
  export MODEL=openai/gpt-oss-20b
  # Optional but recommended:
  export PROMPTAGENT_META_MODEL=openai/gpt-oss-20b   # or a smaller model
"""

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from together import Together  # type: ignore
except ImportError:  # pragma: no cover
    Together = None  # type: ignore


# -------------------------------
# Objective function
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
    - api_calls_per_q = api_calls_total / n_q
    - k_tokens_per_q  = (prompt+completion)/n_q / 1000.0
    """
    if n_q <= 0:
        return -1e9
    api_per_q = api_calls_total / float(n_q)
    k_tok_per_q = (prompt_toks + compl_toks) / float(n_q) / 1000.0
    return float(acc) - alpha_api * api_per_q - beta_tok * k_tok_per_q


# -------------------------------
# Log parsing utilities
# -------------------------------

BOOL_LIST_RE = re.compile(r"^\s*\[(?:\s*(?:True|False)\s*(?:,\s*)?)+\]\s*$")
INT_LINE_RE = re.compile(r"^\s*\d+\s*$")
TOKENS_PAIR_RE = re.compile(
    r"(?:prompt[_\s-]*tokens?\s*[:=]\s*(\d+).*)?(?:completion[_\s-]*tokens?\s*[:=]\s*(\d+))?",
    re.IGNORECASE,
)
CONSENSUS_RE = re.compile(r"Consensus answer:\s*([A-D])", re.IGNORECASE)


@dataclass
class LogMetrics:
    acc: float
    n_questions: int
    api_calls_total: int
    tokens_total: int
    prompt_tokens: int
    completion_tokens: int
    correctness: List[bool]
    predictions: List[str]


def _parse_result_log(log_path: Path) -> LogMetrics:
    """
    Parse an llmlp_listwise_mmlu.py log file.

    We extract:
      - per-question correctness from the last boolean list (e.g. [True, False, ...])
      - predictions from "Consensus answer: X" lines
      - api_calls_total from the last pure integer line
      - prompt/completion tokens heuristically if present
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Missing result log: {log_path}")

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 1) correctness list from last boolean list line
    correctness: List[bool] = []
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if BOOL_LIST_RE.match(line):
            try:
                arr = ast.literal_eval(line)
                if isinstance(arr, list) and all(isinstance(x, bool) for x in arr):
                    correctness = [bool(x) for x in arr]
            except Exception:
                pass
            break
    n_questions = len(correctness)

    # 2) predictions from "Consensus answer: X"
    predictions: List[str] = []
    for line in lines:
        m = CONSENSUS_RE.search(line)
        if m:
            predictions.append(m.group(1).strip().upper())

    # Align lengths (best-effort)
    if predictions and len(predictions) != n_questions:
        predictions = predictions[:n_questions]

    # 3) api_calls_total: last pure integer line
    api_calls_total = 0
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if INT_LINE_RE.match(s):
            try:
                api_calls_total = int(s)
                break
            except Exception:
                continue

    # 4) optional tokens
    prompt_tokens = 0
    completion_tokens = 0

    # pass 1: look for both on the same line, near the bottom
    for i in range(len(lines) - 1, -1, -1):
        m = TOKENS_PAIR_RE.search(lines[i])
        if m:
            p, c = m.group(1), m.group(2)
            if p:
                prompt_tokens = int(p)
            if c:
                completion_tokens = int(c)
            if prompt_tokens or completion_tokens:
                break

    # pass 2: separate "prompt token" / "completion token" lines
    if prompt_tokens == 0:
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].lower()
            if "prompt" in s and "token" in s:
                nums = re.findall(r"\d+", s)
                if nums:
                    prompt_tokens = int(nums[-1])
                    break
    if completion_tokens == 0:
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].lower()
            if "completion" in s and "token" in s:
                nums = re.findall(r"\d+", s)
                if nums:
                    completion_tokens = int(nums[-1])
                    break

    tokens_total = prompt_tokens + completion_tokens

    # 5) accuracy
    if n_questions > 0:
        acc = sum(1 for x in correctness if x) / float(n_questions)
    else:
        acc = 0.0

    return LogMetrics(
        acc=acc,
        n_questions=n_questions,
        api_calls_total=api_calls_total,
        tokens_total=tokens_total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        correctness=correctness,
        predictions=predictions,
    )


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
) -> Tuple[LogMetrics, Path]:
    """
    Run one subject CSV with llmlp_listwise_mmlu.py and parse its log.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    filename = csv_path.stem
    roles_str = "[" + ",".join(f"'{r}'" for r in roles) + "]"
    log_path = out_root / f"{filename}_73.log"  # keep suffix consistent with exp_mmlu.sh

    # If log already exists and appears parseable, reuse
    if log_path.exists():
        try:
            metrics = _parse_result_log(log_path)
            if metrics.n_questions > 0:
                return metrics, log_path
        except Exception:
            pass  # fall through to re-run

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
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logf.write(f"\n[END] status={proc.returncode}\n")

    if proc.returncode != 0:
        raise RuntimeError(f"Runner failed for {csv_path}. See log: {log_path}")

    metrics = _parse_result_log(log_path)
    if metrics.n_questions <= 0:
        raise RuntimeError(f"Could not parse accuracy from {log_path}")
    return metrics, log_path


# -------------------------------
# Error extraction
# -------------------------------

@dataclass
class ErrorExample:
    subject: str
    index: int  # 0-based index in CSV
    question: str
    options: List[str]  # [A,B,C,D]
    gold: str
    pred: str


def _extract_errors_from_subject(
    csv_path: Path,
    metrics: LogMetrics,
    max_errors: int,
) -> List[ErrorExample]:
    """
    Convert wrong answers for a given subject into ErrorExample objects.
    Uses the training CSV to recover question text / gold answers.
    """
    errors: List[ErrorExample] = []
    if max_errors <= 0 or not metrics.correctness:
        return errors

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return errors

    n = min(len(rows), len(metrics.correctness))
    for idx in range(n):
        if metrics.correctness[idx]:
            continue
        row = rows[idx]
        if len(row) < 6:
            continue
        question = row[0].strip()
        # assume 4 options for MMLU format
        options = [c.strip() for c in row[1:5]]
        # last column is gold letter
        gold = row[-1].strip().upper()
        pred = (
            metrics.predictions[idx].strip().upper()
            if idx < len(metrics.predictions)
            else "?"
        )
        errors.append(
            ErrorExample(
                subject=csv_path.stem,
                index=idx,
                question=question,
                options=options,
                gold=gold,
                pred=pred,
            )
        )
        if len(errors) >= max_errors:
            break

    return errors


# -------------------------------
# Evaluation across all subjects
# -------------------------------

@dataclass
class EvalSummary:
    acc: float
    n_questions: int
    api_calls_total: int
    prompt_tokens: int
    completion_tokens: int
    logs: List[str]  # paths as strings for JSON-ability


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
    max_errors: int = 32,
) -> Tuple[EvalSummary, float, List[ErrorExample]]:
    """
    Evaluate a profile on all CSVs in train_dir.

    Returns:
      - EvalSummary
      - scalar objective score
      - a small list of ErrorExample objects (for meta-LLM)
    """
    del seed  # currently unused; left for future stochasticity

    csvs = sorted(train_dir.glob("*.csv"))
    total_correct = 0.0
    total_q = 0
    total_api_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    logs: List[str] = []
    errors: List[ErrorExample] = []

    for csv_path in csvs:
        metrics, log_path = _eval_one_csv(
            scripts_dir, csv_path, model, exp_name, roles, out_root, env
        )
        logs.append(str(log_path))
        total_q += metrics.n_questions
        total_correct += metrics.acc * metrics.n_questions
        total_api_calls += metrics.api_calls_total
        total_prompt_tokens += metrics.prompt_tokens
        total_completion_tokens += metrics.completion_tokens

        if len(errors) < max_errors:
            remaining = max_errors - len(errors)
            errors.extend(_extract_errors_from_subject(csv_path, metrics, remaining))

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

    summary = EvalSummary(
        acc=acc_overall,
        n_questions=total_q,
        api_calls_total=total_api_calls,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        logs=logs,
    )
    return summary, score, errors


# -------------------------------
# Profile IO helpers
# -------------------------------

def _load_profile(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_profile(p: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


# -------------------------------
# Meta-LLM: prompt optimizer
# -------------------------------

def _make_meta_client() -> Any:
    """
    Build a Together client for meta-LLM prompt optimization.

    Priority for API key:
      1) PROMPTAGENT_API_KEY
      2) TOGETHER_API_KEY
    """
    if Together is None:
        raise RuntimeError(
            "The `together` Python package is required for meta-LLM prompt search. "
            "Install it with `pip install together`."
        )

    api_key = os.getenv("PROMPTAGENT_API_KEY") or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found for meta-LLM. Set PROMPTAGENT_API_KEY or TOGETHER_API_KEY."
        )

    base_url = os.getenv("PROMPTAGENT_API_BASE") or os.getenv("TOGETHER_BASE_URL")

    try:
        if base_url:
            client = Together(api_key=api_key, base_url=base_url)  # type: ignore[arg-type]
        else:
            client = Together(api_key=api_key)  # type: ignore[arg-type]
    except TypeError:
        client = Together(api_key=api_key)  # type: ignore[arg-type]

    return client


def _format_errors_for_llm(errors: List[ErrorExample], max_examples: int = 12) -> str:
    """
    Turn a small list of ErrorExample into a readable block for the meta-LLM.
    """
    if not errors:
        return (
            "No explicit error examples are available; please improve the prompts "
            "in general for multi-agent MMLU question answering."
        )

    lines: List[str] = []
    for e in errors[:max_examples]:
        opts = list(e.options) + [""] * (4 - len(e.options))
        q = e.question.replace("\n", " ").strip()
        lines.append(
            f"Subject: {e.subject}\n"
            f"Question #{e.index + 1}: {q}\n"
            f"Options: A) {opts[0]}  B) {opts[1]}  C) {opts[2]}  D) {opts[3]}\n"
            f"Correct answer: {e.gold}\n"
            f"System's answer: {e.pred}"
        )
    return "\n\n".join(lines)


def _propose_profiles_with_meta_llm(
    current_profile: Dict[str, Any],
    errors: List[ErrorExample],
    num_samples: int,
    meta_model: str,
    client: Any,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Call the meta-LLM once to get a small set of new profile JSONs.
    """
    profile_json = json.dumps(current_profile, ensure_ascii=False, indent=2)
    error_block = _format_errors_for_llm(errors)

    system_msg = (
        "You are an expert prompt engineer for a multi-agent exam solver. "
        "The system uses specialized roles (Economist, Doctor, Lawyer, Mathematician, "
        "Psychologist, Programmer, Historian) plus global system prompts and interaction "
        "templates to answer MMLU-style multiple-choice questions.\n"
        "Your job is to rewrite the JSON prompt profile to improve accuracy."
    )

    user_msg = f"""
Current prompt profile JSON:

{profile_json}

Observed mistakes (each shows question, options, correct answer, and the system's wrong answer):

{error_block}

Based on these mistakes, propose {num_samples} new prompt profiles that may improve performance.

Requirements:
- Each profile MUST be a valid JSON object.
- Preserve the same top-level keys as the input profile (e.g., ROLE_MAP, ROLE_MAP_MATH,
  SYSTEM_PROMPT_MMLU, SYSTEM_PROMPT_MATH, AGENT_INTERACTION_SINGLE_CHOICE,
  AGENT_INTERACTION_MATH, RANKER_INSTRUCTION_SINGLE_CHOICE, RANKER_INSTRUCTION_MATH).
- You may rewrite, extend, or clarify the *values* but do not add or remove top-level keys.
- Focus on improving clarity, domain-specific guidance, and how agents cooperate and rank answers.
- Keep the prompts reasonably concise (no more than a few paragraphs per field).

Return ONLY a JSON array:

[
  {{ ...candidate_profile_1... }},
  {{ ...candidate_profile_2... }},
  ...
]

No explanations, no markdown, no backticks.
""".strip()

    resp = client.chat.completions.create(
        model=meta_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=8192,
    )

    # Defensive extraction in case Together returns unexpected structures
    choice0 = None
    if hasattr(resp, "choices") and resp.choices:
        choice0 = resp.choices[0]
    msg = getattr(choice0, "message", None) if choice0 is not None else None
    content = (getattr(msg, "content", None) or "").strip()

    if not content:
        # Gracefully degrade instead of crashing
        print(
            "[PromptAgent] WARNING: meta-LLM returned empty content; "
            "no neighbors will be proposed for this iteration.",
            file=sys.stderr,
        )
        return []

    # Try to parse as JSON array; be forgiving about extra wrapper text.
    # Try to parse as JSON array; be forgiving about extra wrapper text
    # or a truncated last element.
    def _parse_json_array(s: str) -> Any:
        """
        Best‑effort JSON parser for meta‑LLM output.

        - First try to parse the whole string.
        - Then try cropping from the first '[' to the last ']'.
        - If that still fails (e.g., last element is truncated),
          greedily extract as many complete JSON objects `{...}`
          as possible and return them as a list.
        """
        # 1) Direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) Try cropping to outer array
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            subset = s[start: end + 1]
            try:
                return json.loads(subset)
            except Exception:
                # fall through to object‑by‑object parsing
                pass

        # 3) Last resort: extract individual JSON objects `{ ... }`
        objs: List[Any] = []
        buf = ""
        depth = 0
        in_string = False
        escape = False

        for ch in s:
            # Wait until we see the start of an object
            if not buf:
                if ch.isspace():
                    continue
                if ch != "{":
                    continue
                buf = ch
                depth = 1
                in_string = False
                escape = False
                continue

            buf += ch

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        # End of one candidate object
                        try:
                            obj = json.loads(buf.strip().rstrip(","))
                            objs.append(obj)
                        except Exception:
                            # If this object is malformed, just drop it
                            pass
                        buf = ""

        if objs:
            # Treat the extracted objects as the array contents
            return objs

        # Give up if nothing was recoverable
        raise RuntimeError("Could not parse any JSON objects from meta‑LLM output")

    try:
        data = _parse_json_array(content)
    except Exception as e:
        raise RuntimeError(
            f"Meta-LLM did not return valid JSON: {e}\nRaw content:\n{content}"
        )

    if isinstance(data, dict):
        candidates_raw = [data]
    elif isinstance(data, list):
        candidates_raw = data
    else:
        raise RuntimeError(f"Unexpected meta-LLM JSON type: {type(data)}")

    candidates: List[Dict[str, Any]] = []
    for item in candidates_raw:
        if isinstance(item, dict):
            candidates.append(item)
        if len(candidates) >= num_samples:
            break

    return candidates


# -------------------------------
# Hill-climbing optimization loop
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
    """
    Simple hill-climbing over JSON prompt profiles using a meta-LLM.

    At each iteration:
      - Use errors from the *current best* profile to ask the meta-LLM for K neighbors.
      - Evaluate all neighbors.
      - Move to the neighbor with the best score if it beats current best.
      - Stop early if no neighbor improves the score.
    """
    rng = __import__("random").Random(seed)

    base_profile = _load_profile(base_profile_path)

    # Meta-LLM setup
    meta_model = os.getenv("PROMPTAGENT_META_MODEL", model)
    num_neighbors = int(os.getenv("PROMPTAGENT_NUM_NEIGHBORS", "3"))
    meta_disabled = os.getenv("PA_DISABLE_META", "").strip().lower() in {"1", "true", "yes"}

    if meta_disabled:
        meta_client = None
    else:
        meta_client = _make_meta_client()

    env = dict(env)  # copy so we can mutate PROMPT_PROFILE safely

    # 1) Evaluate base profile
    env["PROMPT_PROFILE"] = str(base_profile_path)
    base_out_dir = out_runs_dir / f"{exp_name}_{'_'.join(roles)}_iter000"
    base_summary, base_score, base_errors = evaluate_profile(
        train_dir=train_dir,
        scripts_dir=scripts_dir,
        model=model,
        roles=roles,
        out_root=base_out_dir,
        exp_name=exp_name,
        seed=seed,
        alpha_api=alpha_api,
        beta_tok=beta_tok,
        env=env,
        max_errors=32,
    )

    best = {
        "profile": base_profile,
        "score": base_score,
        "summary": base_summary,
        "profile_path": str(base_profile_path),
        "errors": base_errors,
    }

    # 2) Iterative hill-climbing with meta-LLM neighbors
    tmp_dir = out_profile_path.parent / ".promptagent_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for t in range(1, iters + 1):
        if meta_client is None:
            # No meta-LLM available; nothing more we can do.
            break

        # Sample neighbors from meta-LLM based on current best profile + errors
        neighbors = _propose_profiles_with_meta_llm(
            current_profile=best["profile"],
            errors=best.get("errors", []),
            num_samples=num_neighbors,
            meta_model=meta_model,
            client=meta_client,
            temperature=0.7,
        )

        if not neighbors:
            # Either meta-LLM returned empty / invalid, or no errors;
            # treat as local optimum and stop gracefully.
            print(
            f"[PromptAgent] No neighbor profiles proposed at iteration {t}; "
            "ending hill-climb.",
            file=sys.stderr,
        )
            break

        improved = False

        for k, cand_profile in enumerate(neighbors):
            cand_path = tmp_dir / f"candidate_t{t:02d}_{k:02d}.json"
            _save_profile(cand_profile, cand_path)

            env["PROMPT_PROFILE"] = str(cand_path)
            cand_out_dir = out_runs_dir / f"{exp_name}_{'_'.join(roles)}_iter{t:02d}_cand{k:02d}"

            summary, score, errors = evaluate_profile(
                train_dir=train_dir,
                scripts_dir=scripts_dir,
                model=model,
                roles=roles,
                out_root=cand_out_dir,
                exp_name=exp_name,
                seed=seed,
                alpha_api=alpha_api,
                beta_tok=beta_tok,
                env=env,
                max_errors=32,
            )

            if score > best["score"]:
                best = {
                    "profile": cand_profile,
                    "score": score,
                    "summary": summary,
                    "profile_path": str(cand_path),
                    "errors": errors,
                }
                improved = True

        if not improved:
            # Local optimum reached
            print(
                f"[PromptAgent] No improvement at iteration {t}; "
                f"best score stays {best['score']:.4f}.",
                file=sys.stderr,
            )
            break

    # 3) Write the winner
    _save_profile(best["profile"], out_profile_path)
    best_summary = best["summary"]
    return {
        "profile_path": str(out_profile_path),
        "best_score": float(best["score"]),
        "eval_summary": {
            "acc": float(best_summary.acc),
            "n_questions": int(best_summary.n_questions),
            "api_calls_total": int(best_summary.api_calls_total),
            "prompt_tokens": int(best_summary.prompt_tokens),
            "completion_tokens": int(best_summary.completion_tokens),
            "logs": list(best_summary.logs),
        },
    }


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
    ap.add_argument("--iters", type=int, default=3, help="Number of hill-climb steps.")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument(
        "--roles",
        type=str,
        default="Economist,Doctor,Lawyer,Mathematician,Psychologist,Programmer,Historian",
    )
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

    out_runs_dir = scripts_dir / "promptagent_runs"

    # Environment: inherit .env at repo root if present
    env = os.environ.copy()
    dotenv = repo_root / ".env"
    if dotenv.exists():
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

    print("\n=== PromptAgent result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

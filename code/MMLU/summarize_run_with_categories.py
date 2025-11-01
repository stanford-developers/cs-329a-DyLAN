#!/usr/bin/env python3
import argparse
import ast
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# ---------------------------
# Mappings provided by you
# ---------------------------
SUBCATEGORY = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

# ---------------------------
# Helpers
# ---------------------------
def find_baseline_subject_dir(run_dir: Path) -> Path:
    # The 7-role run folder: starts with mmlu_downsampled_...
    cands = sorted([p for p in run_dir.glob("mmlu_downsampled_*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"Could not find a baseline subject folder under: {run_dir}")
    return cands[0]

def find_eval_dir(run_dir: Path) -> Path:
    # Your evaluation dir is evaluation_results_baseline (or fallback evaluation_results)
    for name in ["evaluation_results_baseline", "evaluation_results"]:
        p = run_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find evaluation results dir under: {run_dir}")

def parse_txt_metrics(txt_path: Path):
    """
    Format produced by llmlp_listwise_mmlu.py:
      Line 0: [True, False, ...] <avg_acc>
      Line 1: <total_resp> <avg_resp_per_q>
      Line 2: [[importance matrix]]
      Line 3: [average importances]
      Line 4: <total_prompt_tokens>
      Line 5: <total_completion_tokens>
    """
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Accuracy
    acc_avg = float(lines[0].rsplit(" ", 1)[1])
    # Responses
    parts = lines[1].split()
    total_resp = int(parts[0])
    avg_resp = float(parts[1]) if len(parts) > 1 else None
    # Tokens
    prompt_tokens = int(lines[4])
    completion_tokens = int(lines[5])

    # #questions = len(list) in line 0
    acc_list_str = lines[0].rsplit(" ", 1)[0]
    acc_list = ast.literal_eval(acc_list_str)
    q_cnt = len(acc_list)

    return {
        "accuracy": acc_avg,
        "avg_responses": avg_resp,
        "total_responses": total_resp,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "q_cnt": q_cnt,
    }

def early_stop_rate_from_jsonl(jsonl_path: Path):
    """
    The JSON file contains one line per question:
      completions = [ [round1/2/3 for agent1], [ ... for agent2], ...]
    Early stop if **every agent** has None in the *last* round, meaning the model stopped before final round.
    """
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return None

    total = 0
    early = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completions = json.loads(line)
            except Exception:
                continue
            total += 1
            # completions: list of length = #agents; each is a list of len = #rounds
            if not completions or not isinstance(completions, list):
                continue
            # Determine last round index from first agent that has a list
            last_round_idx = None
            for row in completions:
                if isinstance(row, list) and row:
                    last_round_idx = len(row) - 1
                    break
            if last_round_idx is None:
                continue
            # Early stop if last round is None for all agents
            all_none_last = True
            for row in completions:
                if isinstance(row, list) and len(row) > last_round_idx and row[last_round_idx] is not None:
                    all_none_last = False
                    break
            if all_none_last:
                early += 1
    return (early / total) if total > 0 else None

def load_eval_overall_json(eval_dir: Path):
    # Written by exp_mmlu_evaluation.sh
    cands = list(eval_dir.glob("evaluation_results_*roles.json"))
    return json.load(open(cands[0])) if cands else None

def load_selected_roles(eval_dir: Path):
    # Written by exp_mmlu_evaluation.sh
    p = eval_dir / "selected_roles_4roles.json"
    if not p.exists():
        # Fall back to any selected_roles file
        cands = list(eval_dir.glob("selected_roles_*roles.json"))
        if not cands:
            return {}
        p = cands[0]
    return json.load(open(p))

def canonical_subject_name(stem: str) -> str:
    """Turn 'abstract_algebra_test_73' -> 'abstract_algebra'"""
    base = stem
    for suffix in ["_test_73", "_test", "_val_73", "_val", "_eval", "_73"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base

def roles_str(lst):
    if not lst:
        return ""
    return "|".join(lst)

def compute_cost(prompt_toks, completion_toks, price_in_per_m, price_out_per_m):
    if price_in_per_m is None or price_out_per_m is None:
        return None
    return (prompt_toks / 1_000_000.0) * price_in_per_m + (completion_toks / 1_000_000.0) * price_out_per_m

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Summarize pre/post selection metrics, tokens, early stopping, and roles.")
    ap.add_argument("--run-dir", required=True, help="e.g., runs/baseline_20251101-1402")
    ap.add_argument("--outfile", default="run_summary.csv", help="Output CSV path")
    ap.add_argument("--price-in-per-m", type=float, default=None, help="Optional $/1M prompt tokens for cost")
    ap.add_argument("--price-out-per-m", type=float, default=None, help="Optional $/1M completion tokens for cost")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    baseline_dir = find_baseline_subject_dir(run_dir)
    eval_dir = find_eval_dir(run_dir)
    # map: test_name -> roles (post-selection)
    selected_roles_map = load_selected_roles(eval_dir)
    overall_eval = load_eval_overall_json(eval_dir)  # has 'by_test'

    # POST lookup from evaluation_results json (tokens/accuracy/responses)
    post_by_test = {}
    if overall_eval and "by_test" in overall_eval:
        post_by_test = overall_eval["by_test"]  # keys like 'abstract_algebra_test_73'

    # Collect per-subject rows
    rows = []

    # Index evaluation subfolders to locate *_eval.json for early-stop computation
    eval_json_index = {}  # subject_stem -> Path
    for sub in eval_dir.glob("eval_*"):
        if sub.is_dir():
            # Find *_eval.json under this subdir
            cands = list(sub.glob("*_eval.json"))
            if cands:
                stem = cands[0].stem.replace("_eval", "")
                eval_json_index[stem] = cands[0]

    # Walk baseline .txt (pre-selection)
    for txt in sorted(baseline_dir.glob("*_73.txt")):
        stem = txt.stem                        # e.g. "abstract_algebra_test_73"
        subject = canonical_subject_name(stem) # "abstract_algebra"
        jsonl_pre = txt.with_suffix(".json")   # same folder, same stem

        # Pre metrics
        pre = parse_txt_metrics(txt)
        pre_early_rate = early_stop_rate_from_jsonl(jsonl_pre) or 0.0

        # Try to find post metrics from overall json
        post_key = stem.replace("_73", "") if stem not in post_by_test else stem
        if post_key not in post_by_test:
            # try variants
            for var in [stem, stem.replace("_test", "_val"), stem.replace("_val", "_test")]:
                if var in post_by_test:
                    post_key = var
                    break

        post = None
        if post_key in post_by_test:
            jd = post_by_test[post_key]
            post = {
                "accuracy": float(jd.get("accuracy", 0.0)),
                "avg_responses": float(jd.get("avg_responses", jd.get("avg_responses_per_question", 0.0))),
                "total_responses": int(jd.get("responses", 0)),
                "prompt_tokens": int(jd.get("prompt_tokens", 0)),
                "completion_tokens": int(jd.get("completion_tokens", 0)),
                "q_cnt": int(jd.get("questions", 0)),
            }
        else:
            # Fallback: parse *_eval.txt in the eval subfolder
            # Locate eval result .txt
            eval_txt = None
            for sub in eval_dir.glob(f"eval_{subject}_*"):
                cand = sub / f"{subject}_eval.txt"
                if cand.exists():
                    eval_txt = cand
                    break
            if eval_txt and eval_txt.exists():
                post = parse_txt_metrics(eval_txt)
            else:
                # if missing, skip post stats (we still keep pre row)
                post = {
                    "accuracy": None, "avg_responses": None, "total_responses": None,
                    "prompt_tokens": None, "completion_tokens": None, "q_cnt": None
                }

        # Post early stopping
        post_early_rate = 0.0
        # Find *_eval.json (already re-homed by the evaluation script)
        post_json = None
        # try exact
        for sub in eval_dir.glob(f"eval_{subject}_*"):
            cand = sub / f"{subject}_eval.json"
            if cand.exists():
                post_json = cand
                break
        if post_json and post_json.exists():
            es = early_stop_rate_from_jsonl(post_json)
            post_early_rate = es if es is not None else 0.0

        # Roles post-selection
        # selected_roles_map uses filename keys (sometimes *_test, sometimes *_val, sometimes without suffix)
        key_variants = [
            stem, stem.replace("_73", ""), stem.replace("_test_73", "_test"), stem.replace("_test_73", ""),
            stem.replace("_test_73", "_val"), stem.replace("_val_73", "_test"),
            subject, f"{subject}_test", f"{subject}_val"
        ]
        picked_roles = None
        for k in key_variants:
            if k in selected_roles_map:
                picked_roles = selected_roles_map[k]
                break
        if picked_roles is None:
            picked_roles = []  # not found

        # Subcat & category
        subcats = SUBCATEGORY.get(subject, ["other"])
        subcat = subcats[0]
        cat = None
        for c_name, members in CATEGORIES.items():
            if subcat in members:
                cat = c_name
                break
        if cat is None:
            cat = "other (business, health, misc.)"

        # Costs (optional)
        pre_cost = compute_cost(pre["prompt_tokens"], pre["completion_tokens"],
                                args.price_in_per_m, args.price_out_per_m)
        post_cost = None
        if post["prompt_tokens"] is not None and post["completion_tokens"] is not None:
            post_cost = compute_cost(post["prompt_tokens"], post["completion_tokens"],
                                     args.price_in_per_m, args.price_out_per_m)

        row = {
            "level": "subject",
            "name": subject,
            "subcategory": subcat,
            "category": cat,

            "q_pre": pre["q_cnt"],
            "acc_pre": pre["accuracy"],
            "avg_resp_pre": pre["avg_responses"],
            "prompt_tok_pre": pre["prompt_tokens"],
            "completion_tok_pre": pre["completion_tokens"],
            "total_tok_pre": pre["prompt_tokens"] + pre["completion_tokens"],
            "early_stop_rate_pre": pre_early_rate,
            "cost_pre": pre_cost,

            "q_post": post["q_cnt"],
            "acc_post": post["accuracy"],
            "avg_resp_post": post["avg_responses"],
            "prompt_tok_post": post["prompt_tokens"],
            "completion_tok_post": post["completion_tokens"],
            "total_tok_post": (None if (post["prompt_tokens"] is None or post["completion_tokens"] is None)
                               else post["prompt_tokens"] + post["completion_tokens"]),
            "early_stop_rate_post": post_early_rate,
            "cost_post": post_cost,

            "acc_delta": (None if (post["accuracy"] is None or pre["accuracy"] is None)
                          else post["accuracy"] - pre["accuracy"]),
            "avg_resp_delta": (None if (post["avg_responses"] is None or pre["avg_responses"] is None)
                               else post["avg_responses"] - pre["avg_responses"]),
            "total_tok_delta": (None if (post["prompt_tokens"] is None or post["completion_tokens"] is None)
                                else (post["prompt_tokens"] + post["completion_tokens"])
                                     - (pre["prompt_tokens"] + pre["completion_tokens"])),
            "early_stop_delta": (None if (post_early_rate is None or pre_early_rate is None)
                                 else post_early_rate - pre_early_rate),
            "cost_delta": (None if (pre_cost is None or post_cost is None)
                           else post_cost - pre_cost),

            "roles_post": roles_str(picked_roles),
            "n_roles_post": len(picked_roles),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------------------
    # Aggregations by subcategory & category
    # Weighted means by question count where appropriate
    # ---------------------------
    def weighted_mean(series, weights):
        try:
            return (series * weights).sum() / weights.sum() if weights.sum() > 0 else None
        except Exception:
            return None

    agg_rows = []

    # Build role counts per group (post-selection)
    def role_counts_for(df_group):
        c = Counter()
        for s in df_group["roles_post"].dropna().astype(str).tolist():
            if not s:
                continue
            roles = [r for r in s.split("|") if r]
            c.update(roles)
        return c

    # Subcategory
    for subcat, g in df.groupby("subcategory"):
        w = g["q_pre"].fillna(0)
        w_post = g["q_post"].fillna(0)

        role_counts = role_counts_for(g)
        top_roles = ", ".join([f"{r}({n})" for r, n in role_counts.most_common(3)]) if role_counts else ""

        agg_rows.append({
            "level": "subcategory",
            "name": subcat,
            "subcategory": subcat,
            "category": None,

            "q_pre": g["q_pre"].sum(),
            "acc_pre": weighted_mean(g["acc_pre"], w),
            "avg_resp_pre": weighted_mean(g["avg_resp_pre"], w),
            "prompt_tok_pre": g["prompt_tok_pre"].sum(),
            "completion_tok_pre": g["completion_tok_pre"].sum(),
            "total_tok_pre": g["total_tok_pre"].sum(),
            "early_stop_rate_pre": weighted_mean(g["early_stop_rate_pre"], w),

            "q_post": g["q_post"].sum(),
            "acc_post": weighted_mean(g["acc_post"], w_post),
            "avg_resp_post": weighted_mean(g["avg_resp_post"], w_post),
            "prompt_tok_post": g["prompt_tok_post"].sum(skipna=True),
            "completion_tok_post": g["completion_tok_post"].sum(skipna=True),
            "total_tok_post": g["total_tok_post"].sum(skipna=True),
            "early_stop_rate_post": weighted_mean(g["early_stop_rate_post"], w_post),

            "acc_delta": weighted_mean(g["acc_post"] - g["acc_pre"], w_post.where(w_post > 0, other=0)),
            "avg_resp_delta": weighted_mean(g["avg_resp_post"] - g["avg_resp_pre"], w_post.where(w_post > 0, other=0)),
            "total_tok_delta": (g["total_tok_post"].sum(skipna=True) - g["total_tok_pre"].sum()),
            "early_stop_delta": weighted_mean(g["early_stop_rate_post"] - g["early_stop_rate_pre"],
                                              w_post.where(w_post > 0, other=0)),
            "roles_post": top_roles,
            "n_roles_post": None,
        })

    # Category
    for cat, g in df.groupby("category"):
        w = g["q_pre"].fillna(0)
        w_post = g["q_post"].fillna(0)

        role_counts = role_counts_for(g)
        top_roles = ", ".join([f"{r}({n})" for r, n in role_counts.most_common(3)]) if role_counts else ""

        agg_rows.append({
            "level": "category",
            "name": cat,
            "subcategory": None,
            "category": cat,

            "q_pre": g["q_pre"].sum(),
            "acc_pre": weighted_mean(g["acc_pre"], w),
            "avg_resp_pre": weighted_mean(g["avg_resp_pre"], w),
            "prompt_tok_pre": g["prompt_tok_pre"].sum(),
            "completion_tok_pre": g["completion_tok_pre"].sum(),
            "total_tok_pre": g["total_tok_pre"].sum(),
            "early_stop_rate_pre": weighted_mean(g["early_stop_rate_pre"], w),

            "q_post": g["q_post"].sum(),
            "acc_post": weighted_mean(g["acc_post"], w_post),
            "avg_resp_post": weighted_mean(g["avg_resp_post"], w_post),
            "prompt_tok_post": g["prompt_tok_post"].sum(skipna=True),
            "completion_tok_post": g["completion_tok_post"].sum(skipna=True),
            "total_tok_post": g["total_tok_post"].sum(skipna=True),
            "early_stop_rate_post": weighted_mean(g["early_stop_rate_post"], w_post),

            "acc_delta": weighted_mean(g["acc_post"] - g["acc_pre"], w_post.where(w_post > 0, other=0)),
            "avg_resp_delta": weighted_mean(g["avg_resp_post"] - g["avg_resp_pre"], w_post.where(w_post > 0, other=0)),
            "total_tok_delta": (g["total_tok_post"].sum(skipna=True) - g["total_tok_pre"].sum()),
            "early_stop_delta": weighted_mean(g["early_stop_rate_post"] - g["early_stop_rate_pre"],
                                              w_post.where(w_post > 0, other=0)),
            "roles_post": top_roles,
            "n_roles_post": None,
        })

    # Overall
    w = df["q_pre"].fillna(0)
    w_post = df["q_post"].fillna(0)
    role_counts_all = Counter()
    for s in df["roles_post"].dropna().astype(str).tolist():
        if s:
            role_counts_all.update([r for r in s.split("|") if r])
    top_all = ", ".join([f"{r}({n})" for r, n in role_counts_all.most_common(3)]) if role_counts_all else ""

    agg_rows.append({
        "level": "overall",
        "name": "overall",
        "subcategory": None,
        "category": None,

        "q_pre": df["q_pre"].sum(),
        "acc_pre": weighted_mean(df["acc_pre"], w),
        "avg_resp_pre": weighted_mean(df["avg_resp_pre"], w),
        "prompt_tok_pre": df["prompt_tok_pre"].sum(),
        "completion_tok_pre": df["completion_tok_pre"].sum(),
        "total_tok_pre": df["total_tok_pre"].sum(),
        "early_stop_rate_pre": weighted_mean(df["early_stop_rate_pre"], w),

        "q_post": df["q_post"].sum(),
        "acc_post": weighted_mean(df["acc_post"], w_post),
        "avg_resp_post": weighted_mean(df["avg_resp_post"], w_post),
        "prompt_tok_post": df["prompt_tok_post"].sum(skipna=True),
        "completion_tok_post": df["completion_tok_post"].sum(skipna=True),
        "total_tok_post": df["total_tok_post"].sum(skipna=True),
        "early_stop_rate_post": weighted_mean(df["early_stop_rate_post"], w_post),

        "acc_delta": weighted_mean(df["acc_post"] - df["acc_pre"], w_post.where(w_post > 0, other=0)),
        "avg_resp_delta": weighted_mean(df["avg_resp_post"] - df["avg_resp_pre"], w_post.where(w_post > 0, other=0)),
        "total_tok_delta": (df["total_tok_post"].sum(skipna=True) - df["total_tok_pre"].sum()),
        "early_stop_delta": weighted_mean(df["early_stop_rate_post"] - df["early_stop_rate_pre"],
                                          w_post.where(w_post > 0, other=0)),
        "roles_post": top_all,
        "n_roles_post": None,
    })

    out_df = pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True)

    out_path = Path(args.outfile).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved summary to: {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Single‑LLM MMLU Evaluation + CI Script (robust CSV loader incl. headerless)
# - One model call per question
# - Reports accuracy, API calls, tokens-in/out with 95% bootstrap CIs
# - Breaks out metrics overall and by meta-category
# - Optional comparison to pre-selection (7 roles) via --importance-csv

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DATASET="${EVAL_DATASET:-$SCRIPT_DIR/../../data/MMLU/evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-single_llm_results}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
N_BOOT="${N_BOOT:-1000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
IMPORTANCE_CSV="${IMPORTANCE_CSV:-}"   # optional

usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Run a single-LLM MMLU baseline (one model call per question) and report:
accuracy, API calls, tokens-in, tokens-out with 95% bootstrap CIs.

OPTIONS:
  -m, --model MODEL              Chat model id (default: ${MODEL})
  -d, --dataset DIR              Path to evaluation dataset dir (default: ${EVAL_DATASET})
  -o, --output DIR               Output directory (default: ${OUTPUT_DIR})
  -p, --max-parallel NUM         Max parallel files (default: ${MAX_PARALLEL})
  --n-boot NUM                   Bootstrap replicates for CI (default: ${N_BOOT})
  -t, --temperature FLOAT        Temperature (default: ${TEMPERATURE})
  --top-p FLOAT                  top_p (default: ${TOP_P})
  -i, --importance-csv FILE      (optional) importance_1to7.csv for 7-role comparison
  -h, --help                     Show help

Examples:
  $0
  $0 --model "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
  N_BOOT=2000 $0 --dataset "../../data/MMLU/evaluation" --max-parallel 8
  $0 --importance-csv runs/my_run/importance_1to7_baseline.csv
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

# -------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2;;
    -d|--dataset) EVAL_DATASET="$2"; shift 2;;
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
    --n-boot) N_BOOT="$2"; shift 2;;
    -t|--temperature) TEMPERATURE="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    -i|--importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -d "$EVAL_DATASET" ]]; then
  log "ERROR: Evaluation dataset not found: $EVAL_DATASET"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

log "Single‑LLM Evaluation"
log "Model: $MODEL"
log "Dataset: $EVAL_DATASET"
log "Output: $OUTPUT_DIR"
log "Max parallel: $MAX_PARALLEL"
log "Bootstrap reps: $N_BOOT"
log "Temp: $TEMPERATURE  |  top-p: $TOP_P"
[[ -n "$IMPORTANCE_CSV" ]] && log "Pre (7‑role) comparison CSV: $IMPORTANCE_CSV"

# ------------------------------------------------------------
# Write the Python evaluator
# ------------------------------------------------------------
cat > "$OUTPUT_DIR/single_llm_eval.py" << 'PY'
#!/usr/bin/env python3
import os, sys, re, json, math, time, traceback
from pathlib import Path
from typing import Dict, List, Tuple
import concurrent.futures

import numpy as np
import pandas as pd

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI client (Together-compatible)
import openai
openai.api_base = os.getenv("TOGETHER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.together.xyz/v1"))
openai.api_key  = os.getenv("TOGETHER_API_KEY", os.getenv("OPENAI_API_KEY"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P       = float(os.getenv("TOP_P", "1.0"))

# ---------------------------
# Subject → subcategory map
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

def subject_key_from_name(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r'_eval$', '', base)
    base = re.sub(r'_(test|val)(_\d+)?$', '', base)
    return base

def meta_for_subject(subject: str) -> str:
    sub = SUBCATEGORY.get(subject, ["other"])[0]
    for meta, subs in CATEGORIES.items():
        if sub in subs:
            return meta
    return "other (business, health, misc.)"

# ---------------------------
# CSV loader (robust + headerless)
# ---------------------------
def _pick_col(cols_lower_map: Dict[str,str], *exact, contains=None):
    for k in exact:
        if k in cols_lower_map:
            return cols_lower_map[k]
    if contains:
        for raw_lower, original in cols_lower_map.items():
            for frag in contains:
                if frag in raw_lower:
                    return original
    return None

def _choice_col(letter: str, cols_lower_map: Dict[str,str]):
    idx = {"A":"1","B":"2","C":"3","D":"4"}[letter]
    candidates = [
        letter.lower(),
        f"choice_{letter.lower()}",
        f"option_{letter.lower()}",
        f"answer_{letter.lower()}",
        f"ans_{letter.lower()}",
        f"opt_{letter.lower()}",
        f"option{idx}",
        f"choice{idx}",
        f"opt{idx}",
        f"ans{idx}",
        f"answer{idx}",
    ]
    for c in candidates:
        if c in cols_lower_map:
            return cols_lower_map[c]
    for raw_lower, original in cols_lower_map.items():
        if raw_lower.strip(".:() ") == letter.lower():
            return original
    return None

def _detect_indexing_numeric(vals: pd.Series):
    try:
        s = pd.to_numeric(vals, errors="coerce").dropna()
        if s.empty: return None
        mn, mx = int(s.min()), int(s.max())
        if mn == 0 and mx <= 3: return "zero"
        if mn >= 1 and mx <= 4: return "one"
    except Exception:
        pass
    return None

def _norm(x: str) -> str:
    x = str(x)
    x = re.sub(r'^[A-Da-d][\).\]:\-]\s*', '', x.strip())
    return ' '.join(x.split()).lower()

def _read_csv_any(path: str) -> pd.DataFrame:
    """
    Try reading with headers; if canonical signals are missing (question + answer + ≥3 choices),
    re‑read as headerless with canonical names [question, A, B, C, D, answer].
    """
    # 1) Try normal read with header
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python")
    except Exception:
        # If even that fails, force headerless
        df = pd.read_csv(
            path, header=None, names=["question","A","B","C","D","answer"],
            dtype=str, keep_default_na=False, engine="python"
        )
        return df

    cols_lower = [str(c).lower().strip() for c in df.columns]
    lower_set = set(cols_lower)

    have_question = any(k in lower_set for k in ["question","prompt","stem","text","query","question_text"])
    have_answer   = any(k in lower_set for k in [
        "answer","target","label","correct","answer_key","answer_index","answer_idx","gold","gold_label","right","gt"
    ])

    # Count how many choice columns we can recognize in the headerful read
    choice_aliases = [
        "a","b","c","d",
        "option_a","option_b","option_c","option_d",
        "choice_a","choice_b","choice_c","choice_d",
        "answer_a","answer_b","answer_c","answer_d",
        "opt_a","opt_b","opt_c","opt_d"
    ]
    num_choice_hits = sum(1 for alias in choice_aliases if alias in lower_set)

    # Heuristic: in the headerless case, the last header often becomes a single letter A/B/C/D
    looks_like_promoted_first_row = (
        df.shape[1] >= 6
        and not have_question
        and not have_answer
        and any(str(c).strip().upper() in {"A","B","C","D"} for c in df.columns)
    )

    # Require both question and answer and at least 3 choices; else treat as headerless
    if looks_like_promoted_first_row or not (have_question and have_answer and num_choice_hits >= 3):
        df = pd.read_csv(
            path,
            header=None,
            names=["question","A","B","C","D","answer"],
            dtype=str,
            keep_default_na=False,
            engine="python"
        )

    return df

def load_mmlu_csv(path: str) -> List[dict]:
    df = _read_csv_any(path)
    cols_lower_map = {c.lower().strip(): c for c in df.columns}

    # question column
    qcol = _pick_col(
        cols_lower_map, "question",
        contains=["question","prompt","stem","query","problem","question_text"]
    )
    if not qcol:
        raise ValueError(f"{path}: missing column 'question'")

    # option columns
    acol = _choice_col("A", cols_lower_map)
    bcol = _choice_col("B", cols_lower_map)
    ccol = _choice_col("C", cols_lower_map)
    dcol = _choice_col("D", cols_lower_map)
    if not all([acol,bcol,ccol,dcol]):
        raise ValueError(f"{path}: missing choice columns A/B/C/D")

    # answer column
    anscol = _pick_col(
        cols_lower_map,
        "answer","target","label","correct","correct_answer","answer_key",
        "answer_index","answer_idx","correct_option","gold","gold_label","right","gt",
        contains=["answer","label","correct"]
    )
    if not anscol:
        raise ValueError(f"{path}: missing column 'answer'")

    indexing = _detect_indexing_numeric(df[anscol])

    rows = []
    for _, r in df.iterrows():
        q = str(r[qcol])
        A = str(r[acol]); B = str(r[bcol]); C = str(r[ccol]); D = str(r[dcol])
        raw_ans = str(r[anscol]).strip()

        ans = ""
        if raw_ans:
            m = re.match(r'^\s*([ABCD])\s*$', raw_ans, flags=re.I)
            if m:
                ans = m.group(1).upper()
            else:
                # numeric?
                try:
                    n = int(float(raw_ans))
                    if indexing == "zero" and 0 <= n <= 3:   ans = "ABCD"[n]
                    elif indexing == "one" and 1 <= n <= 4: ans = "ABCD"[n-1]
                except Exception:
                    pass
                # textual match?
                if not ans:
                    norm_ans = _norm(raw_ans)
                    opts = {"A": _norm(A), "B": _norm(B), "C": _norm(C), "D": _norm(D)}
                    for letter, opt_text in opts.items():
                        if norm_ans == opt_text:
                            ans = letter; break

        rows.append({"question": q, "A": A, "B": B, "C": C, "D": D, "answer": ans})
    return rows

# ---------------------------
# LLM call (return a single letter)
# ---------------------------
LETTER_RE = re.compile(r'\b([A-D])\b', re.IGNORECASE)

def extract_choice(text: str) -> str:
    if not text:
        return ""
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else ""

def ask_single(model: str, q: dict, max_tokens: int = 2) -> Tuple[str, int, int]:
    system = "You are a careful multiple-choice solver. Reply with a single letter only."
    user = f"""Choose exactly one option (A, B, C, or D). Return ONLY the letter.

Question:
{q['question']}

Options:
A. {q['A']}
B. {q['B']}
C. {q['C']}
D. {q['D']}"""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
    )
    content = resp["choices"][0]["message"]["content"]
    choice = extract_choice(content)
    usage = resp.get("usage", {})
    ptok = int(usage.get("prompt_tokens", 0))
    ctok = int(usage.get("completion_tokens", 0))
    return choice, ptok, ctok

def evaluate_file(csv_path: str, model: str, retries: int = 3, backoff: float = 1.0) -> dict:
    data = load_mmlu_csv(csv_path)
    calls = ptok = ctok = correct = 0
    for row in data:
        last_err = None
        for attempt in range(retries):
            try:
                choice, p, c = ask_single(model, row)
                calls += 1; ptok += p; ctok += c
                if row["answer"] and choice == row["answer"]:
                    correct += 1
                break
            except Exception as e:
                last_err = e
                time.sleep(backoff * (2 ** attempt))
        else:
            sys.stderr.write(f"[WARN] Skipped one question in {csv_path} after retries: {last_err}\n")
    return {"questions": len(data), "correct": correct, "responses": calls,
            "prompt_tokens": ptok, "completion_tokens": ctok}

# ---------------------------
# Pre-selection (7-role) reader (optional)
# ---------------------------
def subject_key_from_filename(fname: str) -> str:
    return subject_key_from_name(Path(fname).stem)

def collect_pre_metrics(importance_csv: str) -> pd.DataFrame:
    df = pd.read_csv(importance_csv)
    rows = []
    has_tin  = 'prompt_tokens' in df.columns
    has_tout = 'completion_tokens' in df.columns
    for _, r in df.iterrows():
        fname = r['filename']
        test_name = Path(fname).stem
        subject = subject_key_from_filename(test_name)
        meta = meta_for_subject(subject)
        q_cnt = int(r['q_cnt'])
        correct_float = float(r['acc']) * q_cnt
        responses = int(r['resp'])
        rows.append({
            'test_name': test_name,
            'subject': subject,
            'meta': meta,
            'questions': q_cnt,
            'correct': correct_float,
            'responses': responses,
            'prompt_tokens': int(r['prompt_tokens']) if has_tin else np.nan,
            'completion_tokens': int(r['completion_tokens']) if has_tout else np.nan
        })
    return pd.DataFrame(rows)

# ---------------------------
# Bootstrap & summaries
# ---------------------------
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, scale_tokens_by: float = None):
    if df.empty: return {}
    rng = np.random.default_rng(seed)
    n = len(df)

    def agg(d: pd.DataFrame):
        q = d['questions'].sum()
        acc = float(d['correct'].sum()) / q if q > 0 else float('nan')
        api = d['responses'].sum()
        tin = d['prompt_tokens'].sum(skipna=True)
        tout = d['completion_tokens'].sum(skipna=True)
        if scale_tokens_by is not None:
            tin *= scale_tokens_by; tout *= scale_tokens_by
        return acc, api, tin, tout

    point = agg(df)
    acc_s, api_s, tin_s, tout_s = [], [], [], []
    have_tin  = df['prompt_tokens'].notna().any()
    have_tout = df['completion_tokens'].notna().any()

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = df.iloc[idx]
        a,b,c,d = agg(s)
        acc_s.append(a); api_s.append(b)
        tin_s.append(c if have_tin else np.nan)
        tout_s.append(d if have_tout else np.nan)

    def pct_ci(arr):
        arr = np.array(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        'accuracy': {'point': float(point[0]), 'ci95': pct_ci(acc_s)},
        'api_calls': {'point': int(point[1]),
                      'ci95': [int(np.nanpercentile(api_s, 2.5)), int(np.nanpercentile(api_s,97.5))]},
    }
    out['tokens_in']  = {'point': float(point[2]), 'ci95': pct_ci(tin_s)}  if have_tin  else {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}
    out['tokens_out'] = {'point': float(point[3]), 'ci95': pct_ci(tout_s)} if have_tout else {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}
    return out

def print_block(title: str, res: dict):
    print(f"\n{title}")
    print("-" * len(title))
    if not res:
        print("No data.")
        return
    a = res['accuracy']; k = res['api_calls']
    print(f"Accuracy             : {a['point']:.4f}  [95% CI {a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]")
    print(f"API calls            : {k['point']}  [95% CI {k['ci95'][0]}, {k['ci95'][1]}]")
    ti = res['tokens_in']; to = res['tokens_out']
    if not math.isnan(ti['point']):
        print(f"Tokens in            : {ti['point']:.0f}  [95% CI {ti['ci95'][0]:.0f}, {ti['ci95'][1]:.0f}]")
    else:
        print("Tokens in            : N/A")
    if not math.isnan(to['point']):
        print(f"Tokens out           : {to['point']:.0f}  [95% CI {to['ci95'][0]:.0f}, {to['ci95'][1]:.0f}]")
    else:
        print("Tokens out           : N/A")

def main():
    if len(sys.argv) not in (6,7):
        print("Usage: single_llm_eval.py <dataset_dir> <model> <output_dir> <max_parallel> <n_boot> [importance_csv]")
        sys.exit(1)

    dataset_dir  = sys.argv[1]
    model        = sys.argv[2]
    output_dir   = sys.argv[3]
    max_parallel = int(sys.argv[4])
    n_boot       = int(sys.argv[5])
    importance_csv = sys.argv[6] if len(sys.argv) == 7 else ""

    tests = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    tests.sort()
    print(f"Found {len(tests)} test files; evaluating with model={model}, temp={TEMPERATURE}, top_p={TOP_P}")

    results = []
    def run_one(fp):
        name = Path(fp).stem
        try:
            m = evaluate_file(fp, model=model)
            subject = subject_key_from_name(name)
            meta = meta_for_subject(subject)
            return {'test_name': name, 'subject': subject, 'meta': meta, **m}
        except Exception as e:
            sys.stderr.write(f"✗ {name}: ERROR {e}\n")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fut = {ex.submit(run_one, fp): fp for fp in tests}
        for f in concurrent.futures.as_completed(fut):
            r = f.result()
            if r: results.append(r)

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SINGLE-LLM EVALUATION RESULTS")
    print("="*60)

    if df.empty:
        print("\nOVERALL (Single LLM)\n--------------------\nNo data.")
        print("Hint: Your CSVs are likely headerless. This driver now handles headerless files automatically.\n"
              "If problems persist: run `head -1 <file>`; the first row should be a question, not column names.")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "single_llm_by_test.json"), "w") as f:
        json.dump({r['test_name']: r for r in results}, f, indent=2)

    single_overall = bootstrap_ci(df, n_boot=n_boot, seed=0)
    print_block("OVERALL (Single‑LLM baseline)", single_overall)

    summary = {'notes': {'mode':'single_llm','n_boot': n_boot, 'temperature': TEMPERATURE, 'top_p': TOP_P},
               'overall': {'single_llm': single_overall}, 'by_meta': {}}

    for meta in CATEGORIES.keys():
        sub = df[df['meta'] == meta]
        res = bootstrap_ci(sub, n_boot=n_boot, seed=hash(meta) % (2**32))
        print_block(f"{meta} — Single‑LLM", res)
        summary['by_meta'][meta] = {'single_llm': res}

    if importance_csv and os.path.exists(importance_csv):
        pre_df = collect_pre_metrics(importance_csv)
        R = float(pre_df['responses'].sum()) / float(df['responses'].sum())
        pre_has_tok = pre_df['prompt_tokens'].notna().any() and pre_df['completion_tokens'].notna().any()
        pre_overall = bootstrap_ci(pre_df, n_boot=n_boot, seed=1, scale_tokens_by=None if pre_has_tok else R)
        print_block("OVERALL (Pre‑selection / 7 roles)", pre_overall)
        summary['overall']['pre'] = pre_overall
        for meta in CATEGORIES.keys():
            pre_cat = pre_df[pre_df['meta'] == meta]
            pre_has_tok_cat = pre_cat['prompt_tokens'].notna().any() and pre_cat['completion_tokens'].notna().any()
            res = bootstrap_ci(pre_cat, n_boot=n_boot, seed=(hash(meta)+1)%(2**32),
                               scale_tokens_by=None if pre_has_tok_cat else R)
            print_block(f"{meta} — Pre‑selection", res)
            summary['by_meta'][meta]['pre'] = res
        summary['notes']['pre_tokens_estimated'] = not pre_has_tok
        summary['notes']['response_ratio_R'] = R
        print("\nNOTES\n-----")
        print("* API calls are total successful chat completions (one per question).")
        if not pre_has_tok:
            print(f"* Pre tokens not logged; estimated via R = total_responses_pre / total_responses_single = {R:.4f}")

    out_json = os.path.join(output_dir, "metrics_summary_single_llm.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")

if __name__ == "__main__":
    main()
PY

chmod +x "$OUTPUT_DIR/single_llm_eval.py"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
python "$OUTPUT_DIR/single_llm_eval.py" \
  "$EVAL_DATASET" \
  "$MODEL" \
  "$OUTPUT_DIR" \
  "$MAX_PARALLEL" \
  "$N_BOOT" \
  ${IMPORTANCE_CSV:+$IMPORTANCE_CSV}

log "Single‑LLM evaluation completed!"
log "Results saved in: $OUTPUT_DIR"

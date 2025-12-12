#!/usr/bin/env bash
set -euo pipefail

# DyLAN MMLU Evaluation with Memory Bank
# - Runs reduced-role evaluation (post-selection) with Memory Bank support
# - Uses trained Memory Bank from pre-selection phase (READ-ONLY mode)
# - Computes metrics with 95% bootstrap CIs
# - Reports accuracy, API calls, tokens for pre vs post selection

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
NUM_ROLES="${NUM_ROLES:-4}"   # roles selected per question in evaluation
N_BOOT="${N_BOOT:-1000}"      # bootstrap replicates for CI

# Memory Bank configuration
export USE_MEMORY_BANK=1
export MEMORY_MODE="eval"  # READ-ONLY mode (no updates during evaluation)
MEMORY_BANK_FILE="${MEMORY_BANK_FILE:-memory_bank_gpt-oss-20b/memory_bank_cache.json}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-5}"           # Top K memories to inject per agent
export MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-500}"  # Max tokens for memory context

# Default paths
IMPORTANCE_CSV="${IMPORTANCE_CSV:-importance_1to7.csv}"
EVAL_DATASET="${EVAL_DATASET:-$SCRIPT_DIR/../../data/MMLU/evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results_with_memory}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run reduced-role evaluation with Memory Bank support (READ-ONLY mode).
Reports accuracy, API calls, tokens with 95% bootstrap CIs.

OPTIONS:
    -m, --model MODEL              LLM model to use (default: openai/gpt-oss-20b)
    -i, --importance-csv FILE      Path to importance CSV file (default: importance_1to7.csv)
    -d, --dataset DIR              Path to evaluation dataset directory (default: ../../data/MMLU/evaluation)
    -o, --output DIR               Output directory (default: evaluation_results_with_memory)
    -n, --num-roles NUM            Number of roles to select per question (default: 4)
    -p, --max-parallel NUM         Maximum parallel jobs (default: 4)
    --n-boot NUM                   Bootstrap replicates for CI (default: 1000)
    
    MEMORY BANK OPTIONS:
    --memory-bank FILE             Path to trained memory bank (default: memory_bank_gpt-oss-20b/memory_bank_cache.json)
    --memory-top-k NUM             Top K memories to inject per agent (default: 5)
    --memory-max-tokens NUM        Max tokens for memory context (default: 500)
    
    -h, --help                     Show help

EXAMPLES:
    # Use default memory bank
    $0

    # Use custom memory bank
    $0 --memory-bank my_memory_bank.json

    # Adjust memory injection
    $0 --memory-top-k 10 --memory-max-tokens 1000

    # Custom evaluation settings
    $0 --model "gpt-4" --num-roles 3 --max-parallel 8
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2;;
        -i|--importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
        -d|--dataset) EVAL_DATASET="$2"; shift 2;;
        -o|--output) OUTPUT_DIR="$2"; shift 2;;
        -n|--num-roles) NUM_ROLES="$2"; shift 2;;
        -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
        --n-boot) N_BOOT="$2"; shift 2;;
        --memory-bank) MEMORY_BANK_FILE="$2"; shift 2;;
        --memory-top-k) export MEMORY_TOP_K="$2"; shift 2;;
        --memory-max-tokens) export MEMORY_MAX_TOKENS="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ ! -f "$IMPORTANCE_CSV" ]]; then
    log "ERROR: Importance CSV not found: $IMPORTANCE_CSV"
    exit 1
fi
if [[ ! -d "$EVAL_DATASET" ]]; then
    log "ERROR: Evaluation dataset not found: $EVAL_DATASET"
    exit 1
fi
if [[ ! "$NUM_ROLES" =~ ^[1-7]$ ]]; then
    log "ERROR: Number of roles must be in [1..7], got: $NUM_ROLES"
    exit 1
fi

# Check memory bank file
if [[ ! -f "$MEMORY_BANK_FILE" ]]; then
    log "WARNING: Memory bank file not found: $MEMORY_BANK_FILE"
    log "Will proceed without memory bank (USE_MEMORY_BANK will be disabled)"
    export USE_MEMORY_BANK=0
else
    log "Memory Bank found: $MEMORY_BANK_FILE"
    # Make absolute path for Python
    MEMORY_BANK_FILE="$(cd "$(dirname "$MEMORY_BANK_FILE")" && pwd)/$(basename "$MEMORY_BANK_FILE")"
    export MEMORY_BANK_PATH="$MEMORY_BANK_FILE"
fi

log "Starting DyLAN MMLU Evaluation with Memory Bank"
log "Model: $MODEL"
log "Importance CSV: $IMPORTANCE_CSV"
log "Dataset: $EVAL_DATASET"
log "Output: $OUTPUT_DIR"
log "Roles per question: $NUM_ROLES"
log "Max parallel jobs: $MAX_PARALLEL"
log "Bootstrap reps (95% CI): $N_BOOT"
log "---"
if [[ "$USE_MEMORY_BANK" == "1" ]]; then
    log "Memory Bank: ENABLED (EVAL mode - READ-ONLY)"
    log "Memory Bank file: $MEMORY_BANK_FILE"
    log "Memory Top-K: $MEMORY_TOP_K"
    log "Memory Max Tokens: $MEMORY_MAX_TOKENS"
else
    log "Memory Bank: DISABLED"
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------
# Python driver: selection + evaluation + metrics + bootstrap CIs
# (Same as original, but will use memory bank during evaluation)
# ---------------------------------------------------------------------
cat > "$OUTPUT_DIR/evaluate_roles_with_memory.py" << 'EOF'
#!/usr/bin/env python3
import os, sys, re, json, ast, math, subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import concurrent.futures
from functools import partial

# Subject → subcategory map (same as original)
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

def subcats_for_subject(subject: str) -> List[str]:
    return SUBCATEGORY.get(subject, ["other"])

def meta_for_subject(subject: str) -> str:
    subcats = subcats_for_subject(subject)
    sub = subcats[0] if subcats else "other"
    for meta, subs in CATEGORIES.items():
        if sub in subs:
            return meta
    return "other (business, health, misc.)"

def select_top_roles(importance_csv: str, num_roles: int = 4) -> Tuple[Dict[str, List[str]], List[str], pd.DataFrame]:
    """Select top roles for each test based on importance scores (greedy selection)"""
    df = pd.read_csv(importance_csv)
    role_cols = [c for c in df.columns if c.endswith('_imp')]
    role_names = [c.replace('_imp', '') for c in role_cols]
    selected = {}
    for _, row in df.iterrows():
        fname = row['filename']
        scores = [(role_names[i], row[role_cols[i]]) for i in range(len(role_cols))]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [r for r, _ in scores[:num_roles]]
        selected[fname] = top
    return selected, role_names, df

def run_evaluation(test_file: str, selected_roles: Dict[str, List[str]], model: str, 
                   all_roles: List[str], out_dir: str, memory_bank_path: str = None) -> str:
    """Run evaluation for a single test file"""
    filename = Path(test_file).stem

    # Match importance key
    lookup = filename
    if filename not in selected_roles:
        if filename.endswith('_val'):
            lookup = filename.replace('_val', '_test')
        elif filename.endswith('_test'):
            lookup = filename.replace('_test', '_val')
        elif '_test' not in filename and '_val' not in filename:
            lookup = filename + '_test'

    if lookup not in selected_roles:
        print(f"Warning: No importance for {filename} (tried {lookup}); skipping")
        return None

    roles = selected_roles[lookup]
    test_roles_str = str(roles)

    exp_name = f"eval_{filename}"
    roles_str_clean = test_roles_str.replace(' ', '').replace('[','').replace(']','').replace(',', '_').replace("'", '')
    case_dir = os.path.join(out_dir, f"{exp_name}_{roles_str_clean}")
    os.makedirs(case_dir, exist_ok=True)

    result_file = os.path.join(case_dir, f"{filename}_eval.txt")
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        print(f"Skipping {filename} (already processed)")
        return result_file

    print(f"Evaluating {filename} with roles: {roles} (Memory Bank: {'enabled' if memory_bank_path else 'disabled'})")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmlu_dir   = os.path.dirname(script_dir)
    llmlp = os.path.join(mmlu_dir, 'llmlp_listwise_mmlu.py')
    if not os.path.exists(llmlp):
        raise FileNotFoundError(f"llmlp_listwise_mmlu.py not found at {llmlp}")

    expected_txt = os.path.join(case_dir, f"{exp_name}_{len(roles)}3.txt")
    expected_json = os.path.join(case_dir, f"{exp_name}_{len(roles)}3.json")

    # Set up environment with memory bank path if provided
    env = os.environ.copy()
    if memory_bank_path:
        env['MEMORY_BANK_PATH'] = memory_bank_path

    cmd = ['python', llmlp, test_file, exp_name, model, exp_name, test_roles_str]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir, env=env)
    if res.returncode != 0:
        print(f"Error running evaluation for {filename}:\n{res.stderr}")
        if res.stdout: print("STDOUT:", res.stdout)
        return None

    # Move outputs to stable names
    if os.path.exists(expected_txt):
        import shutil
        if expected_txt != result_file:
            shutil.move(expected_txt, result_file)
        json_out = result_file.replace('.txt', '.json')
        if os.path.exists(expected_json) and expected_json != json_out:
            shutil.move(expected_json, json_out)
        return result_file
    else:
        print(f"Warning: expected file not found: {expected_txt}")
        if res.stdout: print("STDOUT:", res.stdout)
        return None

def parse_result_file(result_file: str):
    """Parse evaluation result file"""
    with open(result_file, 'r') as f:
        lines = f.readlines()
    accs_parts = lines[0].strip().rsplit(' ', 1)
    accs = ast.literal_eval(accs_parts[0])
    resp_parts = lines[1].strip().split(' ', 1)
    total_resp = int(resp_parts[0])
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())
    q = len(accs)
    c = sum(1 for a in accs if a)
    return {
        'questions': q,
        'correct': c,
        'responses': total_resp,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }

def collect_post_metrics(result_files: List[str]) -> pd.DataFrame:
    rows = []
    for rf in result_files:
        if not rf or not os.path.exists(rf): continue
        test_name = Path(rf).stem.replace('_eval', '')
        m = parse_result_file(rf)
        subj = subject_key_from_name(test_name)
        meta = meta_for_subject(subj)
        rows.append({
            'test_name': test_name,
            'subject': subj,
            'meta': meta,
            **m
        })
    return pd.DataFrame(rows)

def collect_pre_metrics(importance_csv: str) -> pd.DataFrame:
    df = pd.read_csv(importance_csv)
    rows = []
    has_tok_in = 'prompt_tokens' in df.columns
    has_tok_out = 'completion_tokens' in df.columns
    for _, r in df.iterrows():
        fname = r['filename']
        subj = subject_key_from_name(fname)
        meta = meta_for_subject(subj)
        q_cnt = int(r['q_cnt'])
        correct_float = float(r['acc']) * q_cnt
        responses = int(r['resp'])
        row = {
            'test_name': Path(fname).stem,
            'subject': subj,
            'meta': meta,
            'questions': q_cnt,
            'correct_float': correct_float,
            'responses': responses
        }
        if has_tok_in:  row['prompt_tokens'] = int(r['prompt_tokens'])
        if has_tok_out: row['completion_tokens'] = int(r['completion_tokens'])
        rows.append(row)
    d = pd.DataFrame(rows)
    if 'correct' not in d.columns:
        d['correct'] = d['correct_float']
    if 'prompt_tokens' not in d.columns:
        d['prompt_tokens'] = np.nan
    if 'completion_tokens' not in d.columns:
        d['completion_tokens'] = np.nan
    return d[['test_name','subject','meta','questions','correct','responses','prompt_tokens','completion_tokens']]

def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, scale_tokens_by: float = None):
    """Bootstrap across tests with optional token scaling"""
    if df.empty:
        return {}

    rng = np.random.default_rng(seed)
    n = len(df)

    q_sum = df['questions'].sum()
    acc_point = float(df['correct'].sum()) / q_sum if q_sum > 0 else float('nan')
    api_point = df['responses'].sum()
    tin_point = df['prompt_tokens'].sum(skipna=True)
    tout_point = df['completion_tokens'].sum(skipna=True)

    if scale_tokens_by is not None:
        tin_point = float(scale_tokens_by) * tin_point
        tout_point = float(scale_tokens_by) * tout_point

    acc_samps, api_samps, tin_samps, tout_samps = [], [], [], []
    have_tin = df['prompt_tokens'].notna().any()
    have_tout = df['completion_tokens'].notna().any()

    for _ in range(n_boot):
        idx = rng.integers(low=0, high=n, size=n)
        s = df.iloc[idx]
        q = s['questions'].sum()
        acc = float(s['correct'].sum()) / q if q > 0 else float('nan')
        api = s['responses'].sum()
        tin = s['prompt_tokens'].sum(skipna=True) if have_tin else np.nan
        tout = s['completion_tokens'].sum(skipna=True) if have_tout else np.nan
        if scale_tokens_by is not None:
            tin = (float(scale_tokens_by) * tin) if not np.isnan(tin) else np.nan
            tout = (float(scale_tokens_by) * tout) if not np.isnan(tout) else np.nan

        acc_samps.append(acc)
        api_samps.append(api)
        tin_samps.append(tin)
        tout_samps.append(tout)

    def pct_ci(arr):
        arr = np.array(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        'accuracy': {'point': float(acc_point), 'ci95': pct_ci(acc_samps)},
        'api_calls': {'point': int(api_point), 'ci95': [int(np.nanpercentile(api_samps, 2.5)), int(np.nanpercentile(api_samps, 97.5))]},
    }

    if have_tin:
        out['tokens_in'] = {'point': float(tin_point), 'ci95': pct_ci(tin_samps)}
    else:
        out['tokens_in'] = {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}
    if have_tout:
        out['tokens_out'] = {'point': float(tout_point), 'ci95': pct_ci(tout_samps)}
    else:
        out['tokens_out'] = {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}

    return out

def print_block(title: str, res: dict, mark_est_tokens: bool = False):
    print(f"\n{title}")
    print("-" * len(title))
    acc = res['accuracy']; api = res['api_calls']
    print(f"Accuracy             : {acc['point']:.4f}  [95% CI {acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
    print(f"API calls            : {api['point']}  [95% CI {api['ci95'][0]}, {api['ci95'][1]}]")

    ti = res.get('tokens_in', None)
    to = res.get('tokens_out', None)
    if ti and not (math.isnan(ti['point']) or math.isnan(ti['ci95'][0])):
        est_tag = " (est.)" if mark_est_tokens else ""
        print(f"Tokens in            : {ti['point']:.0f}{est_tag}  [95% CI {ti['ci95'][0]:.0f}, {ti['ci95'][1]:.0f}]")
    else:
        print("Tokens in            : N/A")
    if to and not (math.isnan(to['point']) or math.isnan(to['ci95'][0])):
        est_tag = " (est.)" if mark_est_tokens else ""
        print(f"Tokens out           : {to['point']:.0f}{est_tag}  [95% CI {to['ci95'][0]:.0f}, {to['ci95'][1]:.0f}]")
    else:
        print("Tokens out           : N/A")

def main():
    if len(sys.argv) not in (7, 8, 9):
        print("Usage: evaluate_roles_with_memory.py <importance_csv> <dataset_dir> <model> <num_roles> <output_dir> <max_parallel> [<n_boot>] [<memory_bank_path>]")
        sys.exit(1)
    
    importance_csv = sys.argv[1]
    dataset_dir   = sys.argv[2]
    model         = sys.argv[3]
    num_roles     = int(sys.argv[4])
    output_dir    = sys.argv[5]
    max_parallel  = int(sys.argv[6])
    n_boot        = int(sys.argv[7]) if len(sys.argv) >= 8 else 1000
    memory_bank_path = sys.argv[8] if len(sys.argv) >= 9 else None

    print(f"Loading importance data from: {importance_csv}")
    selected_roles, all_roles, df_imp = select_top_roles(importance_csv, num_roles)
    print(f"Found importance data for {len(selected_roles)} tests")
    print(f"Selected {num_roles} roles per test from: {all_roles}")
    
    if memory_bank_path:
        print(f"Memory Bank: {memory_bank_path}")
    else:
        print("Memory Bank: Not provided (running without memory)")

    # Gather test files
    tests = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    print(f"Found {len(tests)} test files")
    
    # Filter to only those with importance data
    to_proc = []
    for tf in tests:
        fn = Path(tf).stem
        lookup = fn
        if fn not in selected_roles:
            if fn.endswith('_val'):
                lookup = fn.replace('_val', '_test')
            elif fn.endswith('_test'):
                lookup = fn.replace('_test', '_val')
            elif '_test' not in fn and '_val' not in fn:
                lookup = fn + '_test'
        if lookup in selected_roles:
            to_proc.append(tf)
    print(f"Processing {len(to_proc)} test files with importance data")

    # Run evaluations in parallel
    result_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fn = partial(run_evaluation, selected_roles=selected_roles, model=model, 
                    all_roles=all_roles, out_dir=output_dir, memory_bank_path=memory_bank_path)
        fut = {ex.submit(fn, tf): tf for tf in to_proc}
        for f in concurrent.futures.as_completed(fut):
            rf = f.result()
            if rf: result_files.append(rf)

    # Collect metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS (WITH MEMORY BANK)")
    print("="*60)

    post_df = collect_post_metrics(result_files)
    pre_df  = collect_pre_metrics(importance_csv)

    if post_df.empty:
        print("\n⚠️  ERROR: No evaluation results collected!")
        sys.exit(1)

    # Compute response ratio for token estimation
    total_resp_pre  = pre_df['responses'].sum()
    total_resp_post = post_df['responses'].sum()
    R = (float(total_resp_pre) / float(total_resp_post)) if total_resp_post > 0 else float('nan')

    # Overall summaries
    post_overall = bootstrap_ci(post_df, n_boot=n_boot, seed=0)
    pre_has_tok = pre_df['prompt_tokens'].notna().any() and pre_df['completion_tokens'].notna().any()
    pre_overall = bootstrap_ci(pre_df, n_boot=n_boot, seed=1,
                               scale_tokens_by=None if pre_has_tok else R)

    print_block("OVERALL (Post-selection with Memory Bank)", post_overall, mark_est_tokens=False)
    print_block("OVERALL (Pre-selection / 7 roles)", pre_overall, mark_est_tokens=(not pre_has_tok))

    # Per meta-category
    categories = list(CATEGORIES.keys())
    summary = {
        'notes': {'memory_bank_enabled': memory_bank_path is not None},
        'overall': {'post': post_overall, 'pre': pre_overall},
        'by_meta': {}
    }

    for meta in categories:
        post_cat = post_df[post_df['meta'] == meta]
        pre_cat  = pre_df[pre_df['meta'] == meta]
        post_cat_res = bootstrap_ci(post_cat, n_boot=n_boot, seed=hash(meta) % (2**32))
        pre_cat_has_tok = pre_cat['prompt_tokens'].notna().any() and pre_cat['completion_tokens'].notna().any()
        pre_cat_res = bootstrap_ci(pre_cat, n_boot=n_boot, seed=(hash(meta)+1) % (2**32),
                                   scale_tokens_by=None if pre_cat_has_tok else R)

        print_block(f"{meta} — Post-selection with Memory", post_cat_res, mark_est_tokens=False)
        print_block(f"{meta} — Pre-selection", pre_cat_res, mark_est_tokens=(not pre_cat_has_tok))

        summary['by_meta'][meta] = {
            'post': post_cat_res,
            'pre': pre_cat_res
        }

    # Print notes
    print("\nNOTES")
    print("-----")
    print("* API calls are computed as the total number of model responses.")
    print("* Memory Bank was used in EVAL mode (read-only) during post-selection evaluation.")
    if not pre_has_tok:
        print(f"* Pre-selection tokens estimated as: tokens_pre ≈ {R:.4f} × tokens_post")
        summary['notes']['pre_tokens_estimated'] = True
        summary['notes']['response_ratio_R'] = R
    else:
        summary['notes']['pre_tokens_estimated'] = False

    # Save JSON
    out_json = os.path.join(output_dir, f"metrics_summary_{num_roles}roles_with_memory.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")

if __name__ == "__main__":
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_roles_with_memory.py"

# ---------------------------------------------------------------------
# Run evaluation + metrics (with bootstrap)
# ---------------------------------------------------------------------
log "Running evaluation with $NUM_ROLES roles per question (with Memory Bank)..."

# Pass memory bank path to Python if available
if [[ "$USE_MEMORY_BANK" == "1" && -f "$MEMORY_BANK_FILE" ]]; then
    python "$OUTPUT_DIR/evaluate_roles_with_memory.py" \
        "$IMPORTANCE_CSV" \
        "$EVAL_DATASET" \
        "$MODEL" \
        "$NUM_ROLES" \
        "$OUTPUT_DIR" \
        "$MAX_PARALLEL" \
        "$N_BOOT" \
        "$MEMORY_BANK_FILE"
else
    python "$OUTPUT_DIR/evaluate_roles_with_memory.py" \
        "$IMPORTANCE_CSV" \
        "$EVAL_DATASET" \
        "$MODEL" \
        "$NUM_ROLES" \
        "$OUTPUT_DIR" \
        "$MAX_PARALLEL" \
        "$N_BOOT"
fi

log "Evaluation completed!"
log "Results saved in: $OUTPUT_DIR"


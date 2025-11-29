#!/usr/bin/env bash
set -euo pipefail

# ---------- config & env ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Load .env keys if present (Together/OpenAI, etc.)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

MODEL="${MODEL:-openai/gpt-oss-20b}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/data/MMLU/evaluation}"
ROLES_JSON="${ROLES_JSON:-$REPO_ROOT/code/MMLU/standard_dylan/mmlu_with_local_judge/roles_top4.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/code/MMLU/standard_dylan/mmlu_eval_local_judge}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
BOOTSTRAP="${BOOTSTRAP:-1000}"

mkdir -p "$OUT_DIR"

echo "[info] MODEL=$MODEL"
echo "[info] EVAL_DIR=$EVAL_DIR"
echo "[info] ROLES_JSON=$ROLES_JSON"
echo "[info] OUT_DIR=$OUT_DIR"
echo "[info] MAX_PARALLEL=$MAX_PARALLEL"
echo "[info] BOOTSTRAP=$BOOTSTRAP"

# ---------- tiny helpers ----------
active_jobs() { jobs -rp | wc -l | tr -d ' '; }
wait_any() {
  local pids; mapfile -t pids < <(jobs -rp)
  if [[ ${#pids[@]} -gt 0 ]]; then wait "${pids[0]}" || true; fi
}
# Portable timestamp (macOS/Linux)
ts() { date "+%Y-%m-%dT%H:%M:%S%z"; }

# Retrieve roles (as a JSON string) for a subject key
get_roles_for() {
  local subj="$1"
  python - "$ROLES_JSON" "$subj" <<'PY'
import json, sys
rp, subj = sys.argv[1], sys.argv[2]
m = json.load(open(rp))
roles = m.get(subj) or m.get(subj + ".csv")
if not roles:
    raise SystemExit(f"[ERROR] No roles for {subj} in {rp}")
# Print compact JSON array, no spaces (shell-safe)
print(json.dumps(roles, separators=(',',':')))
PY
}

# ---------- run jobs ----------
shopt -s nullglob
for csv in "$EVAL_DIR"/*.csv; do
  subj="$(basename "$csv" .csv)"
  roles_json="$(get_roles_for "$subj")"   # e.g. ["Economist","Doctor","Lawyer","Mathematician"]

  # llmlp_listwise_mmlu.py writes <subj>_43.{txt,log} when len(roles)==4.
  RES_TXT="$OUT_DIR/${subj}_43.txt"
  RUN_LOG="$OUT_DIR/${subj}_43.run.log"

  # Skip if we already have a reasonable result text file
  if [[ -f "$RES_TXT" ]] && [[ "$(wc -l < "$RES_TXT")" -ge 1 ]]; then
    echo "Skip $subj (already has $RES_TXT)"
    continue
  fi

  while (( $(active_jobs) >= MAX_PARALLEL )); do wait_any; done

  echo ">>> Eval ${subj}  roles=${roles_json}"
  (
    echo "[START] $(ts) subject=$subj model=$MODEL" >> "$RUN_LOG"
    # Important: we pass OUT_DIR directly so there is only one folder.
    python "$REPO_ROOT/code/MMLU/llmlp_listwise_mmlu.py" \
           "$csv" "$subj" "$MODEL" "$OUT_DIR" "$roles_json" \
           >> "$RUN_LOG" 2>&1
    echo "[END] $(ts) subject=$subj" >> "$RUN_LOG"
  ) &
done

fails=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then fails=$((fails+1)); fi
done
echo "All evaluation jobs finished (failures: $fails)."

# ---------- summarize (recursive) ----------
BOOTSTRAP="$BOOTSTRAP" OUT_DIR="$OUT_DIR" python - <<'PY'
import os, re, glob, ast, json, random, sys
ROOT = "code/MMLU/standard_dylan"
OUT_DIR = os.environ.get("OUT_DIR")
BOOTSTRAP = int(os.environ.get("BOOTSTRAP","1000"))

def find_result_txts():
    patterns = [
        os.path.join(OUT_DIR, "*_43.txt"),
        os.path.join(ROOT, "mmlu_eval_local_judge_*", "*_43.txt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def parse_txt(path):
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                left, acc_str = s.rsplit(" ", 1)
                bits = ast.literal_eval(left)
                acc = float(acc_str)
                if isinstance(bits, list) and all(isinstance(x, bool) for x in bits):
                    return bits, acc
            except Exception:
                pass
            break
    return None, None

def parse_log_for_bits(log_path):
    import re
    pat = re.compile(r'^\s*(\[(?:\s*(?:True|False)\s*,?)+\])\s+([0-9]*\.?[0-9]+)\s*$', re.M)
    try:
        with open(log_path, "r") as f:
            text = f.read()
        m = None
        for m in pat.finditer(text):
            pass
        if not m:
            return None, None
        bits = ast.literal_eval(m.group(1))
        acc = float(m.group(2))
        return bits, acc
    except Exception:
        return None, None

def subject_from_file(p):
    base = os.path.basename(p)
    if base.endswith("_43.txt"):
        return base[:-len("_43.txt")]
    if base.endswith("_43.run.log"):
        return base[:-len("_43.run.log")]
    return re.sub(r'_(\d+)\.(txt|log)$','',base)

txts = find_result_txts()
subjects = {}
all_bits = []

for t in txts:
    subj = subject_from_file(t)
    bits, _ = parse_txt(t)
    if bits is None:
        # fall back: corresponding run.log in OUT_DIR
        log_path = os.path.join(OUT_DIR, f"{subj}_43.run.log")
        bits, _ = parse_log_for_bits(log_path)
    if bits is None:
        continue
    subjects[subj] = {"n": len(bits), "acc": sum(bits)/len(bits) if bits else 0.0, "file": t}
    all_bits.extend(bits)

# fallback: try only run logs in OUT_DIR
if not subjects:
    for log in glob.glob(os.path.join(OUT_DIR, "*_43.run.log")):
        subj = subject_from_file(log)
        bits, _ = parse_log_for_bits(log)
        if bits is None:
            continue
        subjects[subj] = {"n": len(bits), "acc": sum(bits)/len(bits) if bits else 0.0, "file": log}
        all_bits.extend(bits)

if not subjects:
    print(f"[ERROR] No *_43.txt (or parsable logs) found under {OUT_DIR}")
    sys.exit(1)

n_subj = len(subjects)
n_q = len(all_bits)
overall = sum(all_bits)/n_q

rng = random.Random(1337)
boots = []
for _ in range(BOOTSTRAP):
    sample = [all_bits[rng.randrange(n_q)] for _ in range(n_q)]
    boots.append(sum(sample)/n_q)
boots.sort()
lo = boots[int(0.025*BOOTSTRAP)]
hi = boots[int(0.975*BOOTSTRAP)]

print()
print("============================================================")
print("MMLU — Evaluation with Local‑Judge‑Selected Roles")
print("============================================================")
print(f"Subjects evaluated : {n_subj}")
print(f"Questions total    : {n_q}\n")
print("OVERALL (Local‑judge selection)")
print("-------------------------------")
print(f"Accuracy           : {overall:.4f}  [95% CI {lo:.4f}, {hi:.4f}]\n")
print(f"Detailed files in  : {OUT_DIR}\n")

summary = {
    "subjects": n_subj,
    "questions": n_q,
    "overall_acc": overall,
    "ci_95": [lo, hi],
    "files_considered": sorted(v["file"] for v in subjects.values()),
}
with open(os.path.join(OUT_DIR, "metrics_summary_local_judge.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved JSON summary → {OUT_DIR}/metrics_summary_local_judge.json")
PY

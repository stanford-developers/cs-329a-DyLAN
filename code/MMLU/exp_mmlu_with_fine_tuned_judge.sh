#!/usr/bin/env bash
# code/MMLU/exp_mmlu_with_fine_tuned_judge.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
# Load keys (.env at repo root)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

MODEL="${MODEL:-openai/gpt-oss-20b}"
JUDGE_CKPT="${JUDGE_CKPT:-$REPO_ROOT/code/MMLU/finetune/ckpts/merged}"
ROLES="${ROLES:-[\"Economist\",\"Doctor\",\"Lawyer\",\"Mathematician\",\"Psychologist\",\"Programmer\",\"Historian\"]}"
SEL_DIR="${SEL_DIR:-$REPO_ROOT/data/MMLU/small_team_selection}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/code/MMLU/standard_dylan/mmlu_with_local_judge}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OVERWRITE="${OVERWRITE:-0}"

echo "[info] MODEL=$MODEL"
echo "[info] JUDGE_CKPT=$JUDGE_CKPT"
echo "[info] SEL_DIR=$SEL_DIR"
echo "[info] OUT_DIR=$OUT_DIR"
echo "[info] ROLES_JSON=$ROLES"
echo "[info] MAX_PARALLEL=$MAX_PARALLEL"
echo "[info] OVERWRITE=$OVERWRITE"

mkdir -p "$OUT_DIR"

active_jobs() { jobs -rp | wc -l; }
wait_any() { local pids=($(jobs -rp)); if [[ ${#pids[@]} -gt 0 ]]; then wait "${pids[0]}" || true; fi; }

shopt -s nullglob
for csv in "$SEL_DIR"/*.csv ; do
  subj="$(basename "$csv" .csv)"
  echo ">>> Subject: $subj"

  while (( $(active_jobs) >= MAX_PARALLEL )); do wait_any; done
  (
    python "$REPO_ROOT/code/MMLU/run_mmlu_with_local_judge.py" \
      --csv "$csv" \
      --subject "$subj" \
      --model "$MODEL" \
      --roles-json "$ROLES" \
      --judge-ckpt "$JUDGE_CKPT" \
      --outdir "$OUT_DIR" \
      $( ((OVERWRITE==1)) && echo --overwrite )
  ) &
done

fails=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then fails=$((fails+1)); fi
done
echo "Done. Outputs in: $OUT_DIR (failures: $fails)"

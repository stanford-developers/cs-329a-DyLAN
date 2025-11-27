#!/usr/bin/env bash
set -euo pipefail

# ---------- user knobs ----------
MODEL="${MODEL:-openai/gpt-oss-20b}"
JUDGE_CKPT="${JUDGE_CKPT:-code/MMLU/finetune/ckpts/merged}"

# Always use the 7 DyLAN roles by default (JSON, not Python list!)
ROLES_JSON_DEFAULT='["Economist","Doctor","Lawyer","Mathematician","Psychologist","Programmer","Historian"]'
ROLES_JSON="${ROLES_JSON:-$ROLES_JSON_DEFAULT}"

# Selection subset only (no fallback to evaluation here)
SEL_DIR="${SEL_DIR:-data/MMLU/small_team_selection}"

# Where to write subject logs + artifacts
OUT_DIR="${OUT_DIR:-code/MMLU/standard_dylan/mmlu_with_local_judge}"
# --------------------------------

mkdir -p "$OUT_DIR"

echo "[info] MODEL=${MODEL}"
echo "[info] JUDGE_CKPT=${JUDGE_CKPT}"
echo "[info] SEL_DIR=${SEL_DIR}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] ROLES_JSON=${ROLES_JSON}"

shopt -s nullglob
for csv in "${SEL_DIR}"/*.csv ; do
  fname=$(basename "$csv" .csv)
  echo ">>> Subject: ${fname}"

  # run_mmlu_with_local_judge.py expects: csv exp model out_dir roles_json  [--judge-ckpt PATH]
  # We also tee stdout/stderr into a subject .log so you don't lose progress output.
  python code/MMLU/run_mmlu_with_local_judge.py \
      "$csv" "$fname" "$MODEL" "$OUT_DIR" "$ROLES_JSON" \
      --judge-ckpt "$JUDGE_CKPT" \
      >"${OUT_DIR}/${fname}.log" 2>&1
done

echo "Done. Outputs in: ${OUT_DIR}"

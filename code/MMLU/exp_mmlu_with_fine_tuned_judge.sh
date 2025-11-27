#!/usr/bin/env bash
set -euo pipefail

# ---------------- user knobs (env overrides supported) ----------------
# Base model for the agents (used inside DyLAN runs)
MODEL="${MODEL:-openai/gpt-oss-20b}"

# Path to your merged local fine-tuned judge checkpoint directory
JUDGE_CKPT="${JUDGE_CKPT:-code/MMLU/finetune/ckpts/merged}"

# 7 default DyLAN roles (JSON string). Keep the single quotes!
ROLES_JSON="${ROLES_JSON:-['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']}"

# Pre-selection input split (ALWAYS this directory — no fallback)
SEL_DIR="${SEL_DIR:-data/MMLU/small_team_selection}"

# Where to write new DyLAN pre-selection results that use the local judge
OUT_DIR="${OUT_DIR:-code/MMLU/standard_dylan/mmlu_with_local_judge}"
# ---------------------------------------------------------------------

echo "[info] MODEL=${MODEL}"
echo "[info] JUDGE_CKPT=${JUDGE_CKPT}"
echo "[info] SEL_DIR=${SEL_DIR}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] ROLES_JSON=${ROLES_JSON}"

if [[ ! -d "${JUDGE_CKPT}" ]]; then
  echo "[error] JUDGE_CKPT does not exist: ${JUDGE_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${SEL_DIR}" ]]; then
  echo "[error] SEL_DIR does not exist: ${SEL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

shopt -s nullglob
for csv in "${SEL_DIR}"/*.csv; do
  subject="$(basename "${csv}" .csv)"
  echo ">>> Subject: ${subject}"

  # IMPORTANT: run_mmlu_with_local_judge.py expects positional args:
  #   csv  exp_name  model  out_dir  roles_json   --judge-ckpt PATH
  python code/MMLU/run_mmlu_with_local_judge.py \
    "${csv}" "${subject}" "${MODEL}" "${OUT_DIR}" "${ROLES_JSON}" \
    --judge-ckpt "${JUDGE_CKPT}"
done

echo
echo "[done] Pre-selection runs finished. Outputs are in: ${OUT_DIR}"
echo "[next] To do the FINAL evaluation on data/MMLU/evaluation using the roles derived here,"
echo "       run your existing evaluation script pointing at ${OUT_DIR} (e.g., set RESULTS_DIR to ${OUT_DIR})."

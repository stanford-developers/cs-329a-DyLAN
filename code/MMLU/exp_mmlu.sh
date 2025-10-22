#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Optional: load API keys from repo-root .env (so Python sees them)
# ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

# ------------------------------------------------------------
# Model + paths
# ------------------------------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct-Turbo-Free}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Your small MMLU slice (3 csvs)
dir="$REPO_ROOT/data/MMLU/one_percent_team_selection"
exp_name="mmlu_downsampled"

# This folder name must match what the Python code and anal_imp.sh expect
OUT_DIR="${exp_name}_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian"
mkdir -p "$OUT_DIR"

ROLES="['Economist', 'Doctor', 'Lawyer', 'Mathematician', 'Psychologist', 'Programmer', 'Historian']"

shopt -s nullglob
for file in "$dir"/*.csv; do
  filename="$(basename -- "$file")"
  filename="${filename%.*}"   # drop .csv

  RES_NAME="$OUT_DIR/${filename}_73.txt"
  LOG_NAME="$OUT_DIR/${filename}_73.log"

  # Skip if we already have a 4-line result file
  if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -eq 4 ]]; then
    echo "Skip $filename (already done)"
    continue
  fi

  echo "Running $filename → $LOG_NAME"
  # IMPORTANT: redirect stdout first, then stderr; otherwise stderr goes to the terminal
  python "$SCRIPT_DIR/llmlp_listwise_mmlu.py" \
         "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" \
         > "$LOG_NAME" 2>&1 &
done

wait
echo "All python jobs finished."

# Run post-processing from the MMLU folder so relative paths match
( cd "$SCRIPT_DIR" && bash anal_imp.sh )

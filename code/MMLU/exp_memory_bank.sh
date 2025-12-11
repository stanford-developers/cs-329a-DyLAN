#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Pre-selection with Memory Bank Training
# - Runs 7-role pre-selection to collect importance scores
# - Trains Memory Bank by extracting agent experiences
# - Outputs: *_73.txt files with importance data + memory_bank.json
# ------------------------------------------------------------

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

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
# Configurable knobs (can override via env)
# ------------------------------------------------------------
MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${DATA_DIR:-$REPO_ROOT/data/MMLU/small_team_selection}"
exp_name="memory_bank"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

# Memory Bank specific configuration
export USE_MEMORY_BANK=1
export MEMORY_MODE="train"  # Training mode for pre-selection
export MEMORY_IMPORTANCE_THRESHOLD="${MEMORY_IMPORTANCE_THRESHOLD:-0.1}"

# Multi-agent configuration (7 roles for pre-selection)
ROLES="${ROLES:-['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']}"

# Output folder name
OUT_DIR="${exp_name}_$(basename "$MODEL" | tr '/' '_' | tr '.' '_')"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Running Pre-selection with Memory Bank Training"
echo "Model: $MODEL"
echo "Data directory: $dir"
echo "Output directory: $OUT_DIR"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Roles (7 for pre-selection): $ROLES"
echo "Memory Bank: ENABLED (Training Mode)"
echo "Memory Importance Threshold: $MEMORY_IMPORTANCE_THRESHOLD"
echo "=========================================="
echo ""

active_jobs() { jobs -rp | wc -l; }

# Wait for any background job to finish (compatible with bash 3.2+)
wait_any() {
  local pids=($(jobs -rp))
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[0]}" || true
  fi
}

shopt -s nullglob
for file in "$dir"/*.csv; do
  filename="$(basename -- "$file")"
  filename="${filename%.*}"

  # Create role-specific output directory
  ROLE_DIR="$OUT_DIR/${filename}_$(echo "$ROLES" | tr -d "[]', " | tr ',' '_')"
  RES_NAME="$ROLE_DIR/${filename}_${exp_name}_73.txt"
  LOG_NAME="$ROLE_DIR/${filename}_${exp_name}.log"
  MEMORY_FILE="$ROLE_DIR/memory_bank.json"
  
  # Create the role-specific directory
  mkdir -p "$ROLE_DIR"

  # Check if experiment is already completed
  # Python writes 6 lines: accs, resp_cnts, importances, avg_importances, prompt_tokens, completion_tokens
  if [[ -f "$RES_NAME" ]] && [[ "$(wc -l < "$RES_NAME")" -ge 6 ]]; then
    echo "Skip $filename (already done)"
    continue
  fi

  # backpressure on background jobs
  while (( $(active_jobs) >= MAX_PARALLEL )); do
    wait_any
  done

  echo "Running $filename → $LOG_NAME"
  (
    echo "[START] $(date -Iseconds) file=$file model=$MODEL roles=$ROLES memory_enabled=1"
    SECONDS=0
    
    # Pass ROLE_DIR so each test has its own memory bank (enables parallel execution)
    python "$SCRIPT_DIR/llmlp_listwise_mmlu.py" \
           "$file" "$filename" "$MODEL" "$ROLE_DIR" "$ROLES" \
           > "$LOG_NAME" 2>&1
    status=$?
    
    # Log completion info
    echo "[END] $(date -Iseconds) file=$file status=$status elapsed_sec=$SECONDS" >> "$LOG_NAME"
    
    # Log memory bank statistics if available
    if [[ -f "$MEMORY_FILE" ]]; then
      memory_count=$(python -c "import json; data=json.load(open('$MEMORY_FILE')); print(len(data.get('entries', [])))" 2>/dev/null || echo "0")
      echo "[MEMORY] Total memories stored: $memory_count" >> "$LOG_NAME"
    fi
    
    exit $status
  ) &
done

# Wait for any remaining jobs; keep the script alive even if some fail
fails=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then fails=$((fails+1)); fi
done

echo ""
echo "=========================================="
echo "All memory bank jobs finished (failures: $fails)."
echo "=========================================="

# Merge all individual memory banks into a single global memory bank
echo ""
echo "Merging individual memory banks..."
GLOBAL_MEMORY_BANK="$OUT_DIR/memory_bank_cache.json"
python3 - "$OUT_DIR" "$GLOBAL_MEMORY_BANK" << 'MERGE_EOF'
import json
import glob
import sys
import os

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
global_bank_path = sys.argv[2] if len(sys.argv) > 2 else "memory_bank_merged.json"

# Find all memory_bank.json files in subdirectories
memory_files = glob.glob(f"{out_dir}/**/memory_bank.json", recursive=True)

if not memory_files:
    print("No memory bank files found to merge.")
    sys.exit(0)

print(f"Found {len(memory_files)} memory bank files to merge")

# Merged data structure: {owner: {id: text}}
merged_data = {}
total_memories = 0
memory_counter = {}  # Track next ID for each owner

for memory_file in memory_files:
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for owner, memories in data.items():
            if owner not in merged_data:
                merged_data[owner] = {}
                memory_counter[owner] = 1
            
            # Add memories with new sequential IDs to avoid conflicts
            for mem_id, mem_text in memories.items():
                new_id = str(memory_counter[owner])
                merged_data[owner][new_id] = mem_text
                memory_counter[owner] += 1
                total_memories += 1
        
        print(f"  Merged: {memory_file}")
    except Exception as e:
        print(f"  Warning: Failed to merge {memory_file}: {e}")

# Save merged memory bank
if merged_data:
    with open(global_bank_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Merged memory bank saved to: {global_bank_path}")
    print(f"   Total memories: {total_memories}")
    print(f"   Agents with memories: {', '.join(merged_data.keys())}")
    for owner, mems in merged_data.items():
        print(f"     - {owner}: {len(mems)} memories")
else:
    print("No memories found to merge.")
MERGE_EOF

# Calculate average accuracy and memory statistics across all test files
if command -v python &> /dev/null; then
  echo ""
  echo "Computing aggregate statistics..."
  python << 'EOF'
import os
import sys
import glob
import json

out_dir = sys.argv[1] if len(sys.argv) > 1 else "memory_bank_*"
txt_files = glob.glob(f"{out_dir}/**/*_memory_bank_73.txt", recursive=True)

if not txt_files:
    print("No results found.")
    sys.exit(0)

accuracies = []
total_prompt_tokens = 0
total_completion_tokens = 0
total_memories = 0
memory_files_found = 0

print(f"Found {len(txt_files)} result files")

for txt_file in txt_files:
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 6:
                # Line 1 has format: [True, False, ...] accuracy
                acc_line = lines[0].strip()
                if ' ' in acc_line:
                    acc = float(acc_line.split()[-1])
                    accuracies.append(acc)
                # Lines 5 and 6 have token counts
                total_prompt_tokens += int(lines[4].strip())
                total_completion_tokens += int(lines[5].strip())
        
        # Check for corresponding memory bank file
        dir_path = os.path.dirname(txt_file)
        memory_file = os.path.join(dir_path, "memory_bank.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                    memory_count = len(memory_data.get('entries', []))
                    total_memories += memory_count
                    memory_files_found += 1
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Failed to process {txt_file}: {e}")

if accuracies:
    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\nAggregate Results:")
    print(f"  Files evaluated: {len(accuracies)}")
    print(f"  Average accuracy: {avg_acc:.4f}")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,}")
    print(f"\nMemory Bank Statistics:")
    print(f"  Memory files found: {memory_files_found}")
    print(f"  Total memories stored: {total_memories}")
    if memory_files_found > 0:
        print(f"  Average memories per file: {total_memories / memory_files_found:.1f}")
else:
    print("No valid results to aggregate.")
EOF
fi

echo ""
echo "=========================================="
echo "Generating importance_1to7.csv..."
echo "=========================================="

# Generate importance CSV from *_memory_bank_73.txt files
python3 << 'GEN_CSV_EOF'
import os
import sys
import glob
import csv
from pathlib import Path

out_dir = sys.argv[1] if len(sys.argv) > 1 else "memory_bank_*"
txt_files = glob.glob(f"{out_dir}/**/*_memory_bank_73.txt", recursive=True)

if not txt_files:
    print("No *_memory_bank_73.txt files found!")
    sys.exit(1)

print(f"Found {len(txt_files)} result files")

# Parse all result files
rows = []
for txt_file in txt_files:
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 6:
            print(f"Skipping {txt_file}: insufficient lines")
            continue
        
        # Extract filename (test name)
        filename = Path(txt_file).parent.name
        # Extract test name from directory name (e.g., abstract_algebra_test_Economist...)
        test_name = filename.split('_Economist')[0] if '_Economist' in filename else filename
        
        # Line 0: [True, False, ...] accuracy
        acc_line = lines[0].strip()
        if ' ' in acc_line:
            acc = float(acc_line.split()[-1])
        else:
            continue
        
        # Line 1: total_responses avg_responses
        resp_line = lines[1].strip().split()
        total_resp = int(resp_line[0])
        
        # Line 3: [avg_importance_role1, avg_importance_role2, ...]
        avg_imp_line = lines[3].strip()
        import ast
        avg_importances = ast.literal_eval(avg_imp_line)
        
        # Line 4: prompt_tokens
        # Line 5: completion_tokens
        prompt_tokens = int(lines[4].strip())
        completion_tokens = int(lines[5].strip())
        
        # Count questions from line 0
        accs_str = acc_line.rsplit(' ', 1)[0]
        accs_list = ast.literal_eval(accs_str)
        q_cnt = len(accs_list)
        
        # Build row
        row = {
            'filename': test_name,
            'acc': acc,
            'resp': total_resp,
            'q_cnt': q_cnt,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
        
        # Add importance scores for each role (7 roles)
        role_names = ['Economist', 'Doctor', 'Lawyer', 'Mathematician', 
                     'Psychologist', 'Programmer', 'Historian']
        for i, role in enumerate(role_names):
            if i < len(avg_importances):
                row[f'{role}_imp'] = avg_importances[i]
            else:
                row[f'{role}_imp'] = 0.0
        
        rows.append(row)
        
    except Exception as e:
        print(f"Error processing {txt_file}: {e}")

if not rows:
    print("No valid data to write!")
    sys.exit(1)

# Sort by filename for consistency
rows.sort(key=lambda x: x['filename'])

# Write CSV
csv_path = "importance_1to7.csv"
fieldnames = ['filename', 'Economist_imp', 'Doctor_imp', 'Lawyer_imp', 
              'Mathematician_imp', 'Psychologist_imp', 'Programmer_imp', 
              'Historian_imp', 'acc', 'resp', 'q_cnt', 'prompt_tokens', 'completion_tokens']

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Generated {csv_path}")
print(f"   Contains {len(rows)} test files")
print(f"   Location: {os.path.abspath(csv_path)}")
GEN_CSV_EOF
python3 - "$OUT_DIR"

echo ""
echo "=========================================="
echo "Pre-selection with Memory Bank Complete!"
echo ""
echo "Results are stored in: $OUT_DIR"
echo "Generated files:"
echo "  - importance_1to7.csv: Role importance scores for all tests"
echo "  - memory_bank_cache.json: Merged memory bank from all tests"
echo ""
echo "Each test subdirectory contains:"
echo "  - *_memory_bank_73.txt: Individual test results"
echo "  - memory_bank.json: Individual test memories"
echo "  - *.log: Detailed execution logs"
echo ""
echo "Next steps:"
echo "  Run evaluation with memory bank:"
echo "    ./exp_mmlu_evaluation_with_memory.sh --memory-bank $OUT_DIR/memory_bank_cache.json"
echo "=========================================="

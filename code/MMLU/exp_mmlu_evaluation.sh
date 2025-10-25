#!/usr/bin/env bash
set -euo pipefail

# DyLAN MMLU Evaluation Script
# Takes eval dataset and importance1to7.csv, reduces roles to 4 per question
# and reports accuracy + extensible metrics
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct-Turbo-Free}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
NUM_ROLES="${NUM_ROLES:-4}"  # Number of roles to select per question

# Default paths
IMPORTANCE_CSV="${IMPORTANCE_CSV:-importance_1to7.csv}"
EVAL_DATASET="${EVAL_DATASET:-$SCRIPT_DIR/../../data/MMLU/test}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DyLAN MMLU Evaluation Script - Reduces roles to 4 per question based on importance scores

OPTIONS:
    -m, --model MODEL              LLM model to use (default: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free)
    -i, --importance-csv FILE      Path to importance CSV file (default: importance_1to7.csv)
    -d, --dataset DIR              Path to evaluation dataset directory (default: ../../data/MMLU/test)
    -o, --output DIR               Output directory for results (default: evaluation_results)
    -n, --num-roles NUM            Number of roles to select per question (default: 4)
    -p, --max-parallel NUM         Maximum parallel jobs (default: 4)
    -h, --help                     Show this help message

EXAMPLES:
    # Basic evaluation with default settings
    $0

    # Custom model and dataset
    $0 --model "gpt-4" --dataset "/path/to/test/data"

    # Select top 3 roles instead of 4
    $0 --num-roles 3

    # Use custom importance file
    $0 --importance-csv "custom_importance.csv"
EOF
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

active_jobs() { 
    jobs -rp | wc -l
}

wait_any() {
    local pids=($(jobs -rp))
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait "${pids[0]}" || true
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -i|--importance-csv)
            IMPORTANCE_CSV="$2"
            shift 2
            ;;
        -d|--dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num-roles)
            NUM_ROLES="$2"
            shift 2
            ;;
        -p|--max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$IMPORTANCE_CSV" ]]; then
    log "ERROR: Importance CSV file not found: $IMPORTANCE_CSV"
    log "Please run the main experiment first to generate importance scores."
    exit 1
fi

if [[ ! -d "$EVAL_DATASET" ]]; then
    log "ERROR: Evaluation dataset directory not found: $EVAL_DATASET"
    exit 1
fi

if [[ ! "$NUM_ROLES" =~ ^[1-7]$ ]]; then
    log "ERROR: Number of roles must be between 1 and 7, got: $NUM_ROLES"
    exit 1
fi

log "Starting DyLAN MMLU Evaluation"
log "Model: $MODEL"
log "Importance CSV: $IMPORTANCE_CSV"
log "Dataset: $EVAL_DATASET"
log "Output: $OUTPUT_DIR"
log "Roles per question: $NUM_ROLES"
log "Max parallel jobs: $MAX_PARALLEL"

mkdir -p "$OUTPUT_DIR"

# Create Python script for role selection and evaluation
cat > "$OUTPUT_DIR/evaluate_roles.py" << 'EOF'
#!/usr/bin/env python3
import sys
import pandas as pd
import json
import os
import ast
from pathlib import Path

def select_top_roles(importance_csv, num_roles=4):
    """Select top N roles per test based on importance scores"""
    df = pd.read_csv(importance_csv)
    
    # Role columns (excluding filename, acc, resp, q_cnt)
    role_cols = [col for col in df.columns if col.endswith('_imp')]
    role_names = [col.replace('_imp', '') for col in role_cols]
    
    selected_roles = {}
    
    for _, row in df.iterrows():
        filename = row['filename']
        
        # Get importance scores for this test
        scores = [(role_names[i], row[role_cols[i]]) for i in range(len(role_cols))]
        
        # Sort by importance score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N roles
        top_roles = [role for role, _ in scores[:num_roles]]
        selected_roles[filename] = top_roles
    
    return selected_roles, role_names

def run_evaluation(test_file, selected_roles, model, roles_list, output_dir):
    """Run evaluation for a single test file"""
    filename = Path(test_file).stem
    
    if filename not in selected_roles:
        print(f"Warning: No importance data for {filename}, skipping")
        return None
    
    # Get selected roles for this test
    test_roles = selected_roles[filename]
    
    # Create roles string for this specific test
    test_roles_str = str(test_roles)
    
    # Output files
    exp_name = f"eval_{filename}"
    out_dir = os.path.join(output_dir, f"{exp_name}_{len(test_roles)}roles")
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, f"{filename}_eval.log")
    result_file = os.path.join(out_dir, f"{filename}_eval.txt")
    
    # Check if already processed
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        print(f"Skipping {filename} (already processed)")
        return result_file
    
    print(f"Evaluating {filename} with roles: {test_roles}")
    
    # Import the main evaluation script
    sys.path.append(os.path.dirname(__file__))
    from llmlp_listwise_mmlu import main as eval_main
    
    try:
        # Run evaluation
        eval_main(test_file, filename, model, exp_name, test_roles_str)
        
        # Move results to our output directory
        expected_result = f"{exp_name}_{len(test_roles)}3.txt"
        if os.path.exists(expected_result):
            os.rename(expected_result, result_file)
        
        return result_file
        
    except Exception as e:
        print(f"Error evaluating {filename}: {e}")
        return None

def calculate_metrics(result_files, importance_csv):
    """Calculate evaluation metrics"""
    df_importance = pd.read_csv(importance_csv)
    
    total_questions = 0
    total_correct = 0
    total_responses = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    results_by_test = {}
    
    for result_file in result_files:
        if not result_file or not os.path.exists(result_file):
            continue
            
        filename = Path(result_file).stem.replace('_eval', '')
        
        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 6:
                # Parse results (same format as original)
                accs = ast.literal_eval(lines[0].strip())
                resp_cnts = ast.literal_eval(lines[1].strip())
                importances = ast.literal_eval(lines[2].strip())
                avg_importances = ast.literal_eval(lines[3].strip())
                prompt_tokens = int(lines[4].strip())
                completion_tokens = int(lines[5].strip())
                
                # Calculate metrics for this test
                test_questions = len(accs)
                test_correct = sum(accs)
                test_responses = sum(resp_cnts)
                
                total_questions += test_questions
                total_correct += test_correct
                total_responses += test_responses
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                results_by_test[filename] = {
                    'accuracy': test_correct / test_questions if test_questions > 0 else 0,
                    'questions': test_questions,
                    'correct': test_correct,
                    'responses': test_responses,
                    'avg_responses': test_responses / test_questions if test_questions > 0 else 0,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }
                
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")
            continue
    
    # Calculate overall metrics
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_responses_per_question = total_responses / total_questions if total_questions > 0 else 0
    
    return {
        'overall': {
            'accuracy': overall_accuracy,
            'total_questions': total_questions,
            'total_correct': total_correct,
            'total_responses': total_responses,
            'avg_responses_per_question': avg_responses_per_question,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens
        },
        'by_test': results_by_test
    }

def main():
    if len(sys.argv) != 6:
        print("Usage: evaluate_roles.py <importance_csv> <dataset_dir> <model> <num_roles> <output_dir>")
        sys.exit(1)
    
    importance_csv = sys.argv[1]
    dataset_dir = sys.argv[2]
    model = sys.argv[3]
    num_roles = int(sys.argv[4])
    output_dir = sys.argv[5]
    
    print(f"Loading importance data from: {importance_csv}")
    selected_roles, all_roles = select_top_roles(importance_csv, num_roles)
    
    print(f"Found importance data for {len(selected_roles)} tests")
    print(f"Selected {num_roles} roles per test from: {all_roles}")
    
    # Find test files
    test_files = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            test_files.append(os.path.join(dataset_dir, file))
    
    print(f"Found {len(test_files)} test files")
    
    # Run evaluations
    result_files = []
    for test_file in test_files:
        filename = Path(test_file).stem
        if filename in selected_roles:
            result_file = run_evaluation(test_file, selected_roles, model, all_roles, output_dir)
            result_files.append(result_file)
    
    # Calculate and report metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    metrics = calculate_metrics(result_files, importance_csv)
    
    # Overall metrics
    overall = metrics['overall']
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['total_correct']}/{overall['total_questions']})")
    print(f"  Average responses per question: {overall['avg_responses_per_question']:.2f}")
    print(f"  Total tokens used: {overall['total_tokens']:,}")
    print(f"  Prompt tokens: {overall['total_prompt_tokens']:,}")
    print(f"  Completion tokens: {overall['total_completion_tokens']:,}")
    
    # Per-test metrics
    print(f"\nPER-TEST METRICS:")
    print(f"{'Test Name':<30} {'Accuracy':<10} {'Questions':<10} {'Avg Resp':<10} {'Tokens':<10}")
    print("-" * 80)
    
    for test_name, test_metrics in metrics['by_test'].items():
        print(f"{test_name:<30} {test_metrics['accuracy']:<10.4f} {test_metrics['questions']:<10} "
              f"{test_metrics['avg_responses']:<10.2f} {test_metrics['total_tokens']:<10,}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"evaluation_results_{num_roles}roles.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save role selection info
    roles_file = os.path.join(output_dir, f"selected_roles_{num_roles}roles.json")
    with open(roles_file, 'w') as f:
        json.dump(selected_roles, f, indent=2)
    
    print(f"Role selection info saved to: {roles_file}")

if __name__ == "__main__":
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_roles.py"

# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------
log "Running evaluation with $NUM_ROLES roles per question..."

python "$OUTPUT_DIR/evaluate_roles.py" \
    "$IMPORTANCE_CSV" \
    "$EVAL_DATASET" \
    "$MODEL" \
    "$NUM_ROLES" \
    "$OUTPUT_DIR"

log "Evaluation completed!"
log "Results saved in: $OUTPUT_DIR"

if [[ -f "$IMPORTANCE_CSV" ]]; then
    log "Generating comparison report..."
    
    cat > "$OUTPUT_DIR/compare_with_full.py" << 'EOF'
#!/usr/bin/env python3
import sys
import pandas as pd
import json

def compare_results(importance_csv, eval_results_file):
    """Compare reduced-role results with full 7-role results"""
    
    # Load importance data (represents full 7-role results)
    df_importance = pd.read_csv(importance_csv)
    
    # Load evaluation results
    with open(eval_results_file, 'r') as f:
        eval_results = json.load(f)
    
    # Calculate full 7-role metrics
    total_questions_full = df_importance['q_cnt'].sum()
    total_correct_full = sum(df_importance['acc'] * df_importance['q_cnt'])
    total_responses_full = df_importance['resp'].sum()
    
    accuracy_full = total_correct_full / total_questions_full if total_questions_full > 0 else 0
    avg_responses_full = total_responses_full / total_questions_full if total_questions_full > 0 else 0
    
    # Get reduced-role metrics
    accuracy_reduced = eval_results['overall']['accuracy']
    avg_responses_reduced = eval_results['overall']['avg_responses_per_question']
    
    print("="*60)
    print("COMPARISON: Full 7-Role vs Reduced-Role Performance")
    print("="*60)
    print(f"{'Metric':<25} {'Full (7 roles)':<15} {'Reduced':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {accuracy_full:<15.4f} {accuracy_reduced:<15.4f} {accuracy_reduced-accuracy_full:<15.4f}")
    print(f"{'Avg Responses':<25} {avg_responses_full:<15.2f} {avg_responses_reduced:<15.2f} {avg_responses_reduced-avg_responses_full:<15.2f}")
    
    # Calculate efficiency metrics
    efficiency_gain = (avg_responses_full - avg_responses_reduced) / avg_responses_full * 100
    accuracy_change = (accuracy_reduced - accuracy_full) / accuracy_full * 100
    
    print(f"\nEFFICIENCY ANALYSIS:")
    print(f"  Response reduction: {efficiency_gain:.1f}%")
    print(f"  Accuracy change: {accuracy_change:+.1f}%")
    
    if efficiency_gain > 0 and accuracy_change >= -5:
        print(f"  ✓ Efficient: Reduced responses by {efficiency_gain:.1f}% with minimal accuracy loss")
    elif accuracy_change > 0:
        print(f"  ✓ Improved: Better accuracy with fewer responses")
    else:
        print(f"  ⚠ Trade-off: Reduced responses but accuracy decreased by {abs(accuracy_change):.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_with_full.py <importance_csv> <eval_results_json>")
        sys.exit(1)
    
    compare_results(sys.argv[1], sys.argv[2])
EOF

    chmod +x "$OUTPUT_DIR/compare_with_full.py"
    
    # Find the results file
    results_file=$(find "$OUTPUT_DIR" -name "evaluation_results_${NUM_ROLES}roles.json" | head -1)
    
    if [[ -n "$results_file" ]]; then
        python "$OUTPUT_DIR/compare_with_full.py" "$IMPORTANCE_CSV" "$results_file"
    else
        log "Warning: Could not find evaluation results file for comparison"
    fi
fi

log "Evaluation script completed successfully!"

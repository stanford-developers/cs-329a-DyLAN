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

    # Handle mismatch between _test and _val suffixes
    # Try exact match first, then try with _val suffix
    lookup_filename = filename
    if filename not in selected_roles:
        # Try replacing _test with _val
        if filename.endswith('_test'):
            lookup_filename = filename.replace('_test', '_val')
        elif not filename.endswith('_val'):
            lookup_filename = filename + '_val'

    if lookup_filename not in selected_roles:
        print(f"Warning: No importance data for {filename} (tried {lookup_filename}), skipping")
        return None

    # Get selected roles for this test (use lookup_filename which matched)
    test_roles = selected_roles[lookup_filename]

    print(f"Matched {filename} → {lookup_filename} in importance data")
    
    # Create roles string for this specific test
    test_roles_str = str(test_roles)
    
    # Output files
    exp_name = f"eval_{filename}"

    # Use the same folder name that llmlp_listwise_mmlu.py creates
    # Format: exp_name_Role1_Role2_Role3_Role4
    roles_str_clean = test_roles_str.replace(' ', '').replace('[', '').replace(']', '').replace(',', '_').replace("'", '')
    out_dir = os.path.join(output_dir, f"{exp_name}_{roles_str_clean}")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, f"{filename}_eval.log")
    result_file = os.path.join(out_dir, f"{filename}_eval.txt")
    
    # Check if already processed
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        print(f"Skipping {filename} (already processed)")
        return result_file
    
    print(f"Evaluating {filename} with roles: {test_roles}")

    # Run llmlp_listwise_mmlu.py as a subprocess
    import subprocess

    # Get the parent directory (where llmlp_listwise_mmlu.py is located)
    # This script is in evaluation_results/, llmlp_listwise_mmlu.py is in parent dir
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llmlp_script = os.path.join(script_dir, 'llmlp_listwise_mmlu.py')

    # Expected output files created by llmlp_listwise_mmlu.py (in the same out_dir)
    expected_txt = os.path.join(out_dir, f"{exp_name}_{len(test_roles)}3.txt")
    expected_json = os.path.join(out_dir, f"{exp_name}_{len(test_roles)}3.json")

    try:
        # Build command
        cmd = [
            'python', llmlp_script,
            test_file,           # QUERY_CSV
            exp_name,            # EXP_NAME
            model,               # MODEL
            exp_name,            # DIR_NAME
            test_roles_str       # ROLES
        ]

        # Run the command in the output directory so files are created there
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

        if result.returncode != 0:
            print(f"Error running evaluation for {filename}:")
            print(result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return None

        # Check if files were created successfully
        # expected_txt and expected_json already contain the full path
        if os.path.exists(expected_txt):
            # Rename the files to match our expected naming convention
            import shutil
            if expected_txt != result_file:
                shutil.move(expected_txt, result_file)

            # Also rename the JSON file if it exists
            json_result_file = result_file.replace('.txt', '.json')
            if os.path.exists(expected_json) and expected_json != json_result_file:
                shutil.move(expected_json, json_result_file)

            return result_file
        else:
            print(f"Warning: Expected output file not found: {expected_txt}")
            print(f"STDOUT: {result.stdout}")
            return None

    except Exception as e:
        print(f"Error evaluating {filename}: {e}")
        import traceback
        traceback.print_exc()
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
    if len(sys.argv) not in [6, 7]:
        print("Usage: evaluate_roles.py <importance_csv> <dataset_dir> <model> <num_roles> <output_dir> [max_parallel]")
        sys.exit(1)

    importance_csv = sys.argv[1]
    dataset_dir = sys.argv[2]
    model = sys.argv[3]
    num_roles = int(sys.argv[4])
    output_dir = sys.argv[5]
    max_parallel = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    
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
    print(f"Running evaluations with max {max_parallel} parallel jobs")

    # Filter test files to only those with importance data
    files_to_process = []
    for test_file in test_files:
        filename = Path(test_file).stem

        # Check if filename matches (with or without _test/_val conversion)
        lookup_filename = filename
        if filename not in selected_roles:
            if filename.endswith('_test'):
                lookup_filename = filename.replace('_test', '_val')
            elif not filename.endswith('_val'):
                lookup_filename = filename + '_val'

        if lookup_filename in selected_roles:
            files_to_process.append(test_file)

    print(f"Processing {len(files_to_process)} test files with importance data")

    # Run evaluations in parallel
    import concurrent.futures
    from functools import partial

    result_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Create partial function with fixed arguments
        eval_func = partial(run_evaluation,
                           selected_roles=selected_roles,
                           model=model,
                           roles_list=all_roles,
                           output_dir=output_dir)

        # Submit all jobs and collect futures
        future_to_file = {executor.submit(eval_func, test_file): test_file
                         for test_file in files_to_process}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            test_file = future_to_file[future]
            try:
                result_file = future.result()
                if result_file:
                    result_files.append(result_file)
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
                import traceback
                traceback.print_exc()
    
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

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

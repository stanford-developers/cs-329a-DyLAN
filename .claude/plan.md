# Plan: Enhance Simple Single-LLM Baseline with Bootstrap CI Metrics

## Objective
Modify the simple single-LLM baseline (`baseline_mmlu.py` + `exp_baseline.sh`) to output the same comprehensive metrics as the enhanced baseline and evaluation scripts, including:
- 95% Bootstrap Confidence Intervals for accuracy, API calls, and tokens
- Per-test and aggregated results in JSON format
- Meta-category breakdowns (STEM, Humanities, Social Sciences, Other)
- Professional output format matching the evaluation script

## Current State Analysis

### Simple Baseline Current Output
**Files:**
- `code/MMLU/baseline_mmlu.py` - Single LLM call per question
- `code/MMLU/exp_baseline.sh` - Parallel execution wrapper

**Current metrics (6-line .txt format):**
1. List of correctness booleans + accuracy
2. Total questions + avg responses per question (always 1.0)
3. Empty importance scores `[]`
4. Empty average importance `[]`
5. Total prompt tokens
6. Total completion tokens

**Current limitations:**
- No confidence intervals
- No meta-category breakdown
- Simple aggregate-only statistics
- No structured JSON output with full details

### Target Output Format (from evaluation scripts)

**Per-test JSON:**
```json
{
  "test_name": "abstract_algebra_test",
  "subject": "abstract_algebra",
  "meta": "STEM",
  "questions": 100,
  "correct": 85,
  "responses": 100,
  "prompt_tokens": 12500,
  "completion_tokens": 3200
}
```

**Summary JSON with Bootstrap CIs:**
```json
{
  "notes": {
    "mode": "single_baseline",
    "n_boot": 1000,
    "temperature": 0.0,
    "model": "meta-llama/..."
  },
  "overall": {
    "accuracy": {"point": 0.7234, "ci95": [0.7012, 0.7456]},
    "api_calls": {"point": 14042, "ci95": [14042, 14042]},
    "tokens_in": {"point": 234567.0, "ci95": [230123.0, 239012.0]},
    "tokens_out": {"point": 45678.0, "ci95": [44123.0, 47234.0]}
  },
  "by_meta": {
    "STEM": { ... },
    "humanities": { ... },
    "social sciences": { ... },
    "other (business, health, misc.)": { ... }
  }
}
```

## Implementation Plan

### Approach: Create Post-Processing Script (Recommended)

Instead of modifying the core `baseline_mmlu.py` and `exp_baseline.sh` which work well, **create a new post-processing script** that:
1. Reuses the existing baseline execution infrastructure
2. Adds a post-processing step to compute CI metrics
3. Maintains backward compatibility with existing 6-line .txt format

**Advantages:**
- No risk of breaking existing baseline
- Can reuse existing code from evaluation scripts
- Clean separation of concerns
- Easy to run both old and new format

### Files to Create/Modify

#### 1. **New File: `code/MMLU/compute_baseline_metrics.py`**
**Purpose:** Post-process baseline results to compute bootstrap CIs

**Reuse from existing code:**
- Subject → meta-category mapping from `exp_mmlu_evaluation.sh` (lines 111-226)
- `bootstrap_ci()` function from `exp_mmlu_evaluation.sh` (lines 460-524)
- `print_block()` function from `exp_mmlu_evaluation.sh` (lines 526-544)
- JSON output structure from `exp_mmlu_single_llm.sh` (lines 558-597)

**Inputs:**
- Directory containing `*_baseline.txt` files (6-line format)
- Model name (for metadata)
- Number of bootstrap replicates (default: 1000)

**Processing logic:**
1. Read all `*_baseline.txt` files in output directory
2. Parse each file to extract:
   - Test name → subject → meta-category
   - Total questions (from line 2)
   - Correctness list (from line 1) → count correct answers
   - Responses (always equal to questions for baseline)
   - Prompt tokens (line 5)
   - Completion tokens (line 6)
3. Build pandas DataFrame with columns:
   - `test_name`, `subject`, `meta`, `questions`, `correct`, `responses`, `prompt_tokens`, `completion_tokens`
4. Apply bootstrap CI computation per meta-category and overall
5. Output results in JSON format

**Outputs:**
- `baseline_by_test.json` - Per-test detailed results
- `metrics_summary_baseline.json` - Aggregated metrics with 95% CIs
- Console output with formatted tables (like evaluation script)

#### 2. **Modify: `code/MMLU/exp_baseline.sh`**
**Changes:** Add optional post-processing step at the end

**Addition at end of script (after line 140):**
```bash
# ------------------------------------------------------------
# Optional: Compute bootstrap CI metrics
# ------------------------------------------------------------
if [[ "${COMPUTE_METRICS:-1}" == "1" ]]; then
  echo ""
  echo "Computing bootstrap confidence intervals..."
  python "$SCRIPT_DIR/compute_baseline_metrics.py" \
    "$OUT_DIR" \
    "$MODEL" \
    "${N_BOOT:-1000}"
fi
```

**Benefits:**
- Default behavior: compute metrics automatically
- Can disable with `COMPUTE_METRICS=0` if only want old format
- Maintains backward compatibility

### Implementation Details

#### Bootstrap CI Computation Strategy

**Reuse from `exp_mmlu_evaluation.sh:460-524`:**
```python
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    """
    Bootstrap across tests (block bootstrap).
    df must have columns: questions, correct, responses, prompt_tokens, completion_tokens
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Point estimates
    q_sum = df['questions'].sum()
    acc_point = df['correct'].sum() / q_sum
    api_point = df['responses'].sum()
    tin_point = df['prompt_tokens'].sum()
    tout_point = df['completion_tokens'].sum()

    # Bootstrap replicates
    acc_samps, api_samps, tin_samps, tout_samps = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # Sample with replacement
        s = df.iloc[idx]
        q = s['questions'].sum()
        acc_samps.append(s['correct'].sum() / q)
        api_samps.append(s['responses'].sum())
        tin_samps.append(s['prompt_tokens'].sum())
        tout_samps.append(s['completion_tokens'].sum())

    # 95% CI using percentile method
    return {
        'accuracy': {'point': acc_point, 'ci95': [np.percentile(acc_samps, 2.5), np.percentile(acc_samps, 97.5)]},
        'api_calls': {'point': int(api_point), 'ci95': [int(np.percentile(api_samps, 2.5)), int(np.percentile(api_samps, 97.5))]},
        'tokens_in': {'point': tin_point, 'ci95': [np.percentile(tin_samps, 2.5), np.percentile(tin_samps, 97.5)]},
        'tokens_out': {'point': tout_point, 'ci95': [np.percentile(tout_samps, 2.5), np.percentile(tout_samps, 97.5)]}
    }
```

#### Subject to Meta-Category Mapping

**Reuse from `exp_mmlu_evaluation.sh:111-226` and `exp_mmlu_single_llm.sh:111-189`:**
- Use exact same `SUBCATEGORY` dictionary (57 subjects)
- Use exact same `CATEGORIES` dictionary (4 meta-categories)
- Use same `subject_key_from_name()` function to strip `_test`, `_val`, etc.

#### Parsing Baseline .txt Files

**Format of `*_baseline.txt` (6 lines):**
```
[True, False, True, ...] 0.7234
100 1.0
[]
[]
12500
3200
```

**Parsing logic:**
```python
def parse_baseline_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Line 1: [True, False, ...] accuracy
    line1 = lines[0].strip()
    accs_str, acc_str = line1.rsplit(' ', 1)
    accs = eval(accs_str)  # [True, False, ...]

    # Line 2: total_questions avg_responses
    total_questions = int(lines[1].strip().split()[0])

    # Line 5-6: tokens
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())

    # Count correct
    correct = sum(accs)

    return {
        'questions': total_questions,
        'correct': correct,
        'responses': total_questions,  # Always 1:1 for baseline
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }
```

### Dependencies

**Required Python packages (already in codebase):**
- `pandas` - DataFrame operations
- `numpy` - Bootstrap sampling and percentiles
- `json` - JSON output
- Standard library: `os`, `sys`, `re`, `pathlib`

**No new dependencies needed!** All functionality exists in current codebase.

### Code Reuse Summary

| Component | Source File | Lines | Reuse Strategy |
|-----------|-------------|-------|----------------|
| `SUBCATEGORY` mapping | `exp_mmlu_evaluation.sh` | 111-219 | Copy directly |
| `CATEGORIES` mapping | `exp_mmlu_evaluation.sh` | 221-226 | Copy directly |
| `subject_key_from_name()` | `exp_mmlu_evaluation.sh` | 231-240 | Copy directly |
| `meta_for_subject()` | `exp_mmlu_evaluation.sh` | 245-252 | Copy directly |
| `bootstrap_ci()` | `exp_mmlu_evaluation.sh` | 460-524 | Adapt (remove `scale_tokens_by` param) |
| `print_block()` | `exp_mmlu_evaluation.sh` | 526-544 | Adapt (remove `mark_est_tokens` param) |
| JSON output structure | `exp_mmlu_single_llm.sh` | 558-597 | Use as template |

**Total code to write:** ~300 lines (mostly copied/adapted from existing code)

## Testing Strategy

1. **Run existing baseline** to generate baseline results:
   ```bash
   cd code/MMLU
   bash exp_baseline.sh
   ```

2. **Run new metrics computation** on existing results:
   ```bash
   python compute_baseline_metrics.py baseline_meta-llama_Llama-3_3-70B-Instruct-Turbo-Free/ meta-llama/Llama-3.3-70B-Instruct-Turbo-Free 1000
   ```

3. **Verify outputs:**
   - Check `baseline_by_test.json` has all 57 tests
   - Check `metrics_summary_baseline.json` has overall + 4 meta-categories
   - Verify CI ranges are reasonable (accuracy CIs should be ~0.01-0.05 wide)
   - Verify point estimates match aggregate statistics from current baseline

4. **Compare with enhanced baseline** (if available):
   - Run both `exp_baseline.sh` and `exp_mmlu_single_llm.sh`
   - Compare accuracy point estimates (should be similar if same model/data)
   - Compare JSON output structures (should match)

## Backward Compatibility

**Guaranteed:**
- Existing `.txt` files unchanged (still 6 lines)
- Existing `.json` files unchanged (per-question completions)
- Existing aggregate statistics script still works
- Can disable new metrics with `COMPUTE_METRICS=0`

**New additions:**
- `baseline_by_test.json` (new)
- `metrics_summary_baseline.json` (new)
- Console output with formatted CI tables (new)

## Alternative Approaches Considered

### Alternative 1: Modify baseline_mmlu.py directly
**Pros:** Single integrated script
**Cons:**
- Risk breaking existing baseline
- Mixes evaluation and metrics computation
- Harder to maintain

### Alternative 2: Replace with exp_mmlu_single_llm.sh
**Pros:** Already has all features
**Cons:**
- max_tokens=2 bug needs fixing
- Different code style/structure
- More complex (embedded Python in bash)
- Loses existing baseline's simplicity

### Alternative 3: Standalone metrics CLI tool
**Pros:** Reusable for any baseline results
**Cons:**
- Extra step for users
- Not integrated into workflow

**Chosen approach (post-processing) balances all concerns.**

## Timeline (No Time Estimates)

**Steps:**
1. Create `compute_baseline_metrics.py` with code reused from evaluation scripts
2. Test metrics computation on existing baseline results
3. Modify `exp_baseline.sh` to call metrics script
4. Run full baseline evaluation with new metrics
5. Verify output matches expected format
6. Document usage in README or inline help

## User Decisions (Confirmed)

1. **Keep the existing 6-line .txt format** ✓
   - Maintain backward compatibility
   - Add new JSON outputs alongside

2. **Metrics computation is automatic** ✓
   - Default behavior in `exp_baseline.sh`
   - Can disable with `COMPUTE_METRICS=0`

3. **Do NOT fix max_tokens=2 bug** ✓
   - That's a separate issue in `exp_mmlu_single_llm.sh`
   - Not part of this work

4. **No testing subset needed** ✓
   - Run on full dataset directly

5. **Output format matches evaluation files** ✓
   - Use same structure as `exp_mmlu_evaluation.sh`
   - Bootstrap CIs, per-test JSON, summary JSON, meta-category breakdowns

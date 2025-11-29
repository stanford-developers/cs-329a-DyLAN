#!/usr/bin/env python3
import argparse, json, os, sys, glob, ast, random, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def read_roles(roles_json_path):
    with open(roles_json_path, "r") as f:
        return json.load(f)

def roles_for(subject, roles_map):
    # keys may be either "subject" or "subject.csv"
    return roles_map.get(subject) or roles_map.get(subject + ".csv")

def run_one(args, csv_path, subject, roles, script_dir):
    """
    Runs llmlp_listwise_mmlu.py for a single subject.
    We run with cwd=OUT_DIR, pass '.' as outdir, then normalize outputs.
    """
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # compact JSON list, one argument (no spaces so bash won't split if used)
    roles_arg = json.dumps(roles, separators=(',', ':'))

    # logs that we always keep in OUT_DIR (even if llmlp writes into a subfolder)
    runner_log = os.path.join(out_dir, f"{subject}_43.run.log")

    # ensure absolute paths
    csv_path = os.path.abspath(csv_path)
    llmlp = os.path.join(script_dir, "llmlp_listwise_mmlu.py")

    cmd = [
        sys.executable, llmlp,
        csv_path, subject, args.model, ".", roles_arg  # outdir="." (under OUT_DIR)
    ]

    with open(runner_log, "w") as lf:
        lf.write(f"[START] {now()} subject={subject} model={args.model} roles={roles_arg}\n")
        lf.flush()
        p = subprocess.run(
            cmd,
            cwd=out_dir,                    # <— force all outputs under OUT_DIR (or its subdirs)
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
        )
        lf.write(f"[END]   {now()} subject={subject} rc={p.returncode}\n")

    # Normalize: if llmlp created a role-suffixed subdir, move {subj}_43.* back to OUT_DIR.
    # We search recursively under OUT_DIR (safe even with parallelism).
    moved = False
    for ext in ("txt", "log"):
        want = f"{subject}_43.{ext}"
        here = os.path.join(out_dir, want)
        if os.path.exists(here):
            continue
        found = None
        for p in glob.glob(os.path.join(out_dir, "**", want), recursive=True):
            found = p
            break
        if found and os.path.abspath(found) != os.path.abspath(here):
            try:
                shutil.move(found, here)
                moved = True
            except Exception:
                # fallback: copy then leave the original (we won’t rmdir in case others write there)
                shutil.copy2(found, here)

    return subject

def collect_bools_from_txt(path):
    with open(path, "r") as f:
        first = f.readline().strip()
    preds_str, acc_str = first.rsplit(" ", 1)
    bools = ast.literal_eval(preds_str)  # [True, False, ...]
    acc = float(acc_str)
    return bools, acc

def summarize(out_dir, bootstrap):
    files = sorted(glob.glob(os.path.join(out_dir, "*_43.txt")))
    if not files:
        print("\n[ERROR] No *_43.txt found in", out_dir, "(did the inner script write elsewhere?)")
        return False

    all_bools = []
    per_subject = []
    for p in files:
        bools, acc = collect_bools_from_txt(p)
        per_subject.append((os.path.basename(p), len(bools), acc))
        all_bools.extend(bools)

    N = len(all_bools)
    overall = sum(1 for b in all_bools if b) / float(N) if N else 0.0

    print("\n============================================================")
    print("MMLU — Evaluation with Local‑Judge‑Selected Roles")
    print("============================================================")
    print(f"Subjects evaluated : {len(per_subject)}")
    print(f"Questions total    : {N}")

    print("\nOVERALL (Local‑judge selection)")
    print("-------------------------------")
    print(f"Accuracy           : {overall:.4f}", end="")

    if bootstrap and N:
        reps = int(bootstrap)
        scores = []
        for _ in range(reps):
            s = sum(1 for _ in range(N) if all_bools[random.randrange(N)]) / float(N)
            scores.append(s)
        scores.sort()
        lo = scores[int(0.025 * reps)]
        hi = scores[int(0.975 * reps) - 1]
        print(f"  [95% CI {lo:.4f}, {hi:.4f}]")
    else:
        print()

    print("\nDetailed files in:", out_dir)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "openai/gpt-oss-20b"))
    ap.add_argument("--eval-dir", default=os.environ.get("EVAL_DIR"))
    ap.add_argument("--roles-json", default=os.environ.get("ROLES_JSON"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "3")))
    ap.add_argument("--bootstrap", type=int, default=int(os.environ.get("BOOTSTRAP", "1000")))
    ap.add_argument("--subjects", nargs="*", help="Optional subset of subject names to run")
    args = ap.parse_args()

    # Resolve defaults if env vars were not set
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not args.eval_dir:
        args.eval_dir = os.path.join(repo_root, "data", "MMLU", "evaluation")
    if not args.roles_json:
        args.roles_json = os.path.join(repo_root, "code", "MMLU", "standard_dylan", "mmlu_with_local_judge", "roles_top4.json")
    if not args.out_dir:
        args.out_dir = os.path.join(repo_root, "code", "MMLU", "standard_dylan", "mmlu_eval_local_judge")

    print(f"[info] MODEL={args.model}")
    print(f"[info] EVAL_DIR={args.eval_dir}")
    print(f"[info] ROLES_JSON={args.roles_json}")
    print(f"[info] OUT_DIR={args.out_dir}")
    print(f"[info] MAX_PARALLEL={args.max_parallel}")
    print(f"[info] BOOTSTRAP={args.bootstrap}")

    # Load roles map
    roles_map = read_roles(args.roles_json)

    # Build subject list from CSVs
    if args.subjects:
        subjects = args.subjects
        csvs = {s: os.path.join(args.eval_dir, f"{s}.csv") for s in subjects}
    else:
        all_csv = sorted(glob.glob(os.path.join(args.eval_dir, "*.csv")))
        csvs = {os.path.basename(p)[:-4]: p for p in all_csv}
        subjects = list(csvs.keys())

    # Validate roles and schedule runs
    todo = []
    for s in subjects:
        r = roles_for(s, roles_map)
        if not r:
            print(f"[WARN] No roles for {s} in {args.roles_json}; skipping.")
            continue
        todo.append((s, csvs[s], r))

    if not todo:
        print("[ERROR] No runnable subjects.")
        sys.exit(2)

    # Run with a small thread pool; each task calls python once (subprocess)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        for s, csvp, roles in todo:
            print(f">>> Eval {s}  roles={json.dumps(roles, separators=(',',':'))}")
            futures.append(ex.submit(run_one, args, csvp, s, roles, script_dir))

        fails = 0
        for f in as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                fails += 1
                print("[FAIL] task crashed:", e)
    print(f"All evaluation jobs finished (failures: {fails}).")

    ok = summarize(args.out_dir, args.bootstrap)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

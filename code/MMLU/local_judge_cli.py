# code/MMLU/local_judge_cli.py
#!/usr/bin/env python3
import argparse, sys
from local_judge import LocalJudge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    args = ap.parse_args()

    prompt = sys.stdin.read()
    # Expect the prompt to already contain Question/Options/Candidate lines.
    # We treat each "Candidate i:" block as a reply and the whole text above as question.
    # For quick testing you can paste the full prompt you used before.
    # Here we simply pass the entire prompt as the "question" and let the judge parse lists.
    # If you want strict behavior, use LocalJudge.score_replies(question, replies) directly.
    judge = LocalJudge(args.ckpt_dir)
    scores = judge.score_replies(prompt, [])
    print(scores)

if __name__ == "__main__":
    main()

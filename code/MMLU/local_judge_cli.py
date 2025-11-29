#!/usr/bin/env python3
import argparse, json
from local_judge import LocalJudge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--prompt-file", default="/dev/stdin")
    args = ap.parse_args()

    block = open(args.prompt_file, "r").read()
    # very small parser for the quick sanity prompt you used
    parts = block.split("Options:")
    q = parts[0].split("Question:")[-1].strip()
    rest = parts[1]
    opts_txt, rest2 = rest.split("Candidate 1:", 1)
    opts = {}
    for line in opts_txt.strip().splitlines():
        line=line.strip()
        if not line: continue
        k=line[1]; v=line[3:].strip()
        opts[k]=v
    cands=["Candidate 1:"+rest2]
    # split on "Candidate i:"
    out=[]
    idx=1
    while True:
        idx+=1
        key=f"Candidate {idx}:"
        if key in cands[0]:
            left, right = cands[0].split(key,1)
            out.append(left.strip())
            cands=[right]
        else:
            out.append(cands[0].strip())
            break
    cands=[x.split("Scores:")[0].strip() for x in out]

    judge = LocalJudge(args.ckpt_dir)
    scores = judge.score(q, opts, cands)
    print(json.dumps(scores))

if __name__=="__main__":
    main()

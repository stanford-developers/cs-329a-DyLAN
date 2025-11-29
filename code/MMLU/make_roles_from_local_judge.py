#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, glob

def load_aip(json_path):
    with open(json_path, "r") as f:
        obj = json.load(f)
    # tolerate both flattened and nested layouts
    aip = obj.get("aip", obj) if isinstance(obj, dict) else None
    if not isinstance(aip, dict):
        raise ValueError(f"bad AIP object in: {json_path}")
    return aip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Folder containing <subject>_<N>.json files from the local judge")
    ap.add_argument("--out", default=None, help="Output JSON (roles map). Default: <in-dir>/roles_top4.json")
    ap.add_argument("--topk", type=int, default=4, help="How many roles to use per subject")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out = args.out or os.path.join(in_dir, "roles_top4.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    roles_map = {}
    jsons = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not jsons:
        raise SystemExit(f"No *.json found under {in_dir}")

    for jp in jsons:
        base = os.path.basename(jp)
        subject = base.rsplit("_", 1)[0]  # strip _N
        aip = load_aip(jp)
        # sort by score desc, take topK
        top = [r for r,_ in sorted(aip.items(), key=lambda kv: kv[1], reverse=True)][:args.topk]
        roles_map[subject] = top

    with open(out, "w") as f:
        json.dump(roles_map, f, indent=2)

    print(f"[OK] wrote roles map for {len(roles_map)} subjects → {out}")
    # pretty print a couple of lines
    shown = 0
    for k in sorted(roles_map):
        print(f"{k}: {roles_map[k]}")
        shown += 1
        if shown >= 5:
            break

if __name__ == "__main__":
    main()

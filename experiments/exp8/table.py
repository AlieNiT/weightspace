"""Markdown table of evaluate.py results: python table.py <eval_dir> [label ...]"""

import glob
import json
import math
import os
import sys


def cell(s: dict) -> str:
    p, n = s["accuracy"], s["generations"]
    return f"{100 * p:.1f} ± {100 * math.sqrt(p * (1 - p) / n):.1f} | {100 * s['cut_share']:.0f}% | {s['mean_tokens']:,}"


def main(out: str, *labels: str) -> None:
    runs = {
        json.load(open(f))["label"]: json.load(open(f))
        for f in glob.glob(os.path.join(out, "*.json"))
        if not f.endswith(".in.json")
    }
    print("| model | thinking | MATH-500 acc | cut | tokens | GSM8K acc | cut | tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for k in labels or sorted(runs):
        if k in runs:
            r = runs[k]
            think = "on, 32k" if r.get("thinking", True) else f"off, {r.get('max_tokens', 8192) // 1024}k"
            print(f"| {k} | {think} | {cell(r['math500'])} | {cell(r['gsm8k'])} |")


if __name__ == "__main__":
    main(*sys.argv[1:])

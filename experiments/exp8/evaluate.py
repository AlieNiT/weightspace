"""Held-out evaluation: MATH-500 and GSM8K test with the Exp. 8 protocol (thinking on, 32,768 tokens).

python evaluate.py <label> <checkpoint dir | edit.npz> <out_dir> [--gsm8k 500] [--samples 1]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import build_ckpt, chat_prompt, generate, grade_rows, summary  # noqa: E402


def bench(n_gsm8k: int):
    from datasets import load_dataset

    rows = [
        {"id": r["unique_id"], "set": "math500", "problem": r["problem"], "answer": r["answer"]}
        for r in load_dataset("HuggingFaceH4/MATH-500", split="test")
    ]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    rows += [
        {
            "id": f"gsm8k_test_{i}",
            "set": "gsm8k",
            "problem": r["question"],
            "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for i, r in enumerate(gsm)
        if i < n_gsm8k
    ]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--gsm8k", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32768)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    base = glob.glob(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"
    )[0]
    ckpt = a.model
    if ckpt.endswith(".npz"):
        edits = dict(np.load(ckpt))
        ckpt = build_ckpt(base, os.path.join(a.out, f"ckpt_{a.label}"), edits)
    tok = AutoTokenizer.from_pretrained(base)
    rows = bench(a.gsm8k)
    items = [{"id": r["id"], "prompt": chat_prompt(tok, r["problem"])} for r in rows]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.label}.jsonl")
    t0 = time.time()
    if generate(ckpt, items, n=a.samples, max_tokens=a.max_tokens, out=path, gpu_mem=0.85).wait() != 0:
        raise SystemExit("generation failed")
    gen = grade_rows([json.loads(x) for x in open(path)], {r["id"]: r["answer"] for r in rows}, a.max_tokens)
    with open(path, "w") as f:
        for r in gen:
            f.write(json.dumps(r) + "\n")
    sets = {r["id"]: r["set"] for r in rows}
    res = {"label": a.label, "model": a.model, "seconds": round(time.time() - t0)}
    for s in ("math500", "gsm8k"):
        res[s] = summary([r for r in gen if sets[r["id"]] == s])
    json.dump(res, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

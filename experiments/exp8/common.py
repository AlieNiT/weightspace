"""Shared pieces of the Exp. 8 replication: prompts, grading, loop signals, checkpoint building, vLLM generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
# non-thinking mode settings from the Qwen3.5-0.8B model card
SAMPLING_NO_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")
HERE = os.path.dirname(os.path.abspath(__file__))


def chat_prompt(tok, problem: str, thinking: bool = True) -> str:
    msgs = [{"role": "user", "content": problem + "\n" + INSTRUCTION}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def last_boxed(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    i = max(text.rfind("\\boxed"), -1)
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k].strip() or None
    return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    from math_verify import parse, verify

    try:
        g, p = parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")
        return bool(g and p and verify(g, p))
    except Exception:
        return False


def loop_end(text: str, max_period: int = 200, min_repeats: int = 3, min_span: int = 40) -> bool:
    """The text ends with one block of words repeated verbatim at least 3 times (and over 40+ words)."""
    w = text.split()[:-1]
    n = len(w)
    for p in range(1, max_period + 1):
        reps = max(min_repeats, -(-min_span // p))
        if reps * p > n:
            break
        blk = w[n - p :]
        if all(w[n - (j + 1) * p : n - j * p] == blk for j in range(1, reps)):
            return True
    return False


def seed_of(item_id: str, sample: int) -> int:
    return int(hashlib.sha256(f"{item_id}:{sample}".encode()).hexdigest()[:8], 16)


def build_ckpt(base_dir: str, out_dir: str, edits: dict[str, np.ndarray]) -> str:
    """base checkpoint files with `edits` (name -> array) cast to bfloat16 in place of the originals."""
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(base_dir):
        src = os.path.join(base_dir, f)
        if not f.endswith(".safetensors") and os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, f))
    ix = os.path.join(base_dir, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(ix))["weight_map"].values())) if os.path.exists(ix) else ["model.safetensors"]
    done = set()
    for shard in shards:
        sd = load_file(os.path.join(base_dir, shard))
        for name in sd:
            if name in edits:
                a = torch.from_numpy(np.ascontiguousarray(edits[name])).to(sd[name].dtype)
                assert a.shape == sd[name].shape, (name, a.shape, sd[name].shape)
                sd[name] = a.contiguous()
                done.add(name)
        save_file(sd, os.path.join(out_dir, shard), metadata={"format": "pt"})
    missing = set(edits) - done
    assert not missing, missing
    return out_dir


def generate(
    ckpt: str, items: list[dict], *, n: int, max_tokens: int, out: str, gpu_mem: float = 0.85, sampling: dict | None = None
):
    """Run gen_worker.py in a fresh process (vLLM). items: {id, prompt}. Returns the rows written to `out`."""
    inp = out + ".in.json"
    job = {"items": items, "n": n, "max_tokens": max_tokens, "gpu_mem": gpu_mem, "sampling": sampling or SAMPLING}
    json.dump(job, open(inp, "w"))
    return subprocess.Popen([sys.executable, os.path.join(HERE, "gen_worker.py"), ckpt, inp, out])


def grade_rows(rows: list[dict], gold: dict[str, str], max_tokens: int) -> list[dict]:
    for r in rows:
        r["pred"] = last_boxed(r["text"])
        r["correct"] = is_correct(r["pred"], gold[r["id"]])
        r["cut"] = r["finish"] == "length" or r["n_tokens"] >= max_tokens
        r["loop_end"] = loop_end(r["text"]) if r["cut"] else False
    return rows


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["id"].split("_train")[0].split("_test")[0], []).append(r["correct"])
    return {
        "generations": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in by.items()},
        "mean_tokens": int(np.mean([r["n_tokens"] for r in rows])),
        "cut_share": round(sum(r["cut"] for r in rows) / n, 4),
        "loop_end_share": round(sum(r["loop_end"] for r in rows) / n, 4),
        "failures": {
            "wrong_answer": sum((not r["correct"]) and r["pred"] is not None for r in rows),
            "no_answer_finished": sum(r["pred"] is None and not r["cut"] for r in rows),
            "cut_at_max_tokens": sum(r["cut"] for r in rows),
        },
    }


def paired(orig: list[dict], edit: list[dict]) -> dict:
    a: dict[str, list] = {}
    b: dict[str, list] = {}
    for r in orig:
        a.setdefault(r["id"], []).append(r["correct"])
    for r in edit:
        b.setdefault(r["id"], []).append(r["correct"])
    ids = sorted(set(a) & set(b))
    d = np.array([np.mean(b[i]) - np.mean(a[i]) for i in ids])
    flips = {"wrong_to_right": 0, "right_to_wrong": 0}
    for i in ids:
        for x, y in zip(a[i], b[i], strict=False):
            flips["wrong_to_right"] += (not x) and y
            flips["right_to_wrong"] += x and (not y)
    return {
        "items": len(ids),
        "diff_edited_minus_original": round(float(d.mean()), 4),
        "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else None,
        **flips,
    }

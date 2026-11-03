"""Time and peak memory of weightspace hot paths on a Qwen-sized synthetic model (CPU).

python benchmarks/profile_hotpaths.py [--small] [--json out.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc

import numpy as np
import torch

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply


def model(vocab: int, hidden: int, layers: int, inter: int) -> torch.nn.Module:
    m = torch.nn.Module()
    m.embed = torch.nn.Embedding(vocab, hidden)
    for i in range(layers):
        blk = torch.nn.Module()
        for n, shape in {
            "q": (2 * hidden, hidden),
            "k": (hidden // 2, hidden),
            "v": (hidden // 2, hidden),
            "o": (hidden, 2 * hidden),
            "gate": (inter, hidden),
            "up": (inter, hidden),
            "down": (hidden, inter),
        }.items():
            setattr(blk, n, torch.nn.Linear(shape[1], shape[0], bias=False))
        blk.norm = torch.nn.LayerNorm(hidden)
        setattr(m, f"layer{i}", blk)
    return m.to(torch.bfloat16)


RESULTS: dict[str, dict] = {}


def bench(label: str, fn, repeat: int = 3):
    gc.collect()
    tracemalloc.start()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    RESULTS[label] = {"ms": round(best * 1e3, 2), "peak_mb": round(peak / 2**20, 1)}
    print(f"{label:48s} {best * 1e3:10.2f} ms  {peak / 2**20:9.1f} MB peak", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="tiny model, for a quick smoke run")
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.manual_seed(0)
    dims = (4096, 256, 2, 896) if a.small else (151936, 1024, 24, 3584)
    m = model(*dims)
    print(f"model: {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params bf16, torch threads={torch.get_num_threads()}")

    ws = wsp.WeightSpace.from_torch(m)
    emb, mlp, attn = "embed.weight", "layer0.gate.weight", "layer0.q.weight"
    for n in ws.names():
        ws[n]  # materialise float32 mirrors once, outside the timings
    s = wsp.Session(ws, ScriptedLLM([]), mode="fix", max_steps=1)
    t = s.tools

    bench("list_tensors", lambda: t["list_tensors"].fn(), 20)
    bench("intro prompt", lambda: s._intro("task"), 20)
    bench("describe mlp (cold)", lambda: (ws._describe_cache.clear(), ws.describe(mlp)), 3)
    bench("describe attn (cold)", lambda: (ws._describe_cache.clear(), ws.describe(attn)), 3)
    bench("describe embedding (cold)", lambda: (ws._describe_cache.clear(), ws.describe(emb)), 2)
    bench("read_tensor 64x64", lambda: t["read_tensor"].fn(name=mlp, index="0:64, 0:64"), 10)
    bench("first edit (snapshot) embedding", lambda: (ws._original.pop(emb, None), ws.edit(emb)), 3)
    for n in ws.names("layer*.weight")[:40]:
        ws.edit(n)
    print(f"touched tensors: {len(ws.touched())}, {sum(ws.numel(n) for n in ws.touched()) / 1e6:.0f}M weights")
    bench("write_tensor 8 values (+diff)", lambda: t["write_tensor"].fn(name=emb, values=[0.5] * 8, index="7, 0:8"), 5)
    bench("diff (one change)", ws.diff, 5)
    bench("patch", ws.patch, 3)
    bench("sync (bf16 flush)", ws.sync, 3)
    bench("checkpoint", lambda: ws.checkpoint("c"), 2)
    bench("restore + diff", lambda: t["restore"].fn(label="c"), 2)
    sb = s.sandbox
    bench("sandbox: W[emb][0,0] += 1", lambda: sb.run(f"W['{emb}'][0, 0] += 1"), 10)
    bench("sandbox: trivial expression", lambda: sb.run("1 + 1"), 50)

    small = wsp.WeightSpace.from_numpy({"w": np.random.randn(64, 64).astype(np.float32)})
    bench("render_space 4k weights", lambda: wsp.render_space(small), 10)
    text = wsp.render_space(small)
    bench("parse_weights 4k weights", lambda: wsp.parse_weights(text), 10)

    calls = [
        reply(None, call("read_tensor", name=mlp, index=f"{i}, 0:16"), call("describe_tensor", name=attn)) for i in range(30)
    ] + [reply(None, call("submit", answer="ok"))]
    bench("agent loop: 30 steps, 60 tool calls", lambda: wsp.read(ws, "q", llm=ScriptedLLM(list(calls)), max_steps=40), 3)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(RESULTS, f, indent=1)


if __name__ == "__main__":
    main()

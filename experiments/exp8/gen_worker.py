"""vLLM generation in its own process: gen_worker.py <ckpt> <in.json> <out.jsonl>."""

import json
import sys

from common import SAMPLING, seed_of
from vllm import LLM, SamplingParams


def main(ckpt, inp, out):
    job = json.load(open(inp))
    items, n, max_tokens = job["items"], job["n"], job["max_tokens"]
    llm = LLM(
        model=ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=job["gpu_mem"],
        max_model_len=max_tokens + 2048,
        seed=0,
        enable_prefix_caching=True,
        max_num_seqs=256,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    prompts, params, keys = [], [], []
    for it in items:
        for s in range(n):
            prompts.append(it["prompt"])
            params.append(SamplingParams(**job.get("sampling", SAMPLING), max_tokens=max_tokens, seed=seed_of(it["id"], s)))
            keys.append((it["id"], s))
    res = llm.generate(prompts, params)
    with open(out, "w") as f:
        for (i, s), r in zip(keys, res, strict=True):
            o = r.outputs[0]
            f.write(
                json.dumps({"id": i, "sample": s, "text": o.text, "n_tokens": len(o.token_ids), "finish": o.finish_reason}) + "\n"
            )


if __name__ == "__main__":
    main(*sys.argv[1:4])

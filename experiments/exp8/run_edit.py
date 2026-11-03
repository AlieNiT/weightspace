"""Exp. 8 replication: an LLM edits Qwen3.5-0.8B's weights to make it better at math (thinking on), through weightspace.

python run_edit.py --llm anthropic:claude-opus-5-5 --budget 15 --work /root/exp/work
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FROZEN, build_ckpt, chat_prompt, generate, grade_rows, paired, summary  # noqa: E402

import weightspace as wsp  # noqa: E402

GEN_MAX, TF_MAX_TEXTS, TF_MAX_TOKENS, TF_TOTAL = 256, 128, 16384, 262144

TASK = """You are given the weights of a language model, Qwen3.5-0.8B (text part: 24 layers of width 1,024; layers 3, 7, 11, 15,
19 and 23 are full-attention layers with 8 query heads and 2 key/value heads of size 256 and a gated output, the other 18
are Gated DeltaNet linear-attention layers with 16 heads of size 128; each layer has a SwiGLU MLP of width 3,584; RMSNorm;
vocabulary 248,320; input embedding and output layer tied). Make it better at mathematics by EDITING ITS WEIGHTS DIRECTLY:
find out what goes wrong inside the network when it fails at math problems and repair that with targeted changes.

HOW IT WILL BE MEASURED. After you finish, the edited and the unedited model are evaluated on held-out math benchmarks
(you never see them) with this protocol: the chat template with thinking enabled (the model reasons between <think> and
</think>, then answers); the user message is the problem, a newline and "Please reason step by step, and put your final
answer within \\boxed{}."; temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens; the
last \\boxed{} answer is graded against the reference. Only weight values can change (same names and shapes).

WHAT COUNTS. Edits that come from understanding the network: find what goes wrong when it fails, identify the components
responsible (neurons, heads, directions, circuits) and change them in a way you can justify with evidence. Repairing a
specific failure mechanism is in scope (e.g. if it often repeats itself and never finishes, fixing what drives that
repetition). Out of scope, and refused by the forward tools: the final norm (model.language_model.norm.weight), the
embedding / output layer (model.language_model.embed_tokens.weight), and any edit that only multiplies a whole tensor by
a constant. Global knobs (sharpening/flattening the whole output distribution) and making the model skip or close its
thinking regardless of the problem are not what we are looking for.

WHAT YOU HAVE.
1. The Python sandbox: `R[name]` (read-only current weights, float32), `W[name]` (writable; edit in place),
   `original(name)` (the unedited tensor), `diff()`, `checkpoint(label)`, `restore(label)`, `revert(name)`,
   `tok` (the tokenizer), `PRACTICE` (dict id -> {id, source, problem, answer, solution?, subject?, level?}),
   `GSM8K`, `MATH`, `AIME` (lists of ids), `WORK` (a read-write folder), `prompt_of(problem)` (the exact evaluation
   prompt text), `read_gen(k)` / `read_tf(k)` (load the results of forward pass k). numpy, torch (CPU) are available.
   Practice problems: 7,469 GSM8K train, 7,275 MATH train (with reference solutions), 714 AIME (answers only); none is in
   the evaluation.
2. forward_generate: the model answers practice problems with the evaluation protocol. model = "original", "edited" or
   "both" (paired: same problems and seeds for both weights; returns the paired accuracy difference and its standard
   error). problems x n <= 256 (with "both": <= 128 per model); max_tokens <= 32768. Texts, answers and grades are
   saved (read_gen(k)).
3. forward_teacher_forced: the model reads up to 128 raw texts (<= 16,384 tokens each, 262,144 in total) and returns
   per-token log-probs (and top-k), and optionally hidden states after given layers (0 = embeddings) and MLP down_proj
   inputs at given layers, saved as .npy (read_tf(k)). To score an answer, send prompt_of(problem) + answer as one text.
Each forward call costs 1 of your budget whatever its size. Small passes are noisy (about 50/sqrt(n) points).

HOW TO WORK. Diagnose first (wrong answer, no answer, cut at the token limit / looping). Localize causally: show that
changing a component changes the behaviour you predicted, with a dose-response. Measure with paired "both" passes on
problems you did not use to choose the edit, and keep an edit only if its gain is clearly larger than its standard
error. Print compact summaries, never whole arrays or long texts (at most ~300 characters of a generated text).

RULES. No gradient descent or any training, no backward passes/autograd/optimizers, no iterative fitting of weights to a
loss; do not run or simulate the network in the sandbox (only the forward tools run it).

When the budget is spent (or you are done), leave your best edit in place and call `submit` with what you changed, the
evidence, its measured paired gain with standard error, and why you expect it to help."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="anthropic:claude-opus-5-5")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--thinking", default="high")
    ap.add_argument("--work", default="/root/exp/work")
    ap.add_argument("--practice", default="/root/exp/practice")
    ap.add_argument("--gpu-mem", type=float, default=0.8, help="vLLM memory share per probe call (split in two for paired runs)")
    a = ap.parse_args()
    work = a.work
    os.makedirs(work, exist_ok=True)

    base = glob.glob(
        os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*")
    )[0]
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    sd = {}
    for f in sorted(glob.glob(f"{base}/*.safetensors")):
        for k, v in load_file(f).items():
            if k.startswith("model.language_model."):
                sd[k] = v.float().numpy()
    ws = wsp.WeightSpace.from_numpy(sd)
    del sd

    practice = {}
    for name in ("gsm8k_train", "math_train", "aime"):
        for line in open(f"{a.practice}/{name}.jsonl"):
            r = json.loads(line)
            practice[r["id"]] = r
    ids = {s: [i for i in practice if i.startswith(s)] for s in ("gsm8k", "math", "aime")}
    gold = {i: str(r["answer"]) for i, r in practice.items()}
    k_counter = [0]

    def check_edits():
        for n in ws.touched():
            new, old = ws[n], ws.original(n)
            if np.array_equal(new, old):
                continue
            if n in FROZEN:
                raise ValueError(f"edit refused: {n} is out of scope; revert it first")
            nz = np.abs(old) > 1e-8
            ratio = new[nz] / old[nz]
            if nz.any() and np.allclose(new[~nz], 0) and np.ptp(ratio) < 1e-4 * max(1.0, abs(ratio.mean())):
                raise ValueError(f"edit refused: {n} is only a constant rescale of the whole tensor")

    def edited_tensors():
        return {n: ws[n] for n in ws.touched() if not np.array_equal(ws[n], ws.original(n))}

    hf = {}

    def hf_model():
        if not hf:
            m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
            params = dict(m.named_parameters())
            names = {}
            for n in ws.names():
                rest = n[len("model.language_model.") :]
                for cand in (f"model.{rest}", f"model.language_model.{rest}", rest):
                    if cand in params:
                        names[n] = cand
                        break
            hf.update(model=m, params=params, names=names, loaded=set())
        return hf

    def set_hf_weights(edited: bool):
        h = hf_model()
        want = set(edited_tensors()) if edited else set()
        for n in want | h["loaded"]:
            src = ws[n] if n in want else ws.original(n)
            with torch.no_grad():
                h["params"][h["names"][n]].copy_(torch.from_numpy(src))
        h["loaded"] = want

    @wsp.probe(
        parameters={
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["original", "edited", "both"]},
                "problems": {"type": "array", "items": {"type": "string"}, "description": "practice ids"},
                "n": {"type": "integer", "default": 1, "description": "samples per problem"},
                "max_tokens": {"type": "integer", "default": 32768},
            },
            "required": ["model", "problems"],
        }
    )
    def generate_(model, problems, n=1, max_tokens=32768):
        """Generation pass with the evaluation protocol on practice problems (see the task)."""
        if model not in ("original", "edited", "both"):
            raise ValueError("model must be original, edited or both")
        bad = [p for p in problems if p not in practice]
        if bad:
            raise ValueError(f"unknown practice ids: {bad[:5]}")
        cap = GEN_MAX // 2 if model == "both" else GEN_MAX
        if len(problems) * n > cap or max_tokens > 32768 or n < 1:
            raise ValueError(f"problems x n must be <= {cap} and max_tokens <= 32768")
        check_edits()
        k_counter[0] += 1
        k = k_counter[0]
        items = [{"id": p, "prompt": chat_prompt(tok, practice[p]["problem"])} for p in problems]
        runs = {}
        if model in ("original", "both"):
            runs["original"] = base
        if model in ("edited", "both"):
            runs["edited"] = build_ckpt(base, f"{work}/ckpt_edited", edited_tensors())
        mem = a.gpu_mem / len(runs)
        t0 = time.time()
        procs = {
            m: (generate(c, items, n=n, max_tokens=max_tokens, out=f"{work}/gen_{k}_{m}.jsonl", gpu_mem=mem))
            for m, c in runs.items()
        }
        for m, p in procs.items():
            if p.wait() != 0:
                k_counter[0] -= 1
                raise RuntimeError(f"generation failed for {m}")
        rows = {}
        for m in runs:
            rows[m] = grade_rows([json.loads(x) for x in open(f"{work}/gen_{k}_{m}.jsonl")], gold, max_tokens)
            with open(f"{work}/gen_{k}_{m}.jsonl", "w") as f:
                for r in rows[m]:
                    f.write(json.dumps(r) + "\n")
        out = {"pass": k, "model": model, "seconds": round(time.time() - t0)}
        out.update({m: summary(r) for m, r in rows.items()})
        if model == "both":
            out["paired"] = paired(rows["original"], rows["edited"])
        out["files"] = {m: f"{work}/gen_{k}_{m}.jsonl" for m in runs}
        return out

    @wsp.probe(
        parameters={
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["original", "edited"]},
                "texts": {"type": "array", "items": {"type": "string"}},
                "topk": {"type": "integer", "default": 0, "description": "top-k next-token log-probs (<= 20)"},
                "hidden_states": {"type": "array", "items": {"type": "integer"}, "description": "layers 0..24"},
                "mlp_act": {"type": "array", "items": {"type": "integer"}, "description": "layers 0..23"},
            },
            "required": ["model", "texts"],
        }
    )
    def teacher_forced(model, texts, topk=0, hidden_states=None, mlp_act=None):
        """Teacher-forced read of raw texts: per-token log-probs, optional top-k, hidden states and MLP activations."""
        hidden_states, mlp_act = hidden_states or [], mlp_act or []
        enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        if len(texts) > TF_MAX_TEXTS or max(map(len, enc)) > TF_MAX_TOKENS or sum(map(len, enc)) > TF_TOTAL:
            raise ValueError(f"at most {TF_MAX_TEXTS} texts of {TF_MAX_TOKENS} tokens, {TF_TOTAL} in total")
        stored = sum(map(len, enc)) * (len(hidden_states) * 1024 + len(mlp_act) * 3584)
        if stored > 2e9:
            raise ValueError("too many activations requested (> 2e9 values); ask for fewer layers or texts")
        check_edits()
        set_hf_weights(model == "edited")
        m = hf["model"]
        k_counter[0] += 1
        k = k_counter[0]
        d = f"{work}/tf_{k}"
        os.makedirs(d, exist_ok=True)
        cap, hooks = {}, []
        layers = m.model.layers
        for li in mlp_act:
            hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, li=li: cap.__setitem__(li, inp[0])))
        mean_lp = []
        t0 = time.time()
        try:
            for i, ids_ in enumerate(enc):
                x = torch.tensor([ids_], device="cuda")
                with torch.no_grad():
                    o = m(input_ids=x, output_hidden_states=bool(hidden_states), use_cache=False)
                lp = torch.log_softmax(o.logits[0, :-1].float(), -1)
                tlp = lp.gather(-1, x[0, 1:, None])[:, 0]
                np.save(f"{d}/{i}_logprobs.npy", tlp.cpu().numpy())
                mean_lp.append(float(tlp.mean()))
                if topk:
                    v, ix = lp.topk(min(int(topk), 20), -1)
                    np.save(f"{d}/{i}_topk_ids.npy", ix.cpu().numpy())
                    np.save(f"{d}/{i}_topk_logprobs.npy", v.cpu().numpy())
                if hidden_states:
                    np.save(
                        f"{d}/{i}_hidden.npy", torch.stack([o.hidden_states[li][0] for li in hidden_states]).half().cpu().numpy()
                    )
                if mlp_act:
                    np.save(f"{d}/{i}_mlp.npy", torch.stack([cap[li][0] for li in mlp_act]).half().cpu().numpy())
        finally:
            for h in hooks:
                h.remove()
        meta = {
            "pass": k,
            "model": model,
            "texts": len(texts),
            "tokens": [len(e) for e in enc],
            "hidden_layers": hidden_states,
            "mlp_layers": mlp_act,
            "topk": topk,
        }
        json.dump(meta, open(f"{d}/meta.json", "w"))
        return {
            **meta,
            "tokens": sum(meta["tokens"]),
            "dir": d,
            "mean_logprob_per_text": [round(v, 4) for v in mean_lp[:64]],
            "seconds": round(time.time() - t0),
        }

    generate_.name = "generate"

    def read_gen(k):
        return {
            os.path.basename(f)[len(f"gen_{k}_") : -6]: [json.loads(x) for x in open(f)]
            for f in glob.glob(f"{work}/gen_{k}_*.jsonl")
            if not f.endswith(".in.json")
        }

    def read_tf(k):
        d = f"{work}/tf_{k}"
        meta = json.load(open(f"{d}/meta.json"))
        out = {"meta": meta}
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy"):
                out[f[:-4]] = np.load(f"{d}/{f}", mmap_mode="r")
        return out

    ns = {
        "tok": tok,
        "PRACTICE": practice,
        "GSM8K": ids["gsm8k"],
        "MATH": ids["math"],
        "AIME": ids["aime"],
        "WORK": work,
        "original": lambda n: ws.original(n).copy(),
        "prompt_of": lambda problem: chat_prompt(tok, problem),
        "read_gen": read_gen,
        "read_tf": read_tf,
    }
    s = wsp.Session(
        ws,
        a.llm,
        mode="improve",
        probes=[generate_, teacher_forced],
        forward_budget=a.budget,
        namespace=ns,
        max_steps=a.steps,
        max_tool_chars=8000,
        sandbox_timeout=900,
        thinking=a.thinking,
        log_path=f"{work}/transcript.jsonl",
        verbose=True,
    )
    try:
        res = s.run(TASK)
        res.save(f"{work}/result")
        print(res)
    finally:
        edits = edited_tensors()
        np.savez(f"{work}/edit.npz", **edits)
        json.dump(
            {n: {"changed": int((edits[n] != ws.original(n)).sum()), "shape": list(edits[n].shape)} for n in edits},
            open(f"{work}/edit_manifest.json", "w"),
            indent=1,
        )
        print("usage", s.usage, "forward", sum(c.cost for c in s.forward_calls), "/", a.budget)


if __name__ == "__main__":
    main()

"""Improve a Hugging Face causal LM by editing weights (Experiment 8 style).

The LLM gets a Python kernel over the weights and a budget of forward passes:
one probe generates answers and reports how often they loop, another returns
only attention patterns (Experiment 7 style).

    python examples/hf_reduce_loops.py --model Qwen/Qwen2.5-0.5B-Instruct --llm openai:gpt-5 --budget 25
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import weightspace as wsp

PROMPTS = [
    "What is 17 * 23? Think step by step.",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Solve for x: 3x + 7 = 22.",
    "How many prime numbers are there below 30?",
]


def repetition(ids: list[int], n: int = 8) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--llm", required=True, help='e.g. "openai:gpt-5", or "dry-run" for a scripted smoke test')
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="runs/hf_reduce_loops")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()

    def chat(p):
        if tok.chat_template:
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        return p

    @wsp.probe
    def generate(prompts: list = None, max_new_tokens: int = args.max_new_tokens):
        """Generate greedy answers for up to 8 prompts (default: a fixed practice set). Returns each answer,
        its 8-gram repetition rate, and whether it hit the token limit."""
        prompts = (prompts or PROMPTS)[:8]
        out = []
        for p in prompts:
            enc = tok(chat(p), return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            ids = gen[0, enc.input_ids.shape[1] :].tolist()
            out.append(
                {
                    "prompt": p,
                    "answer": tok.decode(ids, skip_special_tokens=True)[-1500:],
                    "repetition": round(repetition(ids), 3),
                    "hit_limit": len(ids) >= max_new_tokens,
                }
            )
        return out

    @wsp.probe
    def attention(text: str, layer: int):
        """Attention weights of every head in one layer for a text you write (no outputs are shown)."""
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            att = model(**enc, output_attentions=True).attentions[layer][0]
        toks = tok.convert_ids_to_tokens(enc.input_ids[0])
        return {"tokens": toks, "attention[head][query][key]": att.round(decimals=3).tolist()}

    llm = args.llm
    if llm == "dry-run":
        from weightspace.llm import ScriptedLLM, call, reply

        llm = ScriptedLLM(
            [
                reply(None, call("list_tensors", pattern="mlp")),
                reply(None, call("forward_generate", prompts=PROMPTS[:1], max_new_tokens=32)),
                reply(None, call("forward_attention", text="2 + 2 = 4", layer=0)),
                reply(None, call("submit", answer="dry run", confidence=0)),
            ]
        )

    res = wsp.improve(
        model,
        "Make the model stop getting stuck in repetition loops on math questions, without hurting its answers.",
        llm=llm,
        context=f"{args.model}: {model.config.num_hidden_layers} layers, hidden size {model.config.hidden_size}. "
        f"Tokenizer vocab {len(tok)}. You may use `tok` (the tokenizer) in Python.",
        probes=[generate, attention],
        forward_budget=args.budget,
        namespace={"tok": tok},
        verbose=True,
    )
    print(res)
    res.save(args.out)


if __name__ == "__main__":
    main()

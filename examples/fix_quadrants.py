"""Fix a sabotaged classifier by editing its weights.

A small MLP learns which quadrant a 2D point is in. We then break it so it
never predicts class 3, and ask an LLM to find and fix the cause, with a
budget of 3 forward passes.

    python examples/fix_quadrants.py --llm openai:gpt-5
    python examples/fix_quadrants.py --llm anthropic:claude-sonnet-4-5
    python examples/fix_quadrants.py            # scripted LLM, no API key needed
"""

import argparse

import torch

import weightspace as wsp
from weightspace.llm import ScriptedLLM, call, reply


def train():
    torch.manual_seed(0)
    x = torch.randn(4096, 2)
    y = (x[:, 0] > 0).long() * 2 + (x[:, 1] > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model, x, y


def scripted():
    """What a good run could look like, so the example works offline."""
    return ScriptedLLM(
        [
            reply("Check behaviour first.", call("forward_accuracy")),
            reply(None, call("describe_tensor", name="2.bias"), call("read_tensor", name="2.bias")),
            reply(
                "Output bias for class 3 is far below the others; set it to their mean.",
                call("checkpoint", label="start"),
                call("python", code="b = W['2.bias']; b[3] = np.delete(b, 3).mean(); b"),
            ),
            reply(None, call("forward_accuracy")),
            reply(None, call("submit", answer="Class 3's output bias was -8; reset it to the others' mean.", confidence=0.9)),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", help='e.g. "openai:gpt-5"; omit to use a scripted LLM')
    ap.add_argument("--out", default="runs/fix_quadrants")
    args = ap.parse_args()

    model, x, y = train()
    with torch.no_grad():
        model[2].bias[3] -= 8.0  # the bug

    @wsp.probe
    def accuracy():
        """Per-class accuracy of the network on 4,096 held-out points."""
        with torch.no_grad():
            pred = model(x).argmax(-1)
        return {f"class {c}": round(float((pred[y == c] == c).float().mean()), 3) for c in range(4)}

    print("before:", accuracy())
    res = wsp.fix(
        model,
        "The classifier never predicts class 3. Find out why and fix it.",
        llm=args.llm or scripted(),
        context=(
            "Network: Linear(2->16) -> ReLU -> Linear(16->4). Input: a point (x0, x1) ~ N(0, I). "
            "Output: logits for the quadrant: class = 2*(x0>0) + (x1>0)."
        ),
        probes=[accuracy],
        forward_budget=3,
        verbose=True,
    )
    print("\n" + str(res))
    print("after:", accuracy())
    res.save(args.out)


if __name__ == "__main__":
    main()

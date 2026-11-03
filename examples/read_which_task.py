"""Ask an LLM which task a network was trained on, from its weights alone (no tools).

Two identical MLPs learn different rules over 8 inputs; the LLM gets one
network's weights as text and must say which rule it computes.

    python examples/read_which_task.py --llm openai:gpt-5 --task parity
"""

import argparse

import torch

import weightspace as wsp

TASKS = {
    "majority": lambda x: (x.sum(1) > 4).long(),
    "first_bit": lambda x: x[:, 0].long(),
    "parity_of_first_two": lambda x: (x[:, 0] != x[:, 1]).long(),
    "last_bit_and_not_third": lambda x: ((x[:, 7] == 1) & (x[:, 2] == 0)).long(),
}


def train(rule, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, 2, (4096, 8)).float()
    y = rule(x)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(600):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--task", choices=list(TASKS), default="parity_of_first_two")
    args = ap.parse_args()

    model = train(TASKS[args.task])
    res = wsp.read(
        model,
        "What boolean function of the 8 input bits does this network compute? Answer as a short formula.",
        llm=args.llm,
        context="MLP 8 -> 8 (tanh) -> 2 logits. Inputs are bits in {0,1}. Output 1 means the function is true.",
        text_mode=True,
        precision=3,
    )
    print(res)
    print("truth:", args.task)


if __name__ == "__main__":
    main()

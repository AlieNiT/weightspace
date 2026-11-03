"""weightspace CLI: inspect a weights file, or ask an LLM about it."""

from __future__ import annotations

import argparse
import sys

from .agent import Session
from .render import describe
from .space import WeightSpace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="weightspace")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect", help="list tensors and stats (no LLM)")
    a.add_argument("path")
    a.add_argument("--pattern")
    a.add_argument("--describe", action="store_true")

    for mode in ("read", "fix", "write", "improve"):
        b = sub.add_parser(mode, help=f"{mode} a weights file with an LLM")
        b.add_argument("path")
        b.add_argument("task")
        b.add_argument("--llm", required=True, help='e.g. "openai:gpt-5" or "anthropic:claude-sonnet-4-5"')
        b.add_argument("--context", default="", help="text or @file with architecture/environment notes")
        b.add_argument("--text-mode", action="store_true", help="put all weights in the prompt, no tools")
        b.add_argument("--max-steps", type=int, default=60)
        b.add_argument(
            "--thinking", choices=["off", "low", "medium", "high", "xhigh", "max"], help="reasoning of the editing LLM"
        )
        b.add_argument("--no-thinking", dest="thinking", action="store_const", const="off", help="same as --thinking off")
        b.add_argument("--speed", help='inference speed of the editing LLM, e.g. "fast" (provider fast mode)')
        b.add_argument("--fast", dest="speed", action="store_const", const="fast", help="same as --speed fast")
        b.add_argument("--out", help="directory for transcript, result.json, patch.npz")
        b.add_argument("--save-weights", help="write edited weights to .safetensors/.npz")
        b.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args(argv)
    ws = WeightSpace.from_file(args.path)

    if args.cmd == "inspect":
        for n in ws.names(args.pattern):
            print(describe(n, ws[n], ws.dtype(n)) if args.describe else f"{n} {list(ws.shape(n))} {ws.dtype(n)}")
        print(f"{len(ws.names(args.pattern))} tensors, {ws.numel():,} weights")
        return 0

    context = args.context
    if context.startswith("@"):
        with open(context[1:]) as f:
            context = f.read()
    if args.cmd == "write":
        ws.zero_()
        ws.commit()
    s = Session(
        ws,
        args.llm,
        mode=args.cmd,
        context=context,
        max_steps=args.max_steps,
        thinking=args.thinking,
        speed=args.speed,
        verbose=args.verbose,
    )
    res = s.run_text(args.task) if args.text_mode else s.run(args.task)
    print(res)
    if args.out:
        res.save(args.out)
    if args.save_weights:
        ws.save(args.save_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())

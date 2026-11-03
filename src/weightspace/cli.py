"""weightspace CLI: inspect a weights file, or ask an LLM about it."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys

from . import __version__
from .agent import Session
from .events import EventLog
from .llm import ScriptedLLM
from .probes import Probe
from .render import describe
from .space import WeightSpace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="weightspace")
    ap.add_argument("--version", action="version", version=f"weightspace {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect", help="list tensors and stats (no LLM)")
    a.add_argument("path")
    a.add_argument("--pattern")
    a.add_argument("--describe", action="store_true")

    for mode in ("read", "fix", "write", "improve"):
        b = sub.add_parser(mode, help=f"{mode} a weights file with an LLM")
        b.add_argument("path")
        b.add_argument("task")
        b.add_argument("--llm", help='e.g. "openai:gpt-5" or "anthropic:claude-sonnet-4-5"')
        b.add_argument("--replay", metavar="TRANSCRIPT", help="replay a logged transcript.jsonl instead of calling an LLM")
        b.add_argument("--probe", action="append", default=[], metavar="FILE.py:FN", help="forward-pass probe (repeatable)")
        b.add_argument("--forward-budget", type=int, default=0)
        b.add_argument("--price", help="USD per million tokens as INPUT,OUTPUT[,CACHED_INPUT[,CACHE_WRITE]]")
        b.add_argument("--max-cost", type=float, help="cost budget in USD (needs --price)")
        b.add_argument("--max-tokens", type=int, help="token budget for the whole run")
        b.add_argument("--max-time", type=float, help="time budget in seconds")
        b.add_argument("--events", metavar="FILE.jsonl", help="write every run event to a JSONL file")
        b.add_argument("--json", action="store_true", help="print result.json instead of the text summary")
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

    if not args.llm and not args.replay:
        ap.error("--llm or --replay is required")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    llm = ScriptedLLM.from_transcript(args.replay) if args.replay else args.llm
    context = args.context
    if context.startswith("@"):
        with open(context[1:]) as f:
            context = f.read()
    if args.cmd == "write":
        ws.zero_()
        ws.commit()
    s = Session(
        ws,
        llm,
        mode=args.cmd,
        context=context,
        probes=[load_probe(p) for p in args.probe],
        forward_budget=args.forward_budget,
        pricing=tuple(float(x) for x in args.price.split(",")) if args.price else None,
        cost_budget=args.max_cost,
        token_budget=args.max_tokens,
        time_budget=args.max_time,
        callbacks=[EventLog(args.events)] if args.events else (),
        max_steps=args.max_steps,
        thinking=args.thinking,
        speed=args.speed,
        verbose=args.verbose,
    )
    res = s.run_text(args.task) if args.text_mode else s.run(args.task)
    print(json.dumps(res.to_dict(), indent=2, default=str) if args.json else res)
    if args.out:
        res.save(args.out)
    if args.save_weights:
        ws.save(args.save_weights)
    return 0 if res.ok else 2


def load_probe(spec: str) -> Probe:
    """`path/to/file.py:fn` or `package.module:fn`; `fn` may be a `Probe` or a plain function."""
    where, _, attr = spec.rpartition(":")
    if not where or not attr:
        raise SystemExit(f"--probe expects FILE.py:FN or module:FN, got {spec!r}")
    if where.endswith(".py") or os.sep in where:
        name = f"_weightspace_probe_{os.path.splitext(os.path.basename(where))[0]}"
        mod_spec = importlib.util.spec_from_file_location(name, where)
        if mod_spec is None or mod_spec.loader is None:
            raise SystemExit(f"cannot load {where!r}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    fn = getattr(module, attr)
    return fn if isinstance(fn, Probe) else Probe(fn)


if __name__ == "__main__":
    sys.exit(main())

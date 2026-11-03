"""Persistent Python namespace the LLM uses to analyse and edit weights.

This is a convenience for a trusted LLM, not a security boundary: code runs
in-process. The model object itself is never put in the namespace, so the
only way to run the network is through the budgeted forward tools.
"""

from __future__ import annotations

import ast
import contextlib
import io
import signal
import threading
import traceback


class SandboxTimeout(Exception):
    pass


class Sandbox:
    def __init__(self, namespace: dict | None = None, *, timeout: float | None = 120, max_output: int = 12000):
        self.ns: dict = {"__name__": "__weightspace__"}
        self.ns.update(namespace or {})
        self.timeout = timeout
        self.max_output = max_output

    def run(self, code: str) -> str:
        buf = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            last = None
            tail = tree.body[-1] if tree.body else None
            if isinstance(tail, ast.Expr):
                tree.body.pop()
                last = ast.Expression(tail.value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), self._deadline():
                exec(compile(tree, "<sandbox>", "exec"), self.ns)
                if last is not None:
                    value = eval(compile(last, "<sandbox>", "eval"), self.ns)
                    if value is not None:
                        print(repr(value))
        except SandboxTimeout:
            buf.write(f"\nTimeoutError: code ran longer than {self.timeout}s")
        except BaseException:
            tb = traceback.format_exc().splitlines()
            buf.write("\n".join(l for l in tb if 'File "' not in l or "<sandbox>" in l))
        out = buf.getvalue()
        if len(out) > self.max_output:
            half = self.max_output // 2
            out = out[:half] + f"\n... [{len(out) - self.max_output:,} chars truncated] ...\n" + out[-half:]
        return out or "(no output)"

    @contextlib.contextmanager
    def _deadline(self):
        if not self.timeout or threading.current_thread() is not threading.main_thread():
            yield
            return

        def handler(signum, frame):
            raise SandboxTimeout()

        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

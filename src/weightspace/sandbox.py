"""Persistent Python namespace the LLM uses to analyse and edit weights.

This is a convenience for a trusted LLM, not a security boundary: code runs
in-process. The model object itself is never put in the namespace, so the
only way to run the network is through the budgeted forward tools.
"""

from __future__ import annotations

import ast
import contextlib
import ctypes
import io
import signal
import threading
import traceback


class SandboxTimeout(Exception):
    pass


class Sandbox:
    """`no_grad=True` runs code under `torch.no_grad()` (when torch is installed), so autograd-based training of
    the weights fails instead of silently working. `timeout` works on the main thread (SIGALRM, where available)
    and on worker threads (an async exception raised in the running thread)."""

    def __init__(
        self,
        namespace: dict | None = None,
        *,
        timeout: float | None = 120,
        max_output: int = 12000,
        no_grad: bool = False,
    ):
        self.ns: dict = {"__name__": "__weightspace__"}
        self.ns.update(namespace or {})
        self.timeout = timeout
        self.max_output = max_output
        self.no_grad = no_grad
        self._lock = threading.Lock()

    def run(self, code: str) -> str:
        with self._lock:
            return self._run(code)

    def _guard(self):
        if not self.no_grad:
            return contextlib.nullcontext()
        try:
            import torch
        except ImportError:
            return contextlib.nullcontext()
        return torch.no_grad()

    def _run(self, code: str) -> str:
        buf = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            last = None
            tail = tree.body[-1] if tree.body else None
            if isinstance(tail, ast.Expr):
                tree.body.pop()
                last = ast.Expression(tail.value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), self._deadline(), self._guard():
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
        if not self.timeout:
            yield
            return
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
            with _alarm(self.timeout):
                yield
            return
        with _watchdog(self.timeout):
            yield


@contextlib.contextmanager
def _alarm(seconds: float):
    def handler(signum, frame):
        raise SandboxTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@contextlib.contextmanager
def _watchdog(seconds: float):
    """Raise SandboxTimeout in the current thread after `seconds` (checked between bytecodes, so a single long
    C call such as a huge matmul finishes first)."""
    tid = threading.get_ident()
    done = threading.Event()
    fired = threading.Event()

    def fire():
        if not done.wait(seconds):
            fired.set()
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(SandboxTimeout))

    t = threading.Thread(target=fire, daemon=True, name="weightspace-sandbox-watchdog")
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join()
        if fired.is_set():  # clear a pending async exception that has not been delivered yet
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)

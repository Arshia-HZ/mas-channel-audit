"""Run a two-argument comparison in a worker process that can be abandoned.

``math_verify`` enforces its timeouts with SIGALRM, which Windows lacks, so a
symbolic answer pair can hang a long job forever there. ``GuardedCall`` runs the
comparison in a persistent worker; if a call exceeds ``timeout`` the worker is
killed and replaced, and the call counts as "not equivalent". A comparison that
raises is an error, never a silent False.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp


def _serve(conn, target: str) -> None:
    module, name = target.split(":")
    func = getattr(importlib.import_module(module), name)
    while True:
        try:
            a, b = conn.recv()
        except EOFError:
            return
        try:
            conn.send(("ok", func(a, b)))
        except Exception as exc:  # reported to the caller, never swallowed
            conn.send(("err", repr(exc)))


class GuardedCall:
    def __init__(self, target: str, timeout: float) -> None:
        self.target = target
        self.timeout = timeout
        self.timeouts = 0
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None

    def _start(self) -> None:
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_serve, args=(child, self.target), daemon=True
        )
        self._proc.start()

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(timeout=5)
        self._proc = self._conn = None

    def __call__(self, a: str, b: str) -> bool:
        if self._proc is None:
            self._start()
        try:
            self._conn.send((a, b))
            if not self._conn.poll(self.timeout):
                self.timeouts += 1
                self.close()
                return False
            kind, value = self._conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as exc:
            self.close()
            raise RuntimeError(f"comparison worker died: {exc!r}") from exc
        if kind == "err":
            raise RuntimeError(f"comparison raised {value}")
        return bool(value)

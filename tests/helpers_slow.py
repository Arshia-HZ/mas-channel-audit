"""Importable by a worker process: a comparison that can hang or crash on demand."""

import time


def equal_or_hang(a: str, b: str) -> bool:
    if a == "hang":
        time.sleep(600)
    if a == "boom":
        raise ValueError("boom")
    return a == b

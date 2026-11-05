"""A hung math_verify call must time out instead of freezing a long job."""

from __future__ import annotations

import sys
import time

import pytest

from channel_audit.mv_guard import GuardedCall


def test_a_normal_call_returns_its_value():
    guard = GuardedCall("helpers_slow:equal_or_hang", timeout=5.0)
    try:
        assert guard("1", "1") is True
        assert guard("1", "2") is False
    finally:
        guard.close()


def test_a_hanging_call_times_out_as_false_and_the_worker_is_replaced():
    guard = GuardedCall("helpers_slow:equal_or_hang", timeout=1.0)
    try:
        t0 = time.time()
        assert guard("hang", "x") is False  # timed out
        assert time.time() - t0 < 8
        assert guard.timeouts == 1
        assert guard("2", "2") is True  # a fresh worker serves the next call
    finally:
        guard.close()


def test_a_crashing_worker_is_an_error_not_a_silent_false():
    guard = GuardedCall("helpers_slow:equal_or_hang", timeout=5.0)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            guard("boom", "x")
    finally:
        guard.close()


def test_the_real_comparison_agrees_with_the_direct_one():
    from channel_audit.grade import _mv_equal_direct

    guard = GuardedCall("channel_audit.grade:_mv_equal_direct", timeout=30.0)
    try:
        for a, b in (("0.5", r"\frac{1}{2}"), ("3", "4"), ("x+1", "1+x")):
            assert guard(a, b) is _mv_equal_direct(a, b)
    finally:
        guard.close()


def test_grading_uses_the_guard_on_windows_only():
    import channel_audit.grade as grade

    assert (grade._GUARD is not None) == (sys.platform == "win32")

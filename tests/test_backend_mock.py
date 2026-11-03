"""The mock backend is deterministic and plants known arm effects."""

from __future__ import annotations

from channel_audit.backends.base import GenRequest
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.extract import extract_boxed, extract_confidence
from channel_audit.grade import is_correct

PROBLEMS = fake_problems(4)
WORLD = MockWorld.from_problems(PROBLEMS)


def req(seed=1, arm="round0", pid=None, content="Solve it.", max_tokens=512):
    pid = pid or PROBLEMS[0].problem_id
    return GenRequest(
        messages=[{"role": "user", "content": content}],
        seed=seed,
        temperature=0.7,
        max_tokens=max_tokens,
        meta={"problem_id": pid, "arm": arm, "replicate": 0, "agent": 0},
    )


def test_same_request_same_output_different_seed_different_output():
    mock = MockBackend(WORLD)
    a, b = mock.generate([req(seed=5), req(seed=5)])
    assert a == b
    assert mock.generate([req(seed=6)])[0].text != a.text


def test_result_has_boxed_answer_confidence_and_token_counts():
    r = MockBackend(WORLD).generate([req(content="one two three")])[0]
    assert extract_boxed(r.text) is not None
    assert extract_confidence(r.text) is not None
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 3
    assert r.completion_tokens == len(r.text.split())


def test_max_tokens_truncates_and_reports_length():
    r = MockBackend(WORLD).generate([req(max_tokens=3)])[0]
    assert r.finish_reason == "length"
    assert r.completion_tokens == 3
    assert extract_boxed(r.text) is None


def test_planted_effect_shifts_accuracy():
    gold = PROBLEMS[0].gold
    pid = PROBLEMS[0].problem_id
    up = MockBackend(WORLD, effects={"x": 0.4})
    flat = MockBackend(WORLD, effects={"x": 0.0})
    reqs = [req(seed=s, arm="x", pid=pid) for s in range(600)]
    hit = lambda b: sum(  # noqa: E731
        is_correct(extract_boxed(r.text), gold) for r in b.generate(reqs)
    )
    assert hit(up) - hit(flat) > 0.3 * 600


def test_world_can_pin_base_accuracy_to_one():
    mock = MockBackend(MockWorld.from_problems(PROBLEMS, base_p=1.0))
    out = mock.generate([req(seed=s) for s in range(50)])
    assert all(is_correct(extract_boxed(r.text), PROBLEMS[0].gold) for r in out)


def _failure_req(leak_form_seed=1):
    solution = "Step 1. [trace 9]\n\\boxed{42}\nConfidence: 50"
    return req(seed=leak_form_seed, arm="failure_msg", content=solution)


def test_failure_message_leaks_the_reviewed_answer_when_asked():
    mock = MockBackend(WORLD, leak_rate=1.0, leak_forms=("claim",))
    text = mock.generate([_failure_req()])[0].text
    assert "42" in text


def test_failure_message_is_clean_when_leak_rate_is_zero():
    mock = MockBackend(WORLD, leak_rate=0.0)
    text = mock.generate([_failure_req()])[0].text
    assert "42" not in text and "boxed" not in text

"""Gate G4: a random sample of failure messages, labelled by hand."""

from __future__ import annotations

import json

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.failure_review import (
    G4_MIN_NOT_VAGUE,
    LABELS,
    N_SAMPLE,
    format_label_sheet,
    g4_check,
    load_labels,
    sample_failure_messages,
    write_label_template,
)
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards


@pytest.fixture(scope="module")
def ctxs(tmp_path_factory):
    root = tmp_path_factory.mktemp("fr")
    problems = fake_problems(16)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=8,
        log=lambda s: None,
    )
    return [open_run(d) for d in sorted(root.iterdir())]


def test_the_plan_samples_forty_messages_and_needs_half_not_vague():
    assert N_SAMPLE == 40 and G4_MIN_NOT_VAGUE == 0.5
    assert LABELS == ("useful", "vague", "wrong")


def test_the_sample_is_random_reproducible_and_without_repeats(ctxs):
    a = sample_failure_messages(ctxs, n=10, seed=1)
    b = sample_failure_messages(ctxs, n=10, seed=1)
    c = sample_failure_messages(ctxs, n=10, seed=2)
    assert [x["record_key"] for x in a] == [x["record_key"] for x in b]
    assert [x["record_key"] for x in a] != [x["record_key"] for x in c]
    assert len({x["record_key"] for x in a}) == 10


def test_each_item_shows_what_the_reader_saw_and_what_was_reviewed(ctxs):
    item = sample_failure_messages(ctxs, n=3, seed=0)[0]
    assert {"record_key", "question", "gold", "reviewed_solution", "message"} <= set(
        item
    )
    ctx = next(
        c for c in ctxs if any(p.problem_id == item["problem_id"] for p in c.problems)
    )
    stored = ctx.leakage.get(item["record_key"])
    assert item["message"] == stored.sanitized_text  # exactly the delivered text
    assert "boxed" not in item["message"]


def test_the_sample_never_looks_at_correctness(ctxs, monkeypatch):
    import channel_audit.failure_review as fr

    # the module must not even import the grader to choose examples
    assert not hasattr(fr, "is_correct") and not hasattr(fr, "grade_generation")


def test_asking_for_more_than_exist_returns_them_all(ctxs):
    total = sum(len(c.leakage) for c in ctxs)
    assert len(sample_failure_messages(ctxs, n=10_000, seed=0)) == total


def test_the_label_sheet_defines_the_three_labels_and_lists_every_item(ctxs):
    items = sample_failure_messages(ctxs, n=5, seed=0)
    text = format_label_sheet(items)
    for label in LABELS:
        assert label in text
    for item in items:
        assert item["record_key"] in text


def test_the_template_is_written_blank_and_never_overwritten(ctxs, tmp_path):
    items = sample_failure_messages(ctxs, n=4, seed=0)
    path = tmp_path / "labels.json"
    write_label_template(items, path)
    data = json.loads(path.read_text())
    assert set(data) == {i["record_key"] for i in items} and set(data.values()) == {""}
    with pytest.raises(FileExistsError):
        write_label_template(items, path)


def test_labels_must_be_valid_and_complete(ctxs, tmp_path):
    items = sample_failure_messages(ctxs, n=4, seed=0)
    path = tmp_path / "labels.json"
    write_label_template(items, path)
    with pytest.raises(ValueError, match="unlabelled"):
        load_labels(path, items)
    data = {i["record_key"]: "useful" for i in items}
    data[items[0]["record_key"]] = "great"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="great"):
        load_labels(path, items)


@pytest.mark.parametrize(
    ("labels", "passed"),
    [
        (["useful"] * 10 + ["vague"] * 10, True),  # exactly half is enough
        (["useful"] * 9 + ["vague"] * 11, False),
        (["wrong"] * 12 + ["vague"] * 8, True),  # a wrong message is not vague
        (["vague"] * 20, False),
    ],
)
def test_g4_passes_when_at_least_half_are_not_vague(labels, passed):
    labelled = {f"k{i}": label for i, label in enumerate(labels)}
    r = g4_check(labelled)
    assert r["passed"] is passed
    assert r["n"] == 20 and r["not_vague"] == sum(label != "vague" for label in labels)


def test_g4_refuses_an_empty_label_set():
    with pytest.raises(ValueError):
        g4_check({})

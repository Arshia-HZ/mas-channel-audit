"""Exploratory figures: the numbers behind them, and the files they write."""

from __future__ import annotations

import json

import pytest

from channel_audit.analysis import figures as F


def analysis():
    """The shape of analysis.json that the accuracy-versus-tokens figure reads."""
    arms = {
        "self_reflect": {
            "accuracy": 0.70,
            "tokens": 1000.0,
            "matched_k": 2.0,
            "gain": -0.02,
        },
        "placebo": {
            "accuracy": 0.60,
            "tokens": 1500.0,
            "matched_k": 3.0,
            "gain": -0.10,
        },
    }
    return {
        "model_id": "mock/model",
        "analyses": {"main": {"arms": arms}},
        "sc_curve": {"1": 0.60, "2": 0.68, "3": 0.72},  # keys are strings after JSON
    }


def decomposition():
    def ci(m):
        return {"mean": m, "ci95_low": m - 0.01, "ci95_high": m + 0.01, "n_problems": 5}

    shares = {
        "wrong_to_correct": ci(0.10),
        "correct_to_wrong": ci(0.05),
        "wrong_to_other_wrong": ci(0.02),
        "unchanged_correct": ci(0.5),
        "unchanged_wrong": ci(0.33),
    }
    part = {
        "total": ci(0.2),
        "spontaneous": ci(0.05),
        "form": ci(-0.05),
        "content": ci(0.2),
    }
    return {
        "banner": "EXPLORATORY",
        "changes": {
            "self_reflect": {"shares": shares, "net": ci(0.05)},
            "placebo": {"shares": shares, "net": ci(-0.3)},
        },
        "source": {
            "net": {"placebo": part},
            "wrong_to_correct": {"placebo": part},
            "correct_to_wrong": {"placebo": part},
        },
    }


# ---- the numbers behind the figures (no drawing needed) ----------------------


def test_one_self_consistency_sample_costs_the_arms_tokens_over_its_matched_count():
    d = F.accuracy_vs_tokens_data(analysis())
    assert d["per_sample_tokens"] == pytest.approx(500.0)
    assert d["curve"] == [(500.0, 0.60), (1000.0, 0.68), (1500.0, 0.72)]
    assert d["arms"]["placebo"] == (1500.0, 0.60)


def test_it_also_works_with_integer_curve_keys_as_in_memory():
    a = analysis()
    a["sc_curve"] = {1: 0.60, 2: 0.68, 3: 0.72}
    assert F.accuracy_vs_tokens_data(a)["curve"][1] == (1000.0, 0.68)


def test_change_type_data_keeps_the_five_types_in_order():
    d = F.change_type_data(decomposition())
    assert list(d["placebo"]) == [
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_other_wrong",
        "unchanged_correct",
        "unchanged_wrong",
    ]
    assert d["placebo"]["unchanged_correct"] == 0.5


def test_source_data_gives_the_three_parts_per_arm():
    d = F.source_data(decomposition(), "net")
    assert d["placebo"] == {"spontaneous": 0.05, "form": -0.05, "content": 0.2}


def test_an_unknown_metric_is_an_error():
    with pytest.raises(KeyError):
        F.source_data(decomposition(), "vibes")


def test_the_zoom_box_frames_the_arms_near_the_best_score_and_leaves_out_outliers():
    data = {
        "arms": {"a": (4000.0, 0.72), "b": (4200.0, 0.718), "far": (4100.0, 0.60)},
        "curve": [(1000.0, 0.60), (4100.0, 0.73), (8000.0, 0.78)],
    }
    (x0, x1), (y0, y1) = F.zoom_box(data)
    assert x0 < 4000.0 and x1 > 4200.0 and x1 - x0 < 2000
    assert y0 > 0.70 and y1 < 0.75  # the outlier at 0.60 and the far curve are out
    assert y0 < 0.718 and y1 > 0.73


# ---- drawing -----------------------------------------------------------------

pytest.importorskip("matplotlib")


def is_png(path):
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 3000


def test_each_figure_writes_a_png(tmp_path):
    F.plot_accuracy_vs_tokens(analysis(), tmp_path / "a.png")
    F.plot_change_types(decomposition(), tmp_path / "b.png")
    F.plot_source_of_change(decomposition(), tmp_path / "c.png")
    assert all(is_png(tmp_path / n) for n in ("a.png", "b.png", "c.png"))


def test_make_all_names_the_figures_by_the_plan_and_never_overwrites(tmp_path):
    paths = F.make_all(analysis(), decomposition(), tmp_path)
    assert [p.name for p in paths] == [
        "fig2_accuracy_vs_tokens.png",
        "fig3_change_types.png",
        "fig4_source_of_change.png",
    ]
    assert all(is_png(p) for p in paths)
    with pytest.raises(FileExistsError):
        F.make_all(analysis(), decomposition(), tmp_path)


def test_the_figures_say_they_are_exploratory():
    assert "EXPLORATORY" in F.FOOTER


def test_the_inputs_round_trip_through_json(tmp_path):
    a = json.loads(json.dumps(analysis()))
    d = json.loads(json.dumps(decomposition()))
    assert len(F.make_all(a, d, tmp_path)) == 3

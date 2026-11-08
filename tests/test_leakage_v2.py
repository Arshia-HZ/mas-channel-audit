"""Failure-message variants: one-sentence cap and variant-tagged leakage records."""

from __future__ import annotations

import json

from channel_audit.leakage import LeakageRecord, first_sentence, record_key


def test_only_the_first_sentence_is_kept():
    text = "Step 2 drops a sign when moving 3 across. The rest follows. Done."
    assert first_sentence(text) == "Step 2 drops a sign when moving 3 across."


def test_markdown_markers_are_removed_from_the_sentence():
    assert first_sentence("**Step 3** is where it goes wrong.") == (
        "Step 3 is where it goes wrong."
    )
    assert first_sentence("### Step 3 fails. Then more.") == "Step 3 fails."


def test_a_long_sentence_is_cut_at_a_word_boundary():
    text = ("word " * 200).strip() + "."
    out = first_sentence(text, max_chars=50)
    assert len(out) <= 50 and not out.endswith(" ") and " " in out
    assert text.startswith(out)


def test_a_message_without_a_terminator_is_kept_whole_up_to_the_cap():
    assert first_sentence("Step 4 divides by zero") == "Step 4 divides by zero"
    assert first_sentence("") == ""
    assert first_sentence("   ") == ""


def test_decimals_and_abbreviations_do_not_end_the_sentence_early():
    text = "Step 2 rounds 3.14 to 3.2, which is off. Then it continues."
    assert first_sentence(text) == "Step 2 rounds 3.14 to 3.2, which is off."


def test_newlines_collapse_to_spaces():
    assert first_sentence("Step 2\nis wrong because\n\nof a sign.") == (
        "Step 2 is wrong because of a sign."
    )


def test_record_keys_of_the_two_variants_never_collide():
    assert record_key("p", 0, 1) == "p|0|1"
    assert record_key("p", 0, 1, "v1") == "p|0|1"
    assert record_key("p", 0, 1, "v2") == "v2|p|0|1"


def test_old_records_without_a_variant_still_load_as_v1():
    old = {
        "record_key": "p|0|1",
        "problem_id": "p",
        "replicate": 0,
        "agent": 1,
        "attempts": 1,
        "outcome": "clean",
        "final_gen_key": "k",
        "n_stripped": 0,
        "leaked": [],
        "sanitized_text": "x",
    }
    assert LeakageRecord.model_validate_json(json.dumps(old)).variant == "v1"
    new = LeakageRecord(**{**old, "record_key": "v2|p|0|1", "variant": "v2"})
    assert new.variant == "v2"

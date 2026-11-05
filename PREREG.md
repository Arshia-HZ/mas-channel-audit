# Preregistration (DRAFT, not frozen)

Status: **draft**. It is completed at M5 and committed and timestamped before any
confirmatory generation. Until then it only records decisions already taken, with the
evidence behind them. Hypotheses, gates G1 to G4 and the power calculation are added at M5.
Items marked **[confirm]** are proposed wording and need the owner's sign-off.

## 1. Problems

- Source: MATH test set, `EleutherAI/hendrycks_math`, pinned revision
  `21a5633873b6a120296cce3e2df9d5550074f4a3`. All 5,000 rows were usable.
- Split rule: order problems by `sha256("mas-channel-audit/split/v1|" + problem_id)`.
  The first 300 are the **screen** split (all levels). Of the rest, level 3 to 5 problems
  fill **pilot** (first 200) and then **confirm** (3,250); the 1,250 level 1 and 2
  leftovers are not used. Only the id and the level are read.
- Frozen files and sha256 hashes: `data/problems/split_manifest.json`.
- Primary data for the study: levels 3 to 5. Pilot problems are never pooled into
  confirmatory results.

## 2. Decoding and limits

- Temperature 0.7, top-p 1.0, one `max_tokens` for every arm of a model.
- `max_tokens = 2048` for every candidate model. T4 check, Qwen3-1.7B, 50 problems: 1,024
  truncated 27.3% of round 0; 2,048 truncated 5.3% (8 of 150, on 5 problems), and none of
  the 40 truncated outputs looped.
- `max_model_len = 12288` (Qwen2.5-Math-1.5B keeps its native 4096). A `full_solution`
  prompt holds three solutions of up to 2,048 tokens plus the reply. At 8,192 the check
  flagged 6 requests on 2 problems.

## 3. Truncation and gate G3

- A truncated output counts as incorrect unless a complete `\boxed{}` answer appears
  before the cut-off; such an answer is the model's real response and is graded (owner
  decision; 2 of 41 truncated outputs at 1,024 tokens). Truncation itself is always
  reported separately, by arm.
- Sensitivity analysis: repeat every contrast after dropping the problems on which any
  round-0 agent truncated. Round 0 is shared by all arms, so this filter does not use any
  arm's outcome.
- Gate G3 (evaluated at M5), proposed wording **[confirm]**:
  1. Context overflow must be zero (hard stop).
  2. Each of the seven revision arms must truncate under 5% (hard stop).
  3. Round 0 and the self-consistency pool are reported with no threshold. Every arm
     branches from the same round 0 and shares the same pool, so their truncation cannot
     favour one arm; the sensitivity analysis above covers it.
  4. An arm that differs from `self_reflect` by more than 2 points of truncation is
     flagged, and its contrasts are reported with and without the affected problems.
- Evidence for this wording, T4 check at 2,048 tokens: round 0 5.3%, pool 7.0%, every arm
  at most 2.0%, no arm more than 2 points from `self_reflect`, and no context overflow
  once the window is 12,288.

## 4. Answer extraction

- The answer is the last `\boxed{}` that is not a confidence statement. A box is a
  confidence statement if it opens with the word "confidence", if "confidence" precedes it
  on the same line, or if it holds just the stated `Confidence: N` number. Such boxes are
  skipped from the end while an earlier box exists. A lone "Confidence ..." box is no
  answer; a lone box holding only a number is kept. An unclosed last box means no answer.
- There is no fallback to "the answer is", last-number guessing or any second extractor.
  An output without a usable box is recorded as having no answer and counts as incorrect.
- Why: the M4 screening showed that Qwen3-1.7B often boxes its `Confidence: N` line, and
  the old last-box rule graded that as the answer. 420 of 1,200 screening samples (35%)
  had their extracted answer change, none of them correct before and 303 correct after.
  Qwen3-1.7B accuracy went from 44.3% to 69.6%. The same rule is applied to the answers
  shown to peers in the revision arms.
- Every result produced before this rule (including the M3 T4 check) is invalid for
  accuracy and must be re-graded into a new directory (`scripts/22_regrade.py`).

## 5. Screening (M4), completed 2026-11-05 except Llama

- Protocol: 300 screen problems (all levels), 4 samples each, temperature 0.7, top-p 1.0,
  `max_tokens` 2048, vLLM 0.33.0, fp16 on a Colab T4, round-0 prompt unless stated.
  Graded with the extractor in section 4.

| Model (round-0 prompt) | Accuracy | Levels 3-5 | Disagreement, levels 3-5 | Truncated | No box (finished) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B, non-thinking | 69.6% | 63.5% | 47.0% | 6.0% | 0.9% |
| Qwen2.5-Math-1.5B (base) | 24.8% | 23.7% | 68.9% | 28.7% | 18.1% |

- Qwen2.5-Math-1.5B is a base model: its outputs often repeat the prompt, invent new
  problems or emit stray tokens. The Instruct variant was not screened.
- Llama-3.2-3B-Instruct: not screened. The Hugging Face repo is gated and Meta access was
  requested on 2026-11-05. It is not needed to choose the primary model and is screened
  when access is granted.

### Gate G0 (sanity model Qwen2.5-1.5B-Instruct, published MATH 55.2)

| Prompt and decoding | Accuracy | Difference | No box (finished) | Truncated |
| --- | --- | --- | --- | --- |
| Round-0 prompt (asks for a confidence), T=0.7 | 33.7% | -21.5 | 19.0% | 2.1% |
| Plain prompt, T=0.7 | 49.7% | -5.5 | 0.2% | 1.9% |
| Plain prompt, greedy | 52.0% | -3.2 | 0.3% | 4.0% |

- G0 passes on the plain greedy run: within 5 points of the published score, extraction
  failure under 5%. Greedy is the like-for-like setting, since published scores use greedy
  decoding. The plain prompt is `prompts/v1/sanity.txt`: the question and "Please reason
  step by step, and put your final answer within `\boxed{}`."
- With the round-0 prompt this model fails G0: in 190 of its 228 unboxed outputs it ends on
  the `Confidence:` line without stating a final answer, so asking for a confidence crowds
  out the box. Qwen3-1.7B is not affected (0.9%). Weaker models may drop the box when asked
  for a confidence; this matters for the `answer_conf` arm and for any other model.

### Primary model: Qwen3-1.7B (non-thinking), approved by the owner 2026-11-05

The plan picks Qwen2.5-Math-1.5B or Qwen3-1.7B, whichever has more usable round-0
disagreement. A level 3 to 5 problem has **usable disagreement** when at least two of its
four samples are valid (a boxed answer, not truncated) and at least one valid sample is
right and one wrong. Over the 219 level 3 to 5 screen problems:

| | Qwen3-1.7B | Qwen2.5-Math-1.5B |
| --- | --- | --- |
| Problems with a right and a wrong sample (any samples) | 38.8% | 57.1% |
| Usable disagreement (valid samples only) | 32.9% | 33.3% |
| Samples that are invalid | 9.1% | 45.9% |
| Problems with all four samples valid | 79.9% | 11.4% |
| Right and wrong with all four samples valid | 28.8% | 6.8% |

The raw count favours Qwen2.5-Math only because almost half its samples are broken
outputs. With invalid samples removed the two have the same usable disagreement, while
Qwen3-1.7B has far more clean problems, higher accuracy and fewer broken outputs. Qwen3-1.7B
is therefore the primary model.

## 6. Open before this file is completed

- Freeze `prompts/v1/` (needs the owner's approval after reading the review sample).
- Screen Llama-3.2-3B-Instruct, or choose another second family, when access is settled.
- Align the validator's G3 check with section 3.
- M5: hypotheses and margins, gates G1 to G4 with thresholds, power calculation.

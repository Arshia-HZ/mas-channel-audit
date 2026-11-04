# Preregistration (DRAFT, not frozen)

Status: **draft**. It is completed at M5 and committed and timestamped before any
confirmatory generation. Until then it only records decisions already taken, with the
evidence behind them. Hypotheses, gates G1 to G4 and the power calculation are added at M5.
Items marked **[confirm]** are my reading of a decision and need the owner's sign-off.

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

- Truncated outputs count as incorrect. **[confirm]** As implemented, a complete
  `\boxed{}` answer that appears before the cut-off is still an answer (2 of 41 at 1,024
  tokens); the owner decides whether that stays or truncated outputs are always incorrect.
- Sensitivity analysis: repeat every contrast after dropping the problems on which any
  round-0 agent truncated. Round 0 is shared by all arms, so this filter does not use any
  arm's outcome.
- G3 is evaluated and reported at M5 with the plan's thresholds (zero context overflow, arm
  truncation under 5%). The T4 check at 2,048 tokens gave round 0 5.3%, the
  self-consistency pool 7.0% and every arm at most 2.0%, with no arm differing from
  `self_reflect` by more than the validator's warning gap. A round-0 or pool rate near 5% is
  documented, not treated as a stop. **[confirm]**

## 4. Screening (M4)

- Each candidate is sampled 4 times per screen problem with the round-0 prompt.
- Candidates: Qwen3-1.7B (non-thinking), Qwen2.5-Math-1.5B, Llama-3.2-3B-Instruct, and
  Qwen2.5-1.5B-Instruct as the sanity model.
- Gate G0: Qwen2.5-1.5B-Instruct within 5 points of its published MATH score of 55.2
  (Qwen2.5 release post), and extraction failure under 5%.
- Reported per model: accuracy by level, disagreement rate, truncation, extraction failure.
- Primary model: **not yet chosen.** The plan picks Qwen2.5-Math-1.5B or Qwen3-1.7B, whichever
  has more usable round-0 disagreement on the screen split. The choice and the numbers
  behind it are written here after screening.

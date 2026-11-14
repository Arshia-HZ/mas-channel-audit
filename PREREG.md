# Preregistration (DRAFT, not frozen)

Status: **draft**. It is completed at M5 and committed and timestamped before any
confirmatory generation. Until then it only records decisions already taken, with the
evidence behind them. Hypotheses, gates G1 to G4 and the power calculation are added at M5.
Items marked **[confirm]** are proposed wording and need the owner's sign-off (none left).
Added on 2026-11-09: section 9 now shows the status of the open items, and section 10 records
the confirmatory run and the frozen analysis. Sections 1 to 8 are unchanged.
Added on 2026-11-14: section 11 closes the audit. Sections 1 to 10 are unchanged.

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
- Gate G3 (evaluated at M5), approved by the owner 2026-11-05:
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
- The confidence value is the last `Confidence: N`, also when the model wraps or boxes the
  number (`Confidence: **70**`, `\text{Confidence: } \boxed{90}`, `Confidence level: 75`,
  `Confidence: **N = 100**`). Reading only the plain form missed half the values: for
  Qwen3-1.7B the screening parsed 601 of 1,200; with the wrapped forms it parses 1,104 of
  the 1,128 finished outputs (97.9%). The `answer_conf` arm shows `(confidence not given)`
  only when no number can be read.

## 5. Screening (M4), completed 2026-11-06

- Protocol: 300 screen problems (all levels), 4 samples each, temperature 0.7, top-p 1.0,
  `max_tokens` 2048, vLLM 0.33.0, fp16 on a Colab T4, round-0 prompt unless stated.
  Graded with the extractor in section 4.

| Model (round-0 prompt) | Accuracy | Levels 3-5 | Disagreement, levels 3-5 | Truncated | No box (finished) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B, non-thinking | 69.6% | 63.5% | 47.0% | 6.0% | 0.9% |
| Qwen2.5-Math-1.5B (base) | 24.8% | 23.7% | 68.9% | 28.7% | 18.1% |
| Llama-3.2-3B-Instruct | 10.5% | 7.5% | 22.4% | 3.7% | 70.4% |

- Qwen2.5-Math-1.5B is a base model: its outputs often repeat the prompt, invent new
  problems or emit stray tokens. The Instruct variant was not screened.
- Llama-3.2-3B-Instruct (screened 2026-11-06, T=0.7, round-0 prompt) is **not usable with
  the frozen round-0 prompt**: 845 of 1,200 outputs have no `\boxed{}`, and 842 of those end
  on the `Confidence:` line with the answer stated in plain text. Only 11 contain an
  answer phrase. The model solves problems but drops the box when asked for a confidence,
  as Qwen2.5-1.5B-Instruct partly does (19%) and Qwen3-1.7B does not (0.9%).
  With the plain prompt and greedy decoding it reaches 43.7% (37.0% at levels 3 to 5),
  5.7% without a box and 10.0% truncated.
- The frozen prompts are not changed for it. Whether to screen another family (for example
  Phi-3.5-mini-instruct or Granite-3.3-2B-Instruct), or to adopt a per-model prompt version,
  is decided before M8, when the second family is needed. **[confirm]**
- Finding for the paper: a confidence request costs format compliance in small instruct
  models, which bears on the `answer_conf` arm outside Qwen3.

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

## 6. Prompts

- `prompts/v1/` is frozen as of 2026-11-05, after the owner read the review sample of the
  T4 check and approved it. The manifest `prompts/v1/MANIFEST.json` holds the sha256 of every
  template (combined sha256 `545ebeded7e1ce0b749d03ad4dfd3cca35539176ed525c8c7508b5cb6aab7079`),
  the tests verify it, and every generation stores its template hash.
- Templates: `round0`, `failure_msg` and the seven revision arms, plus `sanity` (G0 only).

## 7. Pilot (M5)

- Design: the 200 pilot problems, 2 replicates, the primary model (Qwen3-1.7B, temperature
  0.7, 2048 tokens), all seven arms plus the self-consistency pool of 6. The pilot is never
  pooled into confirmatory results; pilot problems are not in the confirm split.
- Shards: `scripts/05_pilot_shards.py` runs shards of about 50 problems. Every (subject,
  level) group stays whole inside one shard, so the placebo donor pool is the whole group in
  the split and does not depend on the shard layout. On the pilot split the shards hold 50,
  52, 49 and 49 problems, every group has 3 or more problems, and each finished shard is
  validated and uploaded to the private dataset. Shard assignment uses only subject and
  level.
- Validation per shard: expected counts, no missing keys or parents, every generation graded,
  every arm's budget computed. A shard that fails validation is not marked done.

### Gates (evaluated on the pilot) **[confirm]** for the G1, G2 and G4 definitions

| Gate | Definition | Pass | If it fails |
| --- | --- | --- | --- |
| G1 | Share of replicates whose three round-0 answers contain two or more distinct parsable answers | 15% or more | Restrict to levels 4 and 5, then retry on fresh pilot problems |
| G2 | Share of replicates where at least one round-0 answer is correct but no answer in the round-0 plurality is | 5% or more | Same as G1; if still failing, test mixed teams first |
| G3 | Section 3 | Zero overflow, each revision arm under 5% truncation | Fix limits before running |
| G4 | 40 failure messages drawn uniformly at random with seed 0 (no use of correctness) and labelled useful, vague or wrong by the owner; wrong counts as not vague | At least half not vague | Keep the arm and plan the small-critic fallback |

- If every arm falls within +/-1 point of matched self-consistency, that is not a failure:
  move to the precise-null paper and give H5 priority.
- A retry after G1 or G2 draws fresh problems from the confirm split and removes them from it.

### Hypotheses and decision rules

Analysis unit: the problem. Replicates are averaged within problem first; arms are compared
with a problem-level paired bootstrap of 10,000 resamples. An arm's score is the plurality
credit of its three final answers (1/m on an m-way tie containing the correct answer; an
unparsable answer never votes). "Matched self-consistency" is the interpolated credit of the
9 round-0 plus pool samples at the sample count whose token cost equals the arm's mean
generated tokens per replicate. An arm's gain is its score minus matched self-consistency at
its own budget.

| # | Contrast | Decision rule |
| --- | --- | --- |
| 1 (H1) | `full_solution` gain over matched SC | Equivalent if the 90% CI lies within +/-2 points |
| 2 (H2) | `answer_only` minus `full_solution` | Non-inferior if the lower 95% bound is above -1.5 points |
| 3 (H3) | `placebo` minus `self_reflect`, correct-to-wrong rate | One-sided, Holm-adjusted |
| 4 (H4) | `failure_cause` gain minus `full_solution` gain | Two-sided, Holm-adjusted |
| 5 (H4) | `answer_conf` gain minus `full_solution` gain | Two-sided, Holm-adjusted |
| 6 | Best low-bandwidth arm gain over matched SC | Two-sided, Holm-adjusted |

- The best low-bandwidth arm in contrast 6 is chosen on the pilot only, among `answer_only`,
  `answer_conf`, `dissent_only` and `failure_cause`, by the largest gain.
- The correct-to-wrong rate counts agents correct in round 0 and wrong after revision,
  divided by all agents.
- H5 (mixed teams) is a separate family: the difference in contrast 4 or 5 between mixed and
  same-model teams.
- Reported on every run, by arm: truncation, context overflow, extraction failure and leakage
  rejection. An arm more than 2 points from `self_reflect` in truncation is repeated without
  the affected problems.

### Sample size

`n = (1.96 + 0.84)^2 s^2 / d^2`, with `s` the standard deviation of the pilot's problem-level
paired differences for each contrast (`scripts/05_gate_check.py`), `d` = 3 points unless the
pilot supports 2, capped at the 3,250 confirm problems. Phase 1's s = 0.2654 (614 problems
for 3 points) was only a placeholder.

### Pilot results (run 2026-11-07 to 2026-11-08, Qwen3-1.7B)

200 problems x 2 replicates, four shards, all validated and uploaded. These numbers do not
change any decision rule above; they are the evidence for the sample size and for what
happens next.

| Gate | Result | Need | Verdict |
| --- | --- | --- | --- |
| G1 round-0 disagreement | 33.0% of replicates (L3 20%, L4 31%, L5 48%) | 15% | PASS |
| G2 recoverable headroom | 4.2% of replicates (17 of 400) | 5% | **FAIL** (narrow) |
| G3 | zero overflow; every revision arm under 3% truncation | section 3 | PASS |
| G4 failure messages | 15% not vague (0 useful, 6 wrong, 34 vague of 40) | 50% | **FAIL** |

- Truncation: round 0 10.3%, pool 9.1%, revision arms 0.8% to 2.8%, failure messages 0.1%.
- G4 labels: the 40 messages (seed 0) were labelled by Claude at the owner's request on
  2026-11-08, using the definitions above; the owner reviewed the labels on 2026-11-08 and
  found no problems.
  Reading the sample: 26 messages say the work is correct, 24 contain an "[answer removed]"
  artifact, 34 run past three sentences although the prompt asks for one, and 7 re-solve
  the problem instead of reviewing it (two of those state the gold answer, which the
  leakage filter missed because it only checks the three round-0 answers). Of 14 reviewed
  solutions that were actually wrong, none got a message naming the real error. Even if the
  eight borderline messages were relabelled not vague, the share would be 35%.

| Arm | Accuracy | Mean tokens | Matched SC samples | SC accuracy | Gain over SC |
| --- | --- | --- | --- | --- | --- |
| self_reflect | 73.8% | 4,231 | 5.25 | 77.2% | -3.4% |
| answer_only | 73.8% | 4,233 | 5.26 | 77.2% | -3.5% |
| answer_conf | 73.9% | 4,226 | 5.25 | 77.2% | -3.3% |
| dissent_only | 74.0% | 3,428 | 4.26 | 75.8% | -1.8% |
| failure_cause | 75.3% | 5,724 | 7.11 | 78.9% | -3.6% |
| full_solution | 72.7% | 4,176 | 5.19 | 77.1% | -4.5% |
| placebo | 63.3% | 4,147 | 5.15 | 77.1% | -13.8% |

Best low-bandwidth arm for contrast 6, chosen on the pilot: `dissent_only`.

| Contrast | Mean | 95% CI | SD | Problems for 3 points | for 2 points |
| --- | --- | --- | --- | --- | --- |
| 1. full_solution gain over SC | -4.5% | [-7.1%, -1.8%] | 0.196 | 336 | 756 |
| 2. answer_only minus full_solution | +1.1% | [-1.2%, +3.4%] | 0.164 | 236 | 530 |
| 3. placebo minus self_reflect, correct-to-wrong | +10.2% | [+7.5%, +13.2%] | 0.204 | 362 | 814 |
| 4. failure_cause minus full_solution (gains) | +0.9% | [-1.5%, +3.2%] | 0.169 | 250 | 563 |
| 5. answer_conf minus full_solution (gains) | +1.2% | [-1.2%, +3.7%] | 0.177 | 273 | 614 |
| 6. best low-bandwidth arm gain over SC | -1.8% | [-4.2%, +0.5%] | 0.172 | 258 | 579 |

- At `d` = 3 points the largest requirement is 362 problems (contrast 3); at 2 points it is
  814. Both are far below the 3,250 confirm problems. The sample size for the confirmatory
  run is not fixed yet; it is set after the decisions in section 8.
- In this pilot no arm beats matched self-consistency; every arm sits 1.8 to 4.5 points
  below it, and `full_solution` is worse with a confidence interval that excludes zero.

### Failure-message v2 re-test (planned 2026-11-08, after the G4 failure)

- Why: the pilot's v1 failure messages were mostly vague, often re-solved the problem and in
  two sampled cases stated the gold answer. The v2 prompt (`prompts/v2/failure_msg.txt`) asks
  for the single step most likely to be wrong and one specific reason in a sentence, with no
  solving and no final answer, and it never lets the reviewer say "no error" (a message must
  stay a suspected error, never a correctness signal). The reply is cut to its first sentence
  and then goes through the same leakage filter as v1.
- v2 was written after seeing the pilot's v1 failures; it is therefore judged on a separate
  draw, not on the 40 messages that motivated it.
- Design: the finished pilot shards are copied; only the v2 messages and their revisions
  (`failure_msg_v2`, `failure_cause_v2`) are generated, so round 0, the pool and every other
  arm are shared. The v1 records are never touched. The reviewer is the same Qwen3-1.7B.
- Success criteria, set before the run, on a fresh draw of 40 v2 messages (seed 1):
  at least half not vague; at least 95% a single sentence with no "[answer removed]" artifact;
  at most one message containing the gold answer (an offline audit, never a filter).
- If it fails, the plan's small-critic fallback (a separate, stronger reviewer) is a new
  decision; if that also fails, `failure_cause` is reported as a weak channel.
- `prompts/v2` was frozen on 2026-11-08 after the owner read and approved the wording.
  `failure_msg.txt` sha256 `f376375982756ce1905945d52fcad87d58f3bca62d8217cfa320284a7bf2aa93`,
  combined sha256 `5903dd082b6dd630b1fca7b2cd057cd8d97c7e14e7a947a6b0d091f033d92096`
  (`prompts/v2/MANIFEST.json`).

#### Result of the v2 re-test (2026-11-08)

| Criterion | Target | Result |
| --- | --- | --- |
| Not vague (fresh 40, seed 1) | at least 50% | **27.5%** (1 useful, 10 wrong, 29 vague): FAIL |
| Single sentence, no artifact | at least 95% | 40 of 40, 0 artifacts: PASS |
| Messages containing the gold answer | at most 1 | 1: PASS (at the limit) |

- Labelled by Claude at the owner's request; the owner reviewed the labels on 2026-11-08
  and found no problems.
- Format is fixed, content is not. 16 of 40 messages are a bare step name with no reason, and
  the reasons that were given were mostly mistaken: 10 criticise a step that is correct.
  Of the 14 reviewed solutions that were actually wrong, one message pointed at the real
  error (a "large minus three small semicircles" assumption). The same-model reviewer
  (Qwen3-1.7B) does not find errors, whatever the wording.
- Arm effect: `failure_cause_v2` scores 74.5% against 75.3% for v1, with 4,942 mean tokens
  (v1 5,724) and a gain over matched SC of -3.5% (v1 -3.6%). Against `full_solution` it is
  +0.9% [-1.5%, +3.3%], the same as v1. Better-formed messages changed nothing measurable.
- The small-critic fallback (a stronger separate reviewer) is the plan's next step for G4;
  the owner decides whether it is worth testing.
- Self-consistency accuracy by sample count on the pilot: k=1 66.9%, k=2 69.2%, k=3 73.5%,
  k=4 75.4%, k=5 77.0%, k=6 77.9%, k=7 78.8%, k=8 79.5%, k=9 80.4%. Every revision arm sits
  near the k=3 round-0 vote (73.5%): revision adds under one point over the vote.
- G2 by level: L3 2.5%, L4 3.8%, L5 6.5%; levels 4 and 5 together 14 of 282 replicates
  (4.96%), just under the 5% line.

## 8. Confirmatory design (M6), decided by the owner on 2026-11-08

- Population: levels 3 to 5, as preregistered. The pilot's G2 miss (4.2% against 5%) is
  documented, not acted on: levels 4 and 5 alone would have been 4.96%, still at the line,
  and a fresh pilot would cost another pilot's compute. G1, G2 and the arm results are
  reported by level (3, 4, 5) as a preregistered secondary breakdown.
- Problems: the first 600 problems of the frozen confirm split, in its frozen hash order
  (199 at level 3, 193 at level 4, 208 at level 5; 21 subject-and-level groups of 16 to 52).
  The choice depends on problem ids only. They are disjoint from the screen and pilot splits.
- Replicates: 2, as in the pilot. Model, decoding, `max_tokens`, `max_model_len` and the
  frozen prompts are those of the pilot.
- Arms: the seven preregistered arms plus `failure_cause_v2`. Contrast 4 uses `failure_cause`
  (v1, the preregistered prompt); the v2 contrast is reported beside it. Both are reported
  whatever the outcome.
- Shards: `confirm.ipynb` runs 12 shards of 40 to 54 problems with every group whole, so the
  placebo donor pool is the group inside the chosen 600 and does not depend on the layout.
  Each shard is validated before the next starts and uploaded to the private dataset.
- Best low-bandwidth arm for contrast 6, chosen on the pilot before this run: `dissent_only`.
- Sample size: the pilot's paired SDs are 0.16 to 0.20. At `d` = 3 points the largest
  requirement is 362 problems, so 600 gives a margin; the 95% interval half-width is about
  1.3 to 1.6 points. That is enough to detect a 3-point effect, to test the H2 margin of 1.5,
  and to show that an arm below matched self-consistency is below it.
- Discipline: no accuracy, gain or contrast is computed on confirmatory data until the
  analysis code is frozen (M7). Between now and then only run quality (truncation, overflow,
  extraction failure, leakage, by arm) is looked at. The pilot is never pooled.
- Compute: about 3.5 times the pilot (3 times the problem-replicates, 18% more generations
  for the extra arm).
- Hardware (decided 2026-11-08, before any confirmatory generation): the screening, the
  pilot and the v2 re-test ran in fp16 on a Colab T4. The confirmatory run may instead run
  on an NVIDIA RTX Pro 6000 Blackwell on molab, with the same model revision, vLLM version,
  fp16, decoding and frozen prompts. Seeds are derived from the identity fields and do not
  depend on hardware, but different GPU kernels give slightly different logits, so the
  samples are not bit-identical to a T4 run. The switch is conditional on one check, made
  on data that was already seen: the first pilot shard is generated on the new GPU
  (uploaded under `pilot_check/`, never over `pilot/`) and compared with the T4 copy of
  that shard. It passes if the truncation, overflow and extraction-failure rates of every
  arm are within 2 points of the T4 shard and the round-0 accuracy is within 5 points
  (about the noise of a 50-problem shard). If it fails, the confirmatory run stays on the
  T4. Either way the whole confirmatory run uses one kind of hardware, and the run records
  the GPU, driver and package versions in `HARDWARE.json`.
- Result of that check (2026-11-08, shard 00 of the pilot on the RTX Pro 6000 Blackwell,
  vLLM 0.33.0 in its own environment with transformers 5.20.2): every arm's truncation and
  extraction-failure rates were within 2 points of the T4 shard (largest difference 1.3
  points, `failure_cause` truncation), overflow was 0, and round-0 accuracy was 72.3%
  against 69.7%. The check passes. On the Blackwell run `VLLM_USE_FLASHINFER_SAMPLER=0`
  is set, because vLLM's default FlashInfer sampler needs nvcc and ninja to compile and
  the sandbox has neither; sampling then uses the PyTorch implementation.

## 9. Open items (status on 2026-11-09)

| Item | Status |
| --- | --- |
| Run the confirmatory shards (M6), then freeze the analysis code (M7) before analysing | **Done**: the shards finished on 2026-11-08 (section 10.1) and the analysis code is frozen as recorded in section 10.3. The analysis was run once on 2026-11-09 (section 10.4). |
| Commit and timestamp this file before the first confirmatory generation | **Done**: sections 1 to 8 were committed as `4d7449d` at 2026-11-08 19:15:35 UTC; the first confirmatory generation was created at 19:17:18 UTC. |
| Shards split the large (subject, level) groups, so placebo donors would have to come from the whole confirm split; decide before M6 | **Resolved in section 8**: the run uses the first 600 problems and keeps every group whole, so the donor pool is the group inside those 600. |
| Choose a second family that follows the round-0 format (Llama-3.2-3B-Instruct does not), before M8 | **Open.** |

## 10. Confirmatory run and analysis (added 2026-11-09)

### 10.1 The confirmatory run (record; no outcome is involved)

- Data: the first 600 problems of the frozen confirm split (levels 3, 4 and 5: 199, 193
  and 208), 2 replicates, the seven preregistered arms plus `failure_cause_v2`, in 12
  shards of 40 to 54 problems. Every shard was validated, marked done (its
  `SHARD_DONE.json` holds the sha256 of its generations) and uploaded to the private
  dataset `Arshia-HZ/mas-channel-audit-raw` under `confirm/qwen3-1.7b/shard_00` to
  `shard_11`.
- Hardware and software (`HARDWARE.json`): one NVIDIA RTX PRO 6000 Blackwell Server
  Edition (driver 595.71.05), Qwen3-1.7B in fp16, vLLM 0.33.0, torch 2.14.0, transformers
  5.20.2, Python 3.12.13, `VLLM_USE_FLASHINFER_SAMPLER=0` (section 8). The first generation
  was created on 2026-11-08 at 19:17:18 UTC and the last at 20:57:21 UTC.
- Run quality, pooled over the 12 shards. This is all that was looked at before the
  freeze; no correctness is involved.

| Generation | n | Truncated | Overflow | No box | No answer |
| --- | --- | --- | --- | --- | --- |
| round0 | 3,600 | 8.2% | 0 | 1.1% | 9.0% |
| sc_pool | 7,200 | 7.6% | 0 | 1.1% | 8.4% |
| self_reflect | 3,600 | 1.5% | 0 | 0.0% | 1.4% |
| answer_only | 3,600 | 1.6% | 0 | 0.1% | 1.6% |
| answer_conf | 3,600 | 1.8% | 0 | 0.1% | 1.8% |
| dissent_only | 1,703 | 1.8% | 0 | 0.1% | 1.8% |
| failure_cause | 3,600 | 1.3% | 0 | 0.1% | 1.2% |
| full_solution | 3,600 | 0.6% | 0 | 0.0% | 0.6% |
| placebo | 3,600 | 0.8% | 0 | 0.1% | 0.8% |
| failure_cause_v2 | 3,600 | 1.8% | 0 | 0.1% | 1.7% |
| failure_msg | 4,831 | 0.1% | 0 | - | - |
| failure_msg_v2 | 3,770 | 0.6% | 0 | - | - |

- Failure-message leakage (rejected means not clean on the first try): v1 had 3,600
  messages, 2,369 clean, 492 regenerated and 739 redacted (rejection 34.2%); v2 had 3,600,
  3,430 clean, 70 regenerated and 100 redacted (rejection 4.7%). `dissent_only` has fewer
  generations because only an agent that saw a disagreement revises.
- G3 is judged on the pooled run, as the pilot's was: zero overflow and every revision arm
  under 5% truncation, which holds. Shard 04 failed G3 on its own (`answer_conf` and
  `failure_cause_v2` at or above 5% truncation among its 53 problems, with `self_reflect`
  at 4.1% against 1.3% for `full_solution` and `placebo`). The shard was kept. From the
  rounded rates above no arm is more than 1 point from `self_reflect` over the pooled run,
  so the flagged-arm run of section 10.2 is not expected to trigger; the analysis decides,
  not this sentence.
- Discipline (section 8): until the freeze in section 10.3 only the run quality above was
  looked at. No accuracy, gain or contrast has been computed on the confirmatory shards,
  and the pilot is never pooled with them.

### 10.2 What the analysis computes (frozen)

The code is `src/channel_audit/analysis/contrasts.py`, run by `scripts/30_analyze.py`. The
owner decided on 2026-11-09 that p-values come from the bootstrap, that the Holm family is
contrasts 3 to 6, and that the scope is the contrasts, the sensitivity runs below and
the quality report, with decomposition and figures left for later and labelled exploratory.
The rest of this section records how the code reads sections 7 and 8; the owner reviewed
it on 2026-11-09 and found no problem.

1. **Scores and budgets** are those of section 7. Replicates are averaged within problem.
   An arm's score is the plurality credit of its three final answers. Its budget is its mean
   generated tokens per replicate: round 0, every revision generation and, for the
   failure-cause arms, every failure-message attempt including rejected ones. The matched
   self-consistency sample count is that budget divided by the mean generated tokens of one
   self-consistency sample (round 0 and pool); credit at a fractional count is interpolated
   linearly between integer counts. A count outside 1 to 9 stops the main analysis.
2. **Intervals**: 10,000 problem-level resamples drawn with numpy `default_rng(0)`; percentile
   intervals (95%: the 2.5th and 97.5th percentiles; 90%: the 5th and 95th). Every contrast
   of one analysis uses the same resamples.
3. **p-values** come from the same resamples. One-sided (contrast 3, whose alternative is that
   `placebo` has more correct-to-wrong changes): p = (1 + number of resampled means <= 0) /
   10,001. Two-sided: twice the smaller of the two one-sided values (the other counts means
   >= 0), at most 1. Holm adjustment runs over contrasts 3, 4, 5 and 6 together; an adjusted
   p below 0.05 is a rejection.
4. **Contrasts** (names as in the output):

| # | Output name | Per-problem quantity | Rule |
| --- | --- | --- | --- |
| 1 | `full_solution_vs_sc` | gain of `full_solution` (score minus matched SC) | "equivalent" if the 90% interval lies within +/-2 points (the margins count as within), otherwise "equivalence not shown"; no p-value |
| 2 | `answer_only_vs_full_solution` | credit of `answer_only` minus credit of `full_solution` | "non-inferior" if the lower end of the two-sided 95% interval is above -1.5 points, otherwise "non-inferiority not shown"; no p-value |
| 3 | `placebo_vs_self_reflect_correct_to_wrong` | correct-to-wrong rate of `placebo` minus that of `self_reflect` | one-sided, Holm |
| 4 | `failure_cause_vs_full_solution` | gain of `failure_cause` minus gain of `full_solution` | two-sided, Holm |
| 5 | `answer_conf_vs_full_solution` | gain of `answer_conf` minus gain of `full_solution` | two-sided, Holm |
| 6 | `best_low_bandwidth_vs_sc` | gain of `dissent_only` (fixed on the pilot) | two-sided, Holm |
| - | `failure_cause_v2_vs_full_solution` | gain of `failure_cause_v2` minus gain of `full_solution` | secondary: two-sided, unadjusted, outside the Holm family, reported beside contrast 4 |

   For contrast 2 the lower end of the two-sided 95% interval is the 2.5th percentile. That
   is the stricter reading of "lower 95% bound"; the one-sided 5th percentile is not used.
5. **Sensitivity and secondary analyses.** Each repeats every contrast with its own Holm
   family and with budgets recomputed on its own problems:
   - `level_3`, `level_4`, `level_5`: the problems of one level.
   - `no_round0_truncation`: drops the problems on which any round-0 agent was cut off at
     `max_tokens` (section 3).
   - `no_flagged_arm_truncation`: run only if pooled truncation of some revision arm differs
     from `self_reflect` by more than 2 points; drops the problems where `self_reflect` or a
     flagged arm (and, for a failure-cause arm, its failure messages) was cut off.
   - `prompt_tokens_in_budget`: the budget counts prompt tokens as well as generated tokens.
     An arm whose budget exceeds the 9 drawn samples is "not estimable", and so are the
     contrasts that need it; Holm then runs over the estimable ones.

   An analysis left with no problem is reported as not run. The primary analysis is `main`;
   all the others are secondary and none replaces it.
6. **Also reported**: the arm table (score, tokens, matched sample count, matched SC score
   and gain with its 95% interval), G1 and G2 overall and by level, G3 on the pooled run, run
   quality by arm (truncation, overflow, extraction failure, leakage rejection) and the
   self-consistency curve.
7. **Checks before any number is computed**; the script stops if one fails. The analysis
   files match the freeze and so do the pipeline files (section 10.3). `--expect-problems 600`
   is given and met. Each shard matches the sha256 in its `SHARD_DONE.json`, validates
   (counts, linkage, grading) and holds only stored grades equal to a fresh grading of the
   same text. No problem appears in two shards.
8. **Run once.** The script refuses an output directory that already holds results.
   Whatever the outcome, `main` is the primary result. A later correction would be a dated,
   labelled deviation published beside the original, never a replacement.
9. **Exploratory.** Change types, the source-of-change decomposition, diversity,
   recoverable-headroom recovery, conformity and all figures are descriptive, written after
   the freeze, and never replace a primary table. Anything not listed in this section is
   exploratory and is labelled so.

Operating characteristics, checked on 2026-11-09 and not part of the frozen code. The error
rates come from a one-off simulation of 2,000 analyses of 600 problems (1,000 resamples each)
with differences shaped like the pilot's. The bootstrap p-values reject a true null in 4.4% to
5.1% of analyses at the 0.05 level, and the Holm family errs in 5.4% (standard error 0.5
points). The equivalence rule declares equivalence in 5.1% of analyses when the true gain
sits on the +2 margin, and the non-inferiority rule in 3.3% on its -1.5 margin. The powers
use the normal approximation with the pilot's paired SDs (0.16 to 0.20) and 600 problems. A
two-sided test at 0.05 detects a true 3-point difference with 96% to 99.6% power (88% to 97%
at the strictest Holm threshold, 0.0125). If the true gain of `full_solution` were exactly 0,
contrast 1 would be declared equivalent in 61% of analyses at the pilot's SD of 0.196 (78% at
0.17); if `answer_only` and `full_solution` truly tie, contrast 2 would be declared
non-inferior in 58% to 61%, and in 96% to 97% if `answer_only` is 1.1 points better, as in
the pilot. "Equivalence not shown" and "non-inferiority not shown" are therefore not
evidence of a difference.

### 10.3 Freeze record (2026-11-09)

- `python scripts/29_freeze_analysis.py` wrote `configs/analysis_freeze.json`, once; it
  refuses to overwrite it. The manifest has two groups, hashed with the same
  line-ending-normalised sha256 as the prompts.
- **Analysis group**, frozen for good and checked by the test suite:
  `src/channel_audit/analysis/contrasts.py`, `analysis/integrity.py`, `analysis/stats.py`,
  `src/channel_audit/analysis_freeze.py` and `scripts/30_analyze.py`. New analysis goes in
  new files; these are not edited.
- **Pipeline group**, pinned for the confirmatory run: `scripts/30_analyze.py` refuses to
  run if one of these changed since the freeze and records their hashes in `RUN.json`.
  Later work on the generation side (M8) may change them. Re-running the confirmatory
  analysis later means checking out the freeze commit. The files are `aggregate`, `budget`,
  `channels`, `extract`, `gates`, `grade`, `leakage`, `models`, `mv_guard`, `quality`,
  `runner`, `schema`, `seeds`, `shards`, `store` and `validate`, all under
  `src/channel_audit/`. `backends/base.py`, `backends/mock.py` and `placebo.py` are imported
  but only produce generations, so they are not hashed; a test checks that every module the
  script imports is in one of the three lists.
- The code was tested on mock data only: planted effects are recovered
  (`tests/test_contrasts.py`), and a separately written implementation that reads the raw
  files and imports nothing from the package (`tests/independent_reference.py`) agrees with
  it on every arm and contrast, including the level and truncation runs
  (`tests/test_analysis_independent.py`). It has not been run on the confirmatory shards.
- Combined sha256 of the manifest: `adb3be510348d5bce6964fa9314ba499e9b1ee2beed682ed32e614dc92b5b492`

### 10.4 The confirmatory analysis run (2026-11-09)

- Run once on the 12 confirmatory shards in a molab sandbox (created 2026-11-09T06:10:53+00:00), with
  the code frozen in section 10.3: commit `3b00be0`, combined sha256
  `adb3be510348d5bce6964fa9314ba499e9b1ee2beed682ed32e614dc92b5b492`, checked by the script before it computed anything.
  Command: `scripts/30_analyze.py --root <shards> --out-dir <new directory> --expect-problems 600`
  with the default 10,000 resamples and seed 0.
- Environment: Python 3.12.13, numpy 2.5.4, math-verify 0.9.0, pydantic
  2.14.3, latex2sympy2_extended 1.11.0, Linux. The shards were downloaded from
  `Arshia-HZ/mas-channel-audit-raw` (`confirm/qwen3-1.7b/`). All the script's checks passed: each
  shard matches its `SHARD_DONE.json` hash and validates, every stored grade equals a fresh
  grading, 600 distinct problems.
- Reports, in `results/confirm_qwen3-1.7b_analysis/` (the only files of `results/` kept in git;
  force-added because `results/` is ignored). sha256 with LF line endings:
  `analysis.md` `e082d42ffa822f2b6665666950c000df009481234a59735ef40ad60340674ab6`,
  `analysis.json` `9ebaf31dda2e7807cc6aeb03d8be1e58b9d94f47a31d9bb4a4ded5456e5a6d70`,
  `RUN.json` `31d39708db3cd70b60e5eeae7442a923c5871d9d80febdc86f012e9242bff549`.
- No deviation from sections 7, 8 and 10.2, and no second run. `RUN.json` shows `hardware: null`
  because the generation sandbox's `HARDWARE.json` was not uploaded with the shards; the hardware is
  recorded in section 10.1.
- Primary analysis (`main`), decisions as the rules of section 10.2 give them: contrast 1
  "equivalence not shown", contrast 2 "non-inferiority not shown", contrast 3 rejected, contrasts 4,
  5 and 6 not rejected after Holm. G1 passed (38.6%), G2 was 2.6% (below the 5% line, as in the
  pilot; documented, not acted on, section 8) and G3 passed on the pooled run with no arm flagged,
  so `no_flagged_arm_truncation` did not run. The numbers are in the reports above.
- From here on, anything computed from these shards that is not in the reports is exploratory and
  is labelled so (section 10.2, item 9).

## 11. Closing the audit (added 2026-11-14)

- **Status.** The audit (M0 to M7) is complete. The confirmatory result is a precise null:
  no channel beats matched self-consistency (section 10.4; reports in
  `results/confirm_qwen3-1.7b_analysis/`). The written record is
  `reports/channel_audit_M0-M7_report.md`. The repository is closed at the tag
  `audit-final`; it takes no further changes except corrections recorded in this file.
- **M8 is superseded, not run.** The plan's M8 (second family, Qwen3-4B, mixed teams) and
  the open item of section 9 (choose a second family) are replaced by a follow-up study of
  trained aggregators. The owner changed direction between 2026-11-12 and 2026-11-14 in
  response to the null. No M8 generation was run. The follow-up lives in
  `Arshia-HZ/trained-reader-audit`, which imports this package pinned at `audit-final`
  and has its own preregistration.
- **Exploratory work after the freeze.** The decomposition of section 10.2 item 9 (change
  types, source of change, diversity, headroom, conformity) and its figures were run on
  2026-11-09 with `scripts/31_decompose.py` and `scripts/32_figures.py` (commit `a1bc1d1`).
  They are descriptive, were written after the freeze, and never replace a primary table.
  The reports are in `results/confirm_qwen3-1.7b_exploratory/`.
- **Reuse of the audit data.** The follow-up study reads the 600 confirmatory problems and
  their round-0 and pool generations as development data ("dev"). This happened after the
  analysis had run; nothing in sections 1 to 10 changes, and the analysis is not re-run.

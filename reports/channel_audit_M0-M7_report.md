# Channel Audit: report on milestones M0 to M7

Status on 2026-11-14. Repository `Arshia-HZ/mas-channel-audit`. Code and data as of commit
`a1bc1d1`; the repository is closed at the tag `audit-final`, which only adds this report
and documentation to that commit. Raw
generations are in the private dataset `Arshia-HZ/mas-channel-audit-raw`. The preregistered
record is `PREREG.md`; where this report and `PREREG.md` disagree, `PREREG.md` wins. Numbers
marked **exploratory** were computed after the analysis freeze and are descriptive only.

## 1. Summary

**Question.** Which message content, if any, lets small reasoning agents beat
self-consistency (SC) at the same token budget on maths problems?

**Setting.** Qwen3-1.7B in non-thinking mode, three agents per team, MATH test levels 3 to 5.
Each team writes one shared round 0 and then gets one revision under one of seven message
channels. The final answer is a plurality vote. Every channel is compared with SC drawn
from the same model at the same number of generated tokens.

**Answer (confirmatory, 600 problems × 2 replicates).** No channel beats matched SC. Every
arm's point estimate is below it, by 0.8 to 3.3 points, and the 95% interval excludes zero
for six of the seven real channels. The best is `dissent_only` (−0.8 points, [−2.0, +0.3]).
Showing peers' full solutions is 1.7 points below SC ([−3.3, −0.0]). Showing solutions to
a different problem (placebo) costs 13.2 points and causes 12.8 points more correct-to-wrong
changes than self-reflection. This is the only Holm-significant contrast.

**Why (exploratory).** Revision is not useless, but it is poor value. The three round-0
answers alone vote to 69.7%. One revision round lifts that to about 71.5–72.0% and costs
roughly 2.2 extra samples' worth of tokens. Spending the same tokens on 2.2 more independent
samples reaches 73.5%. Message content does change answers, but in both directions:
full solutions lead agents to adopt a right peer answer 64% of the time and to copy a wrong
one 33% of the time.

| Arm | What each agent sees | Accuracy | Tokens / replicate | Matched SC k | SC accuracy | Gain over SC [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| `self_reflect` | its own solution | 71.7% | 4,252 | 5.19 | 73.5% | −1.8 [−3.0, −0.7] |
| `answer_only` | peers' final answers | 71.6% | 4,271 | 5.21 | 73.6% | −2.0 [−3.2, −0.7] |
| `answer_conf` | answers + confidence | 72.0% | 4,269 | 5.21 | 73.6% | −1.6 [−2.9, −0.3] |
| `dissent_only` | "a peer disagrees" (only if so) | 71.3% | 3,493 | 4.26 | 72.2% | −0.8 [−2.0, +0.3] |
| `failure_cause` | each peer's suspected first error (v1) | 72.0% | 5,799 | 7.07 | 75.3% | −3.3 [−4.8, −1.8] |
| `failure_cause_v2` | the same, stricter prompt | 71.9% | 4,932 | 6.02 | 74.3% | −2.4 [−3.7, −1.1] |
| `full_solution` | peers' full solutions | 71.8% | 4,209 | 5.14 | 73.5% | −1.7 [−3.3, −0.0] |
| `placebo` | full solutions to another problem | 60.2% | 4,165 | 5.08 | 73.4% | −13.2 [−15.1, −11.3] |

SC by sample count (same 600 problems): k=1 63.7%, k=3 69.7%, k=5 73.4%, k=9 76.8%. One SC
sample costs about 819 generated tokens.

## 2. Design in one page

- **Unit.** A replicate is one problem with a shared round 0 from three agents. Each agent
  solves independently, boxes its answer and states `Confidence: N`. Every arm branches from
  that same stored round 0, so differences between arms come from the message, not from
  different starting samples.
- **Arms.** Seven preregistered channels, plus `failure_cause_v2` (section 4.5). Every
  channel gives each agent exactly one revision. `dissent_only` revises only agents who saw
  a disagreement. `failure_cause` first generates one failure message per peer solution,
  and those tokens are charged to the arm.
- **Baseline.** Six extra independent samples per replicate (the SC pool). Together with
  round 0 they give nine samples. Matched SC is the plurality credit at the sample count
  whose token cost equals the arm's mean generated tokens, interpolated linearly between
  integer counts.
- **Scoring.** Plurality vote over the three final answers. An m-way tie that contains the
  correct answer earns 1/m. An unparsable answer never votes.
- **Decoding.** Temperature 0.7, top-p 1.0, `max_tokens` 2,048, `max_model_len` 12,288, and
  the same for every arm.
- **Grading.** One grader (`grade.py`, math-verify) and one extractor (`extract.py`), with no
  fallbacks. A truncated output is wrong unless a complete box appears before the cut-off.
- **Analysis.** The unit is the problem: replicates are averaged within a problem, then
  arms are compared with a paired bootstrap over problems (10,000 resamples, seed 0).
  Holm runs over contrasts 3 to 6. Contrast 1 is an equivalence test (90% CI within ±2
  points). Contrast 2 is non-inferiority (lower 95% bound above −1.5 points).

| # | Hypothesis | Contrast |
| --- | --- | --- |
| 1 | H1: `full_solution` is equivalent to matched SC | `full_solution` gain over SC |
| 2 | H2: `answer_only` is not worse than `full_solution` | `answer_only` − `full_solution` |
| 3 | H3: placebo causes more correct-to-wrong changes than self-reflection | `placebo` − `self_reflect`, correct-to-wrong rate |
| 4 | H4: `failure_cause` beats `full_solution` at matched budget | difference in gains |
| 5 | H4: `answer_conf` beats `full_solution` at matched budget | difference in gains |
| 6 | The best low-bandwidth arm (chosen on the pilot) beats SC | `dissent_only` gain over SC |

## 3. Milestones

### M0: scaffold (`27be537`)

- Phase 1 (`causal-mas-distill`) was frozen at tag `phase1-final`, with a README pointer to
  this repository. Phase 1 student results on 231 held-out problems: BASE 41.1%,
  rejection-sampled solutions 48.9%, full transcripts 49.8%, debate final solutions 47.0%.
  80% of its problems were ceiling problems, and it had no matched non-interactive control.
- A new package `channel_audit`, with `pyproject.toml`, ruff, pytest, GitHub Actions CI and
  `AGENTS.md` (ten rules: never edit frozen files, never add a second grader, never hide a
  failed generation, the store is append-only, no outcome-based filtering, every behaviour
  tested).

### M1: core contracts (`b92b60c`)

- `schema.py` (typed `Problem`, `Generation` and `Graded` records), `seeds.py` (a seed and
  `gen_key` derived from identity fields only), `store.py` (append-only JSONL with
  duplicate-key detection, recovery from a torn last line, resume), `extract.py`, `grade.py`
  (refuses to import without math-verify) and `aggregate.py` (plurality with 1/m tie credit).
- Tests cover nested, missing, truncated and multiple boxes, grader equivalences, key and
  seed stability, a killed writer and resume without duplicates, and tie credit.

### M2: mock end-to-end pipeline (`e1f6f9b`)

- A backend protocol and a deterministic mock backend with planted arm effects. Pure
  message builders (`channels.py`), placebo donor matching (`placebo.py`: same subject and
  level, nearest token length, never the same problem), the failure-message leakage filter
  (`leakage.py`), token accounting with SC matching (`budget.py`) and the runner.
- The whole script chain runs on 20 fake problems without a GPU. Tests show that every arm
  links to the same three round-0 keys, that `self_reflect` holds no peer text, that the
  answer arms hold no peer reasoning, that placebo uses the `full_solution` template
  byte for byte, and that a planted effect is recovered within its interval.

### M3: real backends and the T4 check (`66751b3` to `e4944b3`)

- vLLM backend using the tokenizer's chat template, per-request seeds and exact token
  counts. An HF backend for P100 throughput checks only. Pinned model revisions, with
  Qwen3 non-thinking mode enforced by the config validator. A Colab/Kaggle runner notebook.
- **T4 check, Qwen3-1.7B, 50 problems:** at `max_tokens` 1,024, round 0 truncated 27.3%; at
  2,048 it truncated 5.3%, and none of the 40 truncated outputs looped. So 2,048 became the
  limit for every model. At a context of 8,192, six `full_solution` requests overflowed;
  12,288 fixed that.
- **Reproducibility:** rerunning the same batch on the same backend gave identical text,
  token counts and finish reasons for 150 of 150 generations. Rerunning a 10-problem
  subset gave 23 of 30, so batch composition changes vLLM outputs. This is measured and
  documented, not hidden.
- **Throughput on T4 (vLLM 0.33.0, fp16):** about 420 generated tokens/s for round 0 and the
  pool, and 175 to 190 tokens/s for long-prompt revision and failure-message calls.
- The owner read 30 outputs across all arms, then froze `prompts/v1` (combined sha256
  `545ebede…`).

### M4: problem freeze and screening (`3796e24` to `6e79f1e`)

- MATH test set (`EleutherAI/hendrycks_math` at a pinned revision, 5,000 rows), ordered by a
  fixed hash of the problem id: 300 screen, 200 pilot and 3,250 confirm (levels 3 to 5).
  Only the id and level are read. The split manifest with sha256 hashes is committed.
- **Extraction finding:** Qwen3-1.7B often boxes its `Confidence: N` line. The naive last-box
  rule graded that box as the answer, so 35% of screening answers were wrong for a
  formatting reason. The fixed rule skips confidence boxes and raises Qwen3-1.7B screening
  accuracy from 44.3% to 69.6%. All earlier results were re-graded into new directories.
- **Screening** (300 problems × 4 samples):

| Model, round-0 prompt | Accuracy | Levels 3–5 | Disagreement, levels 3–5 | Truncated | No box |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B, non-thinking | 69.6% | 63.5% | 47.0% | 6.0% | 0.9% |
| Qwen2.5-Math-1.5B (base) | 24.8% | 23.7% | 68.9% | 28.7% | 18.1% |
| Llama-3.2-3B-Instruct | 10.5% | 7.5% | 22.4% | 3.7% | 70.4% |

- **Gate G0** (Qwen2.5-1.5B-Instruct, published MATH score 55.2): it passes with the plain
  prompt and greedy decoding (52.0%, 0.3% unboxed). With the round-0 prompt it scores only
  33.7%, because asking for a confidence crowds out the box. Finding for the paper: a
  confidence request costs format compliance in small instruct models, Llama-3.2-3B
  included (845 of 1,200 outputs unboxed).
- **Primary model: Qwen3-1.7B.** Once broken samples are removed, both Qwen candidates have
  about 33% usable disagreement, but Qwen3-1.7B has 80% fully valid problems against 11%.

### M5: pilot and gates (`59748c4` to `e5cd53b`)

- 200 pilot problems × 2 replicates, run in four validated shards that were uploaded as they
  finished. Every (subject, level) group stays inside one shard, so placebo donors do not
  depend on the shard layout.

| Gate | Result | Need | Verdict |
| --- | --- | --- | --- |
| G1 round-0 disagreement | 33.0% of replicates | ≥ 15% | pass |
| G2 recoverable headroom (a correct answer present, plurality wrong) | 4.2% | ≥ 5% | **fail** (narrow) |
| G3 overflow and truncation | 0 overflow, every arm under 3% | section 3 | pass |
| G4 failure messages not vague (40, seed 0) | 15% | ≥ 50% | **fail** |

- **Pilot arm results:** every arm is 1.8 to 4.5 points below matched SC, and placebo is
  13.8 below. Paired SDs of 0.16 to 0.20 imply at most 362 problems for a 3-point effect
  (814 for 2 points). `dissent_only` was fixed as the best low-bandwidth arm for contrast 6.
- **Failure-message v2 re-test** (after G4 failed): a stricter one-sentence prompt
  (`prompts/v2`, frozen 2026-11-08), judged on a fresh draw (seed 1). Format was fixed (40
  of 40 single sentences, no artifacts), but content was not: only 27.5% were not vague, and
  10 messages criticised a step that was correct. Of 14 truly wrong solutions, one message
  found the real error. The same-model reviewer does not find errors, whatever the wording.
  The arm effect did not change (−3.5 against −3.6 over SC).

### M6: confirmatory run (`03789f5` to `4d7449d`)

- Design decided before any generation (PREREG §8): the first 600 confirm problems (levels
  3, 4 and 5: 199, 193 and 208), 2 replicates, the seven arms plus `failure_cause_v2`, in 12
  shards of 40 to 54 problems. `PREREG.md` §1–8 was committed (`4d7449d`, 19:15:35 UTC)
  before the first confirmatory generation (19:17:18 UTC).
- Hardware moved from the T4 to an RTX PRO 6000 Blackwell (molab), conditional on a check
  made on already-seen pilot shard 00. Every arm's truncation and extraction rates were
  within 1.3 points, and overflow was 0, so the check passed. The whole confirmatory run
  used one GPU type.
- Run quality was the only thing looked at before the freeze: zero overflow, round 0 8.2%
  truncated, revision arms 0.6% to 1.8%, failure-message leakage rejection 34.2% (v1) and
  4.7% (v2).

### M7: frozen analysis, run once (`3b00be0`, `0a4fcce`)

- Analysis code (`contrasts.py`, `integrity.py`, `stats.py`, `analysis_freeze.py`,
  `30_analyze.py`) was frozen in `configs/analysis_freeze.json` (combined sha256
  `adb3be51…`) before any accuracy was computed. Sixteen pipeline modules are pinned too,
  and the script refuses to run if any of them changed.
- **Validation of the analysis before the real run:** planted mock effects are recovered. An
  independently written reference implementation, which imports nothing from the package,
  agrees on every arm and contrast. A simulation of 2,000 null analyses gave bootstrap type
  I error of 4.4% to 5.1% and a Holm family error of 5.4%.
- **Integrity checks before computing:** shard sha256 values match `SHARD_DONE.json`, every
  shard validates, every stored grade equals a fresh re-grade, there are 600 distinct
  problems, and the script refuses an output directory that already holds results.
- The analysis ran once on 2026-11-09, with no deviation. Its reports are in
  `results/confirm_qwen3-1.7b_analysis/`.

### After M7: exploratory decomposition and figures (`a1bc1d1`)

- New files outside the freeze: `analysis/decompose.py`, `analysis/figures.py`, and scripts
  31 and 32. Every output carries an EXPLORATORY banner. Reports are in
  `results/confirm_qwen3-1.7b_exploratory/`. Figures 2 to 4 are drawn locally and are not
  in git.

## 4. Confirmatory results (primary analysis `main`)

### 4.1 Contrasts

| # | Contrast | Mean | 95% CI | p | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `full_solution` gain over SC | −1.7 | [−3.3, −0.0] | – | – | equivalence not shown |
| 2 | `answer_only` − `full_solution` | −0.2 | [−1.6, +1.2] | – | – | non-inferiority not shown |
| 3 | placebo − self_reflect, correct-to-wrong | +12.8 | [+11.4, +14.3] | 0.0001 | 0.0004 | **rejected** |
| 4 | `failure_cause` − `full_solution` (gains) | −1.6 | [−3.1, −0.0] | 0.045 | 0.136 | not rejected |
| 5 | `answer_conf` − `full_solution` (gains) | +0.1 | [−1.2, +1.4] | 0.89 | 0.89 | not rejected |
| 6 | `dissent_only` gain over SC | −0.8 | [−2.0, +0.3] | 0.16 | 0.32 | not rejected |
| – | `failure_cause_v2` − `full_solution` (secondary) | −0.7 | [−2.2, +0.7] | 0.32 | – | unadjusted |

Reading against the hypotheses:

- **H1** (`full_solution` equivalent to SC): equivalence is not shown. The 90% interval
  reaches about −3 points, past the −2 margin. The point estimate is below SC, not equal to
  it.
- **H2** (answers alone are not worse than full solutions): the two are almost identical
  (−0.2), but the 95% interval reaches −1.6, just past the −1.5 margin. Not shown, narrowly.
- **H3** (placebo hurts): strongly confirmed, at every level and in every sensitivity run.
- **H4** (failure cause or confidence beats full solutions): not supported. `failure_cause`
  is, if anything, worse, because its extra failure-message tokens buy nothing.
- **Contrast 6:** the cheapest channel, `dissent_only`, comes closest to SC but does not
  beat it.

### 4.2 Gates on the confirmatory run

- G1: 38.6% (level 3 29%, level 4 34%, level 5 53%). Pass.
- G2: 2.6% of replicates (31 of 1,200). This is below the 5% line, as in the pilot.
  Documented, not acted on.
- G3: zero overflow, highest revision-arm truncation 1.8%, no flagged arm. Pass.

### 4.3 Sensitivity and secondary runs

| Run | Problems | Contrast 1 | Contrast 2 | Contrast 3 | Contrasts 4–6 |
| --- | --- | --- | --- | --- | --- |
| level 3 | 199 | −2.9 [−5.5, −0.4] | +0.6, non-inferior | +13.2, rejected | none rejected |
| level 4 | 193 | −1.6 [−4.2, +0.8] | +0.2, not shown | +13.0, rejected | none rejected |
| level 5 | 208 | −0.5 [−3.7, +2.8] | −1.4, not shown | +12.3, rejected | none rejected (`failure_cause` −3.3, Holm p 0.07) |
| no round-0 truncation | 482 | −2.1 [−3.8, −0.4] | −0.1, not shown | +14.0, rejected | none rejected |
| prompt tokens in budget | 600 | not estimable | −0.2, not shown | +12.8, rejected | `dissent_only` −2.8 [−4.1, −1.6], rejected |

Counting prompt tokens pushes most arms' budgets past the nine drawn samples, so their
contrasts are not estimable. Under that rule `dissent_only` is significantly below SC.
The flagged-arm truncation run was not triggered.

## 5. Exploratory results (descriptive, unadjusted)

**Change types (share of agents).**

| Arm | Wrong → right | Right → wrong | Wrong → other wrong | Net |
| --- | --- | --- | --- | --- |
| `self_reflect` | 3.8% | 0.6% | 7.1% | +3.2 |
| `answer_only` | 3.8% | 0.6% | 6.7% | +3.2 |
| `answer_conf` | 3.9% | 1.4% | 6.8% | +2.5 |
| `dissent_only` | 3.4% | 0.3% | 5.4% | +3.1 |
| `failure_cause` | 4.2% | 0.9% | 8.4% | +3.3 |
| `full_solution` | 10.8% | 4.2% | 13.3% | +6.6 |
| `placebo` | 2.6% | 13.4% | 17.4% | −10.8 |

- **Source of change.** This splits each arm's net change into three parts:
  - spontaneous change (from `self_reflect`): +3.2 points;
  - the form of reading three solutions (placebo minus self_reflect): −14.0;
  - content (arm minus placebo): +13.3 to +17.4.

  Content therefore only cancels the form cost. Every real channel ends up close to the
  spontaneous +3.2, except `full_solution` at +6.6.
- **Agent-level gains do not reach the vote.** `full_solution` doubles the agent-level net
  change, yet its vote accuracy equals self-reflection's (71.8% against 71.7%). It removes
  diversity: 75.8% of teams end with at most one distinct answer, against 56.7% after
  self-reflection and 61.4% in round 0. Convergence that is right half the time does not
  improve the plurality.
- **Conformity.** Agents exposed to a wrong peer answer copied it 33.4% of the time under
  `full_solution`, against 2.7% under `self_reflect`. They adopted a right peer answer
  64.3% of the time, against 19.1%.
- **Headroom.** Only 31 of 1,200 replicates had a correct answer in round 0 that the vote
  missed.
  - `full_solution` recovered 36.6% of them ([20.4, 53.8]), about 11 cases.
  - It kept 97.2% of the 801 replicates the vote already got right, so it broke about 22.
  - Self-reflection kept 99.3% and recovered 20.4%.

  With N = 3, what the vote can gain is small and what it can lose is large.

## 6. What the null means

1. **Revision is poor value at this scale.** The arms land 1.6 to 2.3 points above the
   round-0 vote (69.7%) at a cost of about 2.2 extra samples. The same tokens spent on
   independent samples give 3.8 points. To pay for itself, a channel would need to add
   about twice what any channel adds.
2. **The channel content matters for agents, not for the vote.** Content clearly changes
   agents' answers (contrast 3, and the source decomposition). With three agents and a vote,
   the changes in both directions nearly cancel.
3. **A same-model critic is not a critic.** Two prompt versions produced failure messages
   that were mostly vague or wrong. Even when well formed, they did not change outcomes.
4. **Headroom is the binding limit.** G2 failed twice (4.2% pilot, 2.6% confirm). With
   three round-0 answers there is too little "right answer present but outvoted" for any
   channel to exploit.
5. **The result is a precise null, not an underpowered one.** The interval half-widths are
   about 1.2 to 1.6 points, enough to rule out a 3-point gain for every arm. Per
   PREREG §7, this is the "precise-null paper" outcome, not a failure.

## 7. Deviations, limitations and caveats

- **G2 and G4 failed.** Both are documented and were not acted on, as preregistered. The
  v2 failure-message prompt was a planned re-test, kept outside the Holm family.
- **Shard 04 failed G3 on its own**, with two arms at or above 5% truncation. It was kept,
  and G3 is judged on the pooled run, as for the pilot.
- **Hardware switch** from T4 to Blackwell, checked on pilot shard 00. Samples are not
  bit-identical across GPUs, and vLLM outputs depend on batch composition (M3). Neither
  affects the comparison between arms, because all arms of the confirmatory run share one
  GPU type and one round 0.
- `RUN.json` records `hardware: null`, because `HARDWARE.json` was not uploaded with the
  shards. The hardware is recorded in PREREG §10.1.
- **Scope limits:** one model (Qwen3-1.7B, non-thinking), one dataset (MATH levels 3 to 5),
  teams of three, one revision round, and the same model as reviewer.
- **The plan's M8 has not started**: no second family, no Qwen3-4B and no mixed teams.
  Llama-3.2-3B-Instruct is unusable with the frozen round-0 prompt, and no replacement
  family has been chosen.

## 8. Reproducibility record

| Item | Value |
| --- | --- |
| Problem manifest | `data/problems/split_manifest.json` (MATH revision `21a56338…`) |
| Prompts | `prompts/v1/MANIFEST.json` `545ebede…`; `prompts/v2/MANIFEST.json` `5903dd08…` |
| PREREG timestamp | §1–8 in `4d7449d` (2026-11-08 19:15:35 UTC), before the first confirmatory generation |
| Analysis freeze | `configs/analysis_freeze.json` combined `adb3be51…`, commit `3b00be0` |
| Analysis reports | `results/confirm_qwen3-1.7b_analysis/` (`analysis.md` `e082d42f…`, `analysis.json` `9ebaf31d…`, `RUN.json` `31d39708…`) |
| Exploratory reports | `results/confirm_qwen3-1.7b_exploratory/` (commit `a1bc1d1`) |
| Raw shards | `Arshia-HZ/mas-channel-audit-raw`, `confirm/qwen3-1.7b/shard_00`–`shard_11` |
| Generation stack | vLLM 0.33.0, torch 2.14.0, transformers 5.20.2, fp16, RTX PRO 6000 Blackwell |
| Analysis stack | Python 3.12.13, numpy 2.5.4, math-verify 0.9.0 |
| Test suite | 768 passed, 3 skipped at `a1bc1d1`; ruff clean |

To re-run the confirmatory analysis, check out `3b00be0`, download the shards, and run
`scripts/30_analyze.py --root <shards> --out-dir <new dir> --expect-problems 600`.

## 9. What carries forward

- **Reusable as is:** the grader and extractor (with the confidence-box rule), deterministic
  seeds and keys, the append-only store, sharded runs with validation and upload, token
  accounting with interpolated SC matching, the problem-level paired bootstrap with Holm,
  and the freeze and run-once discipline.
- **Data already paid for:** 600 dev problems with nine independent Qwen3-1.7B samples per
  replicate (round 0 plus pool), and 200 pilot problems, all graded. These can be
  subsampled for SC and pass@k at k ≤ 9 at no new cost.
- **Findings that motivate the next step:** revision loses to sampling, content changes
  agents in both directions, a same-model critic is useless, and headroom at N = 3 is about
  2.6%.

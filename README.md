# mas-channel-audit

**Which message content, if any, lets small reasoning agents beat self-consistency at the same token budget on math problems?**

An inference-only audit of inter-agent message channels (self-reflection, answers only, answers + confidence, dissent only, failure cause, full solutions, placebo) against budget-matched self-consistency. No fine-tuning, no distillation, no teacher API.

Successor to [`causal-mas-distill`](https://github.com/Arshia-HZ/causal-mas-distill) (Phase 1, archived at tag `phase1-final`).

## Status

M0 to M3 are done (the T4 backend check ran; see `PREREG.md` section 2 and 3). M4 (problem freeze and model screening) is done. M5 (pilot and gates) tooling is implemented and tested; the pilot has not been run yet.
See `AGENTS.md` for contributor and coding-agent rules.

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check .
pytest
```

Raw generations are stored in a private Hugging Face dataset, not in Git:
`Arshia-HZ/mas-channel-audit-raw`.

## Mock pipeline (M2, no GPU)

```bash
python scripts/10_round0.py --out runs/mock --n-fake-problems 20 --replicates 2
python scripts/11_sc_pool.py --out runs/mock
python scripts/12_failure_messages.py --out runs/mock
python scripts/13_branch_arms.py --out runs/mock
python scripts/20_grade.py --out runs/mock
python scripts/21_validate_run.py --out runs/mock
```

`prompts/v1/` holds draft prompts; they are frozen after M3.

## M3: real backends and the T4 check

Backends: `vllm` (default, T4) and `hf` (P100, throughput checks only). Do not
mix backends within one model. Model settings live in `configs/models/*.yaml`
(revisions pinned; Qwen3 is forced to non-thinking mode).

**Easiest:** open [`notebooks/runner.ipynb`](notebooks/runner.ipynb) on a GPU runtime and
*Run all*: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arshia-HZ/mas-channel-audit/blob/main/notebooks/runner.ipynb).
It works on Colab and Kaggle, loads the model once, is resumable, and saves a zip of the
results. The same steps as plain commands (Kaggle T4 with internet, one command per cell):

```bash
!git clone https://github.com/Arshia-HZ/mas-channel-audit.git && cd mas-channel-audit && pip install -e ".[vllm,data]"
```
```bash
!python scripts/03_render_prompt.py --model-config configs/models/qwen3-1.7b.yaml
!python scripts/03_make_check_problems.py --n 50 --out check/problems.jsonl
```
```bash
!python scripts/10_round0.py --out runs/check --problems check/problems.jsonl --model-config configs/models/qwen3-1.7b.yaml --replicates 1 --log-prompts
!python scripts/11_sc_pool.py --out runs/check
!python scripts/12_failure_messages.py --out runs/check
!python scripts/13_branch_arms.py --out runs/check
!python scripts/20_grade.py --out runs/check
!python scripts/21_validate_run.py --out runs/check
!python scripts/03_review_sample.py --out runs/check --n 30
```
Read `runs/check/review_sample.md`. For reproducibility, repeat the steps into
`runs/check2` and run `python scripts/03_repro_check.py --a runs/check --b runs/check2`.
Only after the review, and with the owner's approval, run
`python scripts/03_freeze_prompts.py` to freeze `prompts/v1`.

## M4: frozen problems and model screening

```bash
python scripts/04_freeze_problems.py --out data/problems   # needs [data] and internet
```
Splits the MATH test set (pinned revision) into 300 screen, 200 pilot and the rest confirm by a
fixed hash. Only `data/problems/split_manifest.json` is committed; the problem files are
rebuilt and checked against its sha256.

Screen one model (4 samples per problem), on a GPU, then compare:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arshia-HZ/mas-channel-audit/blob/main/notebooks/screen.ipynb)
(run once per `MODEL_CONFIG`; the sanity model also runs with `SCREEN_PROMPTS = ["plain"]` and, for a like-for-like check against its published greedy score, `TEMPERATURE = 0`), or as commands:

```bash
python scripts/04_screen.py --model-config configs/models/qwen3-1.7b.yaml --problems data/problems --out runs/screen_qwen3-1.7b
python scripts/04_screen_report.py --runs runs/screen_qwen3-1.7b runs/screen_qwen2.5-math-1.5b --json screen_summary.json
```

## M5: pilot in shards, gates and power inputs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arshia-HZ/mas-channel-audit/blob/main/notebooks/pilot.ipynb)
runs all arms on the 200 pilot problems (2 replicates) in resumable shards of about 50
problems; each finished shard is validated and uploaded to the private dataset
(`HF_TOKEN` secret with write access). As commands:

```bash
python scripts/05_pilot_shards.py --model-config configs/models/qwen3-1.7b.yaml --problems data/problems --out runs/pilot --upload-repo Arshia-HZ/mas-channel-audit-raw
python scripts/05_failure_sample.py --root runs/pilot --out-dir g4      # 40 messages to label
python scripts/05_gate_check.py --root runs/pilot --labels g4/labels.json --json pilot_report.json
```
`05_gate_check.py` prints gates G1 to G4, the arm table (accuracy, tokens, matched
self-consistency) and the paired-difference SDs with the sample size they imply.

### Re-testing the stricter failure-message prompt (v2)

After the pilot's gate G4 failed, `prompts/v2/failure_msg.txt` asks for one specific sentence.
Once the owner has frozen it (`python scripts/03_freeze_prompts.py --dir prompts/v2`),
[notebooks/retest.ipynb](https://colab.research.google.com/github/Arshia-HZ/mas-channel-audit/blob/main/notebooks/retest.ipynb)
(or `scripts/05_retest_failure.py`) copies the finished pilot shards, generates only the v2
failure messages and revisions, and uploads the copies under `pilot/<model>-v2`. Draw the
fresh sample with `05_failure_sample.py --variant v2 --seed 1` and check it with
`05_gate_check.py --variant v2 --seed 1 --labels ...`.

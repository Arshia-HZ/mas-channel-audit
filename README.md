# mas-channel-audit

**Which message content, if any, lets small reasoning agents beat self-consistency at the same token budget on math problems?**

An inference-only audit of inter-agent message channels (self-reflection, answers only, answers + confidence, dissent only, failure cause, full solutions, placebo) against budget-matched self-consistency. No fine-tuning, no distillation, no teacher API.

Successor to [`causal-mas-distill`](https://github.com/Arshia-HZ/causal-mas-distill) (Phase 1, archived at tag `phase1-final`).

## Status

M0 to M2 are done. M3 (real backends) is implemented and tested against fake engines; the T4 check below has not been run yet.
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

# mas-channel-audit

**Which message content, if any, lets small reasoning agents beat self-consistency at the same token budget on math problems?**

An inference-only audit of inter-agent message channels (self-reflection, answers only, answers + confidence, dissent only, failure cause, full solutions, placebo) against budget-matched self-consistency. No fine-tuning, no distillation, no teacher API.

Successor to [`causal-mas-distill`](https://github.com/Arshia-HZ/causal-mas-distill) (Phase 1, archived at tag `phase1-final`).

## Status

M0 and M1 (core contracts) are done; M2 (mock end-to-end pipeline) is in. Nothing here has touched a real model yet.
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

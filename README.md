# mas-channel-audit

**Which message content, if any, lets small reasoning agents beat self-consistency at the same token budget on math problems?**

An inference-only audit of inter-agent message channels (self-reflection, answers only, answers + confidence, dissent only, failure cause, full solutions, placebo) against budget-matched self-consistency. No fine-tuning, no distillation, no teacher API.

Successor to [`causal-mas-distill`](https://github.com/Arshia-HZ/causal-mas-distill) (Phase 1, archived at tag `phase1-final`).

## Status

M0: repository scaffold. See `AGENTS.md` for contributor and coding-agent rules.

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check .
pytest
```

Raw generations are stored in a private Hugging Face dataset, not in Git.

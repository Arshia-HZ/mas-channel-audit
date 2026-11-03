# Rules for this repository

This is a scientific measurement pipeline. A silent change can invalidate results.

1. Implement only the current milestone. Do not refactor other code.
2. Never edit files in prompts/v1/, PREREG.md, or a frozen config without explicit approval.
3. Never add a second grader or answer extractor. Use grade.py and extract.py.
4. Never catch an exception to hide a failed generation. Record it.
5. Never overwrite or delete a run. The store is append-only.
6. Never use an arm's outcome to filter problems or select examples.
7. Every new behavior needs a test. Run pytest and ruff before finishing.
8. Add no dependency without saying why.
9. Use mock backends in tests. Tests must not download models.
10. Report what you ran and its output. Do not claim a test passed without running it.

# coral-hebbian: agent notes

Research repo: does a Hebbian/fast-weight state matrix survive INT8/INT4 storage, as it would on the Coralboard's Coral NPU? See README.md for the question and plan. Project status lives in the Obsidian vault note `Projects/Coral Hebbian.md`.

## Setup

- Python venv at `.venv/` (CPU torch). Always run through it: `.venv/Scripts/python ...`.
- Tests: `.venv/Scripts/python -m pytest -q`
- Sweep: `.venv/Scripts/python experiments/sweep.py [--quick]` writes to `results/`.

## Rules

- Keep `fastweight.memory.update` pure, with no data-dependent Python control flow. It will be exported as a static graph for the Torq (MLIR) compiler, which needs fixed shapes.
- Quantized state is simulated as float math inside a step plus requantization between steps. Don't quantize intermediates unless you're deliberately modelling the NPU's accumulators.
- All configs in a sweep must see the same stream for a given seed (paired comparisons). Don't add RNG draws to the stream path that depend on the config.
- This is intended to become a public write-up for college applications. Keep results reproducible (seeded, CSV and plots in `results/`) and don't overstate findings in docs.
- Zach logs hours in `HOURS.md`. Leave it alone unless asked.

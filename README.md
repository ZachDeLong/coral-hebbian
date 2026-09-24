# coral-hebbian

Can a Hebbian fast-weight memory stay useful when its state is stored in INT8 or INT4?

NPUs like the Coral NPU on Google's Coralboard (Synaptics Astra SL2619, 1 TOPS) are inference-only: weights are fixed at compile time. But a fast-weight memory doesn't need weight updates. It learns by updating a state matrix every step,

```
S <- λ·S + β·v·kᵀ            (Hebbian)
S <- λ·S + β·(v - λ·S·k)·kᵀ  (delta rule)
```

and to the NPU that's just a tensor carried from one step to the next. So the question is numerical: what happens to that memory when S is rounded to 8 or 4 bits after every write?

## What we expect to break

- **Decay deadzone.** With round-to-nearest, `round(λ·s) == s` whenever `(1-λ)·|s| < 0.5`. At λ = 0.99, every int8 code below 50 never decays, so old memories freeze instead of fading.
- **Write underflow.** Each write adds a small outer product. Entries smaller than half a quantization step round to zero, and the write is lost.
- **Fixed scale.** Torq compiles ahead of time, so the state's quantization scale is probably fixed at compile time. A scale that fits the steady state leaves each write only a few steps of resolution.

Candidate fixes are sweep axes: stochastic rounding (unbiased on average), the delta rule (error-correcting, bounded state), and per-row or per-step (dynamic) scales.

## Experiment

Associative recall. Stream T random `(key, value)` pairs into the memory, then query every key and decode against a codebook of `vocab` values. Accuracy by *age* (writes since a pair was stored) gives the full forgetting curve in one run. Each config sees the identical stream, so comparisons are paired.

Sweep: `{fp32, int8, int4} × {nearest, stochastic} × {static, dynamic scale} × {Hebbian, delta} × λ`.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python -m pytest -q
.venv/Scripts/python experiments/sweep.py --quick   # seconds
.venv/Scripts/python experiments/sweep.py           # full grid, several minutes on CPU
```

Results land in `results/`: `sweep.csv`, `curves.csv`, and `recall_<scale>.png`.

## Plan

1. **Simulate** (this repo, now): fake-quant sweep on a PC.
2. **Validate on the board**: export the step `(x, S) -> (y, S')` as a static graph, compile with Torq, and check that the NPU matches the simulation. Differences in rounding, saturation, or requantization are findings too.
3. **Demo**: a camera few-shot learner. Show the Coralboard a new object a few times and it recognizes it afterward, learning with no backprop.

## Layout

```
fastweight/quant.py     fake quantization (nearest / stochastic, saturating)
fastweight/memory.py    update rules + quantized state carrier
fastweight/recall.py    associative recall task, calibration
experiments/sweep.py    the grid, CSVs, plots
tests/                  deadzone, stochastic unbiasedness, exact recall, delta overwrite
HOURS.md                time log
```

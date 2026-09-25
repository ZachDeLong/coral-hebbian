# Write-size test (2026-09-24)

**Question:** does one rule explain the INT8 and INT4 results, namely that a quantized memory works as long as a single write is at least about half a quantization step?

**Prediction, stated before running:** with no decay (λ = 1) the Hebbian state keeps growing, so writes shrink relative to the fixed INT8 step. INT8 should start failing after about 1–2k writes.

**Runs:** `sweep.py --seeds 3` at T = 512 over all λ, plus T = 1024 / 2048 / 4096 at λ ∈ {0.999, 1.0}. Each quantized run records its mean write size in quantization steps (`write_lsb`) and readout fidelity (mean cosine between readout and stored value). `collapse.py` pairs each config with fp32 on the same stream. Plots: `collapse_recalled.png`, `collapse_fidelity.png`.

## Results

1. **Prediction confirmed.** INT8 Hebbian at λ = 1 with a static scale:

   | T | write size (steps) | recalled ÷ fp32 | fidelity ÷ fp32 |
   |---|---|---|---|
   | 512 | 0.71 | 0.99 | 0.98 |
   | 1024 | 0.51 | 0.94 | 0.95 |
   | 2048 | 0.37 | 0.85 | 0.92 |
   | 4096 | 0.25 | 0.74 | 0.86 |

   The decline is gradual, not a sharp cliff. It becomes visible below about 0.7 steps.
2. **Above about 0.7 steps, everything matches fp32** (ratio 0.98–1.03): INT8 and INT4, Hebbian and delta, every λ and T, with nearest rounding.
3. **Decay protects INT8.** At λ = 0.999 the state stays bounded, so writes never shrink below about 0.64 steps even at T = 4096, and INT8 holds at 0.99–1.00. Only memories with no decay drift into the danger zone as they grow.
4. **Delta-rule writes stay large** (≥ 1.7 steps at INT8 for every T tested), because its state is bounded. It never came near the threshold at INT8.
5. **Below the threshold, write size alone doesn't predict the outcome.** INT4 points at about 0.1–0.2 steps range from 0.12× to 2.25× fp32. λ decides which failure you get. With short λ, stuck decay extends memory (ratio > 1). With long λ, writes vanish (ratio ≪ 1). So the rule predicts *whether* a memory behaves like fp32, but not *how* it fails.
6. **Stochastic rounding is worse than nearest for INT8 near the threshold** (T=4096: 0.60 vs 0.74), and it costs about 5–7% even at 0.64–0.73 steps, where nearest is lossless. It trades bias for noise, and here the noise loses.
7. **Dynamic scale adds its own damage beyond write size.** INT8 dynamic at λ=1, T=4096 has larger writes (0.57 steps) than static at T=1024 (0.51) but does worse (0.88 vs 0.94). Rescaling every step re-rounds every entry.

## Takeaway (draft)

A fast-weight memory stored in low-precision integers behaves like full precision as long as each write is at least about 0.7 of a quantization step. For INT8 that holds for any memory with decay, and for memories without decay up to about 500–1000 writes at d=128. For INT4 it almost never holds, and the failure mode depends on the decay rate.

## Caveats
- 3 seeds, d = 128, one task (random keys, codebook decoding).
- Write size is a mean over entries; the per-entry distribution probably matters near the threshold.
- The static scale is calibrated on data from the same distribution it's tested on (optimistic).

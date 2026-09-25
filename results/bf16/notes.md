# bf16 storage: results (2026-09-24)

The predictions in [`predictions.md`](predictions.md) were committed before this run started (commit 1d20561, 18:07; the run finished at 18:17). Runs: 3 seeds, λ ∈ {0.99, 0.995, 0.997, 0.998, 0.999, 1.0}, recall (T=512) and rebinding (64 keys, T=1024).

## Prediction check

| # | prediction | outcome |
|---|---|---|
| 1 | λ ≤ 0.995: bf16 ≈ fp32 | ✅ held |
| 2 | λ ≥ 0.998: bf16 behaves like λ = 1 | ❌ **wrong** |
| 3 | λ = 0.997: decay stalls partway | ❌ no visible effect |
| 4 | recall barely changes | ✅ held (within ±1.3 items everywhere) |
| 5 | rebinding, Hebbian: bf16 at λ=0.999 looks like fp32 at λ=1 (≈ 8 / 88) | ❌ **wrong**: 23 / 74 vs fp32 24 / 74 |
| 6 | rebinding, delta rule unaffected | ✅ held |

**bf16 matched fp32 on both tasks at every λ** (rebinding within 1 point, recall within about 1 item).

## Why the prediction failed

The dead-zone argument (`round(λ·s) == s` when `(1-λ)·m < 2^-8`) assumed an entry that isn't being written to. But every step adds a write, and the sum `λ·s + w` is rounded once. If the write is several rounding steps wide, the rounding error changes from step to step like noise instead of cancelling the decay every time, so decay survives on average.

Measured on the same streams, the typical write is several bf16 steps wide:

| task | λ | median write ÷ bf16 step | entries whose write is below half a step |
|---|---|---|---|
| recall | 0.99 | 15.2 | 5% |
| recall | 0.999 | 6.6 | 10% |
| recall | 1.0 | 5.5 | 11% |
| rebind | 0.999 | 5.4 | 11% |
| rebind | 1.0 | 3.9 | 14% |

This is the **write-size rule** again ([`../writesize/notes.md`](../writesize/notes.md)): quantized storage behaves like full precision when writes are comfortably larger than the rounding step. The decay dead zone is a special case, and it only bites when writes are smaller than the step. That's what happened with INT4 and never happened with bf16 here. The prediction should have come from the rule, not from the dead zone alone.

## When bf16 should break (next prediction)

bf16's step is *relative* (about 2^-8 of the entry), so writes shrink relative to it as entries grow. Two ways to get there:
- **Very long memories with no decay.** For λ = 1 the median write/step ratio falls roughly as 1/√T: 5.5 at T=512 and 3.9 at T=1024, so it would reach about 0.5 somewhere in the tens of thousands of writes.
- **Frequently repeated associations.** Writing the same key→value over and over makes an entry grow about linearly, so after on the order of 2^8 repeats new writes to it vanish.

## Also in this run

INT8 (static scale, nearest rounding) is slightly staler than fp32 in rebinding at λ = 0.997–0.998 (53 / 46 vs 58 / 41, and 37 / 61 vs 43 / 56). That's a small effect near the write-size threshold, 3 seeds, unconfirmed.

## Takeaway

**bf16, Torq's main path, is safe for this memory at every decay rate we tested.** That makes the board plan simpler: bf16 first, and expect it to match the simulator.

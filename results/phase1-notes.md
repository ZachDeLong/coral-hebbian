# Phase 1 notes (2026-09-24, first full sweep)

Setup: d=128, T=512, vocab=256, 32 trials, **one seed**. Numbers are "items recalled" (sum of accuracy over ages, out of 512). Treat as preliminary until repeated over seeds.

1. **INT8 is effectively lossless.** Every INT8 config (static or dynamic scale, nearest or stochastic rounding, Hebbian or delta, all λ) matches fp32 within noise, including λ = 0.999 and 1.0 where fp32 recalls about 460. This refutes the pre-registered guess that the usable horizon is capped near 2^(bits-1) ≈ 127 for INT8.
2. **INT4 with nearest rounding lengthens short memories.** At λ = 0.9–0.97, INT4 remembers *longer* than fp32 (Hebbian λ=0.97: 156 vs 67). This fits the decay deadzone: small codes never decay, so the effective λ is higher than the one you set. It's not a free lunch. Forgetting is no longer controlled by λ.
3. **INT4 Hebbian with long λ can't write.** Static scale at λ ≥ 0.999 recalls 16–30 of about 460. `unchanged_frac` ≈ 0.9999, so almost no code ever changes and single writes round to zero.
4. **Inverted forgetting curve (dynamic scale, INT4 nearest, λ ≥ 0.99).** Recent items are recalled at 0% and the *oldest* items at about 100%. The memory keeps its first few writes and then stops learning, because the dynamic scale grows with the state and later writes underflow. It looks like a primacy effect.
5. **Delta rule is much more robust than Hebbian at INT4 static.** λ=1: 124 vs 16 (fp32: 162 vs 479). It degrades gracefully instead of freezing.
6. **Stochastic rounding is not a clean fix.** It partly rescues INT4 Hebbian at long λ with static scale (75–99 vs 16–30), but it hurts at short λ, and combined with dynamic scale it's catastrophic (delta λ ≥ 0.999: about 2 recalled, which is chance). Rescaling every step leaves every entry with a fractional part, so fresh rounding noise is injected everywhere at every step.

## Follow-ups
- Repeat over 5+ seeds and add error bands.
- Add a **rebinding task** (the same keys get new values over time). Frozen memories should return stale answers there, turning finding 2 from a bonus into a bug.
- Measure the effective λ of INT4 nearest directly and compare it to the deadzone prediction 0.5/(1-λ).
- Power-of-two dynamic scale: codes only get requantized when the exponent changes. It might fix finding 6 and is NPU-friendly (shifts).
- Try INT6 to find where the cliff between INT8 and INT4 sits.

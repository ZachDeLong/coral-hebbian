# bf16 storage: predictions (written before running)

Torq's main float path is bf16, which keeps 8 significant bits. Storing the state S in bf16 after every write means:

**Decay dead zone.** For an entry `s = m·2^e` with `m ∈ [1, 2)`, rounding to nearest leaves `λ·s == s` whenever `(1-λ)·m < 2^-8`. So:

| λ | 1-λ | entries that never decay |
|---|---|---|
| ≤ 0.995 | ≥ 0.005 | none (2^-8 ≈ 0.0039) |
| 0.997 | 0.003 | mantissa below about 1.30, so an entry decays only until it reaches about 1.3·2^e, then freezes |
| 0.998 | 0.002 | mantissa below about 1.95, almost all of them |
| 0.999 | 0.001 | all of them |

**Writes are fine.** A write gets lost only when it's smaller than about 2^-9 of the entry it lands on. At these stream lengths, entries are far below 2^8 × a typical write.

## Predictions

1. λ ≤ 0.995: bf16 ≈ fp32 on both tasks.
2. λ ≥ 0.998: bf16 behaves like **λ = 1**, meaning no forgetting.
3. At λ = 0.997, decay stalls partway (entries never drop below about 1.3·2^e), so old values are only partly erased.
4. **Recall** barely changes, since fp32 recall at λ = 0.999 and λ = 1 is already similar at T = 512.
5. **Rebinding, Hebbian:** bf16 at λ = 0.999 should look like fp32 at λ = 1 (about 8% current / 88% stale) instead of fp32 at λ = 0.999 (about 24 / 74). At λ = 0.997–0.998 there should be more stale answers than fp32.
6. **Rebinding, delta rule:** unaffected, because it doesn't rely on decay to overwrite.

Runs: `sweep.py` and `rebind_sweep.py` with `--seeds 3 --lams 0.99 0.995 0.997 0.998 0.999 1.0`.

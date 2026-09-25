# Rebinding test (2026-09-24)

**Question:** in plain recall, INT4 with round-to-nearest remembered *longer* than fp32, because stuck decay stops old memories from fading. Is that a bonus or a bug? Rebinding tests it. A pool of 64 keys gets new values over 1024 writes (about 16 rebindings per key), and the right answer for a key is its **latest** value. Decoding an older value of that key counts as *stale*.

**Prediction:** frozen decay should turn into stale answers.

**Runs:** `rebind_sweep.py --seeds 3` (d=128, vocab=256), same grid as the recall sweep. Numbers are current accuracy / stale rate, in %.

## Results

1. **Confirmed for Hebbian.** INT4 nearest with a static scale:
   - λ=0.99: 15 / 57 vs fp32 80 / 12. The memory mostly answers with old values.
   - λ=0.97: 73 / 19 vs 63 / 4. More keys answer, but stale answers go up 5×.

   With a dynamic scale it's worst: λ=0.99 gives **1 / 79**. The memory keeps its early bindings and never takes the new ones, which is the inverted forgetting curve from the recall task showing up as stale answers.
2. **The delta rule is immune to stuck decay (static scale).** INT4 nearest λ=0.99: 94 / 1 vs fp32 86 / 1, which is actually *better*. The delta rule overwrites by subtracting the old value (`v - S·k`), so it doesn't need decay to forget. That's why frozen decay only helps it. At λ ≥ 0.999 it keeps 91–93% current (fp32: 100%).
3. **The delta rule breaks when writes vanish.** With a dynamic scale at λ ≥ 0.999, INT4 nearest gives 0 / 44–47: same frozen-early-bindings failure as Hebbian. When writes underflow, no update rule can save it.
4. **INT8 matches fp32 everywhere** (within 1–4 points), consistent with the write-size rule.
5. fp32 Hebbian can't rebind without decay (λ ≥ 0.999: 74–88% stale). That's the known weakness of Hebbian memory, not a quantization effect, and the reason the delta rule exists.

## Takeaway

Quantization breaks Hebbian memory in the way you'd expect from a brain-style rule that relies on forgetting to make room: when decay freezes, old bindings never clear, and the memory answers from the past. The delta rule is robust because it forgets *on purpose*, by subtracting the old value when it writes the new one. For the board, the recommended configuration is **delta rule + INT8 (or INT4) with a static per-tensor scale**, which is likely what Torq supports anyway.

## Caveats
- 3 seeds, one pool size (64 keys, below d=128).
- The oldest age bin (512–1023 writes since last binding) has very few keys, so its points in the curve plots are noisy.

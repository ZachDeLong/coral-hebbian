"""Associative recall: stream T random (key, value) writes, then query every key.

Keys are random unit vectors. Values are drawn from a codebook of `vocab` random
unit vectors, and a read counts as correct when the nearest codeword (by dot
product) is the one that was stored. Chance is 1/vocab.

Accuracy is reported by age (how many writes ago the pair was stored), so one
run gives the whole forgetting curve. Every config sees the identical stream for
a given seed, so comparisons between configs are paired.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .memory import FastWeightMemory, MemoryConfig

CALIBRATION_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class RecallTask:
    d: int = 128
    T: int = 512
    vocab: int = 256
    batch: int = 32
    seed: int = 0


@dataclass
class RecallResult:
    cfg: MemoryConfig
    acc_by_age: np.ndarray  # (T,), index 0 = most recent write
    cos_by_age: np.ndarray  # (T,), cosine between readout and the stored value
    saturated_frac: float
    unchanged_frac: float
    write_lsb: float  # mean write size in quantization steps (NaN for fp32)
    seed: int = 0

    @property
    def recalled(self) -> float:
        """Expected number of the T stored pairs that can still be read back."""
        return float(self.acc_by_age.sum())

    @property
    def fidelity(self) -> float:
        """Mean cosine between readout and stored value over all ages. Unlike argmax
        decoding, this moves with any damage to the readout, not just decode flips."""
        return float(self.cos_by_age.mean())

    def age50(self, window: int = 8) -> int:
        """First age where accuracy (averaged over `window` ages) falls below 50%."""
        acc = self.acc_by_age
        if len(acc) < window:
            window = len(acc)
        means = np.convolve(acc, np.ones(window) / window, mode="valid")
        below = np.nonzero(means < 0.5)[0]
        return int(below[0]) if len(below) else len(acc)


def make_stream(task: RecallTask, seed: int):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(task.batch, task.T, task.d, generator=g)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    codebook = torch.randn(task.vocab, task.d, generator=g)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)
    ids = torch.randint(task.vocab, (task.batch, task.T), generator=g)
    return keys, ids, codebook


def stream_through(
    cfg: MemoryConfig, keys: torch.Tensor, values: torch.Tensor, rounding_seed: int, static_range=None
) -> FastWeightMemory:
    """Write keys[:, t] -> values[:, t] for every t. keys, values: (B, T, d)."""
    batch, T, d = keys.shape
    rounding_gen = torch.Generator().manual_seed(rounding_seed)
    mem = FastWeightMemory(cfg, batch, d, static_range=static_range, generator=rounding_gen)
    for t in range(T):
        mem.write(keys[:, t], values[:, t])
    return mem


def _stream_through(cfg: MemoryConfig, task: RecallTask, seed: int, static_range=None) -> FastWeightMemory:
    keys, ids, codebook = make_stream(task, seed)
    return stream_through(cfg, keys, codebook[ids], seed + 1, static_range)


def calibrate(cfg: MemoryConfig, task: RecallTask) -> torch.Tensor:
    """Per-row max |S| from a float run on a held-out stream."""
    fp = replace(cfg, bits=None)
    return _stream_through(fp, task, task.seed + CALIBRATION_SEED_OFFSET).amax_row


@torch.no_grad()
def run_recall(cfg: MemoryConfig, task: RecallTask, static_range: torch.Tensor | None = None) -> RecallResult:
    mem = _stream_through(cfg, task, task.seed, static_range)
    keys, ids, codebook = make_stream(task, task.seed)
    readout = torch.einsum("bij,btj->bti", mem.S, keys)  # (B, T, d)
    pred = (readout @ codebook.T).argmax(dim=-1)  # (B, T)
    acc_by_time = (pred == ids).float().mean(dim=0)  # index 0 = first write
    cos_by_time = torch.nn.functional.cosine_similarity(readout, codebook[ids], dim=-1).mean(dim=0)
    return RecallResult(
        cfg=cfg,
        acc_by_age=acc_by_time.flip(0).numpy(),
        cos_by_age=cos_by_time.flip(0).numpy(),
        saturated_frac=mem.saturated_frac,
        unchanged_frac=mem.unchanged_frac,
        write_lsb=mem.write_lsb,
        seed=task.seed,
    )

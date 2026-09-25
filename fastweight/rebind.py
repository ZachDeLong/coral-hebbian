"""Rebinding: a fixed pool of keys keeps getting new values.

Each write picks a random key from a pool of `pool` keys and binds it to a fresh
random value. At the end every key is queried. The right answer is its most
recent value, and decoding to one of its older values is a *stale* answer.

Plain recall rewards a memory that never fades. Here an old binding that doesn't
fade competes with the new one, so stuck decay should show up as stale answers
instead of a longer memory.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .memory import MemoryConfig
from .recall import CALIBRATION_SEED_OFFSET, stream_through


@dataclass(frozen=True)
class RebindTask:
    d: int = 128
    pool: int = 64
    T: int = 1024
    vocab: int = 256
    batch: int = 32
    seed: int = 0


@dataclass
class RebindResult:
    cfg: MemoryConfig
    age: np.ndarray  # (n,) writes since each queried key was last bound
    current: np.ndarray  # (n,) bool: decoded its latest value
    stale: np.ndarray  # (n,) bool: decoded one of its older values
    write_lsb: float
    unchanged_frac: float
    seed: int = 0

    @property
    def current_acc(self) -> float:
        return float(self.current.mean())

    @property
    def stale_rate(self) -> float:
        return float(self.stale.mean())


def make_rebind_stream(task: RebindTask, seed: int):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randn(task.batch, task.pool, task.d, generator=g)
    pool = pool / pool.norm(dim=-1, keepdim=True)
    codebook = torch.randn(task.vocab, task.d, generator=g)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)
    key_idx = torch.randint(task.pool, (task.batch, task.T), generator=g)
    ids = torch.randint(task.vocab, (task.batch, task.T), generator=g)
    keys = torch.gather(pool, 1, key_idx.unsqueeze(-1).expand(-1, -1, task.d))
    return pool, key_idx, keys, ids, codebook


def calibrate_rebind(cfg: MemoryConfig, task: RebindTask) -> torch.Tensor:
    """Per-row max |S| from a float run on a held-out rebinding stream."""
    seed = task.seed + CALIBRATION_SEED_OFFSET
    _, _, keys, ids, codebook = make_rebind_stream(task, seed)
    return stream_through(replace(cfg, bits=None), keys, codebook[ids], seed + 1).amax_row


def score(pred: np.ndarray, key_idx: np.ndarray, ids: np.ndarray, T: int):
    """pred: (B, P) decoded value id per key. Returns (age, current, stale) over keys written at least once."""
    ages, current, stale = [], [], []
    for b in range(pred.shape[0]):
        history = [[] for _ in range(pred.shape[1])]
        for t in range(T):
            history[key_idx[b, t]].append((t, ids[b, t]))
        for p, h in enumerate(history):
            if not h:
                continue
            last_t, latest = h[-1]
            older = {v for _, v in h[:-1]} - {latest}
            ages.append(T - 1 - last_t)
            current.append(pred[b, p] == latest)
            stale.append(pred[b, p] in older)
    return np.array(ages), np.array(current), np.array(stale)


@torch.no_grad()
def run_rebind(cfg: MemoryConfig, task: RebindTask, static_range: torch.Tensor | None = None) -> RebindResult:
    pool, key_idx, keys, ids, codebook = make_rebind_stream(task, task.seed)
    mem = stream_through(cfg, keys, codebook[ids], task.seed + 1, static_range)
    readout = torch.einsum("bij,bpj->bpi", mem.S, pool)  # (B, P, d)
    pred = (readout @ codebook.T).argmax(dim=-1).numpy()
    age, current, stale = score(pred, key_idx.numpy(), ids.numpy(), task.T)
    return RebindResult(cfg=cfg, age=age, current=current, stale=stale, write_lsb=mem.write_lsb,
                        unchanged_frac=mem.unchanged_frac, seed=task.seed)

import torch

from fastweight import FastWeightMemory, MemoryConfig, RecallTask, calibrate, quantize, run_recall, update


def _decay_once(codes: torch.Tensor, lam: float, mode: str, gen=None) -> torch.Tensor:
    """Apply one zero-write step to a column of int8 codes and requantize."""
    n = codes.numel()
    S = codes.view(1, n, 1).float()  # scale 1.0: codes are the values
    S_new = update(S, torch.zeros(1, 1), torch.zeros(1, n), "hebbian", lam, 0.0)
    return quantize(S_new, 1.0, 8, mode, gen).view(-1)


def test_nearest_rounding_decay_deadzone():
    # round(0.99 * s) == s whenever 0.01*|s| < 0.5, i.e. |s| < 50: those entries never decay.
    codes = torch.arange(-127, 128)
    out = _decay_once(codes, 0.99, "nearest")
    small = codes.abs() < 50
    assert torch.equal(out[small], codes[small].float())
    big = codes.abs() > 50
    assert (out[big].abs() < codes[big].abs()).all()


def test_stochastic_rounding_decays_in_expectation():
    codes = torch.full((200_000,), 20)
    out = _decay_once(codes, 0.99, "stochastic", torch.Generator().manual_seed(0))
    assert abs(out.mean().item() - 19.8) < 0.01


def test_quantize_saturates():
    x = torch.tensor([-1000.0, -3.0, 0.0, 3.0, 1000.0])
    assert quantize(x, 1.0, 4).tolist() == [-7.0, -3.0, 0.0, 3.0, 7.0]


def test_hebbian_orthogonal_keys_recall_exactly():
    d = 16
    mem = FastWeightMemory(MemoryConfig(rule="hebbian", lam=1.0), batch=1, d=d)
    keys = torch.eye(d)
    values = torch.randn(d, d, generator=torch.Generator().manual_seed(0))
    for i in range(d):
        mem.write(keys[i : i + 1], values[i : i + 1])
    for i in range(d):
        assert torch.allclose(mem.read(keys[i : i + 1]), values[i : i + 1], atol=1e-5)


def test_delta_rule_overwrites_same_key():
    d = 8
    mem = FastWeightMemory(MemoryConfig(rule="delta", lam=1.0, beta=1.0), batch=1, d=d)
    k = torch.nn.functional.normalize(torch.randn(1, d), dim=-1)
    v1, v2 = torch.randn(1, d), torch.randn(1, d)
    mem.write(k, v1)
    mem.write(k, v2)
    assert torch.allclose(mem.read(k), v2, atol=1e-5)


def test_recall_runs_for_every_scale_mode():
    task = RecallTask(d=16, T=32, vocab=16, batch=4)
    fp = run_recall(MemoryConfig(), task)
    assert fp.acc_by_age.shape == (32,)
    assert fp.acc_by_age[0] == 1.0  # the latest write reads back cleanly in float
    assert fp.cos_by_age[0] > 0.5 and fp.write_lsb != fp.write_lsb  # fp32 has no write size (NaN)
    rng = calibrate(MemoryConfig(), task)
    for scale in ("static", "static_row", "dynamic"):
        res = run_recall(MemoryConfig(bits=8, scale=scale), task, static_range=rng)
        assert 0.0 <= res.recalled <= 32
        assert res.write_lsb > 0


def test_write_size_shrinks_with_fewer_bits():
    task = RecallTask(d=16, T=32, vocab=16, batch=4)
    rng = calibrate(MemoryConfig(), task)
    w8 = run_recall(MemoryConfig(bits=8), task, static_range=rng).write_lsb
    w4 = run_recall(MemoryConfig(bits=4), task, static_range=rng).write_lsb
    assert abs(w8 / w4 - 127 / 7) < 1e-3 * (127 / 7)  # same writes, 127/7 finer grid

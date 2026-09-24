"""Fake quantization for the carried state.

Between steps the NPU stores the state as small integers. We simulate that by
snapping the float state onto an integer grid (codes * scale, |codes| <= qmax)
after every update. Math inside a step stays float, standing in for the NPU's
wide (int32) accumulators, so the only loss is the one we care about: storage.
"""

import torch


def qmax(bits: int) -> int:
    """Largest code for a symmetric signed integer of `bits` bits (int8 -> 127)."""
    return 2 ** (bits - 1) - 1


def round_codes(x: torch.Tensor, mode: str, generator: torch.Generator | None = None) -> torch.Tensor:
    if mode == "nearest":
        # torch.round is round-half-to-even, like most integer hardware.
        return torch.round(x)
    if mode == "stochastic":
        # Round up with probability equal to the fractional part: unbiased in expectation.
        noise = torch.rand(x.shape, generator=generator, dtype=x.dtype, device=x.device)
        return torch.floor(x + noise)
    raise ValueError(f"unknown rounding mode: {mode}")


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor | float,
    bits: int,
    mode: str = "nearest",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integer codes (as a float tensor) for x on a grid of step `scale`, saturating at +-qmax."""
    m = qmax(bits)
    return round_codes(x / scale, mode, generator).clamp_(-m, m)

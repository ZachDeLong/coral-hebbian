"""Fast-weight memory: a state matrix S carried from step to step.

    hebbian:  S <- lam*S + beta * v k^T
    delta:    S <- lam*S + beta * (v - lam*S k) k^T

No weights are learned here; only S changes. In a quantized config, S lives as
integer codes between steps (what the NPU would store). Each step dequantizes,
updates in float, and requantizes.

`update` is kept pure and free of data-dependent control flow so the same step
can later be exported as a static graph for the Torq compiler.
"""

from dataclasses import dataclass

import torch

from .quant import qmax, quantize

RULES = ("hebbian", "delta")
SCALES = ("static", "static_row", "dynamic")
_TINY = 1e-12


@dataclass(frozen=True)
class MemoryConfig:
    rule: str = "hebbian"  # "hebbian" | "delta"
    lam: float = 0.99  # decay per step
    beta: float = 1.0  # write strength
    bits: int | None = None  # None = float32 reference
    rounding: str = "nearest"  # "nearest" | "stochastic"
    scale: str = "static"  # "static" (per tensor, fixed at compile time) | "static_row" | "dynamic"

    @property
    def label(self) -> str:
        if self.bits is None:
            return "fp32"
        return f"int{self.bits}-{self.rounding}-{self.scale}"


def update(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rule: str, lam: float, beta: float) -> torch.Tensor:
    """One write. S: (B, dv, dk), k: (B, dk), v: (B, dv)."""
    S = lam * S
    if rule == "hebbian":
        err = v
    elif rule == "delta":
        err = v - torch.einsum("bij,bj->bi", S, k)
    else:
        raise ValueError(f"unknown rule: {rule}")
    return S + beta * err.unsqueeze(-1) * k.unsqueeze(-2)


class FastWeightMemory:
    """Runs `update` over a stream, optionally storing S as integer codes between steps.

    Static scale modes need `static_range`: per-row max |S| from a float calibration
    run (like calibrating a quantized model on representative data before compiling).
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        batch: int,
        d: int,
        static_range: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        self.cfg = cfg
        self.gen = generator
        self.S = torch.zeros(batch, d, d)  # dequantized view of the state
        self.amax_row = torch.zeros(d)  # running max |S| per row, for calibration
        self.steps = 0
        self._saturated = 0.0
        self._unchanged = 0.0

        if cfg.bits is None:
            return
        self.codes = torch.zeros(batch, d, d)
        m = qmax(cfg.bits)
        if cfg.scale == "dynamic":
            self.scale = torch.full((batch, 1, 1), _TINY)
        elif static_range is None:
            raise ValueError(f"scale={cfg.scale!r} needs a calibrated static_range")
        elif cfg.scale == "static":
            self.scale = (static_range.max() / m).clamp_min(_TINY)
        elif cfg.scale == "static_row":
            self.scale = (static_range / m).clamp_min(_TINY).view(d, 1)
        else:
            raise ValueError(f"unknown scale mode: {cfg.scale}")

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        cfg = self.cfg
        S_new = update(self.S, k, v, cfg.rule, cfg.lam, cfg.beta)
        if cfg.bits is None:
            self.S = S_new
            self.amax_row = torch.maximum(self.amax_row, S_new.abs().amax(dim=(0, 2)))
        else:
            if cfg.scale == "dynamic":
                # Block floating point: rescale every step so the largest entry hits qmax.
                self.scale = (S_new.abs().amax(dim=(1, 2), keepdim=True) / qmax(cfg.bits)).clamp_min(_TINY)
            codes = quantize(S_new, self.scale, cfg.bits, cfg.rounding, self.gen)
            self._saturated += (codes.abs() == qmax(cfg.bits)).float().mean().item()
            self._unchanged += (codes == self.codes).float().mean().item()
            self.codes = codes
            self.S = codes * self.scale
        self.steps += 1

    def read(self, k: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bj->bi", self.S, k)

    @property
    def saturated_frac(self) -> float:
        """Mean fraction of entries pinned at +-qmax per step."""
        return self._saturated / max(self.steps, 1)

    @property
    def unchanged_frac(self) -> float:
        """Mean fraction of entries whose integer code didn't move in a step (frozen memory)."""
        return self._unchanged / max(self.steps, 1)

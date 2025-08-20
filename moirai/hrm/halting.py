from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


try:
    import torch
    from torch import Tensor
    import torch.nn as nn
except Exception:  # pragma: no cover - torch optional for M0
    torch = None
    Tensor = None  # type: ignore
    nn = None  # type: ignore


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828, -x))


@dataclass
class ACTHalter:
    """
    Minimal ACT halter with a 2-layer MLP interface if torch is available.

    For M0 we support two usage modes:
    - Torch mode: a small nn.Module with 2-layer MLP producing a continue logit.
    - Python mode: a callable that returns a fixed continue probability derived
      from bias terms for testing extremes.
    """

    hidden: int = 4
    step_penalty_target: float = 0.01
    force_logit: Optional[float] = None  # if set, overrides MLP output (extremes)

    def torch_module(self, d_model: int):  # pragma: no cover - trivial wrapper
        if torch is None:
            raise RuntimeError("PyTorch not available")
        return _TorchHalter(d_model=d_model, hidden=self.hidden, force_logit=self.force_logit)

    def python_callable(self) -> Callable[[float], Tuple[float, bool]]:
        """
        Returns a function(state_norm) -> (continue_prob, continue_bool)
        Uses a fixed logit via force_logit if provided; else derives from state_norm.
        """
        def fn(state_norm: float) -> Tuple[float, bool]:
            if self.force_logit is not None:
                p = _sigmoid(self.force_logit)
            else:
                # Simple heuristic: lower norm -> stop sooner
                p = _sigmoid(2.0 * (state_norm - 1.0))
            return p, (p > 0.5)

        return fn


def act_rollout(
    step_fn: Callable[[int], float],
    halter: ACTHalter,
    outer_cap: int,
    use_torch: bool = False,
):
    """
    Run a minimal ACT rollout for testing extremes.

    - step_fn(t) should return a float "state_norm" surrogate per step
    - halter decides to continue or halt based on the state_norm and its policy
    - returns steps_taken, halts_early
    """
    steps = 0
    halts_early = False
    if use_torch and torch is not None:
        # Torch path uses a tiny module in BF16/FP32 as available
        mod = halter.torch_module(d_model=8)
        with torch.no_grad():
            for t in range(outer_cap):
                steps += 1
                x = torch.tensor([[step_fn(t)] * 8], dtype=torch.float32)
                logit = mod(x)
                p = torch.sigmoid(logit).item()
                if halter.force_logit is not None:
                    p = _sigmoid(halter.force_logit)
                if p <= 0.5:
                    halts_early = True
                    break
    else:
        decide = halter.python_callable()
        for t in range(outer_cap):
            steps += 1
            p, cont = decide(step_fn(t))
            if not cont:
                halts_early = True
                break
    return steps, halts_early


if nn is not None:
    class _TorchHalter(nn.Module):  # pragma: no cover - thin torch wrapper
        def __init__(self, d_model: int, hidden: int, force_logit: Optional[float]):
            super().__init__()
            self.ff = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.force_logit = force_logit

        def forward(self, x: Tensor) -> Tensor:
            out = self.ff(x).squeeze(-1)
            if self.force_logit is not None:
                out = out * 0 + self.force_logit
            return out
else:
    class _TorchHalter:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")


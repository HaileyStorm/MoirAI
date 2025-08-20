from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch optional for M0
    torch = None
    Tensor = None  # type: ignore


@dataclass
class Top1Router:
    """
    Static top-1 router with shape-stable outputs.

    - Accepts logits as either PyTorch Tensor (preferred) or a Python list of lists.
    - Returns a tuple (indices, one_hot, probs) where:
        indices: [B] integer indices of the selected expert per item
        one_hot: [B, E] float mask with a single 1.0 per row
        probs:   [B] selection softmax probability of the winning expert

    Capacity and load-balance terms are left for later milestones; this is the
    minimal core required for M0.
    """

    temperature: float = 1.0

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def __call__(self, logits) -> Tuple:
        if torch is not None and isinstance(logits, torch.Tensor):  # Torch path
            return self._torch_route(logits)
        # Fallback: Python/numpy-less path
        return self._py_route(logits)

    def _torch_route(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # logits: [B, E]
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)
        indices = torch.argmax(probs, dim=-1)
        b, e = probs.shape
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, indices.view(-1, 1), 1.0)
        chosen_p = (probs * one_hot).sum(dim=-1)
        return indices, one_hot, chosen_p

    def _py_route(self, logits) -> Tuple[list, list, list]:
        # logits: list[list[float]]; returns python lists with stable shapes
        if not isinstance(logits, list) or not logits:
            raise ValueError("logits must be a non-empty list of lists in python mode")
        b = len(logits)
        e = len(logits[0])
        # Validate rectangular
        for row in logits:
            if len(row) != e:
                raise ValueError("logits must be rectangular")

        def softmax(vec):
            # naive stable softmax
            m = max(vec)
            exps = [pow(2.718281828, (v - m) / self.temperature) for v in vec]
            s = sum(exps)
            return [x / s for x in exps]

        indices = [0] * b
        probs = [0.0] * b
        one_hot = [[0.0 for _ in range(e)] for _ in range(b)]
        for i, row in enumerate(logits):
            p = softmax(row)
            j = max(range(e), key=lambda k: p[k])
            indices[i] = j
            probs[i] = p[j]
            one_hot[i][j] = 1.0
        return indices, one_hot, probs


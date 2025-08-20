from __future__ import annotations

from typing import Any, Callable


def compile_if_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a function with torch.compile(dynamic=False) if available, otherwise
    return the function unchanged. Keeps the call signature intact.
    """
    try:  # torch optional for M0
        import torch

        # In PyTorch 2.3+, dynamic arg is deprecated in favor of fullgraph=True for strictness.
        try:
            compiled = torch.compile(fn, fullgraph=True)
        except Exception:
            compiled = torch.compile(fn, dynamic=False)  # type: ignore[arg-type]
        return compiled
    except Exception:
        return fn


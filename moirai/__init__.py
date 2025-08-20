"""
MoirAI core package

Milestone 0 provides minimal infrastructure:
- Reverse-lattice round trip helpers (bytes -> segments -> bytes)
- Static top-1 router (shape-stable, tensorized API)
- ACT halter scaffold (extreme behaviors testable)
- Compile harness (torch.compile if available; no-op otherwise)
"""

__all__ = [
    "__version__",
]

__version__ = "0.0.1"


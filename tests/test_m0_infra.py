import sys

import pytest

from moirai.hnet import segment_bytes, reconstruct_bytes
from moirai.moe import Top1Router
from moirai.hrm import ACTHalter, act_rollout
from moirai.train import compile_if_available


def test_reverse_lattice_round_trip_fuzz():
    rng = random.Random(42)
    for _ in range(1000):
        n = rng.randint(0, 256)
        data = bytes(rng.getrandbits(8) for _ in range(n))
        # Create random boundaries including 0 and n
        cuts = sorted(set([0, n] + [rng.randint(0, n) for _ in range(rng.randint(0, 8))]))
        lat = segment_bytes(data, cuts)
        rec = reconstruct_bytes(data, lat)
        assert rec == data


def test_top1_router_shapes_and_stability_python():
    router = Top1Router()
    logits = [[0.1, 0.9, -0.5], [1.0, -1.0, 0.0]]
    idx1, onehot1, p1 = router(logits)
    idx2, onehot2, p2 = router(logits)
    assert idx1 == idx2
    assert onehot1 == onehot2
    assert p1 == p2
    # Exactly one selection per row
    for row in onehot1:
        assert abs(sum(row) - 1.0) < 1e-6
        assert max(row) == 1.0


def test_act_extremes_python():
    # Always continue -> hit cap
    cap = 5
    steps, halts_early = act_rollout(lambda t: 1.0, ACTHalter(force_logit=10.0), cap, use_torch=False)
    assert steps == cap and not halts_early
    # Always halt -> 1 step
    steps, halts_early = act_rollout(lambda t: 1.0, ACTHalter(force_logit=-10.0), cap, use_torch=False)
    assert steps == 1 and halts_early


def test_no_graph_breaks_under_compile_if_torch():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")

    # Define a small function using router + simple tensor ops
    router = Top1Router()

    def fn(x):
        # x: [B, E]
        idx, onehot, p = router(x)
        return idx, onehot, p

    compiled = compile_if_available(fn)
    x = torch.randn(4, 3)
    idx, onehot, p = compiled(x)
    # Shapes stable and sensible
    assert idx.shape == (4,)
    assert onehot.shape == (4, 3)
    assert p.shape == (4,)
    assert torch.allclose(onehot.sum(dim=-1), torch.ones(4))


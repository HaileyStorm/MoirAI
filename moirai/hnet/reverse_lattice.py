from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Segment:
    """
    Minimal reverse-lattice segment descriptor.

    start, end: byte offsets (end exclusive)
    weight: optional soft weight for pooling (kept for completeness in M0)
    """

    start: int
    end: int
    weight: float = 1.0

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Negative offsets are not allowed")
        if self.end < self.start:
            raise ValueError("Segment end must be >= start")


class ReverseLattice:
    """
    Reverse-lattice container for exact byte reconstruction from v0 segments.

    This is deliberately simple for M0: it stores the source byte length and a
    list of Segment(start, end, weight). The reconstruction reads the original
    bytes by slicing ranges and concatenating, guaranteeing exact round-trip
    when the segments form a partition of the range [0, n).
    """

    def __init__(self, length: int, segments: Iterable[Segment]):
        if length < 0:
            raise ValueError("Byte length must be non-negative")
        self.length = int(length)
        self.segments: List[Segment] = list(segments)
        self._validate()

    def _validate(self) -> None:
        # Segments must be within [0, length] and non-overlapping when treated
        # as an ordered partition. For M0 we only check coverage and order.
        if not self.segments:
            if self.length != 0:
                raise ValueError("Non-empty byte stream requires at least one segment")
            return
        # Ensure sorted by (start, end)
        self.segments.sort(key=lambda s: (s.start, s.end))
        # Validate continuity and coverage
        cursor = 0
        for seg in self.segments:
            if seg.start != cursor:
                raise ValueError("Segments must form a contiguous partition without gaps")
            if seg.end > self.length:
                raise ValueError("Segment extends beyond byte length")
            cursor = seg.end
        if cursor != self.length:
            raise ValueError("Segments do not cover the full byte range")

    def reconstruct(self, data: bytes) -> bytes:
        if len(data) != self.length:
            raise ValueError("Provided data length does not match lattice length")
        # Exact reconstruction via concatenation of covered ranges
        out = bytearray()
        for seg in self.segments:
            out.extend(data[seg.start : seg.end])
        return bytes(out)


def segment_bytes(data: bytes, boundaries: Iterable[int], weights: Iterable[float] | None = None) -> ReverseLattice:
    """
    Build a ReverseLattice from explicit boundary indices.

    boundaries: sorted cut points including 0 and len(data). Any repeated
    boundary is ignored. Weights are optional per segment; default = 1.0.
    """
    n = len(data)
    # Normalize and clamp boundaries
    uniq: List[int] = []
    seen = set()
    for b in boundaries:
        bi = int(b)
        if bi < 0:
            bi = 0
        if bi > n:
            bi = n
        if bi not in seen:
            uniq.append(bi)
            seen.add(bi)
    if not uniq or uniq[0] != 0:
        uniq.insert(0, 0)
    if uniq[-1] != n:
        uniq.append(n)
    uniq.sort()
    # Form segments
    segs: List[Segment] = []
    if weights is None:
        for a, b in zip(uniq[:-1], uniq[1:]):
            segs.append(Segment(a, b, 1.0))
    else:
        w_list = list(weights)
        if len(w_list) != len(uniq) - 1:
            raise ValueError("weights length must equal number of segments")
        for (a, b), w in zip(zip(uniq[:-1], uniq[1:]), w_list):
            segs.append(Segment(a, b, float(w)))
    return ReverseLattice(n, segs)


def reconstruct_bytes(data: bytes, lattice: ReverseLattice) -> bytes:
    """Helper to reconstruct bytes via the provided lattice."""
    return lattice.reconstruct(data)


"""Reliability-guided DIC (RG-DIC) full-field propagation.

Starting from one or more seed points with a known (or zero, for small
deformation) initial guess, subsets are grown outward over the grid, always
expanding next from the currently most-reliable (highest ZNCC) converged
point and using its converged warp parameters as the initial guess for its
unconverged neighbors. This is the strategy behind Ncorr-style DIC and is far
more robust to large/non-uniform deformation than searching every grid point
independently, since a locally propagated initial guess is almost always
already within the IC-GN basin of convergence.

Reference: B. Pan, "Reliability-guided digital image correlation for image
deformation measurement", Applied Optics 48(8), 2009.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .correlation import ImageInterpolator, icgn_correlate, image_gradients
from .subset import Subset, build_subset


@dataclass
class DicPointResult:
    index: int
    x0: float
    y0: float
    p: np.ndarray
    zncc: float
    converged: bool

    @property
    def u(self) -> float:
        return float(self.p[0])

    @property
    def v(self) -> float:
        return float(self.p[3])


def track_reliability_guided(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    points: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    subset_radius: int,
    seed_indices: Sequence[int],
    seed_p_init: Optional[Sequence[np.ndarray]] = None,
    max_iter: int = 40,
    tol: float = 1e-4,
    zncc_threshold: float = 0.5,
    interpolator: Optional[ImageInterpolator] = None,
    grad: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[DicPointResult]:
    """Propagate subset correlation across `points` from `seed_indices`.

    points: (N, 2) array of (x, y) subset centers in the reference image.
    neighbors: adjacency list (e.g. from fields.build_neighbors).
    seed_p_init: initial 6-param guess per seed (defaults to zero, i.e. small
        deformation assumed at the seed; for large motion, provide a guess
        e.g. from correlation.initial_guess_template_match).
    """
    n = len(points)
    if seed_p_init is None:
        seed_p_init = [np.zeros(6) for _ in seed_indices]

    gx, gy = grad if grad is not None else image_gradients(ref_gray)
    interp = interpolator if interpolator is not None else ImageInterpolator(cur_gray)

    results: List[Optional[DicPointResult]] = [None] * n
    computed = np.zeros(n, dtype=bool)
    subset_cache: Dict[int, Subset] = {}

    counter = itertools.count()  # tie-breaker so heap never compares arrays
    heap: List[Tuple[float, int, int, np.ndarray]] = []
    for idx, p0 in zip(seed_indices, seed_p_init):
        heapq.heappush(heap, (-1.0, next(counter), idx, p0))

    while heap:
        neg_priority, _, idx, p_init = heapq.heappop(heap)
        if computed[idx]:
            continue

        x0, y0 = points[idx]
        subset = subset_cache.get(idx)
        if subset is None:
            subset = build_subset(ref_gray, x0, y0, subset_radius, gx, gy)
            subset_cache[idx] = subset

        if not subset.valid:
            computed[idx] = True
            results[idx] = DicPointResult(idx, x0, y0, p_init, -1.0, False)
            continue

        res = icgn_correlate(
            subset, interp, p_init, max_iter=max_iter, tol=tol, zncc_threshold=zncc_threshold
        )
        computed[idx] = True
        results[idx] = DicPointResult(idx, x0, y0, res.p, res.zncc, res.converged)

        if res.converged:
            for nb in neighbors[idx]:
                if not computed[nb]:
                    heapq.heappush(heap, (-res.zncc, next(counter), nb, res.p))

    # Any point never reached by propagation (isolated behind failed subsets)
    for idx in range(n):
        if results[idx] is None:
            x0, y0 = points[idx]
            results[idx] = DicPointResult(idx, x0, y0, np.zeros(6), -1.0, False)

    return results  # type: ignore[return-value]


def track_sequence(
    ref_gray: np.ndarray,
    frames: Sequence[np.ndarray],
    points: np.ndarray,
    neighbors: Sequence[Sequence[int]],
    subset_radius: int,
    first_frame_seed_indices: Optional[Sequence[int]] = None,
    first_frame_seed_p: Optional[Sequence[np.ndarray]] = None,
    **kwargs,
) -> List[List[DicPointResult]]:
    """Track every grid point across an image sequence, always correlating
    against the fixed reference image (no incremental drift).

    Each point's own converged warp from frame t is used as the initial guess
    at frame t+1 (valid when the sequence step is small relative to the
    deformation rate, which is the normal DIC acquisition regime); any point
    that still fails to converge directly is recovered, at that same frame,
    by reliability-guided propagation from its already-converged neighbors.

    If `first_frame_seed_indices` is omitted, every point is seeded at p=0
    for the first frame (assumes the first deformed frame is close to the
    reference). Pass a single seed with an initial guess from
    correlation.initial_guess_template_match for large rigid-body motion.
    """
    n = len(points)
    gx, gy = image_gradients(ref_gray)
    last_p = [np.zeros(6) for _ in range(n)]

    results_per_frame: List[List[DicPointResult]] = []
    for frame_idx, frame in enumerate(frames):
        interp = ImageInterpolator(frame)
        if frame_idx == 0 and first_frame_seed_indices is not None:
            seeds = list(first_frame_seed_indices)
            seed_p = list(first_frame_seed_p) if first_frame_seed_p is not None else [
                np.zeros(6) for _ in seeds
            ]
        else:
            seeds = list(range(n))
            seed_p = [last_p[i] for i in seeds]

        res = track_reliability_guided(
            ref_gray, frame, points, neighbors, subset_radius, seeds, seed_p,
            interpolator=interp, grad=(gx, gy), **kwargs,
        )
        for i, r in enumerate(res):
            if r.converged:
                last_p[i] = r.p
        results_per_frame.append(res)

    return results_per_frame

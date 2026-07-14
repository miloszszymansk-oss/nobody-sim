"""Acceleration backends (SPEC §1.1-1.2, §3).

Contract: every backend has the signature (pos, mass, G, eps) -> (N,3) and identical
physics up to its documented approximation error. Integrators receive a closed-over
accel_fn(pos) -> (N,3) and know nothing about masses or G (SPEC §4.1.3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nbody.constants import G_ASTRO

# Brute force builds an (N,N,3) array: ~100 MB at N=2000. Hard limit per SPEC §4.3.
BRUTE_N_MAX = 3000


def accel_brute(
    pos: np.ndarray, mass: np.ndarray, G: float = G_ASTRO, eps: float = 0.0
) -> np.ndarray:
    """a_i = G sum_{j!=i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^{3/2}; O(N^2).

    Vectorized via broadcasting: diff[i, j, :] = r_j - r_i. The diagonal (i == j)
    is excluded by zeroing its weight; with eps = 0, coincident *distinct* bodies
    are undefined (documented: use eps > 0 for clusters, SPEC §1.2).
    """
    n = pos.shape[0]
    if n > BRUTE_N_MAX:
        raise ValueError(
            f"accel_brute supports N <= {BRUTE_N_MAX} (O(N^2) memory); use the "
            "'barnes_hut' backend for larger systems (SPEC §4.3)"
        )
    diff = pos[None, :, :] - pos[:, None, :]  # (N,N,3), diff[i,j] = r_j - r_i
    d2 = np.einsum("ijk,ijk->ij", diff, diff) + eps * eps
    np.fill_diagonal(d2, 1.0)  # placeholder to avoid 0**-1.5; weight zeroed below
    w = mass[None, :] * d2**-1.5  # w[i,j] = m_j / (|r_ij|^2 + eps^2)^{3/2}
    np.fill_diagonal(w, 0.0)  # no self-interaction
    return G * np.einsum("ij,ijk->ik", w, diff)


def make_accel(
    mass: np.ndarray,
    G: float = G_ASTRO,
    eps: float = 0.0,
    backend: str = "brute",
    theta: float = 0.5,
    leaf_size: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory returning accel_fn(pos) -> (N,3) for the chosen backend.

    theta/leaf_size apply to the 'barnes_hut' backend only (SPEC §3);
    theta = 0 makes Barnes-Hut exact (and pointless — use it in tests only).
    """
    if backend == "brute":
        return lambda pos: accel_brute(pos, mass, G, eps)
    if backend == "barnes_hut":
        from nbody.barnes_hut import accel_barnes_hut

        return lambda pos: accel_barnes_hut(pos, mass, G, eps, theta=theta, leaf_size=leaf_size)
    raise ValueError(f"unknown force backend: {backend!r}")

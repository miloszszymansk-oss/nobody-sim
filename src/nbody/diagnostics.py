"""Conserved quantities and orbital elements — the judges of correctness (SPEC §1.4).

CRITICAL: total_energy must use the SAME eps as the force backend (SPEC §1.2, pitfall #1);
the pair potential is V_ij = -G m_i m_j / sqrt(|r_ij|^2 + eps^2), whose gradient is exactly
the softened force in forces.accel_brute.
"""

from __future__ import annotations

import numpy as np

from nbody.bodies import System
from nbody.constants import G_ASTRO


def kinetic_energy(s: System) -> float:
    """T = sum_i m_i |v_i|^2 / 2."""
    return 0.5 * float(np.sum(s.mass * np.einsum("ij,ij->i", s.vel, s.vel)))


def total_energy(s: System, G: float = G_ASTRO, eps: float = 0.0) -> float:
    """E = sum_i m_i |v_i|^2 / 2  -  G sum_{i<j} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)."""
    kinetic = 0.5 * float(np.sum(s.mass * np.einsum("ij,ij->i", s.vel, s.vel)))
    i, j = np.triu_indices(s.n, k=1)
    d = s.pos[i] - s.pos[j]
    r = np.sqrt(np.einsum("ij,ij->i", d, d) + eps * eps)
    potential = -G * float(np.sum(s.mass[i] * s.mass[j] / r))
    return kinetic + potential


def momentum(s: System) -> np.ndarray:
    """Total momentum P = sum_i m_i v_i -> (3,)."""
    return (s.mass[:, None] * s.vel).sum(axis=0)


def angular_momentum(s: System) -> np.ndarray:
    """Total angular momentum L = sum_i m_i r_i x v_i -> (3,)."""
    return (s.mass[:, None] * np.cross(s.pos, s.vel)).sum(axis=0)


def orbital_elements(rel_pos: np.ndarray, rel_vel: np.ndarray, gm: float) -> tuple[float, float]:
    """(a, e) of the relative two-body orbit from instantaneous state vectors.

    a from vis-viva: 1/a = 2/|r| - |v|^2/mu (SPEC §4.2).
    e as the norm of the eccentricity (Laplace-Runge-Lenz) vector:
    e_vec = (v x l)/mu - r_hat, with l = r x v (specific angular momentum).
    """
    r = float(np.linalg.norm(rel_pos))
    v2 = float(rel_vel @ rel_vel)
    a = 1.0 / (2.0 / r - v2 / gm)
    l = np.cross(rel_pos, rel_vel)
    e_vec = np.cross(rel_vel, l) / gm - rel_pos / r
    return a, float(np.linalg.norm(e_vec))

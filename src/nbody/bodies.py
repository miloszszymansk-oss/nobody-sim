"""System state and initial-condition factories (SPEC §4.2).

State is three NumPy arrays — pos (N,3) [AU], vel (N,3) [AU/yr], mass (N,) [M_sun].
No per-body objects: vectorization is the only loop we allow (SPEC §4.1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbody.constants import G_ASTRO


@dataclass
class System:
    pos: np.ndarray  # (N,3) [AU]
    vel: np.ndarray  # (N,3) [AU/yr]
    mass: np.ndarray  # (N,)  [M_sun]

    @property
    def n(self) -> int:
        return self.mass.shape[0]

    def copy(self) -> "System":
        return System(self.pos.copy(), self.vel.copy(), self.mass.copy())


def kepler_period(a: float, m_total: float, G: float = G_ASTRO) -> float:
    """Kepler III: T = 2 pi sqrt(a^3 / (G m_total)). In AU/yr/M_sun units, a=1, m=1 -> T=1."""
    return 2.0 * np.pi * np.sqrt(a**3 / (G * m_total))


def sun_earth() -> System:
    """Sun + Earth preset: r = 1 AU, v = 2 pi AU/yr (SPEC §1.3), barycentric frame."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 2.0 * np.pi, 0.0]])
    mass = np.array([1.0, 3.0e-6])
    return to_barycentric(System(pos, vel, mass))


def two_body(m1: float, m2: float, a: float, e: float, G: float = G_ASTRO) -> System:
    """Two bodies starting at perihelion of the relative orbit, in the barycentric frame.

    Relative orbit (SPEC §4.2): mu = G (m1 + m2), r_p = a (1 - e),
    v_p = sqrt(mu (1 + e) / (a (1 - e))) tangential (from vis-viva at r = r_p).
    The relative vector is split onto the two bodies with mass weights, so the
    total momentum is exactly zero by construction.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError(f"eccentricity must be in [0, 1), got {e}")
    mu = G * (m1 + m2)
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    rel_pos = np.array([r_p, 0.0, 0.0])
    rel_vel = np.array([0.0, v_p, 0.0])
    w1, w2 = m2 / (m1 + m2), m1 / (m1 + m2)  # body 1 gets -w1 * rel, body 2 gets +w2 * rel
    pos = np.vstack([-w1 * rel_pos, w2 * rel_pos])
    vel = np.vstack([-w1 * rel_vel, w2 * rel_vel])
    return System(pos, vel, np.array([m1, m2], dtype=float))


def uniform_cluster(n: int, radius: float, total_mass: float, seed: int = 0) -> System:
    """Cold (zero-velocity) uniform-density sphere of n equal-mass bodies.

    Sampling: isotropic direction from a 3D normal, radius r = R * u^(1/3)
    (the cube root makes the *density*, not the radius, uniform).
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = radius * rng.random(n) ** (1.0 / 3.0)
    pos = d * r[:, None]
    vel = np.zeros((n, 3))
    mass = np.full(n, total_mass / n)
    return to_barycentric(System(pos, vel, mass))


def to_barycentric(s: System) -> System:
    """Shift to the center-of-mass frame: COM at origin, total momentum exactly zero."""
    m = s.mass
    com_pos = (m[:, None] * s.pos).sum(axis=0) / m.sum()
    com_vel = (m[:, None] * s.vel).sum(axis=0) / m.sum()
    return System(s.pos - com_pos, s.vel - com_vel, m.copy())

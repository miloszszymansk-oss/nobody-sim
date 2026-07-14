"""Plummer sampler against its own math (SPEC §6.4): virial ratio, half-mass radius,
speed cap. Statistical tests use n large enough that sampling noise ~ n^{-1/2} sits
well inside the tolerances (calibrated per SPEC §5 rule)."""

import numpy as np

from nbody import diagnostics as diag
from nbody.bodies import plummer_sphere
from nbody.constants import G_ASTRO


def test_virial_ratio_near_unity():
    """Ergodic Plummer is in virial equilibrium: 2T/|W| = 1 (W = -(3pi/32) G M^2 / a)."""
    m_tot, a = 1.0, 1.0
    s = plummer_sphere(n=2000, total_mass=m_tot, scale_radius=a, seed=1)
    kinetic = 0.5 * float(np.sum(s.mass * np.einsum("ij,ij->i", s.vel, s.vel)))
    # potential from the sampled configuration (pairwise, eps=0)
    i, j = np.triu_indices(s.n, k=1)
    d = np.linalg.norm(s.pos[i] - s.pos[j], axis=1)
    w = -G_ASTRO * float(np.sum(s.mass[i] * s.mass[j] / d))
    ratio = 2.0 * kinetic / abs(w)
    assert 0.85 < ratio < 1.15, f"virial ratio {ratio:.3f}"


def test_half_mass_radius():
    """r_h = a / sqrt(2^(2/3) - 1) ~ 1.3048 a (with a modest tail-clip bias allowance)."""
    a = 1.0
    s = plummer_sphere(n=4000, total_mass=1.0, scale_radius=a, seed=2)
    r = np.linalg.norm(s.pos, axis=1)
    r_h = np.median(r)
    expected = a / np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
    assert abs(r_h - expected) / expected < 0.08, f"r_h {r_h:.3f} vs {expected:.3f}"


def test_speeds_below_escape():
    """Rejection sampler must never exceed the local escape speed."""
    m_tot, a = 1.0, 1.0
    s = plummer_sphere(n=3000, total_mass=m_tot, scale_radius=a, seed=3)
    r = np.linalg.norm(s.pos, axis=1)
    # velocities were assigned before the barycentric shift; the shift is O(n^{-1/2}),
    # so allow a tiny margin above v_esc
    v = np.linalg.norm(s.vel, axis=1)
    v_esc = np.sqrt(2.0 * G_ASTRO * m_tot) * (r**2 + a**2) ** -0.25
    assert np.all(v <= v_esc * 1.02)


def test_barycentric():
    s = plummer_sphere(n=1000, total_mass=1.0, scale_radius=1.0, seed=4)
    assert np.linalg.norm(diag.momentum(s)) < 1e-12
    com = (s.mass[:, None] * s.pos).sum(axis=0)
    assert np.linalg.norm(com) < 1e-12

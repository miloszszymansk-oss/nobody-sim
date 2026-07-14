"""T1-T2: force backend against hand-computed values and Newton's third law (SPEC §5)."""

import numpy as np

from nbody.bodies import uniform_cluster
from nbody.forces import accel_brute


def test_two_body_analytic():
    """T1: two bodies on the x-axis, acceleration equals the hand formula to machine precision."""
    G = 1.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([3.0, 5.0])
    # Hand formula: a_1 = G m_2 / r^2 towards body 2 -> (+5/4, 0, 0); a_2 = G m_1 / r^2 -> (-3/4, 0, 0)
    expected = np.array([[5.0 / 4.0, 0.0, 0.0], [-3.0 / 4.0, 0.0, 0.0]])
    got = accel_brute(pos, mass, G=G, eps=0.0)
    assert np.max(np.abs(got - expected)) < 1e-14


def test_two_body_analytic_softened():
    """T1b: same, with eps > 0 — denominator becomes (r^2 + eps^2)^{3/2}."""
    G, eps = 1.0, 0.5
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([3.0, 5.0])
    d3 = (4.0 + 0.25) ** 1.5
    expected = np.array([[G * 5.0 * 2.0 / d3, 0.0, 0.0], [-G * 3.0 * 2.0 / d3, 0.0, 0.0]])
    got = accel_brute(pos, mass, G=G, eps=eps)
    assert np.max(np.abs(got - expected)) < 1e-14


def test_newton_third_law():
    """T2: sum_i m_i a_i = 0 (pairwise antisymmetric forces), relative to the force scale."""
    s = uniform_cluster(n=50, radius=1.0, total_mass=1.0, seed=42)
    a = accel_brute(s.pos, s.mass, eps=0.05)
    net_force = (s.mass[:, None] * a).sum(axis=0)
    scale = np.abs(s.mass[:, None] * a).sum()  # total |force| as normalization
    assert np.linalg.norm(net_force) / scale < 1e-12

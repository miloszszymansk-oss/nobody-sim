"""T4-T8: physics as executable assertions (SPEC §5). Tolerances calibrated on first
run per SPEC §5 rule, then frozen — any regression turns these red."""

import numpy as np

from nbody import diagnostics as diag
from nbody.bodies import System, kepler_period, two_body
from nbody.constants import G_ASTRO
from nbody.integrators import leapfrog_step
from nbody.forces import make_accel
from nbody.sim import Config, History, run


def _rel_state(h: History, k: int):
    return h.pos[k, 1] - h.pos[k, 0]


def test_t4_circular_orbit():
    """T4: e=0 orbit — separation stays 1 AU; after 100 periods bodies return to start."""
    m1, m2 = 1.0, 3.0e-6
    s = two_body(m1, m2, a=1.0, e=0.0)
    T = kepler_period(1.0, m1 + m2)
    n_orbits, spo = 100, 1000  # steps per orbit
    cfg = Config(dt=T / spo, n_steps=n_orbits * spo, record_every=spo // 10)
    h = run(s, cfg)
    sep = np.linalg.norm(h.pos[:, 1] - h.pos[:, 0], axis=1)
    assert np.max(np.abs(sep - 1.0)) < 1e-4
    # return to initial position after an integer number of periods (phase error O(dt^2 * t))
    assert np.linalg.norm(h.pos[-1] - h.pos[0]) < 1e-2


def test_t5_kepler_ellipse_elements():
    """T5: e=0.5 — orbital elements (a, e) constant.

    Two equivalent checks, both from recorded history:
    (1) instantaneous (a, e) via diagnostics.orbital_elements at the initial state,
    (2) along the whole run: separation confined to [a(1-e), a(1+e)] with both extremes
        reached, and a recovered from total energy E = -G m1 m2 / (2a) at every sample.
    """
    m1, m2 = 1.0, 1.0e-3
    mu = G_ASTRO * (m1 + m2)
    a0, e0 = 1.0, 0.5
    s = two_body(m1, m2, a=a0, e=e0)
    a_i, e_i = diag.orbital_elements(s.pos[1] - s.pos[0], s.vel[1] - s.vel[0], mu)
    assert abs(a_i - a0) < 1e-12 and abs(e_i - e0) < 1e-12  # ICs consistent with elements

    T = kepler_period(a0, m1 + m2)
    cfg = Config(dt=T / 1000, n_steps=100 * 1000, record_every=100)
    h = run(s, cfg)
    sep = np.linalg.norm(h.pos[:, 1] - h.pos[:, 0], axis=1)
    assert abs(sep.min() - a0 * (1 - e0)) < 1e-3 and abs(sep.max() - a0 * (1 + e0)) < 1e-3
    a_from_E = -G_ASTRO * m1 * m2 / (2.0 * h.energy)  # exact two-body relation
    assert np.max(np.abs(a_from_E - a0)) < 1e-3


def test_t6_leapfrog_energy_bounded_no_trend():
    """T6: |dE/E| stays in a bounded band with no secular trend (symplectic signature)."""
    s = two_body(1.0, 1.0e-3, a=1.0, e=0.5)
    T = kepler_period(1.0, 1.001)
    cfg = Config(dt=T / 1000, n_steps=100 * 1000, record_every=100)
    h = run(s, cfg)
    rel_err = np.abs(h.energy / h.energy[0] - 1.0)
    assert np.max(rel_err) < 1e-3
    half = len(rel_err) // 2
    assert np.max(rel_err[half:]) < 1.5 * np.max(rel_err[:half]), "energy error is trending"


def test_t7_rk4_energy_drifts():
    """T7 (negative control): RK4 energy error grows secularly on an eccentric orbit."""
    s = two_body(1.0, 1.0e-3, a=1.0, e=0.6)
    T = kepler_period(1.0, 1.001)
    cfg = Config(dt=T / 300, n_steps=100 * 300, integrator="rk4", record_every=300)
    h = run(s, cfg)
    rel_err = np.abs(h.energy / h.energy[0] - 1.0)
    q = len(rel_err) // 4
    e25, e50, e75, e100 = rel_err[q], rel_err[2 * q], rel_err[3 * q], rel_err[-1]
    assert e100 > 3.0 * rel_err[len(rel_err) // 10 + 1]
    assert e25 < e50 < e75 < e100, "expected monotone secular drift"


def test_t8_momentum_and_L_conserved():
    """T8: leapfrog conserves total P and L to roundoff (kicks: Newton III & central forces;
    drifts: trivially). See SPEC §1.4."""
    s = two_body(1.0, 1.0e-3, a=1.0, e=0.5)
    T = kepler_period(1.0, 1.001)
    cfg = Config(dt=T / 1000, n_steps=50 * 1000, record_every=1000)
    h = run(s, cfg)
    # momentum: zero by construction at t=0, must stay zero
    # (recompute from last snapshot velocities via a 1-step continuation is overkill;
    #  L history is recorded, P we check via COM drift of positions)
    com = (h.mass[None, :, None] * h.pos).sum(axis=1) / h.mass.sum()
    assert np.max(np.linalg.norm(com, axis=1)) < 1e-10  # COM stays at origin <=> P stays 0
    L0 = h.angular_momentum[0]
    dL = np.linalg.norm(h.angular_momentum - L0, axis=1) / np.linalg.norm(L0)
    assert np.max(dL) < 1e-10


def test_fastpath_matches_reference():
    """sim.run leapfrog fast path must equal repeated reference leapfrog_step calls exactly."""
    s = two_body(1.0, 2.0e-2, a=1.0, e=0.3)
    dt, n = 1e-3, 200
    cfg = Config(dt=dt, n_steps=n, record_every=n)
    h = run(s, cfg)
    accel_fn = make_accel(s.mass, cfg.G, cfg.eps, "brute")
    pos, vel = s.pos.copy(), s.vel.copy()
    for _ in range(n):
        pos, vel = leapfrog_step(pos, vel, dt, accel_fn)
    assert np.array_equal(h.pos[-1], pos), "fast path diverged from reference implementation"

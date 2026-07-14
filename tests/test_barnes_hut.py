"""T9: Barnes-Hut against brute force (SPEC §3, §5) + edge cases + energy smoke test."""

import numpy as np

from nbody import diagnostics as diag
from nbody.barnes_hut import accel_barnes_hut, build_octree
from nbody.bodies import uniform_cluster
from nbody.forces import accel_brute
from nbody.sim import Config, run


def _rel_err(a_bh: np.ndarray, a_br: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a_bh - a_br, axis=1) / np.linalg.norm(a_br, axis=1)


def test_tree_mass_and_com_invariants():
    """Every tree level partitions the bodies: root mass = total, root COM = global COM."""
    s = uniform_cluster(n=500, radius=1.0, total_mass=2.0, seed=3)
    t = build_octree(s.pos, s.mass)
    assert abs(t.mass[0] - s.mass.sum()) < 1e-12
    com = (s.mass[:, None] * s.pos).sum(axis=0) / s.mass.sum()
    assert np.max(np.abs(t.com[0] - com)) < 1e-12
    # leaf CSR covers each body exactly once
    assert np.array_equal(np.sort(t.leaf_bodies), np.arange(500))


def test_theta_zero_matches_brute():
    """T9a: theta=0 opens everything -> exact pairwise sum, equal to brute up to
    summation-order roundoff."""
    s = uniform_cluster(n=300, radius=1.0, total_mass=1.0, seed=7)
    a_br = accel_brute(s.pos, s.mass, eps=0.01)
    a_bh = accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=0.0)
    scale = np.abs(a_br).max()
    assert np.max(np.abs(a_bh - a_br)) / scale < 1e-12


def test_two_body_exact_through_tree():
    """A 2-body tree has only leaves -> exact regardless of theta (matches T1 setup)."""
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([3.0, 5.0])
    expected = np.array([[5.0 / 4.0, 0.0, 0.0], [-3.0 / 4.0, 0.0, 0.0]])
    got = accel_barnes_hut(pos, mass, G=1.0, eps=0.0, theta=0.8)
    assert np.max(np.abs(got - expected)) < 1e-14


def test_theta_05_error_percentiles():
    """T9b: theta=0.5 monopole error on a 1000-body cluster.

    Calibrated (SPEC §5 rule) from first run: median 5.3e-3, p99 2.6e-2. The max is a
    deliberately avoided statistic — relative error blows up on bodies whose net force
    nearly cancels (cluster-edge geometry), telling us about cancellation, not the tree."""
    s = uniform_cluster(n=1000, radius=1.0, total_mass=1.0, seed=11)
    a_br = accel_brute(s.pos, s.mass, eps=0.01)
    a_bh = accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=0.5)
    err = _rel_err(a_bh, a_br)
    assert np.median(err) < 1e-2, f"median {np.median(err):.2e}"
    assert np.percentile(err, 99) < 5e-2, f"p99 {np.percentile(err, 99):.2e}"


def test_error_scales_with_theta():
    """Property test, stronger than absolute thresholds: halving theta must shrink the
    median error by well over the ~theta^2 monopole bound (observed scaling ~theta^3:
    with d measured to the COM the dipole term vanishes; measured ratio ~0.11)."""
    s = uniform_cluster(n=1000, radius=1.0, total_mass=1.0, seed=11)
    a_br = accel_brute(s.pos, s.mass, eps=0.01)
    med = {
        th: np.median(_rel_err(accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=th), a_br))
        for th in (0.25, 0.5)
    }
    assert med[0.25] < 0.3 * med[0.5], f"scaling broken: {med}"


def test_coincident_bodies_no_nan():
    """SPEC §3 edge case: identical positions land in a max-depth bucket leaf;
    softened forces stay finite and match brute."""
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0, 1.0])
    a_bh = accel_barnes_hut(pos, mass, G=1.0, eps=0.1, theta=0.5)
    a_br = accel_brute(pos, mass, G=1.0, eps=0.1)
    assert np.all(np.isfinite(a_bh))
    assert np.max(np.abs(a_bh - a_br)) < 1e-12


def test_leaf_size_bucket_consistency():
    """leaf_size > 1 changes the tree, not the physics (exact pairwise inside buckets)."""
    s = uniform_cluster(n=400, radius=1.0, total_mass=1.0, seed=5)
    a1 = accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=0.0, leaf_size=1)
    a8 = accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=0.0, leaf_size=8)
    scale = np.abs(a1).max()
    assert np.max(np.abs(a1 - a8)) / scale < 1e-12


def test_energy_conservation_with_bh_leapfrog():
    """Smoke: leapfrog + BH on a softened cold cluster over a PRE-COLLAPSE horizon.

    Design note (a first version of this test failed for the right reason): the cold
    sphere free-falls in t_ff = (pi/2) sqrt(R^3/(2GM)) ~ 0.18 yr, and integrating
    through core collapse at dt=1e-3 punishes integration stiffness, not the tree.
    Horizon t=0.1 stays pre-collapse. Calibration run: BH 7.9e-4 vs brute 1.2e-3 —
    the monopole approximation does not degrade energy conservation here."""
    n, radius = 200, 1.0
    eps = 0.05 * radius / n ** (1.0 / 3.0)  # SPEC §8.1 heuristic
    s = uniform_cluster(n=n, radius=radius, total_mass=1.0, seed=13)
    cfg = Config(dt=1e-3, n_steps=100, eps=eps, force="barnes_hut", theta=0.5, record_every=20)
    h = run(s, cfg)
    rel = np.abs(h.energy / h.energy[0] - 1.0)
    assert np.all(np.isfinite(h.energy))
    assert rel.max() < 5e-3, f"max |dE/E| = {rel.max():.2e}"

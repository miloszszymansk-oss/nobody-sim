"""Simulation loop, history recording and JSON export for the web player (SPEC §4.2, §8.2).

The leapfrog path caches the acceleration between steps (the closing kick of step n
uses the same a(r_{n+1}) as the opening kick of step n+1), so it costs 1 accel call
per step versus 2 in the reference integrators.leapfrog_step. Same math, bit-for-bit:
enforced by tests/test_conservation.py::test_fastpath_matches_reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from nbody import diagnostics as diag
from nbody.bodies import System
from nbody.constants import G_ASTRO
from nbody.forces import make_accel
from nbody.integrators import STEPPERS


@dataclass
class Config:
    dt: float
    n_steps: int
    G: float = G_ASTRO
    eps: float = 0.0
    integrator: str = "leapfrog"  # key into integrators.STEPPERS
    force: str = "brute"  # backend for forces.make_accel
    theta: float = 0.5  # Barnes-Hut opening angle (used when force="barnes_hut")
    record_every: int = 1


@dataclass
class History:
    time: np.ndarray  # (K,)
    pos: np.ndarray  # (K,N,3)
    energy: np.ndarray  # (K,)
    angular_momentum: np.ndarray  # (K,3)
    mass: np.ndarray  # (N,)

    def to_json(self, path: str, decimals: int = 4, meta: dict | None = None) -> None:
        """Export in the `nbody-history/1` schema (SPEC §4.4) for the HTML player.

        Design for fast browser parsing: numeric payloads are FLAT arrays, not nested
        lists — JS does `Float64Array.from(j.pos)` once, no per-step object churn.
        Layout: pos[(k*n + i)*3 + c] = coordinate c of body i at sample k
        (step-major, body-minor, xyz). energy has length K, angular_momentum 3K.
        `decimals` trades file size for precision (4 -> ~6 bytes/number).
        """
        k, n = self.time.shape[0], self.mass.shape[0]
        payload = {
            "schema": "nbody-history/1",
            "n": n,
            "k": k,
            "meta": meta or {},
            "time": np.round(self.time, decimals).tolist(),
            "mass": self.mass.tolist(),
            "pos": np.round(self.pos, decimals).ravel().tolist(),
            "energy": self.energy.tolist(),
            "angular_momentum": np.round(self.angular_momentum, 10).ravel().tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))


def run(system: System, cfg: Config) -> History:
    """Integrate cfg.n_steps steps, sampling diagnostics every cfg.record_every steps.

    Diagnostics use cfg.eps — the same softening as the forces (SPEC §1.2 pitfall #1).
    """
    if cfg.integrator not in STEPPERS:
        raise ValueError(f"unknown integrator: {cfg.integrator!r}")
    pos = np.asarray(system.pos, dtype=float).copy()
    vel = np.asarray(system.vel, dtype=float).copy()
    mass = np.asarray(system.mass, dtype=float).copy()
    accel_fn = make_accel(mass, cfg.G, cfg.eps, cfg.force, theta=cfg.theta)

    k_samples = cfg.n_steps // cfg.record_every + 1
    time = np.empty(k_samples)
    pos_hist = np.empty((k_samples, mass.shape[0], 3))
    energy = np.empty(k_samples)
    ang_mom = np.empty((k_samples, 3))

    def record(idx: int, t: float, p: np.ndarray, v: np.ndarray) -> None:
        snap = System(p, v, mass)
        time[idx] = t
        pos_hist[idx] = p
        energy[idx] = diag.total_energy(snap, cfg.G, cfg.eps)
        ang_mom[idx] = diag.angular_momentum(snap)

    record(0, 0.0, pos, vel)
    idx = 1
    if cfg.integrator == "leapfrog":
        # Fast path: cached acceleration, 1 accel call/step (see module docstring).
        a = accel_fn(pos)
        for step in range(1, cfg.n_steps + 1):
            v_half = vel + 0.5 * cfg.dt * a
            pos = pos + cfg.dt * v_half
            a = accel_fn(pos)
            vel = v_half + 0.5 * cfg.dt * a
            if step % cfg.record_every == 0:
                record(idx, step * cfg.dt, pos, vel)
                idx += 1
    else:
        stepper = STEPPERS[cfg.integrator]
        for step in range(1, cfg.n_steps + 1):
            pos, vel = stepper(pos, vel, cfg.dt, accel_fn)
            if step % cfg.record_every == 0:
                record(idx, step * cfg.dt, pos, vel)
                idx += 1

    return History(time, pos_hist, energy, ang_mom, mass)

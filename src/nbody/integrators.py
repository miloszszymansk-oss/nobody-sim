"""Time integrators, physics-agnostic (SPEC §2, §4.1.3).

Common signature: step(pos, vel, dt, accel_fn) -> (pos_next, vel_next).
accel_fn: (N,3) positions -> (N,3) accelerations. Pure functions: inputs untouched.

These are the *reference* implementations (clarity first). The production loop in
sim.run() uses a mathematically identical leapfrog fast path that caches the last
acceleration (1 accel call per step instead of 2); equality is enforced by test.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

AccelFn = Callable[[np.ndarray], np.ndarray]


def leapfrog_step(
    pos: np.ndarray, vel: np.ndarray, dt: float, accel_fn: AccelFn
) -> tuple[np.ndarray, np.ndarray]:
    """Kick-drift-kick velocity Verlet: 2nd order, symplectic, time-reversible (SPEC §2.1)."""
    v_half = vel + 0.5 * dt * accel_fn(pos)  # kick
    pos_next = pos + dt * v_half  # drift
    vel_next = v_half + 0.5 * dt * accel_fn(pos_next)  # kick
    return pos_next, vel_next


def rk4_step(
    pos: np.ndarray, vel: np.ndarray, dt: float, accel_fn: AccelFn
) -> tuple[np.ndarray, np.ndarray]:
    """Classical RK4 on y = (pos, vel), f(y) = (vel, a(pos)): 4th order, NOT symplectic (SPEC §2.2)."""
    k1r, k1v = vel, accel_fn(pos)
    k2r, k2v = vel + 0.5 * dt * k1v, accel_fn(pos + 0.5 * dt * k1r)
    k3r, k3v = vel + 0.5 * dt * k2v, accel_fn(pos + 0.5 * dt * k2r)
    k4r, k4v = vel + dt * k3v, accel_fn(pos + dt * k3r)
    pos_next = pos + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
    vel_next = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return pos_next, vel_next


STEPPERS = {"leapfrog": leapfrog_step, "rk4": rk4_step}

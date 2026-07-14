"""T3: convergence orders on the harmonic oscillator, where the exact solution is known.

SHO: a(x) = -omega^2 x, x(0) = x0, v(0) = 0  =>  x(t) = x0 cos(omega t).
The integrators never learn this is not gravity — accel_fn is just a function (SPEC §4.1.3).
Measured global error vs dt on log-log must have slope ~2 (leapfrog), ~4 (RK4).

Measurement-point subtlety (found the hard way, kept as documentation): starting at
x = X0, v = 0 and measuring |x - x_exact| after exactly one period is DEGENERATE.
Both integrators there have (near-)exact amplitude and a pure phase error d_phi, and at
an extremum of cos the phase enters only quadratically: err ~ 1 - cos(d_phi) ~ d_phi^2/2.
Leapfrog (d_phi ~ dt^2) then fakes order 4; RK4 fakes ~5. Fix: evaluate at a generic
phase (0.85 of a period) on the full state norm sqrt(dx^2 + (dv/omega)^2), where the
phase error enters linearly and the true orders 2 and 4 appear.
"""

import numpy as np

from nbody.integrators import leapfrog_step, rk4_step

OMEGA = 2.0 * np.pi
PERIOD = 1.0
X0 = 1.0
PHASE_FRACTION = 0.85  # generic point — NOT an integer/half multiple of the period


def _sho_state_error(stepper, n_per_period: int) -> float:
    accel = lambda pos: -OMEGA**2 * pos  # noqa: E731
    pos = np.array([[X0, 0.0, 0.0]])
    vel = np.zeros((1, 3))
    dt = PERIOD / n_per_period
    n_steps = round(PHASE_FRACTION * n_per_period)
    for _ in range(n_steps):
        pos, vel = stepper(pos, vel, dt, accel)
    t = n_steps * dt
    x_e = X0 * np.cos(OMEGA * t)
    v_e = -X0 * OMEGA * np.sin(OMEGA * t)
    return float(np.hypot(pos[0, 0] - x_e, (vel[0, 0] - v_e) / OMEGA))


def _slope(stepper, steps_list) -> float:
    errs = [_sho_state_error(stepper, n) for n in steps_list]
    log_dt = np.log([PERIOD / n for n in steps_list])
    log_err = np.log(errs)
    return float(np.polyfit(log_dt, log_err, 1)[0])


def test_leapfrog_order_two():
    slope = _slope(leapfrog_step, [50, 100, 200, 400])
    assert abs(slope - 2.0) < 0.1, f"leapfrog convergence slope {slope:.3f}, expected 2.0±0.1"


def test_rk4_order_four():
    slope = _slope(rk4_step, [10, 20, 40, 80])
    assert abs(slope - 4.0) < 0.2, f"RK4 convergence slope {slope:.3f}, expected 4.0±0.2"


def test_rk4_more_accurate_short_term():
    """Locally RK4 beats leapfrog at equal dt — the point of §2.3 is that this is NOT the whole story."""
    assert _sho_state_error(rk4_step, 100) < _sho_state_error(leapfrog_step, 100)

"""Experiment 1 (SPEC §2.3, §6.1): leapfrog vs RK4 energy behavior on an e=0.9 orbit.

Claim under test: leapfrog's |dE/E| oscillates in a bounded band (shadow Hamiltonian),
RK4's drifts secularly despite its higher order. Output: CSV + PNG + JSON summary.

Note on dt: at e=0.9 the perihelion passage timescale is t_p ~ r_p/v_p ~ T/274,
so dt = T/3000 gives ~11 steps per passage — deliberately coarse enough to make
integrator error visible on a 300-orbit horizon without hiding it in roundoff.
"""

from __future__ import annotations

import argparse
import json
import time as walltime
from pathlib import Path

import numpy as np

from nbody.bodies import kepler_period, two_body
from nbody.sim import Config, run

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--e", type=float, default=0.9)
    p.add_argument("--orbits", type=int, default=300)
    p.add_argument("--steps-per-orbit", type=int, default=3000)
    p.add_argument("--m2", type=float, default=1e-3)
    p.add_argument("--only", choices=["leapfrog", "rk4"], default=None)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    m1, m2 = 1.0, args.m2
    T = kepler_period(1.0, m1 + m2)
    dt = T / args.steps_per_orbit
    n_steps = args.orbits * args.steps_per_orbit
    record_every = max(1, n_steps // 3000)

    results = {}
    for integ in [args.only] if args.only else ["leapfrog", "rk4"]:
        s = two_body(m1, m2, a=1.0, e=args.e)
        t0 = walltime.perf_counter()
        h = run(s, Config(dt=dt, n_steps=n_steps, integrator=integ, record_every=record_every))
        wall = walltime.perf_counter() - t0
        rel = np.abs(h.energy / h.energy[0] - 1.0)
        np.savetxt(
            OUT / f"energy_{integ}_e{args.e}.csv",
            np.column_stack([h.time / T, rel]),
            delimiter=",",
            header="t_orbits,abs_rel_energy_error",
            comments="",
        )
        results[integ] = {
            "wall_s": round(wall, 2),
            "max_rel_err": float(rel.max()),
            "final_rel_err": float(rel[-1]),
            "band_first_half": float(rel[: len(rel) // 2].max()),
            "band_second_half": float(rel[len(rel) // 2 :].max()),
        }
        print(f"[{integ}] wall={wall:.1f}s max|dE/E|={rel.max():.3e} final={rel[-1]:.3e}", flush=True)

    (OUT / f"summary_e{args.e}.json").write_text(json.dumps(
        {"e": args.e, "orbits": args.orbits, "steps_per_orbit": args.steps_per_orbit, **results}, indent=2))

    if not args.only:
        plot(args.e, args.orbits, args.steps_per_orbit)


def plot(e: float, orbits: int, spo: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    styles = {"leapfrog": dict(color="#1a7f37", lw=1.0), "rk4": dict(color="#c0392b", lw=1.0)}
    for integ, st in styles.items():
        data = np.loadtxt(OUT / f"energy_{integ}_e{e}.csv", delimiter=",", skiprows=1)
        mask = data[:, 1] > 0  # log scale: drop exact zeros (t=0)
        ax.plot(data[mask, 0], data[mask, 1], label=integ.upper() if integ == "rk4" else "Leapfrog (KDK)", **st)
    ax.set_yscale("log")
    ax.set_xlabel("time [orbital periods]")
    ax.set_ylabel("|E(t) − E₀| / |E₀|")
    ax.set_title(f"Energy error: symplectic leapfrog vs RK4 — two-body, e={e}, dt=T/{spo}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = OUT / f"energy_drift_e{e}.png"
    fig.savefig(out)
    print(f"plot -> {out}", flush=True)


if __name__ == "__main__":
    main()

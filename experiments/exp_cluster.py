"""Experiment 4 (SPEC §6.4): collision of two Plummer spheres — Barnes-Hut + leapfrog.

Scenario: two equal Plummer clusters (each M=0.5, a=0.5, n bodies) on a bound
encounter: centers at (±d/2, ±b/2, 0), velocities (∓v/2, 0, 0). Orbital energy
E = mu v^2/2 - G M1 M2 / d < 0 -> merger within a few crossing times.

Outputs: figures/cluster_merger.png (snapshot montage + energy trace),
experiments/out/cluster_merger.json (nbody-history/1, feeds the D4 HTML player),
and a JSON summary with energy-drift statistics.
"""

from __future__ import annotations

import argparse
import json
import time as walltime
from pathlib import Path

import numpy as np

from nbody.bodies import System, plummer_sphere, to_barycentric
from nbody.constants import G_ASTRO
from nbody.sim import Config, run

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "out"
FIG = ROOT / "figures"


def two_plummer_collision(
    n_each: int, mass_each: float, a: float, d: float, b: float, v_rel: float, seed: int
) -> System:
    c1 = plummer_sphere(n_each, mass_each, a, seed=seed)
    c2 = plummer_sphere(n_each, mass_each, a, seed=seed + 1)
    pos = np.vstack([c1.pos + [-d / 2, -b / 2, 0.0], c2.pos + [d / 2, b / 2, 0.0]])
    vel = np.vstack([c1.vel + [v_rel / 2, 0.0, 0.0], c2.vel + [-v_rel / 2, 0.0, 0.0]])
    return to_barycentric(System(pos, vel, np.concatenate([c1.mass, c2.mass])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-each", type=int, default=400)
    p.add_argument("--a", type=float, default=0.5, help="Plummer scale radius")
    p.add_argument("--d", type=float, default=3.0, help="initial center separation")
    p.add_argument("--b", type=float, default=0.4, help="impact parameter")
    p.add_argument("--v-rel", type=float, default=3.0, help="closing speed")
    p.add_argument("--t-end", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1.25e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    m_each = 0.5
    # orbital energy of the cluster pair: reduced mass mu = m_each/2 for equal masses
    e_orb = 0.5 * (m_each / 2) * args.v_rel**2 - G_ASTRO * m_each**2 / args.d
    eps = 0.05 * args.a / args.n_each ** (1.0 / 3.0)  # SPEC §8.1 heuristic
    n_steps = int(round(args.t_end / args.dt))
    record_every = max(1, n_steps // 200)

    s = two_plummer_collision(args.n_each, m_each, args.a, args.d, args.b, args.v_rel, args.seed)
    cfg = Config(dt=args.dt, n_steps=n_steps, eps=eps, force="barnes_hut", theta=0.5,
                 record_every=record_every)
    t0 = walltime.perf_counter()
    h = run(s, cfg)
    wall = walltime.perf_counter() - t0

    rel = np.abs(h.energy / h.energy[0] - 1.0)
    print(f"run: {n_steps} steps, N={2 * args.n_each}, wall {wall:.1f}s, "
          f"max|dE/E|={rel.max():.3e} (orbital E={e_orb:.2f}, bound={e_orb < 0})", flush=True)

    h.to_json(OUT / "cluster_merger.json", decimals=4, meta={
        "experiment": "two_plummer_collision", "G": G_ASTRO, "eps": eps,
        "theta": 0.5, "n_each": args.n_each, "a": args.a, "d": args.d,
        "b": args.b, "v_rel": args.v_rel, "dt": args.dt,
    })
    (OUT / "cluster_summary.json").write_text(json.dumps({
        "n_steps": n_steps, "wall_s": round(wall, 2),
        "max_rel_energy_err": float(rel.max()),
        "final_rel_energy_err": float(rel[-1]), "bound_encounter": bool(e_orb < 0)}, indent=2))
    montage(h, args.n_each)


def montage(h, n_each: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    k = h.time.shape[0]
    picks = [int(f * (k - 1)) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    fig = plt.figure(figsize=(13, 7.5), dpi=150)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.55])
    lim = 2.6
    for ax_i, ki in enumerate(picks):
        ax = fig.add_subplot(gs[ax_i // 3, ax_i % 3])
        ax.scatter(h.pos[ki, :n_each, 0], h.pos[ki, :n_each, 1], s=1.2, c="#1a7f37", alpha=0.6)
        ax.scatter(h.pos[ki, n_each:, 0], h.pos[ki, n_each:, 1], s=1.2, c="#c0392b", alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"t = {h.time[ki]:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.plot(h.time, np.abs(h.energy / h.energy[0] - 1.0), color="#333", lw=1.0)
    ax_e.set_yscale("log")
    ax_e.set_xlabel("t [yr]")
    ax_e.set_ylabel("|ΔE/E₀|")
    ax_e.grid(alpha=0.25)
    fig.suptitle("Two Plummer spheres, bound encounter — Barnes-Hut θ=0.5 + leapfrog "
                 f"(N={2 * n_each})", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "cluster_merger.png")
    print(f"figure -> {FIG / 'cluster_merger.png'}", flush=True)


if __name__ == "__main__":
    main()

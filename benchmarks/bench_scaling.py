"""SPEC §6.3: wall time per force evaluation, brute O(N^2) vs Barnes-Hut O(N log N).

Protocol: uniform cluster, eps=0.01, theta=0.5; best of REPS repetitions per point
(best-of is the standard for microbenchmarks — it estimates the noise floor).
Brute is capped at its SPEC §4.3 memory limit. Output: CSV + figures/scaling.png
with N^2 and N log N guide lines anchored at the largest common N.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nbody.barnes_hut import accel_barnes_hut
from nbody.bodies import uniform_cluster
from nbody.forces import BRUTE_N_MAX, accel_brute

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "out"
FIG = ROOT / "figures"

NS = [100, 300, 1000, 3000, 10_000, 30_000]
REPS = 3
THETA = 0.5


def best_time(fn, reps: int = REPS) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rows = []
    for n in NS:
        s = uniform_cluster(n=n, radius=1.0, total_mass=1.0, seed=1)
        t_bh = best_time(lambda: accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=THETA))
        t_br = (
            best_time(lambda: accel_brute(s.pos, s.mass, eps=0.01))
            if n <= BRUTE_N_MAX
            else float("nan")
        )
        rows.append((n, t_br, t_bh))
        print(f"N={n:>6}  brute={t_br if t_br == t_br else float('nan'):.4f}s  bh={t_bh:.4f}s",
              flush=True)
    data = np.array(rows)
    np.savetxt(OUT / "scaling.csv", data, delimiter=",",
               header="N,t_brute_s,t_barnes_hut_s", comments="")

    # crossover: first N where BH beats brute (interpolated in log-log)
    valid = ~np.isnan(data[:, 1])
    diff = np.log(data[valid, 1]) - np.log(data[valid, 2])
    cross = None
    for i in range(1, diff.shape[0]):
        if diff[i - 1] < 0 <= diff[i]:
            n0, n1 = np.log(data[valid][i - 1, 0]), np.log(data[valid][i, 0])
            f = -diff[i - 1] / (diff[i] - diff[i - 1])
            cross = float(np.exp(n0 + f * (n1 - n0)))
            break
    (OUT / "summary.json").write_text(json.dumps(
        {"theta": THETA, "reps": REPS, "crossover_N": cross,
         "rows": [{"N": int(a), "brute_s": None if np.isnan(b) else b, "bh_s": c}
                  for a, b, c in rows]}, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.loglog(data[valid, 0], data[valid, 1], "o-", color="#c0392b", label="brute force O(N²)")
    ax.loglog(data[:, 0], data[:, 2], "s-", color="#1a7f37", label=f"Barnes-Hut θ={THETA}")
    # guide lines anchored at the largest common N
    n_anchor = data[valid, 0][-1]
    t2, tnl = data[valid, 1][-1], data[valid, 2][-1]
    ns = np.array(NS, dtype=float)
    ax.loglog(ns, t2 * (ns / n_anchor) ** 2, ":", color="#c0392b", alpha=0.5, label="∝ N²")
    ax.loglog(ns, tnl * (ns * np.log(ns)) / (n_anchor * np.log(n_anchor)), ":",
              color="#1a7f37", alpha=0.5, label="∝ N log N")
    if cross:
        ax.axvline(cross, color="gray", ls="--", lw=1, alpha=0.7)
        ax.annotate(f"crossover ≈ N={cross:,.0f}", (cross, ax.get_ylim()[0] * 3),
                    rotation=90, fontsize=9, ha="right")
    ax.set_xlabel("N bodies")
    ax.set_ylabel("wall time per force evaluation [s]")
    ax.set_title("Force backend scaling (single core, vectorized NumPy)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png")
    print(f"figure -> {FIG / 'scaling.png'}; crossover ~ N={cross}", flush=True)


if __name__ == "__main__":
    main()

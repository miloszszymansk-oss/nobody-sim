"""Final Day-1 figure (SPEC §2.3): two panels proving the symplectic story.

Panel A — e=0.9, dt=T/3000, 300 orbits: leapfrog band vs early RK4 secular growth;
RK4 trend extrapolated (linear in t) to its predicted crossover with the band.
Panel B — e=0.6, dt=T/200, 2500 orbits: the crossover observed empirically,
validating the extrapolation logic of panel A at computationally cheap parameters.

Output: figures/energy_drift.png (committed; experiments/out stays reproducible scratch).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "out"
FIG = ROOT / "figures"

GREEN, RED = "#1a7f37", "#c0392b"


def load(integ: str, e: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(OUT / f"energy_{integ}_e{e}.csv", delimiter=",", skiprows=1)
    m = (d[:, 0] > 0) & (d[:, 1] > 0)  # log axes
    return d[m, 0], d[m, 1]


def panel(ax, e: float, spo: int, extrapolate_to: float | None) -> None:
    t_lf, err_lf = load("leapfrog", e)
    t_rk, err_rk = load("rk4", e)
    band = err_lf.max()
    ax.plot(t_lf, err_lf, color=GREEN, lw=0.9, label="Leapfrog (KDK), order 2, symplectic")
    ax.plot(t_rk, err_rk, color=RED, lw=1.2, label="RK4, order 4, not symplectic")
    ax.axhline(band, color=GREEN, ls=":", lw=1.0, alpha=0.8)
    rate = err_rk[-1] / t_rk[-1]  # secular drift ~ linear in t
    if extrapolate_to is not None:
        t_x = np.logspace(np.log10(t_rk[-1]), np.log10(extrapolate_to), 50)
        ax.plot(t_x, rate * t_x, color=RED, ls="--", lw=1.0, alpha=0.7,
                label="RK4 trend extrapolation")
        t_cross = band / rate
        ax.plot([t_cross], [band], "o", color="black", ms=6, zorder=5)
        ax.annotate(f"predicted crossover\n≈ {t_cross:,.0f} orbits",
                    (t_cross, band), textcoords="offset points", xytext=(-10, -34),
                    ha="right", fontsize=9)
    else:
        above = np.nonzero(err_rk > band)[0]
        t_cross = t_rk[above[0]]
        ax.plot([t_cross], [band], "o", color="black", ms=6, zorder=5)
        ax.annotate(f"observed crossover\n≈ {t_cross:,.0f} orbits",
                    (t_cross, band), textcoords="offset points", xytext=(-8, -36),
                    ha="right", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time [orbital periods]")
    ax.set_title(f"e = {e}, dt = T/{spo}")
    ax.grid(True, which="both", alpha=0.2)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150, sharey=True)
    panel(ax_a, e=0.9, spo=3000, extrapolate_to=2e4)
    panel(ax_b, e=0.6, spo=200, extrapolate_to=None)
    ax_a.set_ylabel("|E(t) − E₀| / |E₀|")
    ax_a.legend(loc="lower right", fontsize=9)
    fig.suptitle("Two-body energy error: bounded symplectic band vs secular RK4 drift "
                 "(leapfrog dotted line = its max band)", y=0.98)
    fig.tight_layout()
    out = FIG / "energy_drift.png"
    fig.savefig(out)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()

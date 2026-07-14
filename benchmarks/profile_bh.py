"""D3 profiling harness (SPEC changelog 0.4): where does a Barnes-Hut step spend time?

Two instruments:
  1. cProfile over a short BH-driven simulation (N=3000) with a pstats digest that
     aggregates the functions we care about (sorting, grouping, tree build, walk).
  2. A/B microbenchmark of the tree-build grouping strategy on IDENTICAL keys:
       A (baseline) : np.unique(return_inverse, return_counts) + separate np.argsort
                      -> the key array is sorted TWICE per tree level;
       B (shipped)  : barnes_hut._group_keys — one argsort, unique/inverse/counts
                      derived from the sorted array, `order` reused for the leaf CSR.
     A is reconstructed here verbatim so the comparison runs on the same inputs
     (the shipped code already uses B; keeping A alive in the harness is the evidence).

Writes benchmarks/out/profile_report.txt and prints the digest to stdout.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from nbody.barnes_hut import _group_keys, accel_barnes_hut, build_octree
from nbody.bodies import plummer_sphere
from nbody.sim import Config, run

OUT = Path(__file__).resolve().parent / "out"
N_PROFILE = 3000
N_STEPS = 20
KEY_SORT_NAMES = ("argsort", "unique", "searchsorted", "sort")


def group_keys_baseline(keys: np.ndarray):
    """The pre-D3 strategy: np.unique + a second, independent argsort (two sorts)."""
    u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(keys, kind="stable")
    return u, inv, cnt, order


def profile_simulation() -> str:
    s = plummer_sphere(n=N_PROFILE, total_mass=1.0, scale_radius=1.0, seed=2)
    cfg = Config(dt=1e-4, n_steps=N_STEPS, eps=0.01, force="barnes_hut", theta=0.5,
                 record_every=N_STEPS)
    prof = cProfile.Profile()
    prof.enable()
    run(s, cfg)
    prof.disable()

    stream = io.StringIO()
    st = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
    st.print_stats(18)
    raw = stream.getvalue()

    # digest: aggregate tottime by interesting function-name substrings
    total = st.total_tt
    buckets: dict[str, float] = {}
    for (filename, _line, funcname), (_cc, _nc, tottime, _cum, _callers) in st.stats.items():
        label = None
        if any(k in funcname for k in KEY_SORT_NAMES):
            label = "sorting/grouping (argsort/unique/searchsorted)"
        elif "build_octree" in funcname or "_group_keys" in funcname or "_assign_children" in funcname:
            label = "tree build (own time)"
        elif "accel_barnes_hut" in funcname or "_ragged_arange" in funcname:
            label = "tree walk (own time)"
        elif "bincount" in funcname or "reduce" in funcname or "einsum" in funcname:
            label = "aggregation (bincount/einsum/ufunc.reduce)"
        if label:
            buckets[label] = buckets.get(label, 0.0) + tottime
    lines = [f"cProfile digest — N={N_PROFILE}, {N_STEPS} BH steps, total tottime {total:.3f}s"]
    for label, t in sorted(buckets.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {t:7.3f}s  {100 * t / total:5.1f}%  {label}")
    return "\n".join(lines) + "\n\n--- raw pstats (top by cumulative) ---\n" + raw


def ab_grouping_benchmark() -> str:
    """Same key streams a real build would see, grouped by both strategies."""
    s = plummer_sphere(n=N_PROFILE, total_mass=1.0, scale_radius=1.0, seed=2)
    tree = build_octree(s.pos, s.mass)  # warm-up + realism check
    rng = np.random.default_rng(0)
    # capture realistic per-level key arrays by re-simulating the digit cascade
    keys_levels = []
    keys = np.zeros(N_PROFILE, dtype=np.int64)
    lo, hi = s.pos.min(0), s.pos.max(0)
    centers = np.broadcast_to(0.5 * (lo + hi), (N_PROFILE, 3)).copy()
    half0 = 0.5 * float((hi - lo).max())
    for depth in range(1, 9):
        bits = s.pos > centers
        keys = keys * 8 + (bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2])
        centers = centers + (bits * 2.0 - 1.0) * (half0 / 2.0**depth)
        keys_levels.append(keys.copy())

    def time_variant(fn) -> float:
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            for kl in keys_levels:
                fn(kl)
            best = min(best, time.perf_counter() - t0)
        return best

    t_a = time_variant(group_keys_baseline)
    t_b = time_variant(_group_keys)
    n_nodes = tree.n_nodes
    return (
        f"A/B grouping on {len(keys_levels)} realistic key levels (N={N_PROFILE}, "
        f"{n_nodes} tree nodes):\n"
        f"  A baseline (unique + argsort, two sorts): {1e3 * t_a:7.2f} ms\n"
        f"  B shipped  (_group_keys, one sort):       {1e3 * t_b:7.2f} ms\n"
        f"  speedup on grouping: {t_a / t_b:.2f}x\n"
    )


def end_to_end_build_benchmark() -> str:
    s = plummer_sphere(n=N_PROFILE, total_mass=1.0, scale_radius=1.0, seed=2)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        build_octree(s.pos, s.mass)
        best = min(best, time.perf_counter() - t0)
    best_full = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        accel_barnes_hut(s.pos, s.mass, eps=0.01, theta=0.5)
        best_full = min(best_full, time.perf_counter() - t0)
    return (f"end-to-end (N={N_PROFILE}): build_octree {1e3 * best:.2f} ms; "
            f"full accel (build+walk) {1e3 * best_full:.2f} ms; "
            f"build share {100 * best / best_full:.0f}%\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = "\n".join([ab_grouping_benchmark(), end_to_end_build_benchmark(),
                        profile_simulation()])
    (OUT / "profile_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

"""Barnes-Hut O(N log N) gravity (SPEC §3) — fully vectorized, no per-body recursion.

Design (why this looks different from textbook pseudocode): a classic recursive
octree walk costs ~1 Python call per (body, node) visit — at N=10^4 that is millions
of interpreter operations per step, slower than vectorized brute force. Instead both
phases here are *level-synchronous* array programs:

  BUILD  — bodies carry an integer octant-path key; at each depth every active body
           appends its octant digit (0..7) to the key, np.unique groups bodies into
           cells, and bincount aggregates cell mass / center of mass. Cells with
           <= leaf_size bodies (or at max_depth — e.g. coincident positions, SPEC §3
           edge case) become bucket leaves; the rest keep subdividing. Child links
           fall out of searchsorted on the next level's parent keys (both sorted).

  WALK   — a frontier of (target body, node) pairs starts at (i, root) for all i and
           is processed one tree level per iteration with pure array ops:
             accept   internal node with s^2 < theta^2 d^2  -> monopole contribution,
             leaf     -> exact softened pairwise over the bucket (self-pairs masked),
             reject   -> pair expands to the node's children (ragged, via repeat).
           Accumulation uses np.bincount per component (fast, deterministic).

Correctness notes:
  * theta = 0 never accepts, so the walk degenerates to exact pairwise summation and
    must match accel_brute to summation-order roundoff (enforced by T9).
  * Acceptance measures d to the node's center of mass with the *unsoftened* metric;
    eps enters only the force denominator, exactly as in forces.accel_brute.
  * Self-safety: for theta < 1/sqrt(3) ~ 0.577, s/d < theta implies d > s*sqrt(3),
    so an accepted node can never contain the target body (max COM-to-corner
    distance inside an edge-s cube is s*sqrt(3)). For larger theta this guarantee
    weakens; we default to theta = 0.5.
  * Barnes-Hut force is NOT pairwise-antisymmetric (each target approximates
    independently), so total momentum is conserved only to O(force error) — the
    Newton-III test T2 applies to brute force alone. Documented trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbody.constants import G_ASTRO

DEFAULT_THETA = 0.5
MAX_DEPTH = 20  # 8^20 < 2^63: octant-path keys stay inside int64


@dataclass
class Octree:
    com: np.ndarray  # (M,3) center of mass per node
    mass: np.ndarray  # (M,)
    size: np.ndarray  # (M,) cube edge length
    is_leaf: np.ndarray  # (M,) bool
    child_start: np.ndarray  # (M,) first child's global node index (internal nodes)
    child_count: np.ndarray  # (M,)
    leaf_start: np.ndarray  # (M,) into leaf_bodies (leaf nodes)
    leaf_count: np.ndarray  # (M,)
    leaf_bodies: np.ndarray  # (sum of leaf_count,) body indices

    @property
    def n_nodes(self) -> int:
        return self.mass.shape[0]


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """[0..c0-1, 0..c1-1, ...] for segment sizes `counts` — the ragged-expansion idiom."""
    total = int(counts.sum())
    ends = np.cumsum(counts)
    return np.arange(total) - np.repeat(ends - counts, counts)


def build_octree(pos: np.ndarray, mass: np.ndarray, leaf_size: int = 1) -> Octree:
    """Level-synchronous octree build over occupied cells only (SPEC §3)."""
    n = pos.shape[0]
    mins, maxs = pos.min(axis=0), pos.max(axis=0)
    center0 = 0.5 * (mins + maxs)
    half0 = 0.5 * float((maxs - mins).max())
    half0 = half0 * (1.0 + 1e-12) if half0 > 0.0 else 1.0  # all-coincident guard

    # global node table (root = node 0)
    com_l = [center0[None, :].copy()]
    mass_l = [np.array([mass.sum()])]
    size_l = [np.array([2.0 * half0])]
    isleaf_l = [np.array([n <= leaf_size])]
    child_start_l = [np.zeros(1, dtype=np.int64)]
    child_count_l = [np.zeros(1, dtype=np.int64)]
    leaf_start_l = [np.zeros(1, dtype=np.int64)]
    leaf_count_l = [np.array([n if n <= leaf_size else 0], dtype=np.int64)]
    leaf_bodies_parts: list[np.ndarray] = []
    if n <= leaf_size:
        leaf_bodies_parts.append(np.arange(n, dtype=np.int64))
        com_l[0] = ((mass[:, None] * pos).sum(axis=0) / mass.sum())[None, :]
        return _assemble(com_l, mass_l, size_l, isleaf_l, child_start_l, child_count_l,
                         leaf_start_l, leaf_count_l, leaf_bodies_parts)
    com_l[0] = ((mass[:, None] * pos).sum(axis=0) / mass.sum())[None, :]

    active = np.arange(n, dtype=np.int64)
    keys = np.zeros(n, dtype=np.int64)  # octant path of each active body
    centers = np.broadcast_to(center0, (n, 3)).copy()  # current cell center per body
    # previous level's internal cells: their sorted keys and global node ids
    prev_keys = np.zeros(1, dtype=np.int64)
    prev_ids = np.zeros(1, dtype=np.int64)
    n_nodes = 1
    n_leaf_bodies = 0

    for depth in range(1, MAX_DEPTH + 1):
        p = pos[active]
        m = mass[active]
        bits = p > centers  # (A,3) octant bits
        digit = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        keys = keys * 8 + digit
        centers = centers + (bits * 2.0 - 1.0) * (half0 / 2.0**depth)

        u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        c_mass = np.bincount(inv, weights=m)
        c_com = np.stack(
            [np.bincount(inv, weights=m * p[:, k]) / c_mass for k in range(3)], axis=1
        )
        c_size = np.full(u.shape[0], half0 * 2.0 ** (1 - depth))
        c_leaf = (cnt <= leaf_size) | (depth == MAX_DEPTH)

        # link children into the previous level's internal cells
        parent_of_u = u >> 3
        left = np.searchsorted(parent_of_u, prev_keys, side="left")
        right = np.searchsorted(parent_of_u, prev_keys, side="right")
        _assign_children(child_start_l, child_count_l, prev_ids, n_nodes + left, right - left)

        # bucket-leaf CSR for this level (bodies sorted by key keep cell grouping)
        order = np.argsort(keys, kind="stable")
        leaf_body_mask_sorted = c_leaf[inv[order]]
        lb = active[order][leaf_body_mask_sorted]
        leaf_cnt = np.where(c_leaf, cnt, 0)
        starts_local = np.cumsum(leaf_cnt) - leaf_cnt  # positions inside lb
        leaf_bodies_parts.append(lb)

        com_l.append(c_com)
        mass_l.append(c_mass)
        size_l.append(c_size)
        isleaf_l.append(c_leaf)
        child_start_l.append(np.zeros(u.shape[0], dtype=np.int64))
        child_count_l.append(np.zeros(u.shape[0], dtype=np.int64))
        leaf_start_l.append(np.where(c_leaf, n_leaf_bodies + starts_local, 0))
        leaf_count_l.append(leaf_cnt.astype(np.int64))

        prev_keys = u[~c_leaf]
        prev_ids = n_nodes + np.nonzero(~c_leaf)[0]
        n_nodes += u.shape[0]
        n_leaf_bodies += lb.shape[0]

        cont = ~c_leaf[inv]
        if not cont.any():
            break
        active = active[cont]
        keys = keys[cont]
        centers = centers[cont]

    return _assemble(com_l, mass_l, size_l, isleaf_l, child_start_l, child_count_l,
                     leaf_start_l, leaf_count_l, leaf_bodies_parts)


def _assign_children(child_start_l, child_count_l, prev_ids, start_vals, count_vals):
    """Write child links for previous-level internal nodes (global ids) into the
    per-level storage lists. Level boundaries are cumulative array lengths."""
    bounds = np.cumsum([0] + [a.shape[0] for a in child_start_l])
    for pid, s, c in zip(prev_ids, start_vals, count_vals):
        lvl = int(np.searchsorted(bounds, pid, side="right")) - 1
        off = int(pid - bounds[lvl])
        child_start_l[lvl][off] = s
        child_count_l[lvl][off] = c


def _assemble(com_l, mass_l, size_l, isleaf_l, cs_l, cc_l, ls_l, lc_l, lb_parts) -> Octree:
    return Octree(
        com=np.concatenate(com_l, axis=0),
        mass=np.concatenate(mass_l),
        size=np.concatenate(size_l),
        is_leaf=np.concatenate(isleaf_l),
        child_start=np.concatenate(cs_l),
        child_count=np.concatenate(cc_l),
        leaf_start=np.concatenate(ls_l),
        leaf_count=np.concatenate(lc_l),
        leaf_bodies=(np.concatenate(lb_parts) if lb_parts else np.zeros(0, dtype=np.int64)),
    )


def accel_barnes_hut(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float = G_ASTRO,
    eps: float = 0.0,
    theta: float = DEFAULT_THETA,
    leaf_size: int = 1,
) -> np.ndarray:
    """Same contract as forces.accel_brute; approximation error controlled by theta
    (~(s/d)^2 per accepted node, SPEC §3). theta = 0 reproduces brute force exactly
    up to summation order. The tree is rebuilt on every call (positions move)."""
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    if n < 2:
        return acc
    tree = build_octree(pos, mass, leaf_size=leaf_size)
    theta2 = theta * theta
    eps2 = eps * eps

    fb = np.arange(n, dtype=np.int64)  # frontier: target body ...
    fn = np.zeros(n, dtype=np.int64)  # ... vs node (start everyone at the root)
    while fb.shape[0]:
        dvec = tree.com[fn] - pos[fb]
        d2 = np.einsum("ij,ij->i", dvec, dvec)
        leaf = tree.is_leaf[fn]
        accept = ~leaf & (tree.size[fn] ** 2 < theta2 * d2)

        if accept.any():
            b, dv, r2 = fb[accept], dvec[accept], d2[accept]
            w = G * tree.mass[fn[accept]] * (r2 + eps2) ** -1.5
            for k in range(3):
                acc[:, k] += np.bincount(b, weights=w * dv[:, k], minlength=n)

        if leaf.any():
            lb, ln = fb[leaf], fn[leaf]
            cnts = tree.leaf_count[ln]
            if int(cnts.sum()):
                srcs = tree.leaf_bodies[np.repeat(tree.leaf_start[ln], cnts) + _ragged_arange(cnts)]
                tgts = np.repeat(lb, cnts)
                nz = srcs != tgts  # mask self-interaction
                srcs, tgts = srcs[nz], tgts[nz]
                dv = pos[srcs] - pos[tgts]
                r2 = np.einsum("ij,ij->i", dv, dv)
                w = G * mass[srcs] * (r2 + eps2) ** -1.5
                for k in range(3):
                    acc[:, k] += np.bincount(tgts, weights=w * dv[:, k], minlength=n)

        rej = ~leaf & ~accept
        if rej.any():
            rb, rn = fb[rej], fn[rej]
            c = tree.child_count[rn]
            fn = np.repeat(tree.child_start[rn], c) + _ragged_arange(c)
            fb = np.repeat(rb, c)
        else:
            break
    return acc

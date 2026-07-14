# nbody-sim

Gravitational N-body simulator built spec-first: symplectic leapfrog vs RK4 (with an empirical
demonstration of why symplecticity matters), Barnes-Hut O(N log N) forces, and a test suite that
enforces physics: energy, momentum and angular-momentum conservation, Kepler orbits, convergence
orders.

Status: Day 1 — architecture contract in [SPEC.md](SPEC.md), core implementation in progress.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

Sections to come (Day 5): Physics · Results (plots) · Reproduce.

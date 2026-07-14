"""Placeholder so pytest collects; real suites arrive with D1 prompt 2 (SPEC §5).

Planned files: test_forces.py (T1-T2), test_integrators.py (T3),
test_conservation.py (T4-T8), test_barnes_hut.py (T9, D2).
"""

import nbody


def test_package_imports_and_G():
    assert abs(nbody.G_ASTRO - 39.478) < 1e-2

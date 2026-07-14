"""Physical constants in astronomical units (SPEC §1.3)."""

import numpy as np

# Gravitational constant in AU^3 / (M_sun * yr^2). Kepler III: T^2 = 4 pi^2 a^3 / (G M).
G_ASTRO: float = 4.0 * np.pi**2

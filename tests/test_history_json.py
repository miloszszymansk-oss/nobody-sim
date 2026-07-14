"""Round-trip test of the nbody-history/1 export (SPEC §4.4)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from nbody.bodies import two_body
from nbody.sim import Config, run


def test_history_json_roundtrip():
    s = two_body(1.0, 1.0e-3, a=1.0, e=0.3)
    cfg = Config(dt=1e-3, n_steps=50, record_every=10)
    h = run(s, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h.json"
        h.to_json(path, decimals=4, meta={"experiment": "unit"})
        j = json.loads(path.read_text())

    assert j["schema"] == "nbody-history/1"
    k, n = j["k"], j["n"]
    assert (k, n) == (len(h.time), 2)
    assert len(j["time"]) == k
    assert len(j["mass"]) == n
    assert len(j["pos"]) == k * n * 3
    assert len(j["energy"]) == k
    assert len(j["kinetic"]) == k
    assert len(j["angular_momentum"]) == k * 3
    # physical consistency of the split: V = E - T must be negative for a bound orbit
    assert all(e - t < 0 for e, t in zip(j["energy"], j["kinetic"]))
    assert j["meta"]["experiment"] == "unit"

    # flatten order contract: pos[(k*n + i)*3 + c]
    pos = np.array(j["pos"]).reshape(k, n, 3)
    assert np.max(np.abs(pos - h.pos)) < 5e-4  # decimals=4 rounding
    assert np.allclose(j["energy"], h.energy)

#!/usr/bin/env python3
"""The event-level spark veto of wft.seed.seeds_from_hits (rerun/model_audit/followups/gd_sparks, 2026-10-28).

    .venv/bin/python -m pytest wft/tests/test_spark_veto.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import seed as ws  # noqa: E402

FEU_X, FEU_Y = 1, 2
POS = {FEU_X: np.arange(512) * ws.PITCH_MM, FEU_Y: np.arange(512) * ws.PITCH_MM}


def _event(eid, nx, ny, x0=100, y0=200):
    """One event: nx contiguous strips in x from x0, ny in y from y0, equal significance."""
    rows = [(eid, FEU_X, x0 + i, 500.0, 50.0) for i in range(nx)] + [(eid, FEU_Y, y0 + i, 500.0, 50.0) for i in range(ny)]
    return pd.DataFrame(rows, columns=['eventId', 'feu', 'channel', 'amplitude', 'significance'])


def test_veto_threshold_is_total_hits():
    df = pd.concat([_event(1, 25, 25), _event(2, 26, 25), _event(3, 6, 5)])
    s = ws.seeds_from_hits(df, POS, FEU_X, FEU_Y)
    assert s[1]['n_hits'] == 50 and not s[1]['spark']          # 50 is kept (> 50 vetoes)
    assert s[2]['n_hits'] == 51 and s[2]['spark']
    assert s[2]['x'] == [] and s[2]['y'] == []                 # a vetoed event is not seeded
    assert not s[3]['spark'] and s[3]['x'] and s[3]['y']


def test_explicit_none_disables_the_veto():
    s = ws.seeds_from_hits(_event(2, 40, 40), POS, FEU_X, FEU_Y, spark_veto=None)
    assert not s[2]['spark'] and s[2]['x'] and s[2]['y']


def test_module_constant_is_read_at_call_time():
    """A wft.diag override moves the module constant after the function is defined; it must be the value used
    (before 2026-10-28 the default was bound at import and the override was recorded but not applied)."""
    old = ws.SPARK_VETO_HITS
    try:
        ws.SPARK_VETO_HITS = 100
        s = ws.seeds_from_hits(_event(2, 40, 40), POS, FEU_X, FEU_Y)
        assert not s[2]['spark']
        ws.SPARK_VETO_HITS = 10
        s = ws.seeds_from_hits(_event(3, 6, 5), POS, FEU_X, FEU_Y)
        assert s[3]['spark']
    finally:
        ws.SPARK_VETO_HITS = old

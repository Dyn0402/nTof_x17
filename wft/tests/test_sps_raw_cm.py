"""RAW SPS events: the cache builder removes the per-block common mode (pass-1 review P2, 2026-10-30)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'rerun', 'model_audit'))
import sps_build_cache as sbc                                   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW                      # noqa: E402


def _event(seed=0, ns=64, drop=0.23):
    rng = np.random.default_rng(seed)
    ped = rng.uniform(344, 2947, 512)
    cm = rng.normal(0, 300, (8, ns)).cumsum(1) / 4                   # wandering within the window, per block
    sig = np.zeros((512, ns))
    t = np.arange(ns)
    for view in ('x', 'y'):
        chs = np.flatnonzero(VIEW == view)
        c0 = chs[np.argsort(POSITION_MM[chs])][150]
        near = chs[np.abs(POSITION_MM[chs] - POSITION_MM[c0]) < 2.5]
        for c in near:
            sig[c] = 800 * np.exp(-0.5 * ((t - 30) / 4) ** 2) * np.exp(-abs(POSITION_MM[c] - POSITION_MM[c0]))
    full = ped[:, None] + cm[np.arange(512) // 64] + sig + rng.normal(0, 10, (512, ns))
    ch, sm = np.meshgrid(np.arange(512), t, indexing='ij')
    keep = rng.random((512, ns)) > drop                              # the FEU's dropped samples are absent
    return ch[keep], sm[keep], full[keep].astype(np.float32), sig[keep]


def test_raw_common_mode_removed():
    ch, sm, am, truth = _event()
    out = sbc._baseline(ch, sm, am, raw=True)
    quiet = truth == 0
    assert np.std(out[quiet]) < 15                                   # ~ the 10 ADC noise, not the ~300 ADC common mode
    assert abs(np.median(out[quiet])) < 3
    big = truth > 200
    assert np.median(np.abs(out[big] - truth[big])) < 15             # the signal survives
    # without the common-mode step (the pass-1 cache) the quiet strips carry the common mode
    assert np.std(sbc._pre_baseline(ch, sm, am)[quiet]) > 50


def test_zs_unchanged():
    ch = np.array([1, 2, 3]); sm = np.array([0, 1, 2]); am = np.array([256.0, 300.0, 512.0], np.float32)
    out = sbc._baseline(ch, sm, am, raw=False)
    assert np.allclose(out, [1e-3, 44.0, 256.0])                     # the flat ZS baseline only; an exact 0 -> 1e-3

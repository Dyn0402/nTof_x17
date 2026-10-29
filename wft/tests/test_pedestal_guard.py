"""pedestal_guard (F11, off by default): a connector whose hit noise is under-stated gets its significance rescaled and the analyzer
floor re-applied; healthy connectors are never touched; off = identity."""
import types

import numpy as np
import pandas as pd

from wft import reco


class _Reader:
    def __init__(self, f):
        self.noise = np.full(512, 10.0, dtype=np.float32)


def _hits(seed=0):
    rng = np.random.default_rng(seed)
    n = 20000
    ch = rng.integers(0, 512, n)
    ped_noise = np.where(ch >= 448, 5.0, 10.0)          # connector 8 pedestal at half the data noise
    amp = rng.uniform(55, 400, n)
    return pd.DataFrame(dict(eventId=rng.integers(0, 3000, n), feu=6, channel=ch, amplitude=amp,
                             significance=amp / ped_noise))


def test_guard_corrects_only_the_bad_connector(monkeypatch):
    monkeypatch.setattr('wft.io.FeuReader', _Reader)
    monkeypatch.setattr(reco, 'PED_MISMATCH_REL', 0.7)
    h = _hits()
    cfg = types.SimpleNamespace()
    out, rep = reco.pedestal_guard(h, cfg, files_by_feu={6: ['f.root']})
    assert list(rep['corrected'][6]) == [8]
    assert abs(rep['corrected'][6][8]['r'] - 0.5) < 0.01
    ok = h['channel'] < 448
    pd.testing.assert_frame_equal(out[out['channel'] < 448], h[ok])     # healthy connectors untouched
    bad = out[out['channel'] >= 448]
    assert (bad['significance'] >= reco.HIT_SIG).all()
    assert len(bad) == int(((h['channel'] >= 448) & (h['amplitude'] / 10.0 >= reco.HIT_SIG)).sum())


def test_guard_off_and_healthy_identity(monkeypatch):
    monkeypatch.setattr('wft.io.FeuReader', _Reader)
    monkeypatch.setattr(reco, 'PED_MISMATCH_REL', 0.7)
    h = _hits()
    h['significance'] = h['amplitude'] / 10.0
    out, rep = reco.pedestal_guard(h, types.SimpleNamespace(), files_by_feu={6: ['f.root']})
    assert out is h and not rep['corrected']
    monkeypatch.setattr(reco, 'PED_MISMATCH_REL', 0)
    out, rep = reco.pedestal_guard(_hits(), types.SimpleNamespace(), files_by_feu={6: ['f.root']})
    assert not rep['corrected']


def test_default_off_is_identity():
    h = _hits()
    out, rep = reco.pedestal_guard(h, types.SimpleNamespace(), files_by_feu={6: ['f.root']})
    assert reco.PED_MISMATCH_REL == 0 and out is h and not rep['corrected']

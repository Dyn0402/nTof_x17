"""wft.io per-sample ADC rails (opt-in, ERROR_MODEL.md "Rails", 2026-10-29): FeuReader.iter_events(with_raw=True),
extract_window(raw_all=...), wft.io.rails_wanted().  Without a consumer switched on nothing changes."""
import numpy as np

from wft import io as wio
from wft import model as wm


class _Arr(dict):
    pass


class _Tree:
    """Stands in for the uproot 'nt' tree of one decoded FEU file."""

    def __init__(self, amps, eids, ftst):
        self.amps, self.eids, self.ftst = amps, eids, ftst

    def arrays(self, names, entry_start=0, entry_stop=None, library='np'):
        s = slice(entry_start, entry_stop)
        out = {'eventId': np.asarray(self.eids)[s], 'ftst': np.asarray(self.ftst)[s]}
        out['amplitude'] = np.array(self.amps[s], dtype=object)
        return out


def _reader(n_ev=3, ns=8, seed=0):
    rng = np.random.default_rng(seed)
    r = wio.FeuReader.__new__(wio.FeuReader)
    r.n_sample = ns
    r.ped = rng.uniform(200, 400, 512).astype(np.float32)
    r.noise = np.full(512, 8.0, np.float32)
    amps = []
    for e in range(n_ev):
        raw = (r.ped[None, :] + rng.normal(0, 8, (ns, 512)) + rng.normal(0, 30, (ns, 1))).round()
        raw[3:5, 100:104] = 4095                         # a clipped pulse
        raw[5, 100] = 0                                  # and the undershoot after it, at the floor
        amps.append(np.clip(raw, 0, 4095).astype(np.uint16).reshape(-1))
    r.tree = _Tree(amps, list(range(10, 10 + n_ev)), [7] * n_ev)
    r.event_ids = np.arange(10, 10 + n_ev)
    r.n_entries = n_ev
    r.incomplete = {}
    return r, amps


def test_with_raw_same_waveform_and_rails_exact():
    r, amps = _reader()
    plain = list(r.iter_events())
    both = list(r.iter_events(with_raw=True))
    assert len(plain) == len(both) == 3
    for (e1, f1, w1), (e2, f2, w2, raw) in zip(plain, both):
        assert e1 == e2 and f1 == f2 and np.array_equal(w1, w2)      # W is untouched
        assert raw.shape == w1.shape and raw.dtype == np.int16
        hi, lo = wio.rail_arrays(w2, raw)
        assert np.allclose(hi[100:104, 3:5], w2[100:104, 3:5])       # clipped samples sit exactly at their rail
        assert np.allclose(lo[100, 5], w2[100, 5])                   # the floor sample sits at the floor
        # rail = 4095 - ped - CM: the same for every sample of a channel only up to the CM
        cm = (raw.astype(np.float32) - r.ped[:, None]) - w2
        assert np.allclose(hi, 4095.0 - r.ped[:, None] - cm, atol=1e-3)


def test_extract_window_carries_rails_only_when_given():
    r, _ = _reader()
    eid, ftst, W, raw = next(r.iter_events(with_raw=True))
    pos = np.arange(512) * 0.78
    a = wio.extract_window(W, r.noise, pos, [100, 101, 102, 103], 3)
    b = wio.extract_window(W, r.noise, pos, [100, 101, 102, 103], 3, raw_all=raw)
    assert a.rail_hi is None and a.rail_lo is None
    assert np.array_equal(a.W, b.W) and np.array_equal(a.ch, b.ch)
    assert b.rail_hi.shape == b.W.shape
    clip = b.W >= b.rail_hi - 0.5
    assert np.array_equal(clip, raw[b.ch] >= 4095)


def test_rails_wanted_follows_consumers():
    old = wio.CARRY_RAILS, wm.SAT_RAIL, wm.SAT_FLOOR
    try:
        wio.CARRY_RAILS, wm.SAT_RAIL, wm.SAT_FLOOR = False, False, 0
        assert not wio.rails_wanted()
        wm.SAT_RAIL = True
        assert wio.rails_wanted()
        wm.SAT_RAIL, wm.SAT_FLOOR = False, 1
        assert wio.rails_wanted()
        wm.SAT_FLOOR, wio.CARRY_RAILS = 0, True
        assert wio.rails_wanted()
    finally:
        wio.CARRY_RAILS, wm.SAT_RAIL, wm.SAT_FLOOR = old

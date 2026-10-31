"""SPS cache: channels coupled to an oscillator are censored in the events where it is ON (rerun/sps_junk, 2026-10-31).

On run55 d700 the oscillating Y ch 373 (157.56 mm) is on in 36 % of the events, and in those events ch 375 (156.0 mm)
carries an in-phase copy above 300 ADC 80 % of the time (0.2 % otherwise).  The copy won the seed whenever the real Y
signal was weaker: 13 % of that row's Y windows sat at 155-160 mm.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'rerun', 'model_audit'))
import sps_build_cache as sbc                                   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW                      # noqa: E402

OSC, COPY = 373, 375


def _peaks(n=3000, on_frac=0.36, copy_frac=0.8, beam_mm=(170.0, 290.0), beam_with_osc=False, rng=None):
    """Per-event peaks [n, 512]: a beam lighting ~3 Y strips per event over beam_mm (and X), the oscillator ON in
    on_frac of the events with an in-phase copy on COPY.  beam_with_osc: the beam is there only in the ON events."""
    rng = rng or np.random.default_rng(3)
    pk = rng.normal(0, 10, (n, 512)).clip(0)
    on = rng.random(n) < on_frac
    ys = np.flatnonzero((VIEW == 'y') & (POSITION_MM >= beam_mm[0]) & (POSITION_MM <= beam_mm[1]))
    xs = np.flatnonzero((VIEW == 'x') & (POSITION_MM >= 100) & (POSITION_MM <= 240))
    for i in range(n):
        if beam_with_osc and not on[i]:
            continue
        for chs in (ys, xs):
            c = rng.choice(chs); pk[i, c] = 800; pk[i, chs[np.abs(POSITION_MM[chs] - POSITION_MM[c]) < 1.6]] += 300
    pk[on, OSC] = 900
    pk[on & (rng.random(n) < copy_frac), COPY] = 350
    return pk


def _osc():
    o = sbc.SEED_BAD.copy(); o[OSC] = True
    return o


def test_copy_is_coupled():
    cp = sbc.coupled_channels(_peaks(), _osc())
    assert cp == {OSC: [COPY]}


def test_beam_strips_next_to_the_oscillator_are_not_coupled():
    # the beam itself covers the oscillator's neighbourhood (150-200 mm): its strips are lit ON and OFF alike
    cp = sbc.coupled_channels(_peaks(beam_mm=(150.0, 200.0), copy_frac=0.0), _osc())
    assert cp == {}


def test_beam_only_while_the_oscillator_is_on_is_not_coupling():
    # an oscillator that is on only while the beam is (beam-off triggers in the OFF events, cf. run60 ov07): every strip
    # is lit more in the ON events; the view's own ON/OFF ratio away from the oscillator removes that
    cp = sbc.coupled_channels(_peaks(beam_mm=(150.0, 200.0), copy_frac=0.0, beam_with_osc=True, on_frac=0.5), _osc())
    assert cp == {}


def _zs(track_y=230.0, osc_on=True, copy_amp=350.0, ns=32):
    t = np.arange(ns); ch, sm, am = [], [], []
    for view, x0 in (('x', 185.0), ('y', track_y)):
        chs = np.flatnonzero(VIEW == view)
        for c in chs[np.abs(POSITION_MM[chs] - x0) < 2.0]:
            a = 250 * np.exp(-0.5 * ((t - 12) / 3) ** 2) * np.exp(-abs(POSITION_MM[c] - x0))
            keep = a > 20; ch += [c] * keep.sum(); sm += list(t[keep]); am += list(256 + a[keep])
    if osc_on:          # the oscillator and its in-phase copies (run55 d700: 375 strongest, then 371, 376, 370)
        for c, amp in ((OSC, 900.0), (COPY, copy_amp), (371, 200.0), (376, 110.0), (370, 90.0)):
            w = amp * np.clip(np.sin(t / 2.4), 0, None); keep = w > 20
            ch += [c] * keep.sum(); sm += list(t[keep]); am += list(256 + w[keep])
    return np.array(ch), np.array(sm), np.array(am, np.float32)


COPIES = [370, 371, COPY, 376]


def test_on_event_copy_does_not_seed_and_is_censored():
    bad = _osc(); couple = {OSC: np.array(COPIES)}
    # a weak real Y track (250 ADC lead) at 230 mm; the copy (350) would win the seed
    ch, sm, am = _zs()
    old = sbc.event_record(ch, sm, am, raw=False, nsmp=32, seed_bad=bad, qmin=100)
    assert old['seed_ch_y'] == COPY
    rec = sbc.event_record(ch, sm, am, raw=False, nsmp=32, seed_bad=bad, qmin=100, couple=couple)
    assert abs(POSITION_MM[rec['seed_ch_y']] - 230.0) < 1.0 and rec['censor_y'] == [] and rec['censor_x'] == []
    # a real track next to the oscillator: the copy sits in the window, blanked and listed for the fit to censor
    ch, sm, am = _zs(track_y=153.0)
    rec = sbc.event_record(ch, sm, am, raw=False, nsmp=32, seed_bad=bad, qmin=100, couple=couple)
    assert sorted(rec['censor_y']) == COPIES
    for c in COPIES:
        assert not np.asarray(rec['y']['W'])[list(rec['y']['ch']).index(c)].any()
    assert abs(rec['ref_mesh_y'] - 153.0) < 1.0


def test_off_event_strip_is_ordinary():
    bad = _osc(); couple = {OSC: np.array(COPIES)}
    ch, sm, am = _zs(track_y=156.0, osc_on=False)          # a real track ON the coupled strip, oscillator off
    rec = sbc.event_record(ch, sm, am, raw=False, nsmp=32, seed_bad=bad, qmin=100, couple=couple)
    assert rec['censor_y'] == [] and abs(POSITION_MM[rec['seed_ch_y']] - 156.0) < 1.0


def test_weak_oscillator_next_to_a_known_one_is_found_but_not_its_copy():
    # run55 d600: ch 373 ON in ~5 % of the events (below the seed rule), its copy 375 in most of those
    pk = _peaks(on_frac=0.05, copy_frac=0.8)
    osc = sbc.near_oscillators(pk, sbc.SEED_BAD.copy())
    assert list(np.flatnonzero(osc)) == [OSC]


def test_beam_strip_next_to_a_known_oscillator_is_not_an_oscillator():
    pk = _peaks(on_frac=0.0, copy_frac=0.0, beam_mm=(150.0, 200.0))
    assert not sbc.near_oscillators(pk, sbc.SEED_BAD.copy()).any()


def test_near_oscillator_needs_enough_events():
    # a 315-event probe (run53 group 003): a handful of ON events is not an oscillator
    pk = _peaks(n=315, on_frac=0.015, copy_frac=0.0)
    assert not sbc.near_oscillators(pk, sbc.SEED_BAD.copy()).any()

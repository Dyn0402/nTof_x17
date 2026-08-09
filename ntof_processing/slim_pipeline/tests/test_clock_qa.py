#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_clock_qa.py -- prove every clock-QA check actually fires.

    python test_clock_qa.py            # 14 cases, ~5 s, no data needed

A monitor that has only ever seen good data is not a monitor: it is a program
that prints "PASS". Each case below synthesises a slim with ONE defect injected
and asserts that the named check reaches the expected level and that the
healthy control does not. If a threshold is ever loosened past the point of
uselessness, a case here goes red.

The synthetic segment mimics the real thing closely enough for the checks --
same tree names and columns, a realistic 6 ns residual core, 80 ms of
time-since-flash, ~1000 bunches -- but it is NOT physics. It exists to exercise
the judgement, not to validate the fit.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ntof_processing.slim_pipeline import clock_qa as Q       # noqa: E402
from ntof_processing.slim_pipeline import config as C         # noqa: E402

RNG = np.random.default_rng(20260809)
N_EV, N_BUNCH = 40_000, 800


def make_segment(d: Path, *, efficiency=0.958, accidental=0.00046,
                 cv_gap=0.0004, resid_mean=0.0, resid_rms=6.3, drift_ns=0.0,
                 K=1.1063e-4, T0=-252.6, arm_shift=None, da_rms=6.5,
                 dk_rms_ppm=0.6, frac_fitted=0.99, flash_hits=0,
                 edge_pileup=False, bootstrap_snr=180.0, with_bunches=True,
                 dream_run='run_00', dream_subrun='synthetic', ntof_run=0,
                 n_ev=None):
    """Write a synthetic ntof_hits directory with the requested properties."""
    d.mkdir(parents=True, exist_ok=True)
    global N_EV
    N_EV = n_ev or 40_000
    n_flash = 500
    n_phys = N_EV - n_flash
    ev_id = np.arange(N_EV, dtype=np.uint64)
    is_flash = np.zeros(N_EV, np.uint8); is_flash[:n_flash] = 1
    phys = is_flash == 0
    bunch = RNG.integers(100, 100 + N_BUNCH, N_EV).astype(np.int32)
    t = RNG.uniform(0, 80e6, N_EV)                     # ns since flash

    matched = np.zeros(N_EV, np.uint8)
    idx = np.flatnonzero(phys)
    take = RNG.choice(idx, int(efficiency * n_phys), replace=False)
    matched[take] = 1
    m = matched == 1

    resid = np.full(N_EV, np.nan, np.float32)
    r = RNG.normal(resid_mean, resid_rms, m.sum())
    if drift_ns:                                       # linear in t
        r = r + drift_ns * (t[m] / 80e6 - 0.5)
    r = np.clip(r, -C.ACCEPT_NS, C.ACCEPT_NS)
    resid[m] = r
    arm = np.full(N_EV, -1, np.int8)
    arm[m] = RNG.integers(0, 4, m.sum())

    events = dict(
        eventId=ev_id, bunch=bunch, t_dream_ns=t, is_flash=is_flash,
        t_pred_ns=t * (1 + K) + T0, matched=matched, residual_ns=resid,
        arm=arm, da_ns=np.zeros(N_EV, np.float32),
        dk=np.zeros(N_EV, np.float32),
        corr_ns=np.zeros(N_EV, np.float32),
        corr_cv_ns=np.zeros(N_EV, np.float32))

    ub = np.unique(bunch)
    nfit = int(frac_fitted * ub.size)
    fitted = np.zeros(ub.size, np.uint8); fitted[:nfit] = 1
    bunches = dict(
        bunch=ub.astype(np.int32),
        n_triggers=np.full(ub.size, N_EV // ub.size, np.int32),
        fitted=fitted,
        da_ns=RNG.normal(0, da_rms, ub.size).astype(np.float32),
        dk=(RNG.normal(0, dk_rms_ppm, ub.size) * 1e-6).astype(np.float32),
        n_core=np.full(ub.size, 40, np.int32))

    # hits: 8 per matched trigger, dt inside the slim window
    W = C.SLIM_NS
    per = 8
    hid = np.repeat(ev_id[m], per)
    ndt = hid.size
    if edge_pileup:                                    # a window that clips
        dt = RNG.uniform(-W, W, ndt)
    else:
        dt = np.clip(RNG.normal(0, 35, ndt), -W, W)
    det = RNG.integers(0, len(C.SCINT_TREES), ndt).astype(np.int16)
    ctl = (RNG.random(ndt) < 0.02).astype(np.uint8)
    if flash_hits:                                     # hits on flash triggers
        hid = np.concatenate([hid, ev_id[:flash_hits]])
        dt = np.concatenate([dt, RNG.uniform(-W, W, flash_hits)])
        det = np.concatenate([det, RNG.integers(0, 12, flash_hits).astype(np.int16)])
        ctl = np.concatenate([ctl, np.zeros(flash_hits, np.uint8)])
    hits = dict(eventId=hid.astype(np.uint64), det=det,
                dt_ns=dt.astype(np.float32), is_control=ctl,
                amp=RNG.uniform(10, 500, dt.size).astype(np.float32),
                satuflag=np.zeros(dt.size, np.uint8))

    fname = f'ntof_hits_{dream_run}_{dream_subrun}_{ntof_run:06d}.root'
    with uproot.recreate(d / fname) as f:
        f['events'] = events
        f['hits'] = hits
        if with_bunches:
            f['bunches'] = bunches

    arm_off = dict(Q.REF_ARM)
    if arm_shift:
        for k, v in arm_shift.items():
            arm_off[k] = arm_off[k] + v
    boot = None
    if bootstrap_snr:
        boot = dict(peak_ns=0.0, counts=int(200 * bootstrap_snr / 10),
                    floor=max(1.0, 200 * bootstrap_snr / 10 / bootstrap_snr),
                    snr=bootstrap_snr, n_candidates=50000, search_ns=50000.0,
                    hist=dict(lo_ns=-50000.0, bin_ns=200.0,
                              counts=[1] * 500))
    (d / 'calibration.json').write_text(json.dumps(dict(
        K=K, T0_ns=T0, arm_offset_ns=arm_off, accept_ns=C.ACCEPT_NS,
        slim_ns=W, control_shift_ns=C.CONTROL_SHIFT_NS,
        n_bunches_fitted=nfit, fit=dict(iters=[], bootstrap=boot))))
    (d / 'qa.json').write_text(json.dumps(dict(
        efficiency=efficiency, efficiency_cv=efficiency - cv_gap,
        accidental=accidental, purity=1 - accidental,
        n_events=N_EV, n_physics=n_phys, n_flash=n_flash,
        n_hits=int(dt.size), hits_per_trigger=float(dt.size / n_phys),
        seconds=1.0)))
    (d / 'provenance.json').write_text(json.dumps(dict(
        dream_run=dream_run, dream_subrun=dream_subrun, ntof_run=ntof_run,
        ntof_processing='synthetic')))
    return d


CASES = [
    # (name, kwargs, check name, expected level)
    ('healthy',            {},                              None,   'PASS'),
    ('efficiency low',     dict(efficiency=0.80),           'match efficiency', 'WARN'),
    ('efficiency dead',    dict(efficiency=0.40),           'match efficiency', 'FAIL'),
    ('accidental high',    dict(accidental=0.004),          'accidental rate', 'WARN'),
    ('accidental huge',    dict(accidental=0.05),           'accidental rate', 'FAIL'),
    ('overfit per-bunch',  dict(cv_gap=0.03),               'cross-validation gap', 'FAIL'),
    ('fit off-centre',     dict(resid_mean=5.0),            'residual centred', 'WARN'),
    ('fit badly off',      dict(resid_mean=10.0),           'residual centred', 'FAIL'),
    ('residuals wide',     dict(resid_rms=14.0),            'residual width', 'WARN'),
    ('clock drifts',       dict(drift_ns=12.0),             'no residual drift in time-since-flash', 'WARN'),
    ('K unphysical',       dict(K=3.0e-4),                  'K in physical range', 'FAIL'),
    ('arm offset moved',   dict(arm_shift={'C': 20.0}),     'arm offsets vs reference', 'WARN'),
    ('per-bunch scatter',  dict(da_rms=22.0),               'per-bunch offset scatter', 'WARN'),
    ('bunches unfitted',   dict(frac_fitted=0.60),          'bunches fitted', 'FAIL'),
    ('hits on flash',      dict(flash_hits=250),            'flash triggers carry no hits', 'FAIL'),
    ('window clipping',    dict(edge_pileup=True),          'coincidence contained in slim window', 'FAIL'),
    ('weak coarse peak',   dict(bootstrap_snr=9.0),         'coarse peak above floor', 'WARN'),
    ('no bootstrap record', dict(bootstrap_snr=0.0),        'coarse peak above floor', 'NA'),
    ('old file, no bunches', dict(with_bunches=False),      'bunches fitted', 'NA'),
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='clockqa_'))
    fails = []
    print(f'{"case":<24} {"check":<38} {"got":<6} {"want":<6}')
    print('-' * 78)
    try:
        for name, kw, check, want in CASES:
            d = make_segment(tmp / name.replace(' ', '_'), **kw)
            q = Q.analyse(d)
            if check is None:
                got = q.verdict
            else:
                hit = [c for c in q.checks if c['name'] == check]
                if not hit:
                    fails.append(f'{name}: no check named {check!r}')
                    continue
                got = hit[0]['level']
            ok = got == want
            print(f'{name:<24} {(check or "VERDICT"):<38} {got:<6} {want:<6}'
                  f'{"" if ok else "   <-- MISMATCH"}')
            if not ok:
                fails.append(f'{name}: {check or "verdict"} was {got}, '
                             f'expected {want}')
            # the healthy control must stay clean in every non-verdict case
            if check is not None and want in ('WARN', 'FAIL'):
                others = [c for c in q.checks
                          if c['name'] != check and c['level'] == 'FAIL']
                if others and name not in ('K unphysical',):
                    fails.append(f'{name}: collateral FAIL in '
                                 f'{[c["name"] for c in others]}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('-' * 78)
    if fails:
        print(f'{len(fails)} PROBLEM(S):')
        for f in fails:
            print(f'  {f}')
        return 1
    print(f'all {len(CASES)} cases behaved as specified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

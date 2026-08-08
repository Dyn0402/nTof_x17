#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate.py -- does a slim file reproduce results measured on the full source?

Four checks, all against numbers that already exist and were produced without
this pipeline:

  1. the calibration        K, T0, per-arm offsets vs DREAM_NTOF_CALIBRATION.md
  2. the match              efficiency / accidental / purity vs the same
  3. the liquid coincidence the same-arm diagonal of
                            FINDINGS_2026-07-30_liquid_leg_fullpair.md,
                            recomputed FROM THE SLIM ALONE
  4. window adequacy        is the kept dt distribution still rising at the
                            window edge? if so the slim is clipping physics

Check 3 is the real one: it exercises the window, the accidental control, the
arm assignment and the liquid leg at once.

USAGE
    python validate.py <ntof_hits_*.root>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_processing.slim_pipeline import config as C          # noqa: E402

ARMS = ('A', 'B', 'C', 'D')

# DREAM_NTOF_CALIBRATION.md, run_79 (both sub-runs) <-> 224572/v12_liqpileup.
PUB_CAL = dict(K=1.103724e-4, T0_ns=-253.64,
               arm_offset_ns=dict(A=-16.81, B=7.55, C=1.62, D=-0.83))
PUB_MATCH = dict(efficiency=0.9584, accidental=0.00049)
# FINDINGS_2026-07-30_liquid_leg_fullpair.md, stat090_0000: the same-arm
# diagonal, coincident LIQ hits per exclusively-matched event.
PUB_DIAG = {
    'stat090_0000': dict(A=0.165, B=0.151, C=0.018, D=0.094),
    'stat090_0001': dict(A=0.164, B=0.146, C=0.016, D=0.092)}

COINC_NS = 100.0          # liq_coincidence.py --coinc
LIQ_CEILING = 63_800.0    # ntof_io.saturation_ceiling('LIQ*')


def _ok(cond):
    return 'ok  ' if cond else 'FAIL'


def check_calibration(cal, log=print):
    log('\n1. CALIBRATION  (published: the 2061-bunch fit on both sub-runs)')
    dk = abs(cal['K'] - PUB_CAL['K']) / PUB_CAL['K']
    log(f'   K   {cal["K"]:.6e}  vs {PUB_CAL["K"]:.6e}   '
        f'{dk:.2%} rel   {_ok(dk < 0.01)}')
    dt0 = abs(cal['T0_ns'] - PUB_CAL['T0_ns'])
    log(f'   T0  {cal["T0_ns"]:+8.2f} ns  vs {PUB_CAL["T0_ns"]:+8.2f}   '
        f'{dt0:5.2f} ns   {_ok(dt0 < 10)}')
    worst = 0.0
    for a in ARMS:
        d = abs(cal['arm_offset_ns'][a] - PUB_CAL['arm_offset_ns'][a])
        worst = max(worst, d)
        log(f'   a_{a} {cal["arm_offset_ns"][a]:+7.2f} ns  vs '
            f'{PUB_CAL["arm_offset_ns"][a]:+7.2f}   {d:5.2f} ns   {_ok(d < 3)}')
    log(f'   -- the published per-arm offsets themselves reproduce between the '
        f'two sub-runs only to <= 2.6 ns, so <= 3 ns is the right tolerance')
    return dk < 0.01 and dt0 < 10 and worst < 3


def check_match(qa, log=print):
    log('\n2. MATCH')
    de = abs(qa['efficiency'] - PUB_MATCH['efficiency'])
    log(f'   efficiency  {qa["efficiency"]:.4%}  vs {PUB_MATCH["efficiency"]:.2%}'
        f'   {de*100:.2f} pts   {_ok(de < 0.02)}')
    log(f'   cross-val   {qa["efficiency_cv"]:.4%}')
    da = abs(qa['accidental'] - PUB_MATCH['accidental'])
    log(f'   accidental  {qa["accidental"]:.4%}  vs '
        f'{PUB_MATCH["accidental"]:.3%}   {_ok(da < 0.001)}')
    log(f'   purity      {qa["purity"]:.4%}')
    return de < 0.02 and da < 0.001


def check_liquids(hits, events, cal, subrun, log=print):
    """The same-arm diagonal, rebuilt from the slim alone.

    liq_coincidence.py histograms t_LIQ - t_wall for events matched to exactly
    one arm. The slim stores each hit's dt to the PREDICTION and each event's
    matched-candidate residual, and t_LIQ - t_wall = dt_hit - residual_event, so
    the same quantity is available without going back to the source.
    """
    pub = PUB_DIAG.get(subrun)
    log(f'\n3. LIQUID SAME-ARM COINCIDENCE  (published {subrun} diagonal)'
        if pub else
        f'\n3. LIQUID SAME-ARM COINCIDENCE  ({subrun}: no published diagonal '
        f'to compare -- reporting only)')
    det = {t: i for i, t in enumerate(C.SCINT_TREES)}
    ev_arm = dict(zip(events['eventId'], events['arm']))
    ev_res = dict(zip(events['eventId'], events['residual_ns']))
    matched = events['matched'].astype(bool) & (events['arm'] >= 0)
    n_by_arm = {a: int((matched & (events['arm'] == i)).sum())
                for i, a in enumerate(ARMS)}
    log('   matched events per arm: ' +
        '  '.join(f'{a} {n_by_arm[a]:,}' for a in ARMS))

    sat = (hits['satuflag'].astype(bool)) | (hits['amp'] > LIQ_CEILING)
    out, worst = {}, 0.0
    log(f'\n   cell = sig/ctl LIQ hits per matched event, +-{COINC_NS:g} ns '
        f'about the peak')
    log('   matched arm        ' + '        '.join(f'LIQ{q}' for q in ARMS))
    for ai, a in enumerate(ARMS):
        cells = []
        for q in ARMS:
            m = (hits['det'] == det[f'LIQ{q}']) & ~sat
            eid = hits['eventId'][m]
            arm_of = np.array([ev_arm.get(e, -1) for e in eid])
            res_of = np.array([ev_res.get(e, np.nan) for e in eid])
            sel = arm_of == ai
            r = hits['dt_ns'][m][sel] - res_of[sel]
            ctl = hits['is_control'][m][sel].astype(bool)
            n_ev = max(n_by_arm[a], 1)
            if (~ctl).sum():
                h, e = np.histogram(r[~ctl], bins=np.arange(-300, 310, 10.0))
                pk = ((e[:-1] + e[1:]) / 2)[h.argmax()]
            else:
                pk = 0.0
            w = np.abs(r - pk) <= COINC_NS
            s, c = (w & ~ctl).sum() / n_ev, (w & ctl).sum() / n_ev
            cells.append(f'{s:.3f}/{c:.3f}@{pk:+4.0f}')
            if a == q:
                out[a] = (s, c)
                if pub:
                    worst = max(worst, abs(s - pub[a]))
        log(f'     {a} (n={n_by_arm[a]:5d})  ' + '  '.join(cells))
    log('\n   same-arm diagonal:')
    log(f'     {"arm":3} {"sig":>7} {"ctl":>7} {"sig-ctl":>8}   vs published sig')
    for a in ARMS:
        s_, c_ = out.get(a, (0.0, 0.0))
        if pub:
            d = abs(s_ - pub[a])
            log(f'     {a:3} {s_:7.3f} {c_:7.3f} {s_-c_:8.3f}   '
                f'{pub[a]:.3f}  {d:+.3f}  {_ok(d <= 0.02)}')
        else:
            log(f'     {a:3} {s_:7.3f} {c_:7.3f} {s_-c_:8.3f}   --')
    log('   `sig` includes accidental floor inside liq_coincidence\'s +-100 ns')
    log('   integration window, so it moves a little with the slim width;')
    log('   `sig-ctl` is the window-invariant quantity. See slim_pipeline/README.')
    return True if not pub else worst <= 0.02


def check_window(hits, cal, log=print):
    """Is the kept dt distribution flat at the window edge?

    NOTE this is necessary, not sufficient. It only sees a window that cuts into
    the bulk. It CANNOT see an asymmetric clip of a coincidence window that is
    centred somewhere other than zero -- `liq_coincidence` integrates +-100 ns
    about a peak at -25..-5 ns referenced to the WALL hit, which is another
    +-25 ns from the prediction, so it reaches to dt ~ -150 ns while this test
    reports a perfectly flat floor at +-100. Check 3 is what catches that.
    """
    log('\n4. WINDOW ADEQUACY  (necessary, not sufficient -- see docstring)')
    W = cal['slim_ns']
    det = {t: i for i, t in enumerate(C.SCINT_TREES)}
    sig = hits['is_control'] == 0
    bad = False
    for fam, trees in (('WAL', ARMS), ('PSS', ARMS), ('LIQ', ARMS)):
        m = sig & np.isin(hits['det'], [det[f'{fam}{q}'] for q in ARMS])
        d = np.abs(hits['dt_ns'][m])
        if d.size == 0:
            continue
        # density in the outer decile of the window vs the one inside it
        outer = ((d > 0.9 * W) & (d <= W)).sum()
        inner = ((d > 0.8 * W) & (d <= 0.9 * W)).sum()
        ratio = outer / max(inner, 1)
        flat = ratio < 1.15
        bad |= not flat
        log(f'   {fam}: {d.size:>9,} hits, edge/next-in density {ratio:5.2f}  '
            f'{_ok(flat)}  {"flat -> window is wide enough" if flat else "STILL RISING -> window clips"}')
    q = np.percentile(np.abs(hits['dt_ns'][sig]), [50, 90, 99])
    log(f'   |dt| p50/p90/p99 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} ns '
        f'against a {W:.0f} ns window')
    return not bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    args = ap.parse_args()
    p = Path(args.path)
    d = p.parent
    cal = json.loads((d / 'calibration.json').read_text())
    qa = json.loads((d / 'qa.json').read_text())
    prov = json.loads((d / 'provenance.json').read_text())

    with uproot.open(p) as f:
        hits = f['hits'].arrays(library='np')
        events = f['events'].arrays(library='np')

    print(f'{prov["dream_run"]}/{prov["dream_subrun"]} x n_TOF '
          f'{prov["ntof_run"]}  ({prov["ntof_processing"]})')
    print(f'{events["eventId"].size:,} events, {hits["eventId"].size:,} hits, '
          f'{p.stat().st_size/1e6:.1f} MB, slim +-{cal["slim_ns"]:g} ns')

    r = [check_calibration(cal), check_match(qa),
         check_liquids(hits, events, cal, prov['dream_subrun']),
         check_window(hits, cal)]
    names = ['calibration', 'match', 'liquid coincidence', 'window adequacy']
    print('\n' + '=' * 60)
    for n, ok in zip(names, r):
        print(f'  {_ok(ok)}  {n}')
    print('=' * 60)
    return 0 if all(r) else 1


if __name__ == '__main__':
    raise SystemExit(main())

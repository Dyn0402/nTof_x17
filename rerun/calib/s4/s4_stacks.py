#!/usr/bin/env python3
"""s4_stacks.py <key> --alignment A --bundle B --out DIR [--nboot 20] [--workers 8] -- S4: the cloud_basics stacks on pass 1.

The physics inputs' own estimators (PASS_PLAN §0, physics_inputs.json provenance), unchanged, re-run on one golden row
with the pass's inputs in place of the pre-pass ones:
  * the reference: M3 rays through the PASS alignment (Analysis_pass1/<row>/wft/alignment/alignment.json, the waveform
    alignment) instead of alignment_tpc_veto50 (the hits chain), via headon_cache's ref-only path;
  * the waveforms: the rr1 decode the pass read (staged by rerun/stage_job.py);
  * the electronics template: the pass bundle's tmpl_x (bench_attach.template's interpolation, same STEP).
Estimators:
  head-on stack   unsat_width.py: x_width.measure on the events whose X window stays below 3500 ADC, |tan_x| < 0.03,
                  free sigma_0, Dd, T, A, dt; bootstrap errors.  Dp = sqrt(2 Dd), v = gap / T.
  gap             gap_extent.py: half-height edges of the per-strip charge of inclined tracks (|tan| 0.2-0.8) against
                  z = (x - x_mesh) / tan_ref, X view, window integral; gaps.py's value is the mean of the default
                  (unsaturated events) and --satmask selections with half their difference in quadrature.
Runs inside stage_job.py (parts run_config,m3,decoded); writes DIR/headon.pkl, DIR/incl.pkl and DIR/s4_stacks.json.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [CB, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

SAT_STACK = 3500.0        # unsat_width.py default
TAN_HEADON = 0.06         # headon_cache.py default
TAN_INCL = (0.12, 0.8)    # gap_extent.py build defaults


def pass_template(bundle):
    """bench_attach.template on the pass bundle's arrays."""
    from rc_diffusion import STEP
    z = np.load(os.path.join(bundle, 'arrays.npz'))
    g, h = z['grid'], z['tmpl_x']
    tg = np.arange(0, g[-1] - g[0], STEP)
    hh = np.interp(tg + g[0], g, h)
    return hh / hh.max()


def build(cfg, best):
    import headon_cache as HC
    ev = HC.ref_only(cfg, best, cfg.m3_tracking_dir, TAN_HEADON)
    head = HC.read_windows(cfg, ev, 5.0, -3.0, 33.0)
    sel = lambda tx, ty: TAN_INCL[0] <= abs(tx) <= TAN_INCL[1] or TAN_INCL[0] <= abs(ty) <= TAN_INCL[1]
    ev = HC.ref_only(cfg, best, cfg.m3_tracking_dir, None, select=sel)
    incl = HC.read_windows(cfg, ev, 5.0, -8.0, 40.0)
    return head, incl


def fit_stack(head, h, nboot, workers, T0):
    import x_width as xw
    uns = {k: e for k, e in head.items() if e['x']['W'].max() < SAT_STACK}
    n = sum(abs(e['tan_x']) < xw.TAN for e in uns.values())
    r = dict(n_headon=int(sum(abs(e['tan_x']) < xw.TAN for e in head.values())), n_unsat=int(n))
    if n < xw.MIN_EV:
        return r
    with ProcessPoolExecutor(workers) as pool:
        m = xw.measure(pool, h, uns, T0, nboot)
    m['Dp'] = float(np.sqrt(2 * m['Dd']))
    m['Dp_err'] = float(m['Dd_err'] / np.sqrt(2 * m['Dd'])) if m['Dd'] > 0 else np.nan
    r['unsat'] = m
    return r


def fit_gap(incl, key, nboot):
    import gap_extent as GE
    ev = list(incl.values())
    out = {}
    for lab, sm in (('default', False), ('satmask', True)):
        GE.SATMASK = sm
        bad = GE.bad_channels(key, 'x')
        n, P, E = GE.profile(ev, 'x', bad, qmode='int')
        if n < 50:
            out[lab] = dict(n=int(n)); continue
        hh = GE.half(P, E)
        hh['gap_err'] = GE.boot(ev, 'x', bad, nboot, est=GE.half, qmode='int')
        sg = {}
        for s in (1, -1):
            ns, Ps, Es = GE.profile(ev, 'x', bad, qmode='int', tsign=s)
            if ns >= 40:
                sg[str(s)] = dict(n=int(ns), **GE.half(Ps, Es))
        out[lab] = dict(n=int(n), half=hh, by_sign=sg, erf=GE.fit(P, E))
    GE.SATMASK = False
    a, b = out.get('default', {}).get('half'), out.get('satmask', {}).get('half')
    if a and b and np.isfinite(a['gap']) and np.isfinite(b['gap']):
        out['gap'] = 0.5 * (a['gap'] + b['gap'])
        out['gap_err'] = float(np.sqrt(0.25 * (a['gap_err'] ** 2 + b['gap_err'] ** 2) + (0.5 * (a['gap'] - b['gap'])) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--alignment', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    import cosmic_micro_tpc_analysis as cm
    setup_paths()
    cfg = get_config(a.key)
    best = cm.load_alignment(a.alignment)
    os.makedirs(a.out, exist_ok=True)
    head, incl = build(cfg, best)
    for name, ev in (('headon', head), ('incl', incl)):
        pickle.dump(ev, open(os.path.join(a.out, f'{name}.pkl'), 'wb'), protocol=4)
    print(f'[s4] {a.key}: {len(head)} head-on, {len(incl)} inclined events', flush=True)
    b = json.load(open(os.path.join(a.bundle, 'bundle.json')))
    T0 = 30.0 / (b['v_drift'] * 1e-3)
    R = dict(key=a.key, alignment=a.alignment, bundle=a.bundle, v_bundle=b['v_drift'],
             commit=os.environ.get('RERUN_CODE_COMMIT'))
    R['stack'] = fit_stack(head, pass_template(a.bundle), a.nboot, a.workers, T0)
    print('[s4] stack', json.dumps(R['stack'].get('unsat', R['stack']), default=float)[:400], flush=True)
    R['gap'] = fit_gap(incl, a.key, a.nboot)
    print('[s4] gap', R['gap'].get('gap'), R['gap'].get('gap_err'), flush=True)
    json.dump(R, open(os.path.join(a.out, 's4_stacks.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()

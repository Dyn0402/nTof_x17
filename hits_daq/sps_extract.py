#!/usr/bin/env python3
"""sps_extract.py <registry key> --out OUT.pkl [--per-plateau 3000] -- Arm D windows of one SPS det4 row (known angle).

The SPS pass row (rerun/model_audit/sps_reco.py) rebuilt exactly: the same cache (sps_build_cache, the pass's RAW / ZS
baseline, brightest strip +-16 per view, peak >= 300 ADC, both views), the same bundle adaptation (_init: H4 channel
layer, v = v_span or the bundle's, K, nsamp, no bench bow / v map / w0, hot seed channels censored) and the same per-plane
fit (wft.reco.fit_plane); for every plane with a fit it keeps the window W (baseline subtracted; on ZS rows the
suppressed samples are 0), noise, ch, pos, the censor mask, the fit (p0, w, t0, q, chi2, dof) and the model M on the window.
There is no undecoded ADC on these rows (ZS: pedestal-subtracted on the FEU), so the analog families run on W.

The point of the SPS sample: det4 sat at a known stage angle (25.4 deg measured), so the ratio of a hits slope to the
waveform fit's w on the same plane measures the ladder compression with no reference and no drift velocity (both are
dx/dt in mm/ns).  Run on lxplus with the code archive (WFT_REPO) and the lock venv, like sps_stage.sh.
"""
import argparse
import csv
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-plateau', type=int, default=3000)
    ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    import sps_reco as SR
    import sps_build_cache as sbc
    import datasets
    from det4_sps_map import VIEW
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    D = datasets.get(row['dataset'])
    nsamp = int(D['n_samples'])
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    raw = bool(int(row['raw']))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.per_plateau - len(evs), half_win=16, tan=tan,
                              only=[row['plateau']], verbose=False).get(row['plateau'], {})
        evs.update(got)
        if len(evs) >= a.per_plateau:
            break
    hot = sorted({c for (r_, s_, L), cs in sbc.HOT_LAST.items() if r_ == row['run'] and L == row['plateau'] for c in cs})
    extra_dead = {p: [c for c in hot if VIEW[c] == p] for p in ('x', 'y')}
    cache = os.path.join(a.work, f'sps_cache_{a.key}.pkl')
    pickle.dump(evs, open(cache, 'wb'), protocol=4)
    v = float(row['v_span_um_ns']) if row['v_span_um_ns'] else None
    k = int(row['K']) if row['K'] else int(np.ceil(28800.0 / (v or 12.0) / 60.0)) + 15
    SR._init(cache, SR.BUNDLE, k, v, nsamp, 'h4', raw, extra_dead)
    os.remove(cache)
    from wft import model as wm, reco as wr
    cal = wm.CAL
    out = []
    for eid in sorted(evs):
        ev = evs[eid]
        for plane in 'xy':
            P = ev.get(plane)
            if P is None:
                continue
            if raw:
                P = dict(P, missing=np.asarray(P['W']) == 0.0)
            cens = ev.get(f'censor_{plane}') or []
            if cens:
                miss = np.asarray(P['missing']) if 'missing' in P else np.zeros(np.shape(P['W']), bool)
                P = dict(P, missing=miss | np.isin(np.asarray(P['ch']), cens)[:, None])
            try:
                f = wr.fit_plane(P, plane, cal)
            except Exception:
                f = None
            if f is None or not np.isfinite(getattr(f, 'w', np.nan)):
                continue
            Wg, nz, ps, st = wm.prep_plane(P, plane)
            ch = np.asarray(P['ch'], int)
            g = np.asarray(wm.GAIN[plane][ch], np.float32)
            q = getattr(f, '_q', None)
            M = None if q is None else (wm.model_waveforms(plane, ps, f.p0, f.w, f.t0, q, cal.hyper) * g[:, None])
            out.append(dict(event_id=int(eid), acq_tag=str(ev.get('file', '')), view=plane,
                            W=np.asarray(P['W'], np.float32), raw=None, ped=None,
                            M=None if M is None else M.astype(np.float32),
                            noise=np.asarray(P['noise'], np.float32), censor=np.asarray(st, bool), gain=g,
                            pos=np.asarray(ps, np.float32), ch=ch, p0=f.p0, w=f.w, t0=f.t0, chi2=f.chi2, dof=f.dof,
                            q=None if q is None else np.asarray(q, np.float32), w_table=None, hits=None,
                            tan_fit=getattr(f, 'tan_theta', np.nan)))
    meta = dict(key=a.key, sps=True, registry_row=row, bundle=SR.BUNDLE, v_drift=float(cal.v_drift), kw=dict(cal.kw),
                w0={'x': 0.0, 'y': 0.0}, hyper={k_: v_ for k_, v_ in cal.hyper.items() if isinstance(v_, (int, float, str))},
                nsamp=int(wm.NSAMP), sns=float(wm.SNS), zs=not raw, stage_deg=25.4 if row['mount'] == '25.64 deg' else
                (15.465 if row['mount'] == '15.465 deg' else 0.0), n=len(out))
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, dir=a.work) as fo:
        pickle.dump(dict(meta=meta, planes=out), fo)
        tmp = fo.name
    if a.out.startswith('root://'):
        subprocess.run(['xrdcp', '-f', tmp, a.out], check=True)
        os.remove(tmp)
    else:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        os.replace(tmp, a.out)
    print(f'[sps-xw] wrote {a.out}: {len(out)} planes from {len(evs)} events (v {cal.v_drift}, K {k}, nsamp {nsamp}, '
          f'{"RAW" if raw else "ZS"})', flush=True)


if __name__ == '__main__':
    main()

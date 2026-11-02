#!/usr/bin/env python3
"""dveto_gate.py --sets DIR --rays rays.parquet --runs DIR --classes sparks_<det>.parquet [--variant on] [--out json]
-- every rescue attempt of one variant against M3, under the rescue gate and its pieces (DISCHARGE_VETO_2026-11-02.md
§3-4).  The .spark_rescue sidecar holds the winner fit of every event the rescue re-seeded and fitted, accepted or
not, with the gate's own columns; so a gate can be judged offline without refitting.  Also: the same gate applied to
the ORDINARY events of the 'off' table (how many real clean tracks it would refuse, how many ordinary fakes it would
let through), which is the reference the rescued events are compared with.

Per attempt: real = winner within 5 mm of M3 (both views fitted), fake = farther, or any winner with no M3 ray.
Classes (discharge / muon+other / shower) are the A1 waveform classes (rerun/satstudy/sparks.py), interior rays only.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO]
R_MM = 5.0


def gates(D, tol, pairing):
    from wft import reco as wr
    g = wr.spark_rescue_gate(D, pairing, tol)
    both = D.x_ok.to_numpy(bool) & D.y_ok.to_numpy(bool)
    q = D.x_quality_ok.to_numpy(bool) & D.y_quality_ok.to_numpy(bool)
    iso = D.x_isochronous.to_numpy(bool) | D.y_isochronous.to_numpy(bool)
    tc = np.abs(g.rg_tc.to_numpy()) <= tol
    z = g.rg_lq_z.to_numpy()
    lq = ~np.isfinite(z) | (np.abs(z) <= wr.SPARK_RESCUE_LQ_NSIG)
    nd = (D.x_n_dropped.to_numpy() <= 2) & (D.y_n_dropped.to_numpy() <= 2)
    return {'none': both, 'full': g.rg_ok.to_numpy(), 'full_nd': g.rg_ok.to_numpy() & nd, 'no_lq': both & q & ~iso & tc, 'no_quality': both & ~iso & tc & lq,
            'no_tc': both & q & ~iso & lq, 'tc_lq_only': both & tc & lq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', required=True)
    ap.add_argument('--rays', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--classes', default=None)
    ap.add_argument('--variant', default='on')
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    from wft.calib import CalibrationBundle
    from wft import reco as wr
    S = pd.read_parquet(os.path.join(a.sets, 'sets.parquet')).drop_duplicates('event_id').set_index('event_id')
    R = pd.read_parquet(a.rays).drop_duplicates('event_id').set_index('event_id')
    K = pd.read_parquet(a.classes).set_index('event_id').cls if a.classes and os.path.exists(a.classes) else pd.Series(dtype=str)
    meta = json.load(open(os.path.join(a.runs, a.variant, 'events.meta.json')))
    cal = CalibrationBundle.load(meta['calibration'])
    tol = wr.tc_tol(cal.xy_pairing)
    out = dict(label=a.label, variant=a.variant, tc_tol=tol, rescue=meta['selection'].get('spark_rescue'))

    def judge(D, tag):
        D = D.copy()
        D['set'] = S.set.reindex(D.event_id).to_numpy()
        m3 = D.set.str.endswith('_m3').to_numpy()
        mm = R.reindex(D.event_id)
        r = np.hypot(D.x_p0.to_numpy() - mm.mesh_x.to_numpy(), D.y_p0.to_numpy() - mm.mesh_y.to_numpy())
        near = m3 & (np.nan_to_num(r, nan=1e9) <= R_MM)
        inter = (mm.region == 'interior').to_numpy()
        cls = K.reindex(D.event_id).fillna('-').to_numpy()
        G = gates(D, tol, cal.xy_pairing)
        res = {}
        for gn, ok in G.items():
            e = dict(real=int((ok & near).sum()), fake_m3=int((ok & m3 & ~near).sum()), fake_nom3=int((ok & ~m3).sum()),
                     real_interior=int((ok & near & inter).sum()), fake_m3_interior=int((ok & m3 & ~near & inter).sum()))
            for c in ('discharge', 'muon+other', 'shower'):
                e[f'real_{c}'] = int((ok & near & (cls == c)).sum())
                e[f'fake_{c}'] = int((ok & m3 & ~near & (cls == c)).sum())
            res[gn] = e
        res['_n'] = dict(rows=int(len(D)), m3=int(m3.sum()), nom3=int((~m3).sum()), interior=int((m3 & inter).sum()))
        out[tag] = res
        return D, G, near, m3

    side = os.path.join(a.runs, a.variant, 'events.spark_rescue.parquet')
    A = pd.read_parquet(side) if os.path.exists(side) else None
    if A is not None and len(A):
        D, G, near, m3 = judge(A, 'rescue_attempts')
        rx = D.x_p0.to_numpy() - R.reindex(D.event_id).mesh_x.to_numpy()
        ry = D.y_p0.to_numpy() - R.reindex(D.event_id).mesh_y.to_numpy()
        dtx = D.x_tan_theta.to_numpy() - R.reindex(D.event_id).tan_x.to_numpy()
        dty = D.y_tan_theta.to_numpy() - R.reindex(D.event_id).tan_y.to_numpy()
        ok = G['full'] & near
        pct = lambda x: float(np.percentile(np.abs(x[np.isfinite(x)]), 68.27)) if np.isfinite(x).any() else np.nan
        out['rescued_quality'] = dict(n=int(ok.sum()), res_s68_x=pct(rx[ok]), res_s68_y=pct(ry[ok]),
                                      dtan_s68_x=pct(dtx[ok]), dtan_s68_y=pct(dty[ok]))
    # the ordinary events of the 'off' table under the same gate (reference)
    offp = os.path.join(a.runs, 'off', 'events.parquet')
    if os.path.exists(offp):
        O = pd.read_parquet(offp)
        O = O[O.event_id.isin(S.index[S.set.str.startswith('ord')])]
        if len(O):
            D, G, near, m3 = judge(O, 'ordinary_off')
            rx = D.x_p0.to_numpy() - R.reindex(D.event_id).mesh_x.to_numpy()
            ry = D.y_p0.to_numpy() - R.reindex(D.event_id).mesh_y.to_numpy()
            dtx = D.x_tan_theta.to_numpy() - R.reindex(D.event_id).tan_x.to_numpy()
            dty = D.y_tan_theta.to_numpy() - R.reindex(D.event_id).tan_y.to_numpy()
            ok = G['none'] & near
            pct = lambda x: float(np.percentile(np.abs(x[np.isfinite(x)]), 68.27)) if np.isfinite(x).any() else np.nan
            out['ordinary_quality'] = dict(n=int(ok.sum()), res_s68_x=pct(rx[ok]), res_s68_y=pct(ry[ok]),
                                           dtan_s68_x=pct(dtx[ok]), dtan_s68_y=pct(dty[ok]))
            out['ordinary_n'] = dict(ord_m3=int((S.set == 'ord_m3').sum()), ord_nom3=int((S.set == 'ord_nom3').sum()))
    out['n_sets'] = S.set.value_counts().to_dict()
    txt = json.dumps(out, indent=1, default=float)
    if a.out:
        open(a.out, 'w').write(txt)
    print(f'== {a.label} {a.variant}  tc_tol {tol:.0f} ns  sets {out["n_sets"]}')
    for tag in ('rescue_attempts', 'ordinary_off'):
        if tag in out:
            print(f'  {tag} {out[tag]["_n"]}')
            for gn in ('none', 'full', 'full_nd', 'no_lq', 'no_quality', 'no_tc', 'tc_lq_only'):
                e = out[tag][gn]
                print(f'    {gn:11s} real {e["real"]:5d} (int {e["real_interior"]:5d})  fake_m3 {e["fake_m3"]:4d} (int {e["fake_m3_interior"]:4d})'
                      f'  fake_nom3 {e["fake_nom3"]:4d}   dis {e["real_discharge"]}/{e["fake_discharge"]}'
                      f'  m+o {e["real_muon+other"]}/{e["fake_muon+other"]}  sh {e["real_shower"]}/{e["fake_shower"]}')
    for k in ('rescued_quality', 'ordinary_quality'):
        if k in out:
            print(' ', k, {x: round(y, 4) if isinstance(y, float) else y for x, y in out[k].items()})


if __name__ == '__main__':
    main()

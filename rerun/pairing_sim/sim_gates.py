#!/usr/bin/env python3
"""sim_gates.py -- S1 of PLAN.md against the cosmic gates (§4): the simulated single muons (MX17_Geant v4c, det3
conditions, 12 gun angles) reconstructed by production wft, scored exactly as gates_data.py scores the data.

    ../../.venv/bin/python sim_gates.py            -> ~/x17/cosmic_bench/satstudy/pairing_sim/gates_sim.json

Inputs: s1/events_v4c_<point>_<tag>.parquet (condor s1_reco.sub: tag dphys = det3 golden-row phys bundle with the t0
prior dropped; simcal = the sim-calibrated bundle of RECO_TRUTH 2026-10-18), and the Stage B truth sidecars
(<SIM>/v4c_<point>/*truth.parquet: per channel the induced charge before the electronics).
Per point: n, median / robust sigma of lq, u50, u90 and lnQ, the same at the truth level (sum of induced charge
per view), and the 1/sqrt(N) scaling of the truth split (robust sigma x sqrt(n_seed)). Then the gate table: sim
point vs the det3 data at the same view inclination (gates_data vs_view_angle; th00 vs both views < 2.5 deg), and
the pooled toy (pairing right vs |ln Q1/Q2|) from the near-vertical points (0-15 deg) as pairing.py builds it.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_data as G  # noqa: E402

OUT = G.OUT
S1 = os.path.join(OUT, 's1')
SIM = os.environ.get('PAIRSIM_SIM', '/media/dylan/data/claude_tmp/v3wet')
POINTS = ['th00', 'thx05', 'thx10', 'thx15', 'thx20', 'thx30', 'thy05', 'thy10', 'thy15', 'thy20', 'thy30', 'txy10']
TOY_POINTS = ['th00', 'thx05', 'thx10', 'thx15', 'thy05', 'thy10', 'thy15', 'txy10']


def singles(point, tag):
    f = os.path.join(S1, f'events_v4c_{point}_{tag}.parquet')
    if not os.path.exists(f):
        return None
    E = pd.read_parquet(f)
    n_all = len(E)
    E = E[E.x_ok.astype(bool) & E.y_ok.astype(bool) & (E.x_n_candidates == 1) & (E.y_n_candidates == 1)
          & (E.x_q_sum > 0) & (E.y_q_sum > 0)]
    w = pd.DataFrame({(c, p): E[f'{p}_{c}'].to_numpy() for c in ('p0', 'tan_theta', 'q_sum', 'q_u50', 'q_u90', 't0')
                      for p in 'xy'}, index=E.event_id.to_numpy())
    w.columns = pd.MultiIndex.from_tuples(w.columns)
    return w, n_all


def truth(point, base=None):
    f = glob.glob(os.path.join(base or os.path.join(SIM, f'v4c_{point}'), '*truth.parquet'))
    if not f:
        return None
    T = pd.read_parquet(f[0])
    qx = np.array([np.sum(q[np.asarray(v) == 'X']) for v, q in zip(T.chan_view, T.chan_q_electrons)])
    qy = np.array([np.sum(q[np.asarray(v) == 'Y']) for v, q in zip(T.chan_view, T.chan_q_electrons)])
    lq = np.log(qx / qy); n = T.n_seed.to_numpy()
    return dict(lq_med=float(np.median(lq)), lq_rsig=G.rsig(lq), lnQ_rsig=G.rsig(0.5 * np.log(qx * qy)),
                n_seed_med=float(np.median(n)), split_per_avalanche=G.rsig(lq) * float(np.sqrt(np.median(n))),
                scaling=[dict(lo=lo, hi=hi, n=int(((n >= lo) & (n < hi)).sum()),
                              rsig=G.rsig(lq[(n >= lo) & (n < hi)]),
                              rsig_sqrt_n=G.rsig(lq[(n >= lo) & (n < hi)]) * float(np.sqrt(np.median(n[(n >= lo) & (n < hi)]))))
                         for lo, hi in ((0, 150), (150, 200), (200, 260), (260, 400), (400, 10000))])


def data_ref(gd, point):
    """det3 data at the sim point's inclination: (lq_rsig, lq_med, u50_rsig, lnQ_rsig, n)."""
    if point == 'th00':
        r = gd['vs_view_angle']['x'][0]
    elif point.startswith('thx') or point.startswith('thy'):
        deg = int(point[3:]); rows = gd['vs_view_angle'][point[2]]
        r = next((x for x in rows if x['deg'] == deg), None)
    else:
        return None
    return r if r and r['lq_rsig'] is not None else None


def main():
    gd = json.load(open(os.path.join(OUT, 'gates_data.json')))
    res = dict(points={}, data_det3=dict(lq=gd['det3']['lq'], u50=gd['det3']['u50'], u90=gd['det3']['u90'],
                                          lnQ=gd['det3']['lnQ'], toy=gd['det3']['toy']))
    for pt in POINTS:
        o = dict(truth=truth(pt))
        for tag in ('dphys', 'simcal'):
            s = singles(pt, tag)
            if s is None:
                continue
            w, n_all = s
            g = G.gate(w); g['n_all'] = n_all; g['single_frac'] = len(w) / max(n_all, 1)
            o[tag] = g
        o['data'] = data_ref(gd['det3'], pt)
        res['points'][pt] = o
        d = o.get('dphys')
        if d:
            ref = o['data']
            print(f"{pt:6s} n {d['n']:4d}  lq {d['lq']['median']:+.3f} sig {d['lq']['rsig']:.3f}"
                  f"  u50 {d['u50']['rsig']:4.0f}  lnQ {d['lnQ']['rsig']:.3f}  truth sig {o['truth']['lq_rsig']:.3f}"
                  + (f"   | data sig {ref['lq_rsig']:.3f} u50 {ref['u50_rsig']:.0f} lnQ {ref['lnQ_rsig']:.3f} (n {ref['n']})"
                     if ref else ''), flush=True)
    for tag in ('dphys', 'simcal'):
        ws = [singles(p, tag) for p in TOY_POINTS]
        ws = [s[0].set_axis(s[0].index + 10000 * k) for k, s in enumerate(ws) if s is not None]
        if not ws:
            continue
        w = pd.concat(ws)
        res[f'pooled_{tag}'] = G.gate(w) | dict(toy=G.toy(w), points=TOY_POINTS)
        t = res[f'pooled_{tag}']
        print(f"pooled {tag}: n {t['n']} lq sig {t['lq']['rsig']:.3f} u50 {t['u50']['rsig']:.0f} lnQ {t['lnQ']['rsig']:.3f}"
              f"  toy q {t['toy']['right']['q']:.3f} q+u {t['toy']['right']['q_u']:.3f}  "
              f"frac |lnQ1/Q2|<0.1 {t['toy']['frac_ratio_lt_0p1']:.3f}")
        print('   by ratio q', [(b['lo'], b['n'], b['right'] and round(b['right'], 3)) for b in t['toy']['by_ratio']['q']])
    print('data det3 toy by ratio q', [(b['lo'], b['n'], b['right'] and round(b['right'], 3))
                                      for b in gd['det3']['toy']['by_ratio']['q']])
    # the data's Landau width with its own gain map taken out (25 mm cells): the sim muons cross one 31 mm spot
    key, w, R, _ = G.singles('det3')
    lnQ = 0.5 * np.log(w.q_sum.x * w.q_sum.y).to_numpy()
    c = (w.p0.x.to_numpy() // 25).astype(int) * 1000 + (w.p0.y.to_numpy() // 25).astype(int)
    d = pd.DataFrame(dict(q=lnQ, c=c)); m = d.groupby('c').q.transform('median'); n = d.groupby('c').q.transform('size')
    res['data_lnQ_local'] = G.rsig((d.q - m)[n >= 15])
    json.dump(res, open(os.path.join(OUT, 'gates_sim.json'), 'w'), indent=1)
    # the x/y split diagnostic (MX17_Geant scratchpad/2026-10-26/pairing_sim/stageBC_xysplit.sub), if pulled
    xs = {}
    for dd in sorted(glob.glob(os.path.join(OUT, 'xysplit', 'xys*_th00'))):
        t = truth(None, dd)
        if t:
            xs[f"H = {os.path.basename(dd)[3:-5]} um"] = t
            print(os.path.basename(dd), f"truth sigma {t['lq_rsig']:.4f}  x sqrt(N) {t['split_per_avalanche']:.2f}")
    if xs:
        json.dump(xs, open(os.path.join(OUT, 'xysplit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

"""q7_window: the flattening of chamber A's gap map below 700 V (audit item 7) is the end of the recorded waveform.

R4 tested the window with a cap on q_u90 itself and with q_u50, and concluded the window was not the cause. Both tests
miss how a cut acts. q_u90 is a quantile of the VISIBLE charge. When a drift column is longer than the record, the
visible charge fills the record from the track start t0 to the record end, whatever the gap is. So every quantile
(q_u50 too) then measures 2019 ns - t0, not the gap, and it sits well before the record end.

Tests:
1. Per drift voltage: the angle-scale velocity v (bundle, kw measured against the telescope), the expected column
   T = 28.2 mm / v, q_u90 / T, and the slope of q_u90 (q_u50, q_uend) against t0. A column cut by the record end gives
   d q_u90 / d t0 = -0.8 (q_u90 = 0.80 x (2019 - t0)); an uncut column gives 0.
2. Emulation: stretch the golden run-1 columns (and the 700 V columns) to the velocity of 500 V and 300 V, cap
   q_u90 at 0.80 x (2019 - t0), refit the quadratic and the golden-ridge amplitude exactly as R4 did.

Usage: ../../../.venv/bin/python q7_window.py   (writes $WORK/audit/q16/q7_window.json)
"""
import json, pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import r4_lib as L

REC_END = 2019.0      # last recorded time (ns): 32 samples x 60 ns, t0 + q_uend never exceeds it
RAIL = REC_END - 60   # column end in the last sample
R = 0.80              # q_u90 / column duration, the same at 700, 900 and 1100 V (0.800-0.804)
V_LIST = [300, 500, 700, 900, 1100]

T = pickle.load(open(f'{L.OUT}/template.pkl', 'rb'))      # R4's golden-ridge template (from golden run 2)
FINE = RegularGridInterpolator(T['GRID'], T['f2'], bounds_error=False, fill_value=None)


def bundle_v(rel):
    return json.load(open(f'{L.EOS}/{rel}/bundle.json'))['v_drift']


def slope(x, y, nboot=300, seed=1):
    rng = np.random.default_rng(seed)
    b = np.polyfit(x, y, 1)[0]
    B = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(nboot))]
    return float(b), float(np.std(B))


def ridge_amp(d, nb=100, seed=4):
    """R4's golden-ridge amplitude: quadratic + covariates free, plus the fine template times an amplitude."""
    y = d.lq.values
    t = FINE(np.vstack([d.U.values, d.W.values]).T)
    A = np.hstack([L.design(d, ('walk', 'tan')), t[:, None]])
    c, _ = L.rfit(A, y)
    rng = np.random.default_rng(seed)
    B = [L.rfit(A, y, np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float))[0][-1]
         for _ in range(nb)]
    return float(c[-1]), float(np.std(B))


def shape(d, nb=200):
    f = L.fit_shape(d, nboot=nb)
    return {k: [float(f['c'][i]), float(f['err'][i])] for i, k in ((3, 'guu'), (4, 'gww'), (5, 'guw'))}


def emulate(t, s):
    e = t.copy()
    for p in 'xy':
        e[f'{p}_q_u90'] = np.minimum(s * e[f'{p}_q_u90'], R * (REC_END - e[f'{p}_t0']))
    e['q'] = 0.5 * (e.x_q_u90 + e.y_q_u90)
    e['lq'] = np.log(e.q)
    return e


def main():
    out = dict(rows={}, emulation={})
    print('V     v     T     q90/T  rail   dq90/dt0 x / y          dq50/dt0 x / y          gww            guu            ridge')
    for V in V_LIST:
        rel = f'{L.SC}/drift_scan_resist_490V_drift_{V}V'
        d = L.load_events(rel)
        v = bundle_v(rel); Tc = L.G0 * 1e3 / v
        end = np.maximum(d.x_t0 + d.x_q_uend, d.y_t0 + d.y_q_uend)
        r = dict(v=v, T=Tc, N=len(d), q90_over_T=float(d.q.median() / Tc), rail=float((end >= RAIL).mean()),
                 **{f'd{c}_dt0_{p}': slope(d[f'{p}_t0'].values, d[f'{p}_{c}'].values)
                    for p in 'xy' for c in ('q_u90', 'q_u50', 'q_uend')},
                 **shape(d), ridge=ridge_amp(d))
        out['rows'][V] = r
        print(f'{V:4d} {v:5.1f} {Tc:5.0f}  {r["q90_over_T"]:.3f}  {r["rail"]:.2f}  '
              f'{r["dq_u90_dt0_x"][0]:+.2f}+-{r["dq_u90_dt0_x"][1]:.2f} / {r["dq_u90_dt0_y"][0]:+.2f}+-{r["dq_u90_dt0_y"][1]:.2f}  '
              f'{r["dq_u50_dt0_x"][0]:+.2f}+-{r["dq_u50_dt0_x"][1]:.2f} / {r["dq_u50_dt0_y"][0]:+.2f}+-{r["dq_u50_dt0_y"][1]:.2f}  '
              f'{r["gww"][0]:+.3f}+-{r["gww"][1]:.3f}  {r["guu"][0]:+.3f}+-{r["guu"][1]:.3f}  '
              f'{r["ridge"][0]:+.2f}+-{r["ridge"][1]:.2f}')
    print('\nemulation: fixed gap shape, stretched to the velocity of V, q_u90 capped at 0.80 x (2019 ns - t0)')
    for tname, trel in (('golden run 1', L.GOLD1), ('700 V', f'{L.SC}/drift_scan_resist_490V_drift_700V')):
        t = L.load_events(trel); vt = bundle_v(trel)
        base = dict(**shape(t, nb=50), ridge=ridge_amp(t, nb=30))
        out['emulation'][tname] = dict(template=base)
        print(f'  template {tname}: gww {base["gww"][0]:+.3f}  guu {base["guu"][0]:+.3f}  ridge {base["ridge"][0]:+.2f}')
        for V in (500, 300):
            s = vt / bundle_v(f'{L.SC}/drift_scan_resist_490V_drift_{V}V')
            e = emulate(t, s)
            capped = float(np.mean(np.maximum(s * t.x_q_u90 / (R * (REC_END - t.x_t0)),
                                              s * t.y_q_u90 / (R * (REC_END - t.y_t0))) > 1))
            r = dict(s=s, capped=capped, med_q90=float(e.q.median()), **shape(e, nb=50), ridge=ridge_amp(e, nb=30),
                     dq90_dt0_x=slope(e.x_t0.values, e.x_q_u90.values, nboot=100))
            out['emulation'][tname][V] = r
            print(f'    -> {V} V: stretch {s:.2f}, capped {capped:.2f}, median q90 {r["med_q90"]:.0f}, '
                  f'gww {r["gww"][0]:+.3f}, guu {r["guu"][0]:+.3f}, ridge {r["ridge"][0]:+.2f}+-{r["ridge"][1]:.2f}, '
                  f'dq90x/dt0 {r["dq90_dt0_x"][0]:+.2f}')
    json.dump(out, open(f'{L.WORK}/audit/q16/q7_window.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()

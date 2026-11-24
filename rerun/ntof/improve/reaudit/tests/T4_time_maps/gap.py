#!/usr/bin/env python3
"""T4 part 4: implied gap.  g_mean = v_angle * T_box = kappa * v_t * (30 mm / v_t) = 30 kappa (per acquisition the box T
is 30 mm / v_t by construction of v_t, so the 30 cancels only through kappa = v_angle / v_t from the A-C cosmic line).
Surface two ways: (T) g = g_mean (T/T_med)^(1/(1+beta)), beta = dln v/dln E from Magboltz 90/10 + water at the measured v
(E ~ 240 V/cm); (vT) g = g_mean (1+s)/(1+s_med) * T/T_med per angle-map cell (the bench product, beta cancels)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tmaps as TM
Z = dict(np.load(HERE / 'tmaps.npz')); SB = dict(np.load(HERE / 's_boot.npz'))
import pandas as pd
vt = pd.read_parquet(HERE / 'cache.parquet', columns=['arm', 'v_t'])
GARF = json.load(open('/home/dylan/PycharmProjects/nTof_x17/garfield_sim/results/drift_9010_contam_cern.json'))['mixtures']
pts = []
for k in ('Ar90_iso10', 'Ar_iso10_H2O0.3', 'Ar_iso10_H2O0.5', 'Ar_iso10_H2O1.0'):
    E = np.array([r['E_Vcm'] for r in GARF[k]]); v = np.array([r['v_um_per_ns'] for r in GARF[k]])
    b = np.gradient(np.log(v), np.log(E))
    pts.append((np.interp(240, E, v), np.interp(240, E, b)))
pts = np.array(sorted(pts))
out = {}
for arm in 'ACDB':
    r = {}
    vmed = float(np.nanmedian(vt.v_t[vt.arm == arm])) if arm != 'B' else np.nan
    kap = TM.KAPPA.get(arm, {'x': np.nan, 'y': np.nan})
    k = 0.5 * (kap['x'] + kap['y'])
    vang = k * vmed
    beta = float(np.interp(vang, pts[:, 0], pts[:, 1])) if np.isfinite(vang) else 0.7
    r.update(v_t_median=vmed, kappa=k, v_angle=vang, beta=beta, beta_note='Magboltz 90/10+H2O at 240 V/cm' if arm != 'B' else 'assumed 0.7 (no v for B)',
             expected_dlnv_dlnT=-beta / (1 + beta), g_mean_mm=30 * k if arm != 'B' else np.nan,
             T_box_ns=30e3 / vmed if arm != 'B' else np.nan)
    m = np.log(Z[f'{arm}_xy_T']); n = Z[f'{arm}_xy_n']
    c = 0.5 * (TM.E25[1:] + TM.E25[:-1])
    inner = (np.abs(c)[:, None] < 150) & (np.abs(c)[None, :] < 150) & (n >= 100) & np.isfinite(m)
    gref = r['g_mean_mm'] if arm != 'B' else 30.0
    g = gref * np.exp((m - np.median(m[inner])) / (1 + beta))
    r['gT_interior'] = dict(min=float(g[inner].min()), p2=float(np.percentile(g[inner], 2)), p98=float(np.percentile(g[inner], 98)),
                            max=float(g[inner].max()), rms=float(g[inner].std()))
    q = TM.quad_fit(np.log(g), Z[f'{arm}_xy_err'] / (1 + beta), TM.E25)
    r['gT_shape_mm'] = dict(tilt_x_per_100mm=gref * q['tilt_x_pct_per_100mm'] / 100, tilt_y_per_100mm=gref * q['tilt_y_pct_per_100mm'] / 100,
                            quad_x_at_150=gref * q['quad_x_pct_at_150'] / 100, quad_y_at_150=gref * q['quad_y_pct_at_150'] / 100,
                            errs=[gref * q[e] / 100 for e in ('tilt_x_err', 'tilt_y_err', 'quad_x_err', 'quad_y_err')])
    if arm in 'ACD':
        tag, nmin = ('50', 80) if arm in 'AC' else ('100', 60)
        ed = TM.E50 if arm in 'AC' else TM.E100
        cc = 0.5 * (ed[1:] + ed[:-1])
        for v in 'xy':
            s, ns = SB[f'{arm}_{v}_s'], SB[f'{arm}_{v}_n']
            mt = np.log(Z[f'{arm}_xy_T{tag}'])
            ok = (ns >= nmin) & np.isfinite(mt)
            if arm in 'AC':
                ok &= (np.abs(cc)[:, None] < 150) & (np.abs(cc)[None, :] < 150)
            gv = 30 * kap[v] * (1 + s) / (1 + np.median(s[ok])) * np.exp(mt - np.median(mt[ok]))
            r[f'gvT_{v}'] = dict(n_cells=int(ok.sum()), min=float(gv[ok].min()), max=float(gv[ok].max()), rms=float(gv[ok].std()))
    out[arm] = r
    print(arm, json.dumps(r, default=float))
json.dump(out, open(HERE / 'results_gap.json', 'w'), indent=1, default=float)

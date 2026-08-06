#!/usr/bin/env python3
"""
gap_study.py — the drift-gap puzzle under the RC-ladder reconstruction.

Measures where the charge column actually ends, from full NNLS charge-arrival
profiles of geometrically CONTAINED tracks (the M3 reference says the track
crosses the whole gap inside the active area, so its column must span the
full drift depth).

Outputs, per plane:
  * stacked normalised charge-arrival profile q(u) for contained tracks
  * endpoint fit, two hypotheses:
      - sharp edge:   plateau * 0.5*erfc((u - T_end)/(sqrt(2) sig_e))
      - attachment:   plateau * exp(-u/tau_att) * (same edge)
  * per-event u_end distribution, contained vs edge-clipping tracks
  * T_end vs |tan_ref| (must be flat) and vs transverse position (topography)

    ../../.venv/bin/python mx_june_wft/bench/gap_study.py sat_det3 \
        --bundle <lp bundle> [--limit 4000] [--jobs 8]

Environment: set WFT_MODEL_FRAC / WFT_PRESCAN as in production.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

GAP_NOMINAL_MM = 30.0

_CAL = None


def _init(bundle):
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle)
    wm.use_calibration(_CAL)


def _fit_one(payload):
    from wft import model as wm
    from wft import reco as wr
    eid, wins, truth = payload
    out = {'eid': eid}
    for plane in ('x', 'y'):
        P = wins.get(plane)
        if P is None:
            continue
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        try:
            p0s, _w, t0s = wm.init_guess(P, plane)
            p0s, w0, t0s = wr._global_start(P, plane, p0s, t0s, wm.HYPER)
            r = wm.fit_plane_raw(P, plane, p0s, w0, t0s)
        except Exception:
            continue
        if r is None or not np.isfinite(r['chi2']):
            continue
        q = np.asarray(r['q'], float)
        out[plane] = dict(q=q, w=float(r['w']), t0=float(r['t0']),
                          p0=float(r['p0']),
                          chi2dof=float(r['chi2'] / max(r['dof'], 1)))
    return out


def endpoint_fits(u, prof, prof_err):
    """Fit the stacked profile with sharp-edge and attachment models."""
    from scipy.optimize import curve_fit
    from scipy.special import erfc

    def sharp(u, A, T, sig):
        return A * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))

    def attach(u, A, T, sig, tau):
        return A * np.exp(-u / tau) * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))

    sel = u < 1050
    res = {}
    try:
        p, c = curve_fit(sharp, u[sel], prof[sel], p0=[prof[:5].mean(), 700, 60],
                         sigma=prof_err[sel], absolute_sigma=True, maxfev=20000)
        chi = float((((sharp(u[sel], *p) - prof[sel]) / prof_err[sel]) ** 2).sum())
        res['sharp'] = dict(A=p[0], T_end=p[1], sig_e=p[2], chi2=chi,
                            T_err=float(np.sqrt(c[1, 1])))
    except Exception as e:
        res['sharp'] = dict(error=str(e))
    try:
        p, c = curve_fit(attach, u[sel], prof[sel],
                         p0=[prof[:5].mean(), 700, 60, 5000],
                         sigma=prof_err[sel], absolute_sigma=True,
                         bounds=([0, 300, 10, 300], [np.inf, 1100, 300, 1e6]),
                         maxfev=20000)
        chi = float((((attach(u[sel], *p) - prof[sel]) / prof_err[sel]) ** 2).sum())
        res['attach'] = dict(A=p[0], T_end=p[1], sig_e=p[2], tau_att=p[3],
                             chi2=chi, T_err=float(np.sqrt(c[1, 1])))
    except Exception as e:
        res['attach'] = dict(error=str(e))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--v-geom', type=float, default=None,
                    help='geometric drift speed for the mm conversion '
                         '(default: bundle kw * v_drift, per plane)')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    cal = CalibrationBundle.load(args.bundle)
    with open(os.path.join(W, 'bench_cache.pkl'), 'rb') as f:
        data = pickle.load(f)
    events, meta = data['events'], data['meta']
    box = meta['box']

    # ---- geometric containment from the reference: the track's transverse
    # excursion over the full gap must stay inside the active box
    def classify(t):
        if not all(np.isfinite([t['ref_x'], t['ref_y'],
                                t['tan_x'], t['tan_y']])):
            return None
        margin = 3.0
        cx = (box['x'][0] + margin + 15.5 * abs(t['tan_x']) <= t['ref_x']
              <= box['x'][1] - margin - 15.5 * abs(t['tan_x']))
        cy = (box['y'][0] + margin + 15.5 * abs(t['tan_y']) <= t['ref_y']
              <= box['y'][1] - margin - 15.5 * abs(t['tan_y']))
        return bool(cx and cy)

    payloads, klass = [], {}
    for eid, ev in sorted(events.items()):
        c = classify(ev['truth'])
        if c is None:
            continue
        wins = {}
        for plane in ('x', 'y'):
            cand = ev['wins'].get(plane)
            s = ev['seeds'].get(plane)
            if cand and s and s[0]['n_dropped'] == 0 and len(cand) == 1:
                wins[plane] = cand[0]
        if not wins:
            continue
        klass[eid] = (c, ev['truth'])
        payloads.append((eid, wins, ev['truth']))
    if args.limit:
        payloads = payloads[:args.limit]
    n_cont = sum(1 for e, _, _ in payloads if klass[e][0])
    print(f'{len(payloads):,} events to fit ({n_cont:,} contained, '
          f'{len(payloads) - n_cont:,} edge-clipping)')

    t0w = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(args.bundle,)) as pool:
        for i, out in enumerate(pool.map(_fit_one, payloads, chunksize=8)):
            rows.append(out)
            if (i + 1) % 1000 == 0:
                print(f'  {i + 1:,}/{len(payloads):,} ({time.time() - t0w:.0f} s)',
                      flush=True)
    print(f'fits done in {time.time() - t0w:.0f} s')

    from wft import model as wm
    wm.use_calibration(cal)
    u = wm.UK.copy()                     # bin centres [ns]
    summary = {'run_key': args.run_key, 'bundle': args.bundle,
               'v_drift': cal.v_drift, 'kw': dict(cal.kw), 'planes': {}}
    out_npz = {}
    for plane in ('x', 'y'):
        v_geom = args.v_geom or cal.kw.get(plane, 1.0) * cal.v_drift
        prof_c, prof_e, uend_c, uend_e, tanr, tend_ev, posr = [], [], [], [], [], [], []
        for out in rows:
            d = out.get(plane)
            if d is None or d['chi2dof'] > 250:
                continue
            q = d['q']
            tot = q.sum()
            if tot <= 0:
                continue
            qn = q / tot
            contained, truth = klass[out['eid']]
            live = np.where(q > 0.05 * q.max())[0]
            ue = u[live[-1]] + 30.0 if len(live) else np.nan
            if contained:
                prof_c.append(qn)
                uend_c.append(ue)
                tanr.append(truth[f'tan_{plane}'])
                tend_ev.append(ue)
                posr.append((truth['ref_x'], truth['ref_y']))
            else:
                prof_e.append(qn)
                uend_e.append(ue)
        prof_c, prof_e = np.array(prof_c), np.array(prof_e)
        mc = prof_c.mean(axis=0)
        ec = prof_c.std(axis=0) / np.sqrt(len(prof_c))
        fits = endpoint_fits(u, mc, np.maximum(ec, 1e-5))
        pl = dict(n_contained=len(prof_c), n_edge=len(prof_e),
                  v_geom=v_geom,
                  uend_med_contained=float(np.median(uend_c)),
                  uend_med_edge=float(np.median(uend_e)) if len(uend_e) else np.nan,
                  profile=mc.tolist(), profile_err=ec.tolist(),
                  u=u.tolist(), fits=fits)
        for k, f in fits.items():
            if 'T_end' in f:
                f['gap_mm'] = f['T_end'] * v_geom / 1000.0
        summary['planes'][plane] = pl
        out_npz[f'prof_{plane}'] = mc
        out_npz[f'err_{plane}'] = ec
        # T_end stability vs |tan| and position (median per-event uend)
        tanr = np.array(tanr); tend_ev = np.array(tend_ev)
        posr = np.array(posr) if len(posr) else np.zeros((0, 2))
        bins = [(0, 0.08), (0.08, 0.16), (0.16, 0.28), (0.28, 0.5)]
        pl['uend_vs_tan'] = [
            float(np.median(tend_ev[(np.abs(tanr) >= lo) & (np.abs(tanr) < hi)]))
            if ((np.abs(tanr) >= lo) & (np.abs(tanr) < hi)).sum() > 30 else None
            for lo, hi in bins]
        # coarse 3x3 topography of median per-event uend
        topo = []
        if len(posr):
            xe = np.percentile(posr[:, 0], [0, 33, 66, 100])
            ye = np.percentile(posr[:, 1], [0, 33, 66, 100])
            for i in range(3):
                row = []
                for j in range(3):
                    s = ((posr[:, 0] >= xe[i]) & (posr[:, 0] < xe[i + 1])
                         & (posr[:, 1] >= ye[j]) & (posr[:, 1] < ye[j + 1]))
                    row.append(float(np.median(tend_ev[s])) if s.sum() > 30
                               else None)
                topo.append(row)
        pl['uend_topo_3x3'] = topo

        s, a = fits.get('sharp', {}), fits.get('attach', {})
        print(f"\n== {plane} (n_cont={len(prof_c)}, v_geom={v_geom:.2f})")
        if 'T_end' in s:
            print(f"  sharp:  T_end {s['T_end']:.0f}+-{s['T_err']:.0f} ns  "
                  f"sig_e {s['sig_e']:.0f}  chi2 {s['chi2']:.1f}  "
                  f"-> gap {s['gap_mm']:.2f} mm")
        if 'T_end' in a:
            print(f"  attach: T_end {a['T_end']:.0f}+-{a['T_err']:.0f} ns  "
                  f"sig_e {a['sig_e']:.0f}  tau_att {a['tau_att']:.0f} ns  "
                  f"chi2 {a['chi2']:.1f}  -> gap {a['gap_mm']:.2f} mm")
        print(f"  per-event uend med: contained {pl['uend_med_contained']:.0f}"
              f"  edge {pl['uend_med_edge']:.0f} ns")
        print(f"  uend vs |tan| bins: {pl['uend_vs_tan']}")

    out_dir = os.path.join(W, 'gap_study')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'gap_study.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez(os.path.join(out_dir, 'profiles.npz'), u=u, **out_npz)

    # per-event dump for offline position-binned endpoint fits
    import pandas as pd
    recs = []
    for out in rows:
        contained, truth = klass[out['eid']]
        for plane in ('x', 'y'):
            d = out.get(plane)
            if d is None:
                continue
            q = d['q']
            recs.append(dict(eid=out['eid'], plane=plane,
                             contained=contained,
                             ref_x=truth['ref_x'], ref_y=truth['ref_y'],
                             tan=truth[f'tan_{plane}'],
                             chi2dof=d['chi2dof'], qsum=float(q.sum()),
                             **{f'q{i}': float(v) for i, v in enumerate(q)}))
    pd.DataFrame(recs).to_parquet(os.path.join(out_dir, 'event_profiles.parquet'),
                                  index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.special import erfc
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for k, plane in enumerate(('x', 'y')):
        pl = summary['planes'][plane]
        ax = axs[k]
        ax.errorbar(u, pl['profile'], yerr=pl['profile_err'], fmt='o', ms=3,
                    label='stacked profile (contained)')
        for nm, style in (('sharp', 'r--'), ('attach', 'g-')):
            f = pl['fits'].get(nm, {})
            if 'T_end' in f:
                uu = np.linspace(0, 1080, 300)
                mdl = f['A'] * 0.5 * erfc((uu - f['T_end']) /
                                          (np.sqrt(2) * f['sig_e']))
                if nm == 'attach':
                    mdl *= np.exp(-uu / f['tau_att'])
                ax.plot(uu, mdl, style, lw=1.2,
                        label=f"{nm}: T={f['T_end']:.0f} ns -> "
                              f"{f['gap_mm']:.1f} mm")
        vg = pl['v_geom']
        ax.axvline(GAP_NOMINAL_MM / vg * 1000, color='k', ls=':', lw=1,
                   label=f'30 mm at v={vg:.1f}')
        ax.set_xlabel('charge arrival time after t0 [ns]')
        ax.set_title(f'{plane}: n={pl["n_contained"]}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'gap_profiles.png'), dpi=110)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()

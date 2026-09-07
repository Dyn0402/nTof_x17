#!/usr/bin/env python3
"""
07_t0_prior_gate.py — the T1.1 gate: is a per-event absolute-t0 prior real?

The cosmic bench triggers on a scintillator-pair coincidence (sigma ~5 ns for
the pair, measured end to end — PLAN_42), and `ftst` records the trigger's
phase against the free-running DREAM clock in 10 ns steps. The forward fit's
t0 (arrival of the mesh charge) is therefore predictable per event up to that
jitter: t0_pred(ftst) = const - 10 ns * ftst. Today t0 is fitted freely and
the prediction is unused (HANDOFF_FABLE_2026-08-11 T1.1, F26).

This script measures whether the prior is real and how tight it can be:

1. join the production reconstruction's fitted t0 with each plane's own ftst
   (read from decoded_root — wft carries ftst but never writes it out);
2. per plane, per ftst class: the spread of fitted t0 about the class centre;
3. the fit's OWN t0 uncertainty (t0_err, chi2 curvature) on the 400 cached
   calibration events, fitted with the production bundle.

Reading the result (per the handoff — do NOT judge on an absolute threshold):
  spread ~ fit's own t0_err  => trigger jitter unresolvably small; the prior
                                can be as tight as the budget allows (GOOD).
  spread >> fit's own t0_err => the quadrature excess IS sigma_t0; use it.

    ../.venv/bin/python mx_june_wft/07_t0_prior_gate.py sat_det3
Outputs: <OUT_BASE>/wft/t0_prior/{t0_prior_gate.json, *.png}
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths            # noqa: E402
setup_paths()
import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
import pandas as pd                                      # noqa: E402

from wft import io as wio                                # noqa: E402

FTST_NS = 10.0            # one ftst step [ns]
CHI2DOF_MAX = 50.0        # exclude showers/sparks from the timing sample


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def read_ftst(cfg):
    """event_id -> (ftst_x, ftst_y) for the whole subrun, from decoded_root."""
    import uproot
    out = {}
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        for path in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            arr = uproot.open(path)['nt'].arrays(['eventId', 'ftst'],
                                                 library='np')
            for eid, ft in zip(arr['eventId'], arr['ftst']):
                out.setdefault(int(eid), {})[plane] = int(ft)
    return out


# ---------------------------------------------------------------- part 2
_CACHE_EVENTS = None
_BUNDLE_PATH = None


def _t0err_init(bundle_path, cache_path):
    global _CACHE_EVENTS, _CAL
    import pickle
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)
    with open(cache_path, 'rb') as f:
        _CACHE_EVENTS = pickle.load(f)


def _t0err_one(eid):
    """Free-fit one cached calibration event with the production bundle and
    return each plane's (t0, t0_err, chi2/dof, tan)."""
    from wft import reco as wreco
    ev = _CACHE_EVENTS[eid]
    row = {'eid': eid}
    for plane in ('x', 'y'):
        P = ev.get(plane)
        if P is None:
            continue
        try:
            fit = wreco.fit_plane(P, plane, _CAL)
        except Exception:
            fit = None
        if fit is None:
            continue
        row[plane] = dict(t0=fit.t0, t0_err=fit.t0_err,
                          chi2dof=fit.chi2 / max(fit.dof, 1),
                          tan=fit.tan_theta, ftst=ev.get(f'ftst_{plane}'))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--table', default=None)
    ap.add_argument('--bundle', default=None,
                    help='calibration bundle (default: the one in events.meta)')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    table = args.table or os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    out_dir = args.out or cfg.out_dir('wft', 't0_prior')
    os.makedirs(out_dir, exist_ok=True)

    with open(table.replace('.parquet', '.meta.json')) as f:
        meta = json.load(f)
    bundle_path = args.bundle or meta['calibration']
    # the meta records the path on whichever host ran the reco
    if not os.path.isdir(bundle_path):
        bundle_path = os.path.join(os.path.dirname(table),
                                   os.path.basename(bundle_path))

    df = pd.read_parquet(table)
    print(f'[gate] {len(df):,} events in {table}')
    ftst = read_ftst(cfg)
    print(f'[gate] ftst read for {len(ftst):,} events')

    summary = {'run_key': args.run_key, 'table': table, 'bundle': bundle_path,
               'planes': {}}

    fig1, axs1 = plt.subplots(2, 2, figsize=(13, 9))
    fig2, axs2 = plt.subplots(2, 2, figsize=(13, 9))

    resid_all = {}
    for i, plane in enumerate(('x', 'y')):
        ok = (df[f'{plane}_ok'].astype(bool)
              & df[f'{plane}_quality_ok'].astype(bool)
              & (df[f'{plane}_chi2'] / df[f'{plane}_dof'].clip(lower=1)
                 < CHI2DOF_MAX))
        sub = df[ok]
        ft = np.array([ftst.get(int(e), {}).get(plane, -1)
                       for e in sub['event_id']])
        t0 = sub[f'{plane}_t0'].to_numpy()
        tan = sub[f'{plane}_tan_theta'].to_numpy()
        good = ft >= 0
        ft, t0, tan = ft[good], t0[good], tan[good]
        eid = sub['event_id'].to_numpy()[good]

        classes = sorted(set(ft.tolist()))
        pl = {'n': int(len(t0)), 'classes': {}}
        med_by_class = {}
        for c in classes:
            m = ft == c
            med = float(np.median(t0[m]))
            sig = robust_sigma(t0[m])
            s68 = float(np.percentile(np.abs(t0[m] - med), 68))
            med_by_class[c] = med
            pl['classes'][int(c)] = dict(n=int(m.sum()), median=med,
                                         robust_sigma=sig, s68=s68)

        # within-class residual, pooled
        resid = t0 - np.array([med_by_class[c] for c in ft])
        resid_all[plane] = (resid, tan, eid)
        pl['pooled_sigma'] = robust_sigma(resid)
        pl['pooled_s68'] = float(np.percentile(np.abs(resid), 68))

        # does the class centre march at -10 ns/step?
        cs = np.array(classes, float)
        ms = np.array([med_by_class[c] for c in classes])
        slope = float(np.polyfit(cs, ms, 1)[0]) if len(cs) > 2 else np.nan
        pl['class_median_slope_ns_per_step'] = slope
        summary['planes'][plane] = pl

        ax = axs1[0, i]
        for c in classes:
            m = ft == c
            ax.hist(t0[m], bins=np.linspace(np.median(t0) - 150,
                                            np.median(t0) + 150, 120),
                    histtype='step', lw=1.2, label=f'ftst {c} (n={m.sum():,})')
        ax.set_xlabel(f'{plane}: fitted t0 [ns]')
        ax.set_title(f'{plane}: t0 by ftst class')
        ax.legend(fontsize=7)

        ax = axs1[1, i]
        ax.plot(cs, ms, 'o-', label=f'slope {slope:+.1f} ns/step')
        ax.plot(cs, ms[0] - FTST_NS * (cs - cs[0]), 'k--', lw=1,
                label='-10 ns/step (expected)')
        ax.set_xlabel('ftst class')
        ax.set_ylabel('class median t0 [ns]')
        ax.legend(fontsize=8)

        ax = axs2[0, i]
        ax.hist(resid, bins=np.linspace(-120, 120, 160), histtype='step', lw=2,
                label=(f'sigma {pl["pooled_sigma"]:.1f} ns, '
                       f's68 {pl["pooled_s68"]:.1f} ns'))
        ax.set_xlabel(f'{plane}: t0 - class median [ns]')
        ax.set_title(f'{plane}: within-class t0 spread (n={len(resid):,})')
        ax.legend(fontsize=8)

        # controls: residual vs |tan| and vs event index (run time)
        ax = axs2[1, i]
        at = np.abs(tan)
        bins = np.linspace(0, 0.5, 11)
        ctr, mres, sres = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (at >= lo) & (at < hi)
            if m.sum() < 20:
                continue
            ctr.append(0.5 * (lo + hi))
            mres.append(float(np.median(resid[m])))
            sres.append(robust_sigma(resid[m]))
        ax.errorbar(ctr, mres, yerr=sres, fmt='o-', capsize=3,
                    label='median +- robust sigma')
        ax.set_xlabel(f'{plane}: |tan theta|')
        ax.set_ylabel('t0 - class median [ns]')
        ax.axhline(0, color='gray', lw=0.8)
        ax.legend(fontsize=8)

    fig1.suptitle(f'{args.run_key}: fitted t0 vs ftst (production reco)')
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 't0_by_ftst.png'), dpi=130)
    fig2.suptitle(f'{args.run_key}: within-ftst-class t0 spread')
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 't0_class_spread.png'), dpi=130)

    # ------------------------------------------------ part 2: the fit's own t0_err
    cache_path = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work',
                              'calib_cache.pkl')
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, 'rb') as f:
            eids = sorted(pickle.load(f).keys())
        print(f'[gate] refitting {len(eids)} cached calibration events '
              f'for t0_err ...', flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=_t0err_init,
                                 initargs=(bundle_path, cache_path)) as pool:
            for r in pool.map(_t0err_one, eids, chunksize=8):
                rows.append(r)
        fig3, axs3 = plt.subplots(1, 2, figsize=(13, 4.5))
        for i, plane in enumerate(('x', 'y')):
            te = np.array([r[plane]['t0_err'] for r in rows
                           if plane in r and np.isfinite(r[plane]['t0_err'])
                           and r[plane]['chi2dof'] < CHI2DOF_MAX])
            pl = summary['planes'][plane]
            pl['fit_t0_err_median'] = float(np.median(te)) if len(te) else np.nan
            pl['fit_t0_err_p90'] = (float(np.percentile(te, 90))
                                    if len(te) else np.nan)
            pl['n_t0_err'] = int(len(te))
            # the quadrature excess: measured spread minus the fit's own noise
            s, e = pl['pooled_sigma'], pl['fit_t0_err_median']
            pl['excess_quadrature_ns'] = (float(np.sqrt(max(s * s - e * e, 0.0)))
                                          if np.isfinite(s) and np.isfinite(e)
                                          else np.nan)
            ax = axs3[i]
            ax.hist(te, bins=np.linspace(0, 60, 90), histtype='step', lw=2,
                    label=f'median {pl["fit_t0_err_median"]:.1f} ns')
            ax.axvline(pl['pooled_sigma'], color='r', ls='--',
                       label=f'within-class spread {pl["pooled_sigma"]:.1f} ns')
            ax.set_xlabel(f'{plane}: fit t0_err [ns]')
            ax.legend(fontsize=8)
        fig3.suptitle(f'{args.run_key}: fit t0 uncertainty vs measured spread')
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, 't0_err_vs_spread.png'), dpi=130)
    else:
        print(f'[gate] no calib cache at {cache_path} — skipping t0_err leg')

    with open(os.path.join(out_dir, 't0_prior_gate.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(summary, indent=1))
    for plane in ('x', 'y'):
        pl = summary['planes'][plane]
        s, e = pl.get('pooled_sigma'), pl.get('fit_t0_err_median')
        x = pl.get('excess_quadrature_ns')
        print(f'[gate] {plane}: within-class spread {s:.1f} ns, '
              f'fit t0_err {e if e is None else round(e, 1)} ns, '
              f'excess (=real jitter) {x if x is None else round(x, 1)} ns')


if __name__ == '__main__':
    main()

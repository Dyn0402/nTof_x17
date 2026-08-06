#!/usr/bin/env python3
"""
gap_compare.py — is the charge-visible drift-gap map a property of the CHAMBER?

Takes every dataset that has a gap study (bench/gap_study.py output) and asks,
per detector:

  * do independent runs / slots / subruns of the same chamber give the same
    global column, and the same (x, y) topography?
  * how does the run-to-run agreement compare with the SPLIT-HALF agreement
    inside one dataset (the statistical noise floor of the estimator)?

A real cathode topography must reproduce between runs as well as it reproduces
between two halves of the same run. A reconstruction or reference artefact
need not.

    ../../.venv/bin/python mx_june_wft/bench/gap_compare.py [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

ANALYSIS = '/home/dylan/x17/cosmic_bench/Analysis'
U = (np.arange(18) + 0.5) * 60.0
KERNEL_R = 45.0
GRID_STEP = 10.0
MIN_EVENTS = 60
GAP_MECH = 30.0

# label -> (run_key, detector, short description). The gap study writes into
# <OUT_BASE>/wft/gap_study/, so the run key is enough to find everything.
DATASETS = [
    ('det3 6-27 sat (top)',   'sat_det3',    'det3', 'reference map'),
    ('det3 6-27 P2 (top)',    'g_det3_wknd', 'det3', 'next-day repeat, same slot'),
    ('det3 6-22 long (bot)',  'g_det3',      'det3', 'different day AND slot'),
    ('det2 6-22 longer (top)', 'o22_long_det2', 'det2', 'control chamber'),
    ('det2 6-22 long (top)',  'g_det2',      'det2', 'same run, 8x stats subrun'),
    ('det4 6-24 long',        'g_det4',      'det4', 'first map'),
    ('det6 6-26 long',        'g_det6_long', 'det6', 'first map'),
    ('det7 6-26 long',        'g_det7_long', 'det7', 'first map'),
]


def sharp(u, A, T, sig):
    return A * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))


def fit_T(P, max_err=60.0):
    """erfc endpoint of a stack of normalised charge-arrival profiles.

    Bounded, and fits whose endpoint error exceeds `max_err` ns are dropped —
    on thin sub-samples the unbounded fit can run away to non-physical T.
    """
    if len(P) < 20:
        return np.nan, np.nan
    m = P.mean(axis=0)
    e = np.maximum(P.std(axis=0) / np.sqrt(len(P)), 1e-5)
    sel = U < 1050
    try:
        p, c = curve_fit(sharp, U[sel], m[sel], p0=[m[:5].mean(), 700, 60],
                         sigma=e[sel], absolute_sigma=True, maxfev=20000,
                         bounds=([0, 200, 10], [np.inf, 1100, 300]))
        T, Te = float(p[1]), float(np.sqrt(c[1, 1]))
        if not np.isfinite(Te) or Te > max_err:
            return np.nan, np.nan
        return T, Te
    except Exception:
        return np.nan, np.nan


def gap_vs_charge(d, nbins=5):
    """Endpoint in bins of total fitted charge — the estimator's amplitude
    systematic, and the way to compare chambers at MATCHED signal size."""
    q = d['qsum']
    edges = np.percentile(q, np.linspace(0, 100, nbins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (q >= lo) & (q <= hi)
        if s.sum() < 100:
            continue
        T, Te = fit_T(d['Q'][s])
        out.append(dict(q_med=float(np.median(q[s])), n=int(s.sum()),
                        gap=T * d['v_geom'] / 1e3 if np.isfinite(T) else np.nan,
                        err=Te * d['v_geom'] / 1e3 if np.isfinite(Te) else np.nan))
    return out


def load(run_key, plane='x'):
    from qa_config import get_config
    cfg = get_config(run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft', 'gap_study')
    par = os.path.join(W, 'event_profiles.parquet')
    js = os.path.join(W, 'gap_study.json')
    if not (os.path.exists(par) and os.path.exists(js)):
        return None
    meta = json.load(open(js))
    v_geom = meta['planes'][plane]['v_geom']
    df = pd.read_parquet(par)
    g = df[(df.plane == plane) & df.contained & (df.chi2dof < 250)].copy()
    Q = g[[f'q{i}' for i in range(18)]].to_numpy()
    Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-9)
    return dict(key=run_key, v_geom=v_geom, meta=meta, Q=Q,
                x=g.ref_x.to_numpy(), y=g.ref_y.to_numpy(),
                qsum=g.qsum.to_numpy(), tan=np.abs(g.tan.to_numpy()),
                n=len(g), det_name=cfg.DET_NAME, run=cfg.RUN,
                sub_run=cfg.SUB_RUN, z=cfg.DET_PLANE_Z)


def grid_map(d, sel=None, xs=None, ys=None):
    """Sliding-kernel endpoint map [mm] on a fixed detector-local grid."""
    Q, x, y = d['Q'], d['x'], d['y']
    if sel is not None:
        Q, x, y = Q[sel], x[sel], y[sel]
    if xs is None:
        xs = np.arange(40, 380 + 1e-6, GRID_STEP)
        ys = np.arange(40, 380 + 1e-6, GRID_STEP)
    M = np.full((len(ys), len(xs)), np.nan)
    E = np.full_like(M, np.nan)
    N = np.zeros_like(M)
    for j, yc in enumerate(ys):
        dy2 = (y - yc) ** 2
        for i, xc in enumerate(xs):
            s = dy2 + (x - xc) ** 2 < KERNEL_R ** 2
            N[j, i] = s.sum()
            if s.sum() < MIN_EVENTS:
                continue
            T, Te = fit_T(Q[s])
            if np.isfinite(T):
                M[j, i] = T * d['v_geom'] / 1000.0
                E[j, i] = Te * d['v_geom'] / 1000.0
    return xs, ys, M, E, N


def agreement(A, B):
    """Compare two maps on the same grid."""
    m = np.isfinite(A) & np.isfinite(B)
    if m.sum() < 10:
        return dict(n=int(m.sum()))
    a, b = A[m], B[m]
    return dict(n=int(m.sum()), mean_diff=float(np.mean(a - b)),
                rms_diff=float(np.std(a - b)),
                corr=float(np.corrcoef(a, b)[0, 1]),
                spread_a=float(np.std(a)), spread_b=float(np.std(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'mx_june_wft'))
    ap.add_argument('--plane', default='x')
    args = ap.parse_args()

    data, rows = {}, []
    for label, key, det, note in DATASETS:
        d = load(key, args.plane)
        if d is None:
            print(f'-- {label:24} ({key}): no gap study yet, skipped')
            continue
        d.update(label=label, det=det, note=note)
        data[label] = d
        T, Te = fit_T(d['Q'])
        d['T_global'], d['T_global_err'] = T, Te
        d['gap_global'] = T * d['v_geom'] / 1000.0
        d['gap_global_err'] = Te * d['v_geom'] / 1000.0
        xs, ys, M, E, N = grid_map(d)
        d['xs'], d['ys'], d['M'], d['E'] = xs, ys, M, E
        # split-half (even/odd events): the estimator's own noise floor
        idx = np.arange(d['n'])
        _, _, MA, _, _ = grid_map(d, sel=idx % 2 == 0, xs=xs, ys=ys)
        _, _, MB, _, _ = grid_map(d, sel=idx % 2 == 1, xs=xs, ys=ys)
        d['split'] = agreement(MA, MB)
        rows.append(dict(label=label, det=det, key=key, note=note,
                         n=d['n'], v_geom=round(d['v_geom'], 2),
                         gap_mm=round(d['gap_global'], 2),
                         err=round(d['gap_global_err'], 2),
                         map_median=round(float(np.nanmedian(M)), 2),
                         map_p5=round(float(np.nanpercentile(M, 5)), 2),
                         map_p95=round(float(np.nanpercentile(M, 95)), 2),
                         split_rms=round(d['split'].get('rms_diff', np.nan), 2),
                         split_corr=round(d['split'].get('corr', np.nan), 2)))
    if not rows:
        raise SystemExit('no gap studies found')
    os.makedirs(args.out, exist_ok=True)

    # ---- the estimator's amplitude systematic, and the charge-matched
    # chamber comparison: a low-gain chamber reads a slightly shorter column
    # even at identical geometry, so chambers must be compared at matched qsum.
    qc = []
    for label, d in data.items():
        d['q_curve'] = gap_vs_charge(d)
        for b in d['q_curve']:
            qc.append(dict(label=label, det=d['det'], q_med=round(b['q_med']),
                           n=b['n'], gap=round(b['gap'], 2),
                           err=round(b['err'], 2)))
    if qc:
        print('\n== endpoint vs total fitted charge (quintiles) ==')
        print(pd.DataFrame(qc).to_string(index=False))
        pd.DataFrame(qc).to_csv(
            os.path.join(args.out, 'gap_vs_charge.csv'), index=False)

    # ---- systematic null tests inside each dataset: if the topography were a
    # signal-size or track-angle artefact of the fit, splitting the SAME events
    # on charge or on |tan| would move it. Geometry must not care.
    syst = []
    for label, d in data.items():
        for name, var in (('charge', d['qsum']), ('|tan|', d['tan'])):
            med = np.nanmedian(var)
            lo, hi = var <= med, var > med
            Tl, _ = fit_T(d['Q'][lo])
            Th, _ = fit_T(d['Q'][hi])
            _, _, Ml, _, _ = grid_map(d, sel=lo, xs=d['xs'], ys=d['ys'])
            _, _, Mh, _, _ = grid_map(d, sel=hi, xs=d['xs'], ys=d['ys'])
            ag = agreement(Ml, Mh)
            syst.append(dict(label=label, split=name,
                             gap_low=round(Tl * d['v_geom'] / 1e3, 2),
                             gap_high=round(Th * d['v_geom'] / 1e3, 2),
                             d_gap=round((Tl - Th) * d['v_geom'] / 1e3, 2),
                             map_rms=round(ag.get('rms_diff', np.nan), 2),
                             map_corr=round(ag.get('corr', np.nan), 2),
                             split_half_rms=round(
                                 d['split'].get('rms_diff', np.nan), 2)))
    if syst:
        print('\n== systematic null tests (same events split in two) ==')
        print(pd.DataFrame(syst).to_string(index=False))
        pd.DataFrame(syst).to_csv(
            os.path.join(args.out, 'gap_consistency_systematics.csv'),
            index=False)

    tab = pd.DataFrame(rows)
    print('\n== global column and map summary (X plane) ==')
    print(tab.to_string(index=False))

    # ---- pairwise agreement between datasets of the same detector
    pairs = []
    labels = list(data)
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            A, B = data[la], data[lb]
            if A['det'] != B['det']:
                continue
            ag = agreement(A['M'], B['M'])
            # a chamber can be mounted mirrored between slots; if a flipped
            # map correlates far better, that is bookkeeping, not physics
            flips = {'as-is': A['M'], 'flip-x': A['M'][:, ::-1],
                     'flip-y': A['M'][::-1, :], 'flip-xy': A['M'][::-1, ::-1]}
            fc = {k: agreement(v, B['M']).get('corr', np.nan)
                  for k, v in flips.items()}
            ag['best_orientation'] = max(fc, key=lambda k: (fc[k] if
                                         np.isfinite(fc[k]) else -9))
            ag['corr_flips'] = fc
            # split-half floors of the two datasets, added in quadrature/2
            floor = np.sqrt((A['split'].get('rms_diff', np.nan) ** 2 +
                             B['split'].get('rms_diff', np.nan) ** 2) / 2) / np.sqrt(2)
            pairs.append(dict(det=A['det'], a=la, b=lb, n=ag['n'],
                              d_global=round(A['gap_global'] - B['gap_global'], 2),
                              mean_diff=round(ag.get('mean_diff', np.nan), 2),
                              rms_diff=round(ag.get('rms_diff', np.nan), 2),
                              noise_floor=round(float(floor), 2),
                              corr=round(ag.get('corr', np.nan), 2),
                              best_orient=ag.get('best_orientation', ''),
                              corr_flipx=round(ag.get('corr_flips', {})
                                               .get('flip-x', np.nan), 2),
                              corr_flipy=round(ag.get('corr_flips', {})
                                               .get('flip-y', np.nan), 2)))
    if pairs:
        print('\n== same-chamber, different dataset: does the map reproduce? ==')
        print(pd.DataFrame(pairs).to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    tab.to_csv(os.path.join(args.out, 'gap_consistency_summary.csv'), index=False)
    if pairs:
        pd.DataFrame(pairs).to_csv(
            os.path.join(args.out, 'gap_consistency_pairs.csv'), index=False)
    np.savez(os.path.join(args.out, 'gap_consistency_maps.npz'),
             **{f'{d["key"]}_M': d['M'] for d in data.values()},
             **{f'{d["key"]}_E': d['E'] for d in data.values()},
             xs=list(data.values())[0]['xs'], ys=list(data.values())[0]['ys'])

    # ---- figure: one map per dataset, grouped by detector
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(data)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.3 * nrow),
                            layout='constrained', squeeze=False)
    im = None
    for k, (label, d) in enumerate(data.items()):
        ax = axs[k // ncol][k % ncol]
        im = ax.pcolormesh(d['xs'], d['ys'], d['M'], cmap='RdBu',
                           vmin=25.0, vmax=35.0, shading='nearest')
        ax.set_title(f"{label}\nmedian {np.nanmedian(d['M']):.1f} mm  "
                     f"(n={d['n']:,})", fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]', fontsize=8)
        ax.set_ylabel('y [mm]', fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(data), nrow * ncol):
        axs[k // ncol][k % ncol].axis('off')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label('charge-visible drift gap [mm]  (white = 30 mm mechanical)')
    fig.suptitle('Charge-visible drift-gap maps, every dataset '
                 f'(X plane, contained tracks, kernel r = {KERNEL_R:.0f} mm)',
                 fontsize=12)
    out = os.path.join(args.out, 'gap_consistency_maps.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('\nwrote', out)

    # ---- figure: repeat scatter, map value vs map value, per detector pair
    same = [p for p in pairs if np.isfinite(p['corr'])]
    if same:
        fig2, axs2 = plt.subplots(1, len(same), figsize=(4.0 * len(same), 4.0),
                                  layout='constrained', squeeze=False)
        for k, p in enumerate(same):
            A, B = data[p['a']], data[p['b']]
            m = np.isfinite(A['M']) & np.isfinite(B['M'])
            ax = axs2[0][k]
            ax.plot(A['M'][m], B['M'][m], '.', ms=4, alpha=0.6)
            lo, hi = 24, 33
            ax.plot([lo, hi], [lo, hi], 'k-', lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(p['a'], fontsize=8)
            ax.set_ylabel(p['b'], fontsize=8)
            ax.set_title(f"{p['det']}: r = {p['corr']:.2f}, "
                         f"rms {p['rms_diff']:.2f} mm\n"
                         f"(split-half floor {p['noise_floor']:.2f} mm)",
                         fontsize=9)
            ax.grid(alpha=0.3)
        out2 = os.path.join(args.out, 'gap_consistency_repeat.png')
        fig2.savefig(out2, dpi=140, bbox_inches='tight')
        print('wrote', out2)

    # ---- gap vs charge: chambers compared at matched signal size
    fig3, ax3 = plt.subplots(figsize=(7.2, 5.0), layout='constrained')
    cols = {}
    for label, d in data.items():
        cur = d.get('q_curve') or []
        if not cur:
            continue
        c = cols.setdefault(d['det'], f'C{len(cols)}')
        ax3.errorbar([b['q_med'] for b in cur], [b['gap'] for b in cur],
                     yerr=[b['err'] for b in cur], marker='o', ms=4, lw=1.2,
                     color=c, ls='-' if 'sat' in label or 'longer' in label
                     else '--', label=label)
    ax3.axhline(GAP_MECH, color='k', ls=':', lw=1)
    ax3.text(ax3.get_xlim()[0], GAP_MECH + 0.1, '30 mm mechanical', fontsize=8)
    ax3.set_xlabel('median total fitted charge in the bin')
    ax3.set_ylabel('charge-visible drift gap [mm]')
    ax3.set_title('Endpoint vs signal size: chambers at matched charge')
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7)
    out3 = os.path.join(args.out, 'gap_vs_charge.png')
    fig3.savefig(out3, dpi=140, bbox_inches='tight')
    print('wrote', out3)


if __name__ == '__main__':
    main()

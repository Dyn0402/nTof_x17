#!/usr/bin/env python3
"""
15_explain_figures.py — the T1.4 explanatory figures (F7 + F8) for MPGD26.

Three figures, all from real sat_det3 events through the shipped wft/ chain:

  vd_estimators.png   (F7)  one inclined event display with the hit-time
                            ladder and the forward fit drawn on it, and the
                            seven "drift velocities" of the corpus listed
                            beside it, grouped by what they actually measure.
  deconv_transfer.png (F8a) the impulse response's transfer function against
                            the measured noise floor — where the inversion
                            has no information.
  deconv_scatter.png  (F8b) per-event deconvolved-ladder angle vs the
                            forward-fit angle on the same events: ensemble
                            agreement, per-event blow-up.

The deconvolution is the same 2-D Tikhonov structure as the threading
displays (37_threading_displays.py), ported onto the shipped wft.model time
tensors (the R&D forward_model2 artefacts are gone).

    ../.venv/bin/python mx_june_wft/15_explain_figures.py sat_det3
Output: <OUT_BASE>/wft/explain/
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

KDEEP = 20
LAMBDA = 1.0
QFRAC = 0.05

# F7 table (FEEDBACK_2026-08-11 §F7), det3 @ 1000 V
ESTIMATORS = [
    ('1 · hit-ladder slope (production)', '47–50', 'dead — compression'),
    ('2 · best hit estimators (CFD/MF/…)', '41.8–42.8', 'dead — same S-curve'),
    ('3 · unshared / deconvolved ladder', '34.3', 'biased low'),
    ('4 · gap-filling (assume 29 mm)', '~42', 'inherits −3 to −7 %'),
    ('5 · duration / gap', '28–30', 'understood, biased'),
    ('6 · forward-fit χ²(v) scan', '36.7 ± 0.3 ± 0.9', 'the basis'),
    ('7 · Magboltz (gas + 0.8 % H₂O)', 'fits the scan', 'independent leg'),
]


def deconv2d(P, plane, t0, hyper, wm, lam=LAMBDA, Kd=KDEEP):
    """Line-free 2-D charge deconvolution on shipped-model time tensors.
    Returns (Q (n_strip, Kd), uk, pos) or None."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    n, ns = W.shape
    k_save = wm.K
    wm.set_depth_bins(Kd)
    try:
        t0q = round(t0 / wm.T0_STEP) * wm.T0_STEP
        H0, H1, H2 = wm._time_tensors(plane, t0q, hyper)   # (NSAMP, Kd)
    finally:
        wm.set_depth_bins(k_save)
    kY = hyper.get('kY', 1.0) if plane == 'y' else hyper.get('cX', 1.0)
    c1, c2 = hyper['c1'] * kY, hyper['c2'] * kY

    A = np.zeros((n, ns, n, Kd))
    for j in range(n):
        A[j, :, j, :] += H0
        for off, c, H in ((1, c1, H1), (2, c2, H2)):
            if j + off < n:
                A[j + off, :, j, :] += c * H
            if j - off >= 0:
                A[j - off, :, j, :] += c * H
    A = A.reshape(n * ns, n * Kd)
    ok = ~sat.reshape(-1)
    wgt = np.repeat(1.0 / noise, ns)
    Aw = (A * wgt[:, None])[ok]
    yw = (W / noise[:, None]).reshape(-1)[ok]
    rows = []
    for j in range(n):
        for k in range(Kd - 2):
            r = np.zeros(n * Kd)
            r[j * Kd + k:j * Kd + k + 3] = (1.0, -2.0, 1.0)
            rows.append(r)
    L = np.asarray(rows)
    try:
        q, _ = nnls(np.vstack([Aw, lam * L]),
                    np.concatenate([yw, np.zeros(len(L))]),
                    maxiter=60 * n * Kd)
    except Exception:
        return None
    uk = (np.arange(Kd) + 0.5) * wm.DT
    return q.reshape(n, Kd), uk, pos


def deconv_angle(Q, uk, pos, v):
    """Charge-weighted straight line through the deconvolved cluster."""
    qk = Q.sum(axis=0)
    keep = qk > QFRAC * qk.max()
    if keep.sum() < 3:
        return np.nan
    pc = (Q[:, keep] * pos[:, None]).sum(axis=0) / Q[:, keep].sum(axis=0)
    wgt = qk[keep]
    u = uk[keep]
    # weighted LSQ slope dpc/du [mm/ns]
    um = np.average(u, weights=wgt)
    pm = np.average(pc, weights=wgt)
    w = (wgt * (u - um) * (pc - pm)).sum() / (wgt * (u - um) ** 2).sum()
    return w * 1e3 / v          # tan theta, same convention as the fit


def rise_times(W, noise, frac=0.2, nsig=8.0):
    """Per-strip rising-edge crossing time [ns] (linear interp), NaN if dim."""
    t = np.full(W.shape[0], np.nan)
    for i, w in enumerate(W):
        m = w.max()
        if m < nsig * noise[i]:
            continue
        thr = frac * m
        idx = np.argmax(w >= thr)
        if idx == 0:
            continue
        f = (thr - w[idx - 1]) / (w[idx] - w[idx - 1])
        t[i] = (idx - 1 + f) * 60.0
    return t


def pick_event(events, cal, wr, plane='x', tan_lo=0.22, tan_hi=0.40):
    """A clean inclined event: many live strips, no saturation, good fit."""
    best = None
    for eid in sorted(events):
        ev = events[eid]
        t = ev['truth']
        tan = t.get(f'tan_{plane}', np.nan)
        if not (np.isfinite(tan) and tan_lo <= abs(tan) <= tan_hi) \
                or ev.get('spark'):
            continue
        cand = ev['wins'].get(plane) or []
        if not cand:
            continue
        P = max(cand, key=lambda c: np.asarray(c['W']).max())
        W = np.asarray(P['W'])
        if W.max() >= cal.sat_adc or W.shape[0] < 8:
            continue
        f = wr.fit_plane(P, plane, cal,
                         t0_prior=wr.t0_prior_for(
                             cal, plane, (ev.get('ftst') or {}).get(plane)))
        if f is None or not f.quality_ok or not f.slope_reliable:
            continue
        score = W.shape[0] + 5 * (abs(tan) - tan_lo)
        if best is None or score > best[0]:
            best = (score, eid, P, f, tan)
        if best and best[0] > 14:
            break
    return best[1:], plane


def fig_vd(eid, P, fit, tan_ref, plane, cal, wm, out):
    import matplotlib.pyplot as plt
    W, noise, pos, _ = wm.prep_plane(P, plane)
    v = cal.v_drift
    ts = np.arange(W.shape[1]) * 60.0

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    ext = [pos[0] - 0.39, pos[-1] + 0.39, ts[-1] + 30, -30]
    ax.imshow(W, aspect='auto', extent=ext, cmap='Greys',
              vmin=0, vmax=0.7 * W.max(), interpolation='nearest')

    # (1)-(3): a slope read off per-strip times — the compressed ladder
    tr = rise_times(W, noise)
    m = np.isfinite(tr)
    ax.plot(pos[m], tr[m], 'o', ms=5, color='crimson', zorder=5,
            label='per-strip hit times (rising edge)')
    cf = np.polyfit(pos[m], tr[m], 1)
    pp = np.array([pos[m].min() - 0.5, pos[m].max() + 0.5])
    ax.plot(pp, np.polyval(cf, pp), '-', color='crimson', lw=2,
            label=f'ladder slope → v ≈ {abs(1e3 / cf[0] / tan_ref):.0f} µm/ns'
                  ' (estimators 1–3)')

    # (6): the forward fit's track line t(p) = t0 + (p - p0)/w
    t_line = fit.t0 + (pp - fit.p0) / fit.w
    ax.plot(pp, t_line, '-', color='#1668b4', lw=2.5,
            label=f'forward fit: never converts to v until the last line '
                  f'(6: v = {v:.1f})')
    # the fitted column extent on the time axis
    ax.axhline(fit.t0, color='#1668b4', lw=1, ls=':')
    ax.axhline(fit.t0 + fit.q_uend, color='#1668b4', lw=1, ls=':')

    # (4)-(5): duration/gap-filling brackets
    x_br = ext[0] + 0.4
    ax.annotate('', xy=(x_br, fit.t0), xytext=(x_br, fit.t0 + fit.q_uend),
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
    ax.text(x_br + 0.25, fit.t0 + 0.5 * fit.q_uend,
            'column duration —\nestimators 4–5 assume\nthis spans a known gap',
            fontsize=8.5, color='darkorange', va='center')

    ax.set_xlabel(f'{plane.upper()} strip position [mm]')
    ax.set_ylabel('time [ns]')
    if np.isfinite(fit.q_uend):
        ax.set_ylim(fit.t0 + fit.q_uend + 260, max(-30.0, fit.t0 - 260))
    ax.set_title(f'event {eid}, tan θ_ref = {tan_ref:+.2f} — what each '
                 '"drift velocity" reads off this display')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)

    axr = fig.add_subplot(gs[1])
    axr.axis('off')
    yy = 0.97
    axr.text(0, yy, 'the seven "v_D" of the corpus (det3 @ 1000 V, µm/ns)',
             fontsize=10.5, weight='bold', va='top')
    yy -= 0.075
    groups = [('read a slope off a SQUEEZED ladder', ESTIMATORS[0:3],
               'crimson'),
              ('assume the column fills a known gap', ESTIMATORS[3:5],
               'darkorange'),
              ('fit the waveforms forward', ESTIMATORS[5:6], '#1668b4'),
              ('independent physics', ESTIMATORS[6:7], '0.35')]
    for title, rows, col in groups:
        axr.text(0, yy, title, fontsize=9, color=col, style='italic',
                 va='top')
        yy -= 0.055
        for name, val, status in rows:
            axr.text(0.03, yy, name, fontsize=8.6, va='top')
            axr.text(0.70, yy, val, fontsize=8.6, va='top', weight='bold')
            axr.text(0.70, yy - 0.038, status, fontsize=7.8, va='top',
                     color='0.45')
            yy -= 0.093
        yy -= 0.012
    axr.text(0, yy, 'validator, not an estimator: implied v = w / tan θ_ref\n'
                    'must be FLAT against angle (§8.1) — the judge used for\n'
                    'every model decision.', fontsize=8.8, va='top',
             color='#1668b4')
    yy -= 0.14
    axr.text(0, yy, 'caveat for the talk: the sim mismatch is NOT a constant\n'
                    'tan-scale — a second angle-dependent bias exists\n'
                    '(ANGLED_LADDER_2026-08-09). Do not draw one arrow.',
             fontsize=8.8, va='top', color='0.35')
    fig.savefig(os.path.join(out, 'vd_estimators.png'), dpi=140,
                bbox_inches='tight')
    print('wrote vd_estimators.png')


def fig_transfer(events, cal, wm, out):
    """|H(f)| of the impulse response over the 32-sample window vs the
    measured noise floor, and the implied inversion amplification."""
    import matplotlib.pyplot as plt
    hyper = dict(cal.hyper)
    tmpl, _ = wm._templates('x', hyper['sigma_s'])
    ts = np.arange(32) * 60.0
    h = np.interp(ts, wm.TGRID, tmpl, left=0, right=0)

    # typical signal amplitude and per-sample noise from the cache
    amps, sigs = [], []
    for eid in sorted(events)[:400]:
        for P in (events[eid]['wins'].get('x') or [])[:1]:
            W, noise, _, _ = wm.prep_plane(P, 'x')
            a = W.max(axis=1)
            m = a > 8 * noise
            amps.extend(a[m])
            sigs.extend(noise[m])
    a_med, s_med = float(np.median(amps)), float(np.median(sigs))

    f = np.fft.rfftfreq(32, d=60e-9) / 1e6          # MHz
    H = np.abs(np.fft.rfft(h * a_med / h.max()))
    nfloor = s_med * np.sqrt(32)                     # white noise per mode

    # the quantity un-sharing must resolve: the RC-dispersed neighbour COPY
    base = wm.TS[:, None] - np.array([[0.0]])
    copies = {}
    for plane, cc in (('x', hyper['c1'] * hyper.get('cX', 1.0)),
                      ('y', hyper['c1'] * hyper.get('kY', 1.0))):
        h1, _ = wm._copy_responses(plane, base, hyper)
        hc = h1[:, 0] / max(h.max(), 1e-9)
        copies[plane] = (cc, np.abs(np.fft.rfft(hc * a_med * cc)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.semilogy(f, H, 'o-', color='#1668b4',
                label=f'central strip, A_typ = {a_med:.0f} ADC')
    for plane, col in (('y', 'darkorange'), ('x', 'crimson')):
        cc, Hc = copies[plane]
        ax.semilogy(f, Hc, 'o-', ms=4, color=col,
                    label=f'{plane.upper()} ±1 neighbour copy '
                          f'(coeff {cc:.2f}, RC-dispersed)')
    ax.axhline(nfloor, color='0.2', ls='--',
               label=f'noise floor per mode (σ = {s_med:.1f} ADC/sample)')
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('amplitude per mode [ADC]')
    ax.set_title('what un-sharing must resolve, per event: the copy '
                 'is at the floor')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    amp = np.where(np.abs(H) > 0, H.max() / np.maximum(H, 1e-9), np.inf)
    ax.semilogy(f, amp, 'o-', color='0.3')
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('noise amplification of a naive inverse')
    ax.set_title('a naive inverse amplifies noise ×30')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'deconv_transfer.png'), dpi=140)
    n_lost = {p: int((copies[p][1] < nfloor).sum()) for p in copies}
    print(f'wrote deconv_transfer.png (A_typ {a_med:.0f}, sigma {s_med:.1f}, '
          f'copy modes below floor: X {n_lost["x"]}/{len(f)}, '
          f'Y {n_lost["y"]}/{len(f)})')


def fig_scatter(events, cal, wm, wr, out, n_ev=300):
    """Deconvolved-ladder angle vs forward-fit angle, per event."""
    import matplotlib.pyplot as plt
    hyper = dict(cal.hyper)
    v = cal.v_drift
    rows = []
    for eid in sorted(events):
        if len(rows) >= n_ev:
            break
        ev = events[eid]
        t = ev['truth']
        tan_ref = t.get('tan_x', np.nan)
        if not np.isfinite(tan_ref) or abs(tan_ref) < 0.08 or ev.get('spark'):
            continue
        cand = ev['wins'].get('x') or []
        if not cand:
            continue
        P = max(cand, key=lambda c: np.asarray(c['W']).max())
        if np.asarray(P['W']).shape[0] > 24:
            continue                       # keep the deconvolution tractable
        f = wr.fit_plane(P, 'x', cal,
                         t0_prior=wr.t0_prior_for(
                             cal, 'x', (ev.get('ftst') or {}).get('x')))
        if f is None or not f.quality_ok or not f.slope_reliable:
            continue
        d = deconv2d(P, 'x', f.t0, hyper, wm)
        if d is None:
            continue
        tan_dec = deconv_angle(*d, v)
        if not np.isfinite(tan_dec):
            continue
        rows.append((tan_ref, f.tan_theta, tan_dec))
    r = np.array(rows)
    th_ref, th_fit, th_dec = (np.degrees(np.arctan(r[:, i])) for i in range(3))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    lim = 32
    ax.plot([-lim, lim], [-lim, lim], '-', color='0.8', lw=1)
    ax.scatter(th_fit, th_dec, s=9, alpha=0.45, color='crimson', lw=0,
               label='deconvolved ladder (2-D Tikhonov)')
    bins = np.linspace(-lim, lim, 9)
    bi = np.digitize(th_fit, bins)
    bx = [th_fit[bi == i].mean() for i in range(1, len(bins)) if (bi == i).sum() > 4]
    by = [np.median(th_dec[bi == i]) for i in range(1, len(bins)) if (bi == i).sum() > 4]
    ax.plot(bx, by, 'o-', color='k', ms=5, label='ensemble medians')
    ax.set_xlabel('forward-fit angle [deg]')
    ax.set_ylabel('deconvolved-ladder angle [deg]')
    ax.set_title(f'{len(r)} events, X plane — ensemble fine, per event not')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    d_dec = th_dec - th_ref
    d_fit = th_fit - th_ref
    rng = (-15, 15)
    ax.hist(d_fit, bins=40, range=rng, histtype='stepfilled', alpha=0.5,
            color='#1668b4',
            label=f'forward fit − ref  (σ68 = '
                  f'{np.percentile(np.abs(d_fit - np.median(d_fit)), 68):.1f}°)')
    ax.hist(d_dec, bins=40, range=rng, histtype='step', lw=2, color='crimson',
            label=f'deconvolved − ref  (σ68 = '
                  f'{np.percentile(np.abs(d_dec - np.median(d_dec)), 68):.1f}°)')
    ax.set_xlabel('angle − reference [deg]')
    ax.set_ylabel('events')
    ax.set_title('the same events, against the reference')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'deconv_scatter.png'), dpi=140)
    med = (float(np.median(d_dec)), float(np.median(d_fit)))
    print(f'wrote deconv_scatter.png ({len(r)} events; median dev '
          f'dec {med[0]:+.2f} vs fit {med[1]:+.2f} deg)')
    return dict(n=len(r), sigma68_dec=float(np.percentile(
        np.abs(d_dec - np.median(d_dec)), 68)), sigma68_fit=float(
        np.percentile(np.abs(d_fit - np.median(d_fit)), 68)),
        median_dec=med[0], median_fit=med[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--n-scatter', type=int, default=300)
    ap.add_argument('--figs', default='vd,transfer,scatter',
                    help='comma list: vd, transfer, scatter')
    args = ap.parse_args()
    which = set(args.figs.split(','))

    from qa_config import get_config, setup_paths
    setup_paths()
    import matplotlib
    matplotlib.use('Agg')
    from wft.calib import CalibrationBundle
    from wft import model as wm
    from wft import reco as wr

    cfg = get_config(args.run_key)
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft',
                                         'calib_bundle_lp2_t0p')
    out = cfg.out_dir('wft', 'explain')
    os.makedirs(out, exist_ok=True)
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal)
    wm.set_nsamp(32)

    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft',
                                       'bench_cache_ftst.pkl')
    with open(cache, 'rb') as f:
        events = pickle.load(f)['events']
    print(f'{len(events):,} events, bundle {os.path.basename(bundle)}')

    if 'vd' in which:
        (eid, P, fit, tan_ref), plane = pick_event(events, cal, wr)
        print(f'display event: {eid} (tan_ref {tan_ref:+.2f}, '
              f'{np.asarray(P["W"]).shape[0]} strips)')
        fig_vd(eid, P, fit, tan_ref, plane, cal, wm, out)
    if 'transfer' in which:
        fig_transfer(events, cal, wm, out)
    if 'scatter' in which:
        stats = fig_scatter(events, cal, wm, wr, out, n_ev=args.n_scatter)
        with open(os.path.join(out, 'explain.json'), 'w') as f:
            json.dump(dict(scatter=stats, bundle=bundle), f, indent=1)
    print(f'figures in {out}')


if __name__ == '__main__':
    main()

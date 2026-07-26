#!/usr/bin/env python3
"""Three checks in one:
 A) 1000V det3 refit with K=24 (basis to 1440ns): coherent late-charge bump
    (slow-field tail) or flat junk beyond the edge?
 B) 500V det3 with K=26, mean/trim estimators (25 vs 29 mm discriminator).
 C) det4 endpoint: ref-pinned q profiles at v=34.2, mean estimators.
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2

B3 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
      'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
B4 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det4_day_6-24-26/long_run/'
      'mx17_4/waveform_first')
DT = 60.0

def U50g(uk, prof, pb=(1, 6)):
    plat = np.median(prof[pb[0]:pb[1]])
    if plat <= 0:
        return np.nan
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= pb[1] - 1]
    if len(below) == 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1

_ctx = {}

def fit_q(args):
    eid, v, K = args
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = _ctx['events'][eid]
    H = _ctx['H']
    out = []
    for plane in ('x', 'y'):
        tn = ev[f'tan_{plane}']
        if abs(tn) < 0.08:
            continue
        P = ev[plane]
        if P['W'].size == 0:
            continue
        W, noise, pos, sat = fm2.prep_plane(P, plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat,
                                  ev[f'ref_mesh_{plane}'], tn * v * 1e-3,
                                  float(t0), H)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append(q / q.sum())
    return out


def run_case(label, events, hyper, v, K, n_ev, sel_pred, tmpl=None, gain_unit=False):
    if tmpl is not None:
        fm2.TGRID = tmpl['grid']
        fm2.TMPL = {'x': tmpl['tmpl_x'], 'y': tmpl['tmpl_y']}
        fm2._smear_cache.clear()
    if gain_unit:
        fm2.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
    sel = [e for e, ev in events.items() if sel_pred(ev)][:n_ev]
    _ctx['events'] = {e: events[e] for e in sel}
    _ctx['H'] = hyper
    profs = []
    with ProcessPoolExecutor(max_workers=14) as pool:
        for o in pool.map(fit_q, [(e, v, K) for e in sel], chunksize=6):
            profs.extend(o)
    L = max(len(p) for p in profs)
    P = np.zeros((len(profs), L))
    for i, p in enumerate(profs):
        P[i, :len(p)] = p
    uk = (np.arange(L) + 0.5) * DT
    mean = P.mean(axis=0)
    trim = np.mean(np.sort(P, axis=0)[int(.05 * len(P)):int(.95 * len(P))], axis=0)
    Um = U50g(uk, mean)
    Ut = U50g(uk, trim)
    cum = np.cumsum(mean)
    u95 = uk[np.argmax(cum >= 0.95)]
    print(f'{label}: n={len(P)}  U50(mean)={Um:.0f}  U50(trim)={Ut:.0f} ns  '
          f'col(mean)={v*Um*1e-3:.1f} mm  u95={u95:.0f} ({v*u95*1e-3:.1f} mm)',
          flush=True)
    return uk, mean, dict(U_mean=Um, U_trim=Ut, u95=u95, v=v, n=len(P))

if __name__ == '__main__':
    results = {}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    # A) det3 1000V, K=24 tail
    hj = json.load(open(os.path.join(B3, 'hyper_v2.json')))
    H3 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
    ev3 = pickle.load(open(os.path.join(B3, 'wfcache.pkl'), 'rb'))['events']
    pred = lambda ev: 0.10 < np.hypot(ev['tan_x'], ev['tan_y']) < 0.45
    uk, mean, r = run_case('det3 1000V K=24', ev3, H3, hj['v'], 24, 700, pred)
    results['det3_1000_K24'] = r
    axs[0].plot(uk, mean, 'o-', ms=3)
    axs[0].axvline(25000 / hj['v'], ls=':', color='k')
    axs[0].axvline(29000 / hj['v'], ls='--', color='k')
    axs[0].set_yscale('log'); axs[0].set_ylim(1e-4, 0.2)
    axs[0].set_title('det3 1000V, basis to 1440 ns (log)')
    axs[0].set_xlabel('u [ns]'); axs[0].grid(alpha=0.3)

    # B) det3 500V K=26
    ffv = json.load(open(os.path.join(B3, 'drift_scan_v_lowhv.json')))
    ev5 = pickle.load(open(os.path.join(B3, 'wfcache_500V.pkl'), 'rb'))
    uk5, mean5, r5 = run_case('det3 500V K=26', ev5, H3, ffv['500']['v'], 26, 300, pred)
    results['det3_500_K26'] = r5
    axs[1].plot(uk5, mean5, 'o-', ms=3)
    axs[1].axvline(25000 / ffv['500']['v'], ls=':', color='k', label='25 mm')
    axs[1].axvline(29000 / ffv['500']['v'], ls='--', color='k', label='29 mm')
    axs[1].set_title(f'det3 500V (v={ffv["500"]["v"]:.1f})')
    axs[1].set_xlabel('u [ns]'); axs[1].legend(); axs[1].grid(alpha=0.3)

    # C) det4 endpoint
    hj4 = json.load(open(os.path.join(B4, 'hyper_det4.json')))
    H4 = {k: hj4[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
    ev4 = pickle.load(open(os.path.join(B4, 'wfcache.pkl'), 'rb'))['events']
    tz4 = np.load(os.path.join(B4, 'templates_perplane.npz'))
    ns4 = next(iter(ev4.values()))['x']['W'].shape[1]
    fm2.set_nsamp(ns4)
    pred4 = lambda ev: (ev['x']['W'].size > 0 and ev['y']['W'].size > 0 and
                        0.10 < np.hypot(ev['tan_x'], ev['tan_y']) < 0.45)
    uk4, mean4, r4 = run_case('det4 K=22', ev4, H4, hj4['v'], 22, 700, pred4,
                              tmpl=dict(grid=tz4['grid'], tmpl_x=tz4['tmpl_x'],
                                        tmpl_y=tz4['tmpl_y']), gain_unit=True)
    results['det4_K22'] = r4
    axs[2].plot(uk4, mean4, 'o-', ms=3)
    axs[2].axvline(25000 / hj4['v'], ls=':', color='k', label='25 mm')
    axs[2].axvline(29000 / hj4['v'], ls='--', color='k', label='29 mm')
    axs[2].set_title(f'det4 (v={hj4["v"]:.1f})')
    axs[2].set_xlabel('u [ns]'); axs[2].legend(); axs[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(B3, 'tail_and_dets.png'), dpi=110)
    json.dump(results, open(os.path.join(B3, 'tail_and_dets.json'), 'w'), indent=1)
    print('saved tail_and_dets.png')

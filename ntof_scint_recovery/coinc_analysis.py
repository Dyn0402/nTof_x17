#!/usr/bin/env python3
"""
coinc_analysis.py -- the fixed-reference layer of the recovery: template-fit
pulses, wall-plastic coincidences and the two ends of each wall bar, from the v2
pulse histograms (pulses.py --hist, condor -> flash_tail_pulsehist_v2).

    python -m ntof_scint_recovery.coinc_analysis [--hists DIR]

Production runs only; a run enters an arm's coincidence sums only if every wall
end and both plastic bars of that arm found pulses (live).  Writes
``out_dir('coinc')/coinc_curves.npz`` and ``coinc_recovery.csv``:

* **shape** -- median residual rms / amplitude of the template fit (isolated
  pulses), per family and class, against time.  ``t_shape``: the first time after
  which it stays within +25 % of its 1-10 ms value.
* **fit scale** -- the 90 % quantile of fitted amplitudes above the family floor,
  over its 1-10 ms value (as pulse_analysis, from the template fit).
* **plastic | wall** -- plastic pulses with a wall pulse of the same arm 5-25 ns
  before them, accidentals subtracted with the sideband: their median and their
  upper edge (amplitude above which 5 % of the net pairs lie), over 1-10 ms.
* **bar ends** -- per wall bar and position (dt between the ends, 2 ns bins; to mm
  with ntof_scint_stack's MM-track calibration, ~74 mm/ns, per-bar offsets of up to 7 ns):
  the median ln(A1/A2) against time minus its 1-10 ms value, averaged over the
  central positions: a gain change on ONE end shows here directly, independent
  of what particles arrive.  And the geometric-mean amplitude scale per position.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.pulse_analysis import AMP_FLOOR, cond_quantile, t_within
from ntof_scint_recovery.quick import DETS, PRODUCTION_FROM

CLS = ('par', 'ded')
#: position bands along the wall bars, mm from the bar centre (MM-track calibration)
V_BAND = 300.0
V_EDGES = np.arange(-1500, 1501, 50.0)


def _wall_cal():
    """(arm index, bar) -> (a_dt ns, b_dt ns/mm), ntof_scint_stack's wall position fit."""
    try:
        import pandas as pd
        from sept26_prelim_analysis import paths
        d = pd.read_parquet(paths.spell('scint', 'ana', 'wallpos_cal.parquet'))
        return {('ABCD'.index(r.arm), int(r.grp)): (float(r.dt_icpt), float(r.dt_slope)) for r in d.itertuples()}
    except Exception as e:                      # noqa: BLE001
        print(f'no wall position calibration ({e}); bands in raw dt at 0.0136 ns/mm')
        return {}


CAL = _wall_cal()
PLATEAU = (1e3, 1e4)
SUM_KEYS = ('HF', 'HR', 'HA', 'PW_sig', 'PW_side', 'BAR_r', 'BAR_g', 'BAR_r_acc', 'BAR_g_acc')


def load(hdir: Path, min_run=PRODUCTION_FROM):
    acc, meta, nb, live_arm = {}, None, np.zeros(2), None
    files = sorted(hdir.glob('pulsehist_*.npz'))
    for f in files:
        r = int(f.stem.split('_')[1])
        if r < min_run:
            continue
        z = np.load(f)
        if 'BAR_r' not in z.files:
            continue
        meta = z
        chans = [tuple(c) for c in z['channels']]
        per_ch = z['HA'].sum(axis=(0, 2, 3))
        alive = {c: per_ch[i] > 0 for i, c in enumerate(chans)}
        arm_ok = np.array([all(alive.get((a, e), False) for e in range(1, 9))
                           and all(alive.get((4 + a, b), False) for b in (1, 2)) for a in range(4)])
        for k in SUM_KEYS:
            v = z[k].astype(np.float64)
            if k.startswith(('PW', 'BAR')):
                v = v * arm_ok[None, :, *([None] * (v.ndim - 2))]
            acc[k] = acc.get(k, 0) + v
        nb += (int(z['n_par']), int(z['n_ded']))
    acc.update({k: meta[k] for k in ('t_edges_us', 'a_edges', 'rf_edges', 'tc_edges_us', 'dt_edges', 'lr_edges',
                                     'pw_side_scale', 'channels')})
    acc['nb'] = nb
    return acc


def med_from_hist(h, edges):
    ctr = 0.5 * (edges[:-1] + edges[1:])
    c = np.cumsum(h, axis=-1)
    tot = c[..., -1]
    idx = np.argmax(c >= 0.5 * np.maximum(tot, 1)[..., None], axis=-1)
    v = ctr[idx].astype(float)
    v[tot < 30] = np.nan
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--hists', type=Path, default=(out_dir('pulsehist_v3') if any(out_dir('pulsehist_v3').glob('*.npz'))
                                                    else out_dir('pulsehist_v2')))
    ap.add_argument('--min-run', type=int, default=PRODUCTION_FROM)
    a = ap.parse_args(argv)
    H = load(a.hists, a.min_run)
    te, ae, rfe = H['t_edges_us'], H['a_edges'], H['rf_edges']
    tc = np.sqrt(te[:-1] * te[1:])[:100]
    tce = H['tc_edges_us']
    tcc = np.sqrt(tce[:-1] * tce[1:])
    nlc = 25                                  # coarse log bins; then [1e4, 3e4], ref
    chans = [tuple(c) for c in H['channels']]
    out = {'tc_us': tc, 'tcc_us': tcc[:nlc], 'dt_edges': H['dt_edges'], 'lr_edges': H['lr_edges'], 'a_edges': ae}
    rows = []
    pl = (tc > PLATEAU[0]) & (tc < PLATEAU[1])
    plc = (tcc[:nlc] > PLATEAU[0]) & (tcc[:nlc] < PLATEAU[1])
    rfc = np.sqrt(rfe[1:] * np.maximum(rfe[:-1], rfe[1] / 2))
    for fam in ('WAL', 'PSS', 'LIQ'):
        cis = [i for i, c in enumerate(chans) if DETS[c[0]].startswith(fam)]
        for k, c in enumerate(CLS):
            hr = H['HR'][k, cis].sum(axis=0)
            hf = H['HF'][k, cis].sum(axis=0)
            c_ = np.cumsum(hr, axis=1)
            tot = c_[:, -1]
            idx = np.argmax(c_ >= 0.5 * np.maximum(tot, 1)[:, None], axis=1)
            mrf = rfc[idx].astype(float)
            mrf[tot < 50] = np.nan
            mrf = mrf[:100]
            g = cond_quantile(hf, ae, AMP_FLOOR[fam], 0.9)
            gpl = np.nanmedian(g[:100][pl])
            out[f'{fam}_{c}_resfrac'] = mrf
            out[f'{fam}_{c}_fitscale'] = g[:100] / gpl
            out[f'{fam}_{c}_fitscale_ref'] = g[-1] / gpl
            rp = np.nanmedian(mrf[pl])
            rows.append(dict(what='shape', fam=fam, cls=c, plateau=rp,
                             t_us=t_within(tc, mrf, 0, 1.25 * rp)))
            # robust to a small late bump: 80 % of the early excess over the plateau gone
            early = np.nanmax(mrf[(tc > 0.5) & (tc < 100)])
            rows.append(dict(what='shape80', fam=fam, cls=c, plateau=rp,
                             t_us=t_within(tc, mrf, 0, rp + 0.2 * (early - rp))))
            rows.append(dict(what='fit_scale90', fam=fam, cls=c, plateau=gpl,
                             t_us=t_within(tc, out[f'{fam}_{c}_fitscale'], 0.9, 1e9)))
    # plastic | wall, pooled over arms and bars
    for k, c in enumerate(CLS):
        net = H['PW_sig'][k].sum(axis=(0, 1)) - H['pw_side_scale'] * H['PW_side'][k].sum(axis=(0, 1))   # (t, amp)
        net = np.clip(net, 0, None)
        sig_n = H['PW_sig'][k].sum(axis=(0, 1, 3))
        acc_n = H['pw_side_scale'] * H['PW_side'][k].sum(axis=(0, 1, 3))
        med = cond_quantile(net, ae, 20, 0.5, nmin=50)
        edge = cond_quantile(net, ae, 20, 0.95, nmin=50)
        out[f'pw_{c}_net'] = net
        out[f'pw_{c}_purity'] = (sig_n - acc_n) / np.maximum(sig_n, 1)
        for nm, v in (('median', med), ('edge95', edge)):
            p0 = np.nanmedian(v[:nlc][plc])
            out[f'pw_{c}_{nm}'] = v[:nlc] / p0
            out[f'pw_{c}_{nm}_ref'] = v[-1] / p0
            out[f'pw_{c}_{nm}_mV'] = p0
            rows.append(dict(what=f'plastic_given_wall_{nm}', fam='PSS', cls=c, plateau=p0,
                             t_us=t_within(tcc[:nlc], v[:nlc] / p0, 0.9, 1.1)))
    # the plastic upper edge per bar (it follows each bar's gain): the common-mode gain reference
    for k, c in enumerate(CLS):
        per = np.full((4, 2, nlc), np.nan)
        for arm in range(4):
            for bar in range(2):
                net = np.clip(H['PW_sig'][k, arm, bar] - H['pw_side_scale'] * H['PW_side'][k, arm, bar], 0, None)
                e = cond_quantile(net, ae, 20, 0.95, nmin=40)[:nlc]
                per[arm, bar] = e / np.nanmedian(e[plc])
        out[f'pw_{c}_edge_per_bar'] = per
        mean = np.nanmean(per.reshape(8, -1), axis=0)
        out[f'pw_{c}_edge_bar_mean'] = mean
        for tt in (10, 100, 300):
            j = int(np.argmin(np.abs(tcc[:nlc] - tt)))
            rows.append(dict(what=f'plastic_edge_per_bar_at_{tt}us', fam='PSS', cls=c,
                             plateau=float(np.nanstd(per.reshape(8, -1)[:, j])), t_us=float(mean[j])))
    # wall bars
    dte = H['dt_edges']
    dtc = 0.5 * (dte[:-1] + dte[1:])
    lre = H['lr_edges']
    central = np.abs(dtc) <= 10
    for k, c in enumerate(CLS):
        net_r = np.clip(H['BAR_r'][k] - H['BAR_r_acc'][k], 0, None)          # (wall, bar, t, dt, lr)
        net_g = np.clip(H['BAR_g'][k] - H['BAR_g_acc'][k], 0, None)
        out[f'bar_{c}_dt'] = net_r.sum(axis=(0, 1, 4))                         # (t, dt): pairs per position
        mlr = med_from_hist(net_r, lre)                                       # (wall, bar, t, dt)
        drift = mlr[:, :, :nlc, :] - np.nanmedian(mlr[:, :, :nlc, :][:, :, plc, :], axis=2, keepdims=True)
        out[f'bar_{c}_lnratio_drift'] = np.nanmean(drift[..., central], axis=-1)   # (wall, bar, t)
        out[f'bar_{c}_lnratio_drift_all'] = np.nanmedian(out[f'bar_{c}_lnratio_drift'].reshape(16, -1), axis=0)
        # geometric-mean scale per position band.  Position along the bar from the
        # scint_stack MM-track calibration (dt = a_dt + b_dt * v, per arm and bar),
        # so the bands are in mm and the same for every bar.
        out[f'bar_{c}_v_hist'] = np.zeros(len(V_EDGES) - 1)
        bands = {'end 1': (-1e9, -V_BAND), 'centre': (-V_BAND / 2, V_BAND / 2), 'end 2': (V_BAND, 1e9)}
        for bn, (vlo, vhi) in bands.items():
            hg = np.zeros((net_g.shape[2], net_g.shape[4]))
            for w in range(4):
                for b in range(4):
                    a_dt, b_dt = CAL.get((w, b), (0.0, -0.0136))
                    v = (dtc - a_dt) / b_dt
                    bm = (v >= vlo) & (v < vhi)
                    hg += net_g[w, b][:, bm, :].sum(axis=1)
                    if bn == 'end 1':
                        out[f'bar_{c}_v_hist'] += np.histogram(v, V_EDGES, weights=net_g[w, b].sum(axis=(0, 2)))[0]
            v90 = cond_quantile(hg, ae, 30, 0.9)
            p0 = np.nanmedian(v90[:nlc][plc])
            out[f'bar_{c}_gm_{bn}'] = v90[:nlc] / p0
            out[f'bar_{c}_gm_{bn}_n'] = hg[:nlc].sum(axis=1)
            rows.append(dict(what=f'bar_gm90_{bn}', fam='WAL', cls=c, plateau=p0,
                             t_us=t_within(tcc[:nlc], v90[:nlc] / p0, 0.9, 1.1)))
        dmax = np.nanmax(np.abs(out[f'bar_{c}_lnratio_drift_all'][tcc[:nlc] > 2]))
        rows.append(dict(what='bar_end_ratio_max_drift_after_2us', fam='WAL', cls=c, plateau=0.0, t_us=float(dmax)))
        bands_ = np.array([out[f'bar_{c}_gm_{bn}'] for bn in ('end 1', 'centre', 'end 2')])
        late3 = tcc[:nlc] > 3
        rows.append(dict(what='bar_band_spread_after_3us', fam='WAL', cls=c, plateau=0.0,
                         t_us=float(np.nanmax(np.nanmax(bands_[:, late3], axis=0) - np.nanmin(bands_[:, late3], axis=0)))))
        rows.append(dict(what='bar_gm_min_after_3us', fam='WAL', cls=c, plateau=0.0,
                         t_us=float(np.nanmin(np.nanmean(bands_[:, late3], axis=0)))))
    o = out_dir('coinc')
    np.savez_compressed(o / 'coinc_curves.npz', **out)
    with open(o / 'coinc_recovery.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'bunches {H["nb"]} -> {o}')
    for r in rows:
        print(f"  {r['what']:34s} {r['fam']} {r['cls']}: plateau {r['plateau']:8.3f}   t {r['t_us']:9.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

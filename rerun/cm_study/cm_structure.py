#!/usr/bin/env python3
"""cm_structure.py -- what the common mode on the June bench FEUs is made of (rerun C9 / W11, 2026-10-23).

Every June production row was taken RAW: no on-FEU pedestal, common-mode or zero suppression (run_config
dream_daq_info, all False), and the decoder does no baseline work.  The common mode (CM) is removed only in software:
by the hits analyzer and by wft.io, both as the plain per-sample median of each 64-channel block (= one DREAM chip =
one connector).  Before choosing a better estimator, this measures the CM itself, on events with no signal:

  * size per block (rms of the per-sample block median), raw noise vs after CM;
  * is it chip-local or FEU-wide: correlation between the 8 blocks, and what is left of each block's CM after
    removing a FEU-wide one (median of the other blocks) -- if little, quiet blocks can stand in for a block that
    holds a track;
  * is it the same for every channel of a chip: per-channel slope against its block's CM;
  * is it shared between the X and Y FEUs of one detector (paired by eventId).

    python cm_structure.py <decoded_x.root> <decoded_y.root> <label> [--n 1500] [--out results/]
"""
import argparse
import json
import os

import numpy as np
import uproot

NB, BW = 8, 64


def load(path, n):
    t = uproot.open(path)['nt']
    a = t.arrays(['eventId', 'amplitude'], entry_stop=n, library='np')
    lens = np.array([len(x) for x in a['amplitude']])
    ns = int(np.bincount(lens // 512).argmax())
    ok = lens == ns * 512
    X = np.stack([x.reshape(ns, 512) for x, k in zip(a['amplitude'], ok) if k]).astype(np.float32)
    return a['eventId'][ok], X


def signal_blocks(X, noise, thr=5.0):
    """(E, NB) bool: block holds a cluster (>= 2 adjacent channels over thr sigma in 2 consecutive samples)."""
    cm = np.median(X.reshape(*X.shape[:2], NB, BW), axis=3)
    U = X - np.repeat(cm, BW, axis=2)
    above = U > thr * noise[None, None, :]
    pers = (above[:, 1:] & above[:, :-1]).any(1)            # (E, 512)
    adj = pers[:, 1:] & pers[:, :-1]
    adj = np.concatenate([adj, np.zeros((len(X), 1), bool)], 1) | np.concatenate([np.zeros((len(X), 1), bool), adj], 1)
    # also flag strongly negative excursions (undershoot of a big pulse)
    neg = ((U < -thr * noise[None, None, :]).sum(1) >= 3)
    return (adj | neg).reshape(len(X), NB, BW).any(2)


def study(path, n):
    eid, X = load(path, n)
    ped = np.median(X[:300], axis=(0, 1))
    X -= ped[None, None, :]
    cm = np.median(X.reshape(*X.shape[:2], NB, BW), axis=3)                 # (E, S, NB)
    U = X - np.repeat(cm, BW, axis=2)
    noise = 1.4826 * np.median(np.abs(U[:300]), axis=(0, 1))
    sig = signal_blocks(X, noise)
    quiet_ev = ~sig.any(1)
    r = dict(file=os.path.basename(path), n_events=int(len(X)), n_samples=int(X.shape[1]),
             frac_events_quiet=float(quiet_ev.mean()), frac_blocks_with_signal=float(sig.mean()))
    Q = X[quiet_ev]
    cq = cm[quiet_ev]                                                       # (Eq, S, NB)
    r['raw_noise_adc_median_ch'] = float(np.median(Q.std(axis=(0, 1))))
    Uq = Q - np.repeat(cq, BW, axis=2)
    r['post_cm_noise_adc_median_ch'] = float(np.median(Uq.std(axis=(0, 1))))
    r['cm_rms_per_block'] = [float(x) for x in cq.std(axis=(0, 1))]
    # FEU-wide part: each block against the median of the OTHER seven
    resid = []
    for b in range(NB):
        others = np.median(np.delete(cq, b, axis=2), axis=2)
        resid.append(float((cq[..., b] - others).std()))
    r['cm_resid_vs_other_blocks_rms'] = resid
    flat = cq.reshape(-1, NB)
    r['cm_block_corr'] = np.corrcoef(flat.T).round(3).tolist()
    # does CM vary within the 32-sample window (per-event constant vs per-sample)?
    r['cm_event_mean_rms'] = float(cq.mean(1).std())
    r['cm_within_event_rms'] = float((cq - cq.mean(1, keepdims=True)).std())
    # per-channel slope against its block CM (quiet events)
    xs = Q.reshape(-1, 512)
    cs = np.repeat(cq.reshape(-1, NB), BW, axis=1)
    cs0 = cs - cs.mean(0)
    k = ((xs - xs.mean(0)) * cs0).sum(0) / np.maximum((cs0 ** 2).sum(0), 1e-9)
    r['ch_slope_pctiles'] = [float(np.percentile(k, p)) for p in (1, 10, 50, 90, 99)]
    res_unit = (xs - cs).std(0)
    res_slope = (xs - xs.mean(0) - k * cs0).std(0)
    r['noise_unit_cm_median'] = float(np.median(res_unit))
    r['noise_slope_cm_median'] = float(np.median(res_slope))
    # slope structure along the chip: mean slope per channel index within the block
    r['slope_by_chip_channel_q'] = [float(np.median(k.reshape(NB, BW)[:, i::16])) for i in range(16)]
    return eid, cm, sig, quiet_ev, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fx'); ap.add_argument('fy'); ap.add_argument('label')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ex, cmx, sx, qx, rx = study(a.fx, a.n)
    ey, cmy, sy, qy, ry = study(a.fy, a.n)
    common, ix, iy = np.intersect1d(ex, ey, return_indices=True)
    both_q = qx[ix] & qy[iy]
    mx = cmx[ix][both_q].mean(2).ravel()           # FEU-average CM per event-sample
    my = cmy[iy][both_q].mean(2).ravel()
    out = dict(label=a.label, x=rx, y=ry, xy_paired_quiet_events=int(both_q.sum()),
               xy_cm_corr=float(np.corrcoef(mx, my)[0, 1]) if len(mx) > 10 else None)
    json.dump(out, open(os.path.join(a.out, f'structure_{a.label}.json'), 'w'), indent=1)
    for v, r in (('x', rx), ('y', ry)):
        print(f"{a.label} {v} {r['file']}: quiet {r['frac_events_quiet']:.2f}  raw {r['raw_noise_adc_median_ch']:.1f}"
              f" -> CM {r['post_cm_noise_adc_median_ch']:.1f} ADC; CM rms/block {np.round(r['cm_rms_per_block'],1)}"
              f"; resid vs other blocks {np.round(r['cm_resid_vs_other_blocks_rms'],1)}; within-event"
              f" {r['cm_within_event_rms']:.1f} event-mean {r['cm_event_mean_rms']:.1f}; slope p1/50/99"
              f" {np.round(r['ch_slope_pctiles'],2)}; noise unit/slope {r['noise_unit_cm_median']:.2f}/"
              f"{r['noise_slope_cm_median']:.2f}")
    print(f"{a.label} X-Y CM corr (FEU-average, quiet in both): {out['xy_cm_corr']}  n={out['xy_paired_quiet_events']}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""bench_driftscan.py -- the bench's arriving-current stack at six drift fields (det3, 6-27).

The bench version of run_71's three fields: det3, resist 490 V, drift 100/300/500/700/
900/1100 V (35-382 V/cm, beam convention 700 V <-> 243 V/cm), 15 min of cosmics each,
FEU 7 = X, FEU 8 = Y.  Straight from decoded_root:
  * pedestal: per-channel median of the first 300 events; noise: 1.4826 MAD after CNS
  * common mode: per-sample median of each 64-channel block with the signal channels
    (max > 6 sigma after a first unmasked pass, +-3 channels) masked out
  * one cluster per view (signal strips grouped by < 3 mm gaps; a second cluster above
    10 % of the first rejects the event), cluster extent <= 12 mm (the ladder is
    contained), no saturation (< 3400 ADC)
  * the sum over ALL strips within the cluster +-3 mm = the arriving current
  * time = sample * 60 ns - t0_abs[view][ftst] (the det3 bundle's trigger calibration,
    same DAQ and same run)
No M3, no hit times, no pulse alignment.

    bench_driftscan.py <bundle_dir>
Output: results/bench_driftscan.json
"""
import json
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa')]

BASE = '/home/dylan/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26'
VOLTS = (100, 300, 500, 700, 900, 1100)
FEU = {'x': 7, 'y': 8}
SNS = 60.0
GRID = np.arange(-400.0, 1900.0, 20.0)
NBOOT = 100


def posmap():
    from qa_config import get_config
    from wft.io import strip_position_map
    cfg = get_config('sat_det3')
    cfg.BASE_PATH = '/home/dylan/x17/cosmic_bench/det3/'
    return strip_position_map(cfg)


def view_rows(path, pos, t0a):
    T = uproot.open(path)['nt']
    A = T.arrays(['eventId', 'amplitude', 'ftst'], library='np')
    amps, ftst = A['amplitude'], A['ftst']
    lens = np.array([len(a) // 512 for a in amps]); ns = int(np.bincount(lens).argmax())
    ok_len = np.flatnonzero(lens == ns)
    ped_stack = np.stack([amps[i].reshape(ns, 512) for i in ok_len[:300]]).astype(np.float32)
    ped = np.median(ped_stack, axis=(0, 1))
    sub = ped_stack - ped
    cm = np.median(sub.reshape(len(sub), ns, 8, 64), axis=3)
    noise = 1.4826 * np.median(np.abs(sub - np.repeat(cm, 64, axis=2)), axis=(0, 1))
    noise = np.maximum(noise, 3.0)
    good = np.isfinite(pos)
    order = np.argsort(np.where(good, pos, np.inf))
    rows = []
    for i in ok_len:
        W = amps[i].reshape(ns, 512).astype(np.float32) - ped          # (ns, 512)
        cm0 = np.median(W.reshape(ns, 8, 64), axis=2)
        W0 = W - np.repeat(cm0, 64, axis=1)
        sig = (W0.max(0) > 6 * noise) & good
        if sig.sum() < 2:
            continue
        mask = sig.copy()
        for d in (1, 2, 3):
            mask[d:] |= sig[:-d]; mask[:-d] |= sig[d:]
        Wm = np.where(mask[None, :], np.nan, W)
        cm = np.nanmedian(Wm.reshape(ns, 8, 64), axis=2)
        W1 = W - np.repeat(np.nan_to_num(cm), 64, axis=1)
        # clusters of signal strips in position
        ch = np.flatnonzero(sig); p = pos[ch]; o = np.argsort(p); ch, p = ch[o], p[o]
        q = W1[:, ch].max(0)
        brk = np.flatnonzero(np.diff(p) > 3.0)
        groups = np.split(np.arange(len(ch)), brk + 1)
        gq = np.array([q[g].sum() for g in groups])
        k = int(np.argmax(gq))
        if (np.delete(gq, k) > 0.1 * gq[k]).any():
            continue
        g = groups[k]
        lo, hi = p[g].min(), p[g].max()
        if hi - lo > 12.0 or W1[:, ch[g]].max() > 3400:
            continue
        win = good & (pos >= lo - 3.0) & (pos <= hi + 3.0)
        s = W1[:, win].sum(1)
        s = s - s[:4].mean()
        t0 = t0a.get(str(int(ftst[i])))
        if t0 is None:
            continue
        t = np.arange(ns) * SNS - t0
        rows.append(np.interp(GRID, t, s, left=np.nan, right=np.nan))
    return np.array(rows)


def stack(R):
    with np.errstate(invalid='ignore'):
        S = np.nanmean(R, axis=0)
    return S - np.nanmean(S[GRID < -150])


def main():
    B = json.load(open(os.path.join(sys.argv[1], 'bundle.json')))
    pm = posmap()
    rng = np.random.default_rng(29)
    res = {'grid': GRID.tolist(), 'volts': list(VOLTS), 'E_Vcm': [round(v / 2.88, 1) for v in VOLTS]}
    for V in VOLTS:
        sd = os.path.join(BASE, f'drift_scan_resist_490V_drift_{V}V', 'decoded_root')
        res[str(V)] = {}
        for view, feu in FEU.items():
            f = [x for x in os.listdir(sd) if x.endswith(f'_{feu:02d}.root')][0]
            R = view_rows(os.path.join(sd, f), pm[feu], B['t0_abs'][view])
            S = stack(R); pk = np.nanmax(S)
            band = np.nanstd([stack(R[rng.integers(0, len(R), len(R))]) / pk for _ in range(NBOOT)], axis=0)
            res[str(V)][view] = dict(n=int(len(R)), stack=(S / pk).tolist(), band=band.tolist(), peak_adc=float(pk))
            print(f'{V:5d} V ({V / 2.88:5.0f} V/cm) {view}: n={len(R):5d} peak {pk:7.1f} ADC  every 100 ns: ' +
                  ' '.join(f'{x:.2f}' for x in (S / pk)[::5]))
    json.dump(res, open(os.path.join(HERE, 'results', 'bench_driftscan.json'), 'w'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""ga_check.py <posmaps.json> <bench_root> <out_dir> -- pre-flight F7: the 14 channels the GA gain map added to the masks.

For each GA channel (and its position neighbours, as a reference) on its own golden row, from the rr1 inputs:
  * flank test on the rr1_cm0 hits (flank_census cuts: both +-1 neighbours > 300 ADC, fired = > 50 ADC);
  * amplitude ratio on those flank events: A_c / mean(A_L, A_R), hits and waveform peak (fa80c06 decode, pedestal +
    plain block-median CM as wft.io), against the view's distribution of the same ratio over every judged channel
    (local median within +-32 strips in position);
  * a waveform display: channel and +-2 neighbours on six flank events.
Verdict per channel: dead if fired fraction < 0.25 (the flank rule); weak if the waveform ratio is < 0.6 x the local
median AND the hits ratio agrees (< 0.6 x); hot if the ratio is > 2 x the local median; else healthy -> drop from the
mask.  Writes ga_check.json and figures/ga_<det>_<view>_<ch>.png.
"""
import glob
import json
import os
import sys

import numpy as np

PITCH = 0.78
FIRE, BIG, MIN_FLANK = 50.0, 300.0, 20
N_PED, CNS = 300, 64
MAX_FILES = 4

GA = {'sat_det3': {'x': {'dead': [], 'hot': [62]}, 'y': {'dead': [41, 510], 'hot': []}},
      'g_det4': {'x': {'dead': [94, 125, 196, 376, 400, 435, 439, 455, 472, 498], 'hot': [274]}}}
RUN_DIR = {'sat_det3': 'june_tests/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',
           'g_det4': 'june_tests/mx17_det4_day_6-24-26/long_run'}


def neighbours(pos):
    L = np.full(512, -1); R = np.full(512, -1)
    ok = np.flatnonzero(np.isfinite(pos)); o = ok[np.argsort(pos[ok])]; ps = pos[o]
    for i, c in enumerate(o):
        if i >= 1 and abs(ps[i] - ps[i - 1] - PITCH) < 0.3 * PITCH:
            L[c] = o[i - 1]
        if i + 1 < len(o) and abs(ps[i + 1] - ps[i] - PITCH) < 0.3 * PITCH:
            R[c] = o[i + 1]
    return L, R, o


def ratios(A, L, R):
    """per channel: (n flank events, fired fraction, median A_c / mean(A_L, A_R)) on events with both neighbours > BIG."""
    n = np.zeros(512, int); f = np.full(512, np.nan); r = np.full(512, np.nan)
    for c in range(512):
        if L[c] < 0 or R[c] < 0:
            continue
        m = (A[:, L[c]] > BIG) & (A[:, R[c]] > BIG)
        n[c] = m.sum()
        if n[c] >= MIN_FLANK:
            f[c] = np.mean(A[m, c] > FIRE)
            r[c] = np.median(A[m, c] / (0.5 * (A[m, L[c]] + A[m, R[c]])))
    return n, f, r


def local_median(v, o):
    ref = np.full(512, np.nan)
    for i, c in enumerate(o):
        nb = o[max(0, i - 32):i + 33]; x = v[nb]; x = x[np.isfinite(x)]
        if len(x):
            ref[c] = np.median(x)
    return ref


def hits_matrix(files):
    import uproot
    rows = []
    for i, fn in enumerate(files):
        a = uproot.open(fn)['hits'].arrays(['eventId', 'channel', 'amplitude'], library='np')
        rows.append((a['eventId'].astype(np.int64) + i * 10_000_000, a['channel'].astype(int), a['amplitude']))
    ev = np.concatenate([x[0] for x in rows]); ch = np.concatenate([x[1] for x in rows]); am = np.concatenate([x[2] for x in rows])
    ue, inv = np.unique(ev, return_inverse=True)
    A = np.zeros((len(ue), 512), np.float32); np.maximum.at(A, (inv, ch), am)
    return A


def wave_peaks(files, keep_wf):
    """peak matrix [n_ev, 512] from decoded waveforms (pedestal + plain block-median CM, as wft.io), and a few events' W."""
    import uproot
    P, W = [], []
    for fn in files:
        t = uproot.open(fn)['nt']
        amp = t.arrays(['amplitude'], library='np')['amplitude']
        lens = np.array([len(a) // 512 for a in amp]); ns = int(np.bincount(lens).argmax())
        good = [a.reshape(ns, 512).astype(np.float32) for a in amp if len(a) == ns * 512]
        st = np.stack(good[:N_PED]); ped = np.median(st, axis=(0, 1))
        for w in good:
            w = w - ped
            cm = np.median(w.reshape(ns, 512 // CNS, CNS), axis=2); w -= np.repeat(cm, CNS, axis=1)
            P.append(w.max(0))
            if len(W) < keep_wf:
                W.append(w.T)
    return np.array(P, np.float32), W


def display(path, title, Wpk, Wall, c, L2, pos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o = sorted([x for x in L2 if x >= 0], key=lambda x: pos[x])
    sel = [i for i in range(len(Wall)) if all(Wpk[i, x] > BIG for x in o if x != c and abs(pos[x] - pos[c]) < 1.2 * PITCH)][:6]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for k, i in enumerate(sel):
        a = ax.flat[k]
        for x in o:
            a.plot(Wall[i][x], lw=2.2 if x == c else 1.0, color='k' if x == c else None,
                   label=f'ch {x} ({pos[x] - pos[c]:+.2f} mm)')
        a.set_title(f'event {i}', fontsize=8)
        if k == 0:
            a.legend(fontsize=6)
    for a in ax[-1]:
        a.set_xlabel('sample (60 ns)')
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    return len(sel)


def main():
    pm = json.load(open(sys.argv[1])); root = sys.argv[2]; out = sys.argv[3]
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    res = {}
    for key, views in GA.items():
        info = pm[key]; sd = os.path.join(root, RUN_DIR[key])
        for view, lists in views.items():
            feu = info['feu_x'] if view == 'x' else info['feu_y']
            pos = np.array([np.nan if p is None else p for p in info['pos'][str(feu)]])
            L, R, o = neighbours(pos)
            hf = sorted(glob.glob(os.path.join(sd, 'hits_root_rr1_cm0', f'*_datrun_*_{feu:02d}_hits.root')))[:MAX_FILES]
            df = sorted(glob.glob(os.path.join(sd, 'decoded_root_fa80c06', f'*_datrun_*_{feu:02d}.root')))[:MAX_FILES]
            Ah = hits_matrix(hf)
            Aw, Wall = wave_peaks(df, keep_wf=4000)
            nh, fh, rh = ratios(Ah, L, R); nw, fw, rw = ratios(Aw, L, R)
            refh, refw = local_median(rh, o), local_median(rw, o)
            for kind in ('dead', 'hot'):
                for c in lists[kind]:
                    qh = rh[c] / refh[c]; qw = rw[c] / refw[c]
                    if np.isfinite(fh[c]) and fh[c] < 0.25:
                        verdict = 'dead'
                    elif qw < 0.6 and qh < 0.6:
                        verdict = 'weak'
                    elif qw > 2.0 or qh > 2.0:
                        verdict = 'hot'
                    elif not (np.isfinite(qw) or np.isfinite(qh)):
                        verdict = 'not judged'
                    else:
                        verdict = 'healthy'
                    L2 = [L[L[c]] if L[c] >= 0 else -1, L[c], c, R[c], R[R[c]] if R[c] >= 0 else -1]
                    nd = display(os.path.join(out, 'figures', f'ga_{key}_{view}_{c}.png'),
                                 f'{key} {view} ch {c} (GA: {kind}) -> {verdict}; hits ratio {qh:.2f}, waveform ratio {qw:.2f} '
                                 f'x local median', Aw[:len(Wall)], Wall, c, L2, pos)
                    r = dict(key=key, view=view, feu=feu, ch=c, ga=kind, pos_mm=float(pos[c]),
                             n_flank_hits=int(nh[c]), fired_hits=float(fh[c]), ratio_hits=float(rh[c]), ref_hits=float(refh[c]),
                             n_flank_wf=int(nw[c]), fired_wf=float(fw[c]), ratio_wf=float(rw[c]), ref_wf=float(refw[c]),
                             q_hits=float(qh), q_wf=float(qw), verdict=verdict, n_display=nd,
                             n_events_hits=int(len(Ah)), n_events_wf=int(len(Aw)))
                    res[f'{key}:{view}:{c}'] = r
                    print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(out, 'ga_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

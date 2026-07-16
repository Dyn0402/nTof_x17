"""Duplication-veto test on the wall-plastic coincident amplitude spectra.

Veto rule (per wall hit): flag the hit if a SAME-SIDE channel one group over
(c +- 2, i.e. the odd or even neighbor) has a prompt hit within +-4 ns with
amplitude ratio in [1/3, 3] (the equal-amplitude duplication band).

For each arm and wall channel, accumulate sideband-subtracted coincident
spectra three ways: all hits (as-is), veto survivors, vetoed component.
Also report, per channel, the flagged fraction of signal-window pairs ->
does duplication correlate with the deformed spectra?

Arms: A, D (affected) + B (control). Windows as in 07: |dt_cal|<=8 ns signal,
+20..+120 ns sideband; pss late (tof - tflash > 0.1 ms); wall late as in 06.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/dylan/PycharmProjects/nTof_x17/mx_july_beam_qa')
sys.path.insert(0, str(BASE))
import hitcache

spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224460.root'
W_SIG, SB_LO, SB_HI = 8.0, 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)
DT_MAX = SB_HI + 25
LATE_TOF = 1e5
XT_DT = 4.0
RATIO_LO, RATIO_HI = 1 / 3.0, 3.0
AMP_EDGES = np.geomspace(40, 8e4, 201)
NB = len(AMP_EDGES) - 1
ARMS = ['A', 'D', 'B']

intensity = coinc.bunch_intensity(RUN_FILE)
good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
offs = json.loads((BASE / 'calib' / f'time_offsets_{RUN_FILE.stem}.json').read_text())['stations']
print(f'{len(good_bunches)} good bunches', flush=True)

results = {}
for st in ARMS:
    wal = hitcache.load(RUN_FILE, f'WAL{st}', ['BunchNumber', 'tof', 'detn', 'amp'],
                        good_bunches=good_bunches)
    late = wal['tof'] > 1.2e4 + LATE_TOF
    wal = {k: v[late] for k, v in wal.items()}
    key_w = hitcache.bunch_key(wal['BunchNumber'], wal['tof'])

    pss = hitcache.load(RUN_FILE, f'PSS{st}',
                        ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                        good_bunches=good_bunches)
    late = (pss['tof'] - pss['tflash']) > LATE_TOF
    pss = {k: v[late] for k, v in pss.items()}
    key_p = hitcache.bunch_key(pss['BunchNumber'], pss['tof'])
    print(f'arm {st}: {len(key_w):,} wall / {len(key_p):,} late pss', flush=True)

    # --- duplication flags: same-side neighbor (c +- 2) prompt + ratio band
    flagged = np.zeros(len(key_w), dtype=bool)
    ch_idx = [np.nonzero(wal['detn'] == c + 1)[0] for c in range(8)]
    for c in range(8):
        for c2 in (c - 2, c + 2):
            if not (0 <= c2 < 8):
                continue
            ia, ib = ch_idx[c], ch_idx[c2]
            if not len(ia) or not len(ib):
                continue
            for ri, oi in hitcache.iter_pairs(key_w[ia], key_w[ib],
                                              -XT_DT, XT_DT,
                                              wal['tof'][ia], wal['tof'][ib]):
                ratio = wal['amp'][ib][oi] / np.maximum(wal['amp'][ia][ri], 1)
                ok = (ratio >= RATIO_LO) & (ratio <= RATIO_HI)
                flagged[ia[ri[ok]]] = True
    frac_ch = [flagged[ch_idx[c]].mean() for c in range(8)]
    print(f'  flagged fraction of ALL wall hits: ' +
          ' '.join(f'{f:.3f}' for f in frac_ch), flush=True)

    # --- wall-plastic pairing, offsets applied
    off = np.zeros((8, 2))
    for wc in range(8):
        for pc in range(2):
            off[wc, pc] = offs[st][f'WAL{st}{wc + 1}_PSS{st}{pc + 1}']['offset_ns']

    # (variant: 0=all, 1=survivor, 2=vetoed) x (sig/side) x ch x amp
    h = np.zeros((3, 2, 8, NB))
    n_sig_all = np.zeros(8)
    n_sig_flag = np.zeros(8)
    for ri, oi in hitcache.iter_pairs(key_p, key_w, -DT_MAX, DT_MAX,
                                      pss['tof'], wal['tof']):
        ch_p = pss['detn'][ri] - 1
        ch_w = wal['detn'][oi] - 1
        dt_cal = (wal['tof'][oi] - pss['tof'][ri]) - off[ch_w, ch_p]
        ai = np.clip(np.digitize(wal['amp'][oi], AMP_EDGES) - 1, 0, NB - 1)
        fl = flagged[oi]
        sig_m = np.abs(dt_cal) <= W_SIG
        np.add.at(n_sig_all, ch_w[sig_m], 1)
        np.add.at(n_sig_flag, ch_w[sig_m & fl], 1)
        for j, wm in enumerate((sig_m, (dt_cal >= SB_LO) & (dt_cal <= SB_HI))):
            np.add.at(h, (0, j, ch_w[wm], ai[wm]), 1)
            m = wm & ~fl
            np.add.at(h, (1, j, ch_w[m], ai[m]), 1)
            m = wm & fl
            np.add.at(h, (2, j, ch_w[m], ai[m]), 1)
    results[f'{st}_h'] = h
    results[f'{st}_flagfrac_hits'] = np.array(frac_ch)
    results[f'{st}_flagfrac_sig'] = n_sig_flag / np.maximum(n_sig_all, 1)
    print(f'  flagged fraction of SIGNAL-window pairs: ' +
          ' '.join(f'{f:.3f}' for f in results[f'{st}_flagfrac_sig']), flush=True)

np.savez_compressed(BASE / 'cache' / f'17_dupveto_{RUN_FILE.stem}.npz',
                    amp_edges=AMP_EDGES, sb_scale=SB_SCALE, **results)
print('saved npz', flush=True)

cen = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])
kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.) ** 2)
kern /= kern.sum()
print('\n=== subtracted-spectrum MIP mode: as-is -> survivors (excess kept) ===')
for st in ARMS:
    h = results[f'{st}_h']
    row = []
    for c in range(8):
        vals = []
        for v in (0, 1):
            sub = h[v, 0, c] - SB_SCALE * h[v, 1, c]
            sm = np.convolve(sub, kern, mode='same')
            m = cen > 300
            vals.append((cen[m][np.argmax(sm[m])], sub.sum()))
        row.append(f'{vals[0][0]:.0f}->{vals[1][0]:.0f} ({vals[1][1] / max(vals[0][1], 1):.2f})')
    print(f'WAL{st}: ' + '  '.join(row))

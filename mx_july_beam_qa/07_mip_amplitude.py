"""
07_mip_amplitude.py — Amplitude spectra of true WAL-PSS coincidences -> MIP peaks.

Per arm, pairs every wall-channel hit with every same-arm plastic-bar hit, applies
the per-channel time offsets from calib/time_offsets_<run>.json, and accumulates
amplitude spectra in two dt_cal regions:
    signal    |dt_cal| <= 8 ns
    sideband  +20 ns <= dt_cal <= +120 ns   (positive side only: avoids the
                                             satellite bump at dt_cal ~ -50 ns)
True-coincidence spectrum = signal - sideband * (16/100). Uses high-purity hits
(pss tof > tflash + 0.1 ms), post-recovery bunches.

Also accumulates a 2D (wall amp x pss amp) matrix per arm, same subtraction.

Usage: python 07_mip_amplitude.py [run_file]
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import hitcache

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
CACHE = BASE / 'cache'

W_SIG = 8.0
SB_LO, SB_HI = 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)
DT_MAX = SB_HI + 25
LATE_TOF = 1e5
AMP_EDGES = np.geomspace(40, 8e4, 301)
AMP2D_EDGES = np.geomspace(40, 8e4, 81)


def load_arm(run_file, st, good_bunches):
    wal = hitcache.load(run_file, f'WAL{st}', ['BunchNumber', 'tof', 'detn', 'amp'],
                        good_bunches=good_bunches)
    pss = hitcache.load(run_file, f'PSS{st}',
                        ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                        good_bunches=good_bunches)
    late = (pss['tof'] - pss['tflash']) > LATE_TOF
    pss = {k: v[late] for k, v in pss.items()}
    return wal, pss


def main():
    intensity = coinc.bunch_intensity(RUN_FILE)
    good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
    offs_json = json.loads((BASE / 'calib' / f'time_offsets_{RUN_FILE.stem}.json').read_text())
    print(f'{len(good_bunches)} good bunches; window +-{W_SIG} ns, '
          f'sideband [{SB_LO},{SB_HI}] ns (scale {SB_SCALE:.3f})')

    results = {}
    nb = len(AMP_EDGES) - 1
    for st in 'ABCD':
        off = np.zeros((8, 2))
        for wc in range(8):
            for pc in range(2):
                off[wc, pc] = offs_json['stations'][st][f'WAL{st}{wc + 1}_PSS{st}{pc + 1}']['offset_ns']
        wal, pss = load_arm(RUN_FILE, st, good_bunches)
        print(f'arm {st}: {len(wal["tof"]):,} wall / {len(pss["tof"]):,} late pss hits',
              flush=True)

        t_p, t_w = pss['tof'], wal['tof']
        det_p = pss['detn'].astype(np.int64)
        det_w = wal['detn'].astype(np.int64)
        key_p = hitcache.bunch_key(pss['BunchNumber'], t_p)
        key_w = hitcache.bunch_key(wal['BunchNumber'], t_w)
        nb2 = len(AMP2D_EDGES) - 1
        h_wal = np.zeros((2, 8, nb))     # (sig/side, wall ch, amp)
        h_pss = np.zeros((2, 2, nb))     # (sig/side, bar, amp)
        h_2d = np.zeros((2, nb2, nb2))
        for ri, oi in hitcache.iter_pairs(key_p, key_w, -DT_MAX, DT_MAX, t_p, t_w):
            ch_p = det_p[ri] - 1
            ch_w = det_w[oi] - 1
            dt_cal = (t_w[oi] - t_p[ri]) - off[ch_w, ch_p]
            a_w = wal['amp'][oi]
            a_p = pss['amp'][ri]
            for j, mask in enumerate((np.abs(dt_cal) <= W_SIG,
                                      (dt_cal >= SB_LO) & (dt_cal <= SB_HI))):
                if not mask.any():
                    continue
                aw = np.clip(np.digitize(a_w[mask], AMP_EDGES) - 1, 0, nb - 1)
                ap = np.clip(np.digitize(a_p[mask], AMP_EDGES) - 1, 0, nb - 1)
                h_wal[j] += np.bincount(ch_w[mask] * nb + aw,
                                        minlength=8 * nb).reshape(8, nb)
                h_pss[j] += np.bincount(ch_p[mask] * nb + ap,
                                        minlength=2 * nb).reshape(2, nb)
                aw2 = np.clip(np.digitize(a_w[mask], AMP2D_EDGES) - 1, 0, nb2 - 1)
                ap2 = np.clip(np.digitize(a_p[mask], AMP2D_EDGES) - 1, 0, nb2 - 1)
                h_2d[j] += np.bincount(aw2 * nb2 + ap2,
                                       minlength=nb2 * nb2).reshape(nb2, nb2)
        results[f'{st}_wal'] = h_wal
        results[f'{st}_pss'] = h_pss
        results[f'{st}_2d'] = h_2d
        print(f'  arm {st}: {h_wal[0].sum():,.0f} sig pairs, '
              f'{h_wal[1].sum():,.0f} sideband pairs', flush=True)

    np.savez_compressed(CACHE / f'07_mip_{RUN_FILE.stem}.npz',
                        amp_edges=AMP_EDGES, amp2d_edges=AMP2D_EDGES,
                        sb_scale=SB_SCALE, **results)
    print(f'Cached -> {CACHE / f"07_mip_{RUN_FILE.stem}.npz"}')

    # MIP peak table: mode of smoothed subtracted spectrum above 150 ADC
    cen = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kern /= kern.sum()
    print(f'\nMIP peak candidates (subtracted spectra):')
    print(f'{"channel":12s} {"peak amp":>9s} {"excess hits":>12s}')

    def report(name, sig, sb):
        sub = sig - SB_SCALE * sb
        sm = np.convolve(sub, kern, mode='same')
        m = cen > 150
        pk = cen[m][np.argmax(sm[m])]
        print(f'{name:12s} {pk:9.0f} {sub.sum():12,.0f}')

    for st in 'ABCD':
        for wc in range(8):
            report(f'WAL{st}{wc + 1}', results[f'{st}_wal'][0, wc],
                   results[f'{st}_wal'][1, wc])
        for pc in range(2):
            report(f'PSS{st}{pc + 1}', results[f'{st}_pss'][0, pc],
                   results[f'{st}_pss'][1, pc])


if __name__ == '__main__':
    main()

"""
03_time_offsets.py — Per-channel WAL-PSS time-offset calibration.

For each station (WALX, PSSX) measures the coincidence-peak position dt = t_wall -
t_pss for every (wall channel 1-8) x (pss channel 1-2) combination, using only the
high-purity sample: post-recovery bunches, pss hits >0.1 ms after the gamma flash.
Peak position = sideband-subtracted centroid within +-12 ns of the maximum.

Offsets go to calib/time_offsets_<run>.json; subsequent coincidence analyses
subtract them so all pairs peak at dt' = dt - offset = 0.

Usage: python 03_time_offsets.py [run_file]
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import uproot

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
CALIB = BASE / 'calib'
CALIB.mkdir(exist_ok=True)

STATIONS = ['A', 'B', 'C', 'D']
LATE_TOF = 1e5          # ns after flash: >0.1 ms -> purity ~0.7-0.9
DT_EDGES = coinc.DT_EDGES
DT_CEN = 0.5 * (DT_EDGES[:-1] + DT_EDGES[1:])
SIDE = np.abs(DT_CEN) > 100


def load_station_tree(f, tree_name, good_bunches):
    good = np.asarray(good_bunches)
    out = {k: [] for k in ('BunchNumber', 'tof', 'detn', 'tflash')}
    branches = list(out) if tree_name.startswith('PSS') else ['BunchNumber', 'tof', 'detn']
    for chunk in f[tree_name].iterate(branches, library='np', step_size='300 MB'):
        keep = np.isin(chunk['BunchNumber'], good)
        for k in branches:
            out[k].append(chunk[k][keep])
    arrs = {k: np.concatenate(v) for k, v in out.items() if v}
    order = np.lexsort((arrs['tof'], arrs['BunchNumber']))
    return {k: v[order] for k, v in arrs.items()}


def peak_centroid(hh):
    """Sideband-subtracted centroid and rms within +-12 ns of the peak bin."""
    base = hh[SIDE].mean()
    sub = hh - base
    ipk = np.argmax(sub)
    win = np.abs(DT_CEN - DT_CEN[ipk]) <= 12
    w = np.clip(sub[win], 0, None)
    if w.sum() < 50:
        return np.nan, np.nan, 0.0
    x = DT_CEN[win]
    mu = np.average(x, weights=w)
    rms = np.sqrt(np.average((x - mu) ** 2, weights=w))
    return float(mu), float(rms), float(w.sum())


def main():
    f = uproot.open(RUN_FILE)
    idx = f['index'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    intensity = idx['PulseIntensity'][np.argsort(idx['BunchNumber'])]
    good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
    print(f'{len(good_bunches)} good bunches')

    offsets = {}
    hists = {}
    for st in STATIONS:
        wal = load_station_tree(f, f'WAL{st}', good_bunches)
        pss = load_station_tree(f, f'PSS{st}', good_bunches)
        print(f'station {st}: {len(wal["tof"]):,} wall / {len(pss["tof"]):,} pss hits', flush=True)

        late = (pss['tof'] - pss['tflash']) > LATE_TOF
        h = np.zeros((8, 2, len(DT_CEN)))
        for bn in good_bunches:
            bn = int(bn)
            pa, pb = np.searchsorted(pss['BunchNumber'], [bn, bn + 1])
            wa, wb = np.searchsorted(wal['BunchNumber'], [bn, bn + 1])
            if pa == pb or wa == wb:
                continue
            sel = late[pa:pb]
            t_pss = pss['tof'][pa:pb][sel]
            ch_pss = pss['detn'][pa:pb][sel]
            t_wal, ch_wal = wal['tof'][wa:wb], wal['detn'][wa:wb]
            dt, ref_idx = coinc.pair_dts(t_pss, t_wal)
            if len(dt) == 0:
                continue
            # recover wall indices from dt and ref: pair_dts returns other-idx implicitly;
            # recompute here for channel tags
            lo = np.searchsorted(t_wal, t_pss - coinc.DT_MAX)
            hi = np.searchsorted(t_wal, t_pss + coinc.DT_MAX)
            counts = hi - lo
            within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            wal_idx = np.repeat(lo, counts) + within
            np.add.at(h, (ch_wal[wal_idx] - 1, ch_pss[ref_idx] - 1,
                          np.clip(np.digitize(dt, DT_EDGES) - 1, 0, len(DT_CEN) - 1)), 1)

        hists[st] = h
        offsets[st] = {}
        for wc in range(8):
            for pc in range(2):
                mu, rms, n = peak_centroid(h[wc, pc])
                offsets[st][f'WAL{st}{wc + 1}_PSS{st}{pc + 1}'] = {
                    'offset_ns': round(mu, 2) if np.isfinite(mu) else None,
                    'rms_ns': round(rms, 2) if np.isfinite(rms) else None,
                    'n_excess': int(n)}

    np.savez_compressed(BASE / 'cache' / f'03_offsets_hists_{RUN_FILE.stem}.npz',
                        dt_edges=DT_EDGES, **{st: hists[st] for st in STATIONS})
    meta = {'run': RUN_FILE.stem, 'definition': 'dt = t_wall - t_pss, subtract offset_ns '
            'so calibrated pairs peak at 0', 'sample': f'tof-tflash > {LATE_TOF:.0f} ns, '
            'post-recovery bunches', 'stations': offsets}
    out = CALIB / f'time_offsets_{RUN_FILE.stem}.json'
    out.write_text(json.dumps(meta, indent=2))
    print(f'\nCalibration -> {out}\n')

    print(f'{"pair":18s} {"offset[ns]":>10s} {"rms[ns]":>8s} {"n_excess":>10s}')
    for st in STATIONS:
        for k, v in offsets[st].items():
            print(f'{k:18s} {str(v["offset_ns"]):>10s} {str(v["rms_ns"]):>8s} {v["n_excess"]:>10d}')


if __name__ == '__main__':
    main()

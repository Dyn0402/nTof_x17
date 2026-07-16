"""
01_signal_qa.py — First-look QA of SiPM wall (WALA-D) and plastic scintillator (PSSA-D)
signals from the official n_TOF processed root files (July 2026 EAR2 X17 beam).

Per channel: amplitude/area spectra, signal-width (fwhm) spectra, time-of-flight
distributions (full 20 ms window + gamma-flash zoom), hits per bunch split by
dedicated/parasitic pulse intensity, saturation and pileup fractions.

Histograms are accumulated in a single chunked pass per tree and cached to npz so
plots can be re-styled without re-reading the 13 GB file.

Usage: python 01_signal_qa.py [run_file]
"""

import sys
from pathlib import Path

import numpy as np
import uproot

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
OUT_DIR = Path(__file__).parent / 'figures' / '01_signal_qa'
CACHE = Path(__file__).parent / 'cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
N_CH = {**{t: 8 for t in WALL_TREES}, **{t: 2 for t in PSS_TREES}}

HIT_BRANCHES = ['detn', 'tof', 'amp', 'area', 'fwhm', 'BunchNumber', 'satuflag', 'pileup1']

# Log-spaced binning: amp threshold is ~50, 16-bit full scale ~65k
AMP_EDGES = np.geomspace(40, 8e4, 301)
AREA_EDGES = np.geomspace(1e1, 1e8, 301)
FWHM_EDGES = np.geomspace(1, 1e4, 201)
TOF_EDGES = np.geomspace(1e3, 2.1e7, 601)          # full acquisition, log
FLASH_EDGES = np.linspace(9e3, 16e3, 701)          # gamma-flash zoom, 10 ns bins
DEDICATED_THRESH = 6e12                            # protons; dedicated ~8.5e12, parasitic ~4.1e12


def channel_hist(detn, values, n_ch, edges):
    """(n_ch, n_bins) histogram of values split by channel number (detn is 1-based)."""
    h, _, _ = np.histogram2d(detn, values, bins=[np.arange(0.5, n_ch + 1), edges])
    return h


def process_tree(f, tree_name, bunch_intensity):
    n_ch = N_CH[tree_name]
    tree = f[tree_name]
    h = {
        'amp': np.zeros((n_ch, len(AMP_EDGES) - 1)),
        'area': np.zeros((n_ch, len(AREA_EDGES) - 1)),
        'fwhm': np.zeros((n_ch, len(FWHM_EDGES) - 1)),
        'tof': np.zeros((n_ch, len(TOF_EDGES) - 1)),
        'flash': np.zeros((n_ch, len(FLASH_EDGES) - 1)),
        'amp_vs_tof': np.zeros((len(TOF_EDGES) - 1, len(AMP_EDGES) - 1)),
    }
    n_bunches = len(bunch_intensity)
    hits_per_bunch = np.zeros((n_ch, n_bunches + 1))
    n_satu = np.zeros(n_ch)
    n_pileup = np.zeros(n_ch)
    n_tot = np.zeros(n_ch)

    for chunk in tree.iterate(HIT_BRANCHES, library='np', step_size='200 MB'):
        detn, tof, amp = chunk['detn'], chunk['tof'], chunk['amp']
        h['amp'] += channel_hist(detn, amp, n_ch, AMP_EDGES)
        h['area'] += channel_hist(detn, chunk['area'], n_ch, AREA_EDGES)
        h['fwhm'] += channel_hist(detn, chunk['fwhm'], n_ch, FWHM_EDGES)
        h['tof'] += channel_hist(detn, tof, n_ch, TOF_EDGES)
        h['flash'] += channel_hist(detn, tof, n_ch, FLASH_EDGES)
        h['amp_vs_tof'] += np.histogram2d(tof, amp, bins=[TOF_EDGES, AMP_EDGES])[0]
        np.add.at(hits_per_bunch, (detn - 1, chunk['BunchNumber']), 1)
        n_satu += np.bincount(detn - 1, weights=chunk['satuflag'] != 0, minlength=n_ch)
        n_pileup += np.bincount(detn - 1, weights=chunk['pileup1'] != 0, minlength=n_ch)
        n_tot += np.bincount(detn - 1, minlength=n_ch)

    h['hits_per_bunch'] = hits_per_bunch[:, 1:]  # BunchNumber is 1-based
    h['n_satu'], h['n_pileup'], h['n_tot'] = n_satu, n_pileup, n_tot
    return h


def main():
    print(f'Reading {RUN_FILE}')
    f = uproot.open(RUN_FILE)
    idx = f['index'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    order = np.argsort(idx['BunchNumber'])
    bunch_intensity = idx['PulseIntensity'][order]
    print(f'{len(bunch_intensity)} bunches, '
          f'{np.sum(bunch_intensity > DEDICATED_THRESH)} dedicated / '
          f'{np.sum(bunch_intensity <= DEDICATED_THRESH)} parasitic')

    results = {'bunch_intensity': bunch_intensity}
    for tn in WALL_TREES + PSS_TREES:
        print(f'  processing {tn} ({f[tn].num_entries:,} hits)...', flush=True)
        h = process_tree(f, tn, bunch_intensity)
        for k, v in h.items():
            results[f'{tn}_{k}'] = v

    np.savez_compressed(CACHE / f'01_qa_{RUN_FILE.stem}.npz',
                        amp_edges=AMP_EDGES, area_edges=AREA_EDGES, fwhm_edges=FWHM_EDGES,
                        tof_edges=TOF_EDGES, flash_edges=FLASH_EDGES, **results)
    print(f'Cached histograms -> {CACHE / f"01_qa_{RUN_FILE.stem}.npz"}')

    # Summary table
    print(f'\n{"tree":6s} {"ch":>3s} {"hits":>12s} {"hits/ded.bunch":>14s} {"med amp":>9s} '
          f'{"satu%":>7s} {"pileup%":>8s}')
    ded = bunch_intensity > DEDICATED_THRESH
    amp_centers = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])
    for tn in WALL_TREES + PSS_TREES:
        for ch in range(N_CH[tn]):
            n = results[f'{tn}_n_tot'][ch]
            cum = np.cumsum(results[f'{tn}_amp'][ch])
            med = amp_centers[np.searchsorted(cum, cum[-1] / 2)] if cum[-1] > 0 else 0
            hpb = results[f'{tn}_hits_per_bunch'][ch][ded].mean()
            print(f'{tn:6s} {ch + 1:3d} {int(n):12,d} {hpb:14.0f} {med:9.0f} '
                  f'{100 * results[f"{tn}_n_satu"][ch] / max(n, 1):7.3f} '
                  f'{100 * results[f"{tn}_n_pileup"][ch] / max(n, 1):8.2f}')


if __name__ == '__main__':
    main()

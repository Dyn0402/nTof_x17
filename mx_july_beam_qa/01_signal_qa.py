"""
01_signal_qa.py — First-look QA of SiPM wall (WALA-D) and plastic scintillator (PSSA-D)
signals from the official n_TOF processed root files (July 2026 EAR2 X17 beam).

Per channel: amplitude/area spectra, signal-width (fwhm) spectra, time-of-flight
distributions (full 20 ms window + gamma-flash zoom), hits per bunch split by
dedicated/parasitic pulse intensity, saturation and pileup fractions.

Histograms are accumulated in a single pass per tree and cached to npz so
plots can be re-styled without re-reading the 13 GB file. Hit data come from
the binary hit cache when present (see hitcache.py / fastread/), else from a
chunked uproot read of the root file.

Usage: python 01_signal_qa.py [run_file]
"""

import sys
from pathlib import Path

import numpy as np

import hitcache

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
OUT_DIR = Path(__file__).parent / 'figures' / '01_signal_qa'
CACHE = Path(__file__).parent / 'cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
LIQ_TREES = ['LIQA', 'LIQB', 'LIQC', 'LIQD']    # liquid scintillators (from run224489)
N_CH = {**{t: 8 for t in WALL_TREES}, **{t: 2 for t in PSS_TREES},
        **{t: 1 for t in LIQ_TREES}}

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


def process_tree(run_file, tree_name, bunch_intensity):
    n_ch = N_CH[tree_name]
    d = hitcache.load(run_file, tree_name, HIT_BRANCHES)
    detn = d['detn'].astype(np.int64)
    tof, amp = d['tof'], d['amp']
    h = {
        'amp': channel_hist(detn, amp, n_ch, AMP_EDGES),
        'area': channel_hist(detn, d['area'], n_ch, AREA_EDGES),
        'fwhm': channel_hist(detn, d['fwhm'], n_ch, FWHM_EDGES),
        'tof': channel_hist(detn, tof, n_ch, TOF_EDGES),
        'flash': channel_hist(detn, tof, n_ch, FLASH_EDGES),
        'amp_vs_tof': np.histogram2d(tof, amp, bins=[TOF_EDGES, AMP_EDGES])[0],
    }
    n_bunches = len(bunch_intensity)
    hits_per_bunch = np.bincount(
        (detn - 1) * (n_bunches + 1) + d['BunchNumber'],
        minlength=n_ch * (n_bunches + 1)).reshape(n_ch, n_bunches + 1)
    h['hits_per_bunch'] = hits_per_bunch[:, 1:].astype(float)  # BunchNumber is 1-based
    h['n_satu'] = np.bincount(detn - 1, weights=d['satuflag'].astype(float), minlength=n_ch)
    h['n_pileup'] = np.bincount(detn - 1, weights=d['pileup1'].astype(float), minlength=n_ch)
    h['n_tot'] = np.bincount(detn - 1, minlength=n_ch).astype(float)
    return h


def main():
    print(f'Reading {RUN_FILE}'
          + (' (hit cache)' if hitcache.cache_dir(RUN_FILE) else ' (uproot)'))
    bunch_intensity = hitcache.bunch_intensity(RUN_FILE)
    print(f'{len(bunch_intensity)} bunches, '
          f'{np.sum(bunch_intensity > DEDICATED_THRESH)} dedicated / '
          f'{np.sum(bunch_intensity <= DEDICATED_THRESH)} parasitic')

    # LIQ trees only exist from run224489 on — process them when present
    d = hitcache.cache_dir(RUN_FILE)
    if d is not None:
        liq = [t for t in LIQ_TREES if (d / f'{t}_bunch.npy').exists()]
    else:
        import uproot
        keys = set(k.split(';')[0] for k in uproot.open(RUN_FILE).keys())
        liq = [t for t in LIQ_TREES if t in keys]

    results = {'bunch_intensity': bunch_intensity}
    for tn in WALL_TREES + PSS_TREES + liq:
        print(f'  processing {tn}...', flush=True)
        h = process_tree(RUN_FILE, tn, bunch_intensity)
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
    for tn in WALL_TREES + PSS_TREES + liq:
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

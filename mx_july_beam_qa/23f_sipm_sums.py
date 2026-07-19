"""23f_sipm_sums.py — SiPM-wall TOP+BOTTOM SUM distributions and half-MIP
thresholds, for the trigger (which fires on the top+bottom sum of each bar
group, not single channels).

Bar group g (g=1..4) = top channel detn 2g-1 (odd) + bottom detn 2g (even);
sum channels: (1+2)(3+4)(5+6)(7+8). Per group we match a top hit to its nearest
bottom hit within +-15 ns (same particle, both ends fire) and SUM the two
amplitudes (mV).

Two distributions per sum channel:
  * Y-88 699 keVee Compton edge of the SUM   (source runs 224476-79, per arm)
  * beam MIP peak of the SUM                  (run 224503, wall top-bottom
    coincidence; the MIP is the Landau bump above the gamma background)
Then the trigger threshold = 0.5 * (MIP-sum peak), per sum channel.

Output: cache/23f_sipm_sums.npz, figures/21_y88/sipm_sums.png,
        calib/sipm_sum_thresholds.json
Usage: python 23f_sipm_sums.py [n_bunches_224503]   (default 500)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hitcache
from adc_mv import mv_factors

BASE = Path(__file__).parent
DATA = Path.home() / 'x17' / 'beam_july' / 'data'
OUT = BASE / 'figures' / '21_y88'
CACHE = BASE / 'cache'
CALIB = BASE / 'calib'

RUN_ARM = {'run224476': 'A', 'run224477': 'B', 'run224478': 'C', 'run224479': 'D'}
TB_MAX = 15.0                              # top-bottom match window (ns)
MV_EDGES = np.linspace(0, 300, 301)        # 1 mV bins for the sum
CEN = 0.5 * (MV_EDGES[:-1] + MV_EDGES[1:])
N503 = int(sys.argv[1]) if len(sys.argv) > 1 else 500
KERN = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); KERN /= KERN.sum()


def sum_spectrum(run, arm, good_bunches=None):
    """Top+bottom summed amplitude (mV) histogram per bar group (4,) for WAL{arm}."""
    fac = mv_factors(run)[f'WAL{arm}']
    d = hitcache.load(run, f'WAL{arm}', ['BunchNumber', 'tof', 'detn', 'amp'],
                      good_bunches)
    mv = d['amp'] * fac[(d['detn'] - 1).astype(int)]
    h = np.zeros((4, len(MV_EDGES) - 1))
    for g in range(4):
        top = d['detn'] == (2 * g + 1)
        bot = d['detn'] == (2 * g + 2)
        if top.sum() == 0 or bot.sum() == 0:
            continue
        kt = hitcache.bunch_key(d['BunchNumber'][top], d['tof'][top])
        kb = hitcache.bunch_key(d['BunchNumber'][bot], d['tof'][bot])
        tt, tb = d['tof'][top], d['tof'][bot]
        mt, mb = mv[top], mv[bot]
        for ri, oi in hitcache.iter_pairs(kt, kb, -TB_MAX, TB_MAX, tt, tb):
            s = mt[ri] + mb[oi]
            h[g] += np.histogram(s, bins=MV_EDGES)[0]
    return h


def mip_bump(sub, lo=38, hi=115):
    """MIP-bump peak of the summed spectrum: the most prominent local maximum in
    [lo, hi] (the MIP Landau, a distinct bump sitting above the falling low-
    energy shoulder). Falls back to the smoothed max if find_peaks finds none."""
    from scipy.signal import find_peaks
    sm = np.convolve(sub, KERN, 'same')
    m = (CEN >= lo) & (CEN <= hi)
    seg = sm[m]
    if seg.max() <= 0:
        return np.nan
    pk, props = find_peaks(seg, prominence=seg.max() * 0.03)
    if len(pk):
        return CEN[m][pk[np.argmax(props['prominences'])]]
    return CEN[m][np.argmax(seg)]


def edge_descent(sub, lo=30, hi=110):
    """699 keVee Compton edge of the Y-88 SUM continuum: steepest fractional
    descent (min of d ln(N)/dA) over [lo, hi]."""
    sm = np.convolve(sub, KERN, 'same')
    m = (CEN > lo) & (CEN < hi)
    if not m.any() or sm[m].max() <= 0:
        return np.nan
    floor = max(sm[m].max() * 1e-2, 1.0)
    d = np.gradient(np.log(np.clip(sm, floor, None)))
    idx = np.where(m)[0]
    j = idx[np.argmin(d[idx])]
    return CEN[j]


def main():
    cachef = CACHE / 'sipm_sums.npz'
    if cachef.exists() and '--recompute' not in sys.argv:
        z = np.load(cachef)
        y88 = {a: z[f'y88_{a}'] for a in 'ABCD'}
        mip = {a: z[f'mip_{a}'] for a in 'ABCD'}
        print('loaded sum spectra from cache')
    else:
        y88 = {arm: sum_spectrum(DATA / f'{run}.root', arm)
               for run, arm in RUN_ARM.items()}
        gb = np.arange(N503)
        mip = {arm: sum_spectrum(DATA / 'run224503.root', arm, gb) for arm in 'ABCD'}
        np.savez_compressed(cachef, mv_edges=MV_EDGES,
                            **{f'y88_{a}': y88[a] for a in 'ABCD'},
                            **{f'mip_{a}': mip[a] for a in 'ABCD'})

    # --- landmarks + thresholds ---
    out = {'note': 'SiPM top+bottom SUM distributions. y88_edge_mv = 699 keVee '
                   'Compton edge of the sum; mip_peak_mv = beam MIP bump of the '
                   'sum (224503); threshold_mv = 0.5 * mip_peak_mv. Groups '
                   '(1+2)(3+4)(5+6)(7+8) = top+bottom of bar-groups 1-4.',
           'channels': {}}
    print(f'\n{"sum ch":8s} {"Y88 edge":>9s} {"MIP peak":>9s} {"0.5*MIP thr":>11s}')
    for arm in 'ABCD':
        for g in range(4):
            ye = edge_descent(y88[arm][g], lo=30, hi=110)
            mp = mip_bump(mip[arm][g], lo=35, hi=130)
            name = f'WAL{arm}_{2*g+1}+{2*g+2}'
            out['channels'][name] = dict(y88_edge_mv=round(float(ye), 1),
                                         mip_peak_mv=round(float(mp), 1),
                                         threshold_mv=round(float(mp) / 2, 1))
            print(f'{name:8s} {ye:9.0f} {mp:9.0f} {mp / 2:11.0f}')
    (CALIB / 'sipm_sum_thresholds.json').write_text(json.dumps(out, indent=2))

    # per-arm summary
    print('\nper-arm median (over 4 sum channels):')
    for arm in 'ABCD':
        mps = [out['channels'][f'WAL{arm}_{2*g+1}+{2*g+2}']['mip_peak_mv'] for g in range(4)]
        print(f'  WAL{arm}: MIP-sum {np.median(mps):.0f} mV -> threshold {np.median(mps)/2:.0f} mV')

    figure(y88, mip)
    print('\n-> calib/sipm_sum_thresholds.json & figures/21_y88/sipm_sums.png')


def figure(y88, mip):
    fig, ax = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    for j, arm in enumerate('ABCD'):
        for g in range(4):
            ax[0, j].plot(CEN, np.convolve(y88[arm][g], np.ones(3) / 3, 'same'),
                          label=f'{2*g+1}+{2*g+2}')
            ax[1, j].plot(CEN, np.convolve(mip[arm][g], np.ones(3) / 3, 'same'))
        ax[0, j].set_title(f'WAL{arm}  Y-88 sum (699 keVee edge)', fontsize=10)
        ax[1, j].set_title(f'WAL{arm}  224503 sum (MIP bump)', fontsize=10)
        ax[0, j].set_yscale('log'); ax[0, j].set_xlim(0, 150)
        ax[1, j].set_xlim(0, 150); ax[1, j].set_xlabel('top+bottom sum [mV]')
        ax[0, j].legend(fontsize=7, title='group'); ax[0, j].grid(alpha=0.3)
        ax[1, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel('Y-88 counts (log)'); ax[1, 0].set_ylabel('MIP net coincidences')
    fig.suptitle('SiPM wall top+bottom SUM: Y-88 Compton edge (top) & 224503 MIP (bottom)')
    fig.tight_layout()
    fig.savefig(OUT / 'sipm_sums.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Where does WAL/PSS saturation show up in the amplitude distribution?

The walls never set `satuflag` (their saturation is a negative undershoot,
outside any found pulse) and PSSD never sets it either. So: is there a hard
ceiling visible in the amplitude spectrum itself -- a pile-up or an edge below
the ~63 800 ADC limit -- that a post-processing flag could catch?

Calibration control: LIQA/LIQD, where `satuflag` is truth.
"""
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARTS = [5, 10, 15]
BASE = '/media/dylan/data/x17/ntof_reproc/v12_liqpileup/run224572_%04d.root'
TREES = [f'{g}{q}' for g in ('WAL', 'PSS', 'LIQ') for q in 'ABCD']
CEIL = 63_800.0
FLASH_END_NS = 1e6          # 1 ms: everything before is flash/early

def load(tree):
    amp, tof, sf, fwhm = [], [], [], []
    for p in PARTS:
        with uproot.open(BASE % p) as fh:
            a = fh[tree].arrays(['amp', 'tof', 'satuflag', 'fwhm'], library='np')
        amp.append(a['amp'].astype(float)); tof.append(a['tof'].astype(float))
        sf.append(a['satuflag'].astype(bool)); fwhm.append(a['fwhm'].astype(float))
    return (np.concatenate(amp), np.concatenate(tof),
            np.concatenate(sf), np.concatenate(fwhm))

print(f'{"tree":5} {"hits":>11} {"phys":>11} | physics-time amp percentiles      '
      f'| max amp      | above ceiling | flash amp p99.9 / max')
rows = {}
for t in TREES:
    amp, tof, sf, fwhm = load(t)
    phys = tof > FLASH_END_NS
    ap, af = amp[phys], amp[~phys]
    pct = np.percentile(ap, [99, 99.9, 99.99]) if ap.size else [np.nan]*3
    fpct = (np.percentile(af, 99.9), af.max()) if af.size else (np.nan, np.nan)
    print(f'{t:5} {amp.size:11,} {phys.sum():11,} | '
          f'p99 {pct[0]:8.0f}  p99.9 {pct[1]:8.0f}  p99.99 {pct[2]:8.0f} | '
          f'{ap.max():9.0f}    | {int((ap > CEIL).sum()):5d} phys / '
          f'{int((amp > CEIL).sum()):6d} all | {fpct[0]:9.0f} / {fpct[1]:11.0f}')
    rows[t] = (amp, tof, sf, fwhm, phys)

# ---- pile-up / edge test: fine-binned spectrum in the approach to the rails ----
print('\nfine-binned spectrum, 2 000-ADC bins, ALL times (flash included).')
print('A hard front-end ceiling would show as a spike or an abrupt edge:')
edges = np.arange(20_000, 70_001, 2_000.0)
hdr = '  '.join(f'{int(e/1000):>3d}k' for e in edges[:-1])
print(f'{"tree":5} {hdr}')
for t in TREES:
    amp = rows[t][0]
    h = np.histogram(amp, bins=edges)[0]
    print(f'{t:5} ' + '  '.join(f'{int(v):>4d}' for v in h))

# ---- figure ----
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
for i, fam in enumerate(('WAL', 'PSS', 'LIQ')):
    axL, axR = axes[i]
    for q in 'ABCD':
        amp, tof, sf, fwhm, phys = rows[f'{fam}{q}']
        b = np.logspace(2, np.log10(max(amp.max(), 1e5)), 160)
        axL.hist(amp[phys], bins=b, histtype='step', label=f'{fam}{q} physics')
        axL.hist(amp[~phys], bins=b, histtype='step', ls=':', alpha=.7)
        m = amp > 5_000
        if m.sum() > 50:
            bb = np.logspace(np.log10(5_000), np.log10(max(amp.max(), 1e5)), 40)
            idx = np.digitize(amp[m], bb)
            xs, ys = [], []
            for k in range(1, bb.size):
                s = idx == k
                if s.sum() >= 10:
                    xs.append(np.sqrt(bb[k-1]*bb[k])); ys.append(np.median(fwhm[m][s]))
            axR.plot(xs, ys, 'o-', ms=3, label=f'{fam}{q}')
    for ax in (axL, axR):
        ax.set_xscale('log'); ax.axvline(CEIL, color='r', ls='--', lw=1)
        ax.axvline(34_600, color='k', ls='-.', lw=1)
    axL.set_yscale('log'); axL.set_xlabel('amp [ADC]'); axL.set_ylabel('hits')
    axL.set_title(f'{fam}: amplitude (solid = physics time, dotted = flash)')
    axL.legend(fontsize=7)
    axR.set_xlabel('amp [ADC]'); axR.set_ylabel('median fwhm [ns]')
    axR.set_title(f'{fam}: pulse width vs amplitude '
                  '(red = ADC ceiling, black = wall front-end limit)')
    axR.legend(fontsize=7)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else 'wal_pss_saturation.png'
fig.savefig(out, dpi=110)
print(f'\nwrote {out}')

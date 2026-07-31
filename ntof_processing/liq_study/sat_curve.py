"""Median pulse width vs amplitude, per channel, in fixed bins.

An unsaturated pulse has an amplitude-independent width, so the saturation onset
is where the median fwhm leaves its plateau. Printed rather than auto-detected:
the shape differs by family (the walls dip before they broaden), and an automatic
"1.2x the plateau" rule mis-called it on both LIQ and PSS. Read the table.

Calibration: the liquids, where satuflag is verified truth, stay flat to 0.1 ns
right up to the ADC ceiling -- so a departure below the ceiling is real. Inside
the flash the liquids stay flat too, so a departure there is not a flash artifact.

    python sat_curve.py [physics|flash]      (default physics)

Produces the tables in FINDINGS_2026-07-30_saturation_walls_plastics.md.
"""
import sys
import numpy as np, uproot

REGION = (sys.argv[1] if len(sys.argv) > 1 else 'physics').lower()
assert REGION in ('physics', 'flash'), REGION
PARTS = [5, 10, 15]
BASE = '/media/dylan/data/x17/ntof_reproc/v12_liqpileup/run224572_%04d.root'
EDGES = np.array([8e3,12e3,16e3,20e3,25e3,30e3,34.6e3,40e3,45e3,50e3,55e3,60e3,63.8e3,1e10])
LAB = ['8-12k','12-16k','16-20k','20-25k','25-30k','30-34.6k','34.6-40k','40-45k',
       '45-50k','50-55k','55-60k','60-63.8k','>63.8k']
for fam in ('WAL','PSS','LIQ'):
    print(f'\n===== {fam}: median fwhm [ns] (n) per amp bin — {REGION} region =====')
    print(f'{"ch":4} {"plateau":>8} ' + ' '.join(f'{l:>13}' for l in LAB))
    for qq in 'ABCD':
        t = f'{fam}{qq}'
        A,W,T = [],[],[]
        for p in PARTS:
            with uproot.open(BASE % p) as fh:
                a = fh[t].arrays(['amp','fwhm','tof'], library='np')
            A.append(a['amp'].astype(float)); W.append(a['fwhm'].astype(float))
            T.append(a['tof'].astype(float))
        amp,fw,tof = np.concatenate(A),np.concatenate(W),np.concatenate(T)
        phys = (tof > 1e6) if REGION == 'physics' else (tof <= 1e6)
        pl = np.median(fw[phys & (amp>=8e3) & (amp<20e3)])
        cells=[]
        for lo,hi in zip(EDGES[:-1],EDGES[1:]):
            m = phys & (amp>=lo) & (amp<hi)
            cells.append(f'{np.median(fw[m]):6.1f}({m.sum():5d})' if m.sum()>=5
                         else f'{"-":>6}({m.sum():5d})')
        print(f'{t:4} {pl:8.1f} ' + ' '.join(f'{c:>13}' for c in cells))

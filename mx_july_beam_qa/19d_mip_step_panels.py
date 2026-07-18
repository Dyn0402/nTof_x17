"""19d_mip_step_panels.py — per-PMT, per-HV-step triple-tagged plastic
spectra on a LINEAR amplitude axis with propagated Poisson errors, AND the
authoritative linear-space MIP (MPV) calibration.

Why linear: the 19b numbers were modes of the LOG-binned spectrum
(dN/dlogA = A dN/dA), which sit systematically 20--40% above the true
linear-space MPV of dN/dA — the quantity a Landau MPV convention means.
This script fits the MPV on linear histograms per step (parabola through the
peak, error-weighted, V >= 1400 V where the peak is clear of the acquisition
threshold), transports each to nominal V with the coincident-median n, and
REWRITES calib/pss_mip_calib_<run>.json with the MPV-based values (the old
log-mode value is kept in field 'mip_mv_logmode' for traceability).

Per panel: x-range 0..5x the expected MPV at that step, 25 linear bins;
net = SS - s_l*SB - s_w*BS + s_w*s_l*BB with per-bin sigma; dashed line =
power-law-expected MPV; triangle = per-step fitted MPV (fit steps only);
gray band = acquisition threshold (~50 ADC).

If the bump tracks the dashed line panel after panel while staying clear of
the gray band, it is a gain-scaling physical peak; if it hugged the gray band
it would be a threshold artifact.

Output: figures/19_triples/steps_linear/mip_steps_<PMT>.png  (8 figures)
Usage: python 19d_mip_step_panels.py [run_stem]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224489'
OUT = BASE / 'figures' / '19_triples' / 'steps_linear'
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(BASE / 'cache' / f'19_triples_{RUN_STEM}.npz')
FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']
MIPCAL = json.loads((BASE / 'calib' / f'pss_mip_calib_{RUN_STEM}.json').read_text())['pmts']
AE = d['amp_edges']
CEN = np.sqrt(AE[:-1] * AE[1:])
SWP, SLQ = float(d['s_wp']), float(d['s_lq'])
LABELS = list(d['step_labels'])
VOLT = {l: (int(l.split('_')[1][:-1]) if l[0] == 's' and l[1].isdigit() else None)
        for l in LABELS}
PANELS = [l for l in LABELS if VOLT[l]] + ['end_nominal']
THRESH_ADC = 50.0                       # acquisition amp threshold (~50 ADC)
NBINS = 25
V_MIN_FIT = 1400
MIP_MEV = 5.05                          # normal-incidence deposit, 2.5 cm PVT


def linear_net(h_fine, edges):
    """(wp, lq, fine-log-bins) -> (net, sigma) on linear bins `edges`."""
    coarse = np.array([np.histogram(CEN, bins=edges, weights=h_fine[i, j])[0]
                       for i in range(2) for j in range(2)]).reshape(2, 2, -1)
    net = (coarse[0, 0] - SLQ * coarse[0, 1]
           - SWP * coarse[1, 0] + SWP * SLQ * coarse[1, 1])
    sig = np.sqrt(coarse[0, 0] + SLQ ** 2 * coarse[0, 1]
                  + SWP ** 2 * coarse[1, 0]
                  + (SWP * SLQ) ** 2 * coarse[1, 1])
    return net, sig


def fit_mpv(cen, net, sig, lo_adc):
    """Error-weighted parabola through the peak region (argmax +- 3 bins),
    peak restricted above lo_adc. Returns MPV (ADC) or nan."""
    ok = cen > lo_adc
    if not ok.any() or net[ok].max() <= 0:
        return np.nan
    ipk = np.where(ok)[0][np.argmax(net[ok])]
    sel = slice(max(ipk - 3, 0), ipk + 4)
    x, y, w = cen[sel], net[sel], 1.0 / np.clip(sig[sel], 1, None) ** 2
    if len(x) < 4:
        return np.nan
    c = np.polyfit(x, y, 2, w=np.sqrt(w))
    if c[0] >= 0:                       # not concave -> fall back to argmax bin
        return float(cen[ipk])
    v = -c[1] / (2 * c[0])
    return float(v) if x[0] <= v <= x[-1] else float(cen[ipk])


new_cal = {}
print(f'{"PMT":7s} {"per-step MPV@nom [ADC]":>36s} {"MPV[ADC]":>9s} '
      f'{"MPV[mV]":>8s} {"spread":>7s} {"logmode[mV]":>11s}')
for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        n_g = MIPCAL[pmt]['n_powerlaw']
        vn = MIPCAL[pmt]['V_nominal']
        mip_old = MIPCAL[pmt]['mip_mv'] / f_mv          # ADC, log-mode (range seed)
        trans = []
        for l in PANELS:
            v = VOLT[l]
            if not v or v < V_MIN_FIT:
                continue
            mip_exp = mip_old * (v / vn) ** n_g
            edges = np.linspace(0, 5 * mip_exp, NBINS + 1)
            h = d['pss_amp'][LABELS.index(l), ai, :, :, b]
            net, sig = linear_net(h, edges)
            cen_c = 0.5 * (edges[:-1] + edges[1:])
            mpv = fit_mpv(cen_c, net, sig, 1.5 * THRESH_ADC)
            if np.isfinite(mpv):
                trans.append(mpv * (vn / v) ** n_g)
        mpv_nom = float(np.exp(np.mean(np.log(trans))))
        spread = float(np.std(np.log(trans)))
        new_cal[pmt] = dict(V_nominal=vn, n_powerlaw=n_g,
                            mip_mv=round(mpv_nom * f_mv, 2),
                            mv_per_mev=round(mpv_nom * f_mv / MIP_MEV, 3),
                            log_spread=round(spread, 3),
                            mip_mv_logmode=MIPCAL[pmt]['mip_mv'])
        print(f'{pmt:7s} '
              + ' '.join(f'{t:7.0f}' for t in trans).rjust(36)
              + f' {mpv_nom:9.0f} {mpv_nom * f_mv:8.1f} {100 * spread:6.0f}%'
              + f' {MIPCAL[pmt]["mip_mv"]:11.1f}')

cal_path = BASE / 'calib' / f'pss_mip_calib_{RUN_STEM}.json'
cal = json.loads(cal_path.read_text())
cal['method'] = ('linear-space MPV (error-weighted parabola through the peak) '
                 'of WALxPSSxLIQ triple-tagged plastic spectra per HV step '
                 f'(V>={V_MIN_FIT}), transported to nominal V with the '
                 'coincident-median power-law n; dE assumes normal incidence '
                 'through 2.5 cm PVT. mip_mv_logmode preserves the earlier '
                 'log-binned-mode estimate (biased high, superseded).')
cal['pmts'] = new_cal
cal_path.write_text(json.dumps(cal, indent=2))
print(f'-> {cal_path} (mip_mv now = linear MPV)')

for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        n_g = new_cal[pmt]['n_powerlaw']
        vn = new_cal[pmt]['V_nominal']
        mip_nom = new_cal[pmt]['mip_mv'] / f_mv          # ADC (linear MPV)
        fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
        for k, l in enumerate(PANELS):
            ax = axes.flat[k]
            v = VOLT[l] or vn
            mip_exp = mip_nom * (v / vn) ** n_g          # ADC at this step
            edges = np.linspace(0, 5 * mip_exp, NBINS + 1)
            h = d['pss_amp'][LABELS.index(l), ai, :, :, b]   # (wp, lq, fine)
            net, sig = linear_net(h, edges)
            cen_c = 0.5 * (edges[:-1] + edges[1:]) * f_mv
            ax.errorbar(cen_c, net, yerr=sig, fmt='o-', ms=2.5, lw=1,
                        elinewidth=0.7, color='steelblue', ecolor='gray')
            ax.axvline(mip_exp * f_mv, color='crimson', lw=1.3, ls='--')
            if VOLT[l] and VOLT[l] >= V_MIN_FIT:
                mpv = fit_mpv(cen_c / f_mv, net, sig, 1.5 * THRESH_ADC)
                if np.isfinite(mpv):
                    ax.plot(mpv * f_mv, 0, marker='^', ms=9, color='darkorange',
                            clip_on=False, zorder=5)
            ax.axvspan(0, THRESH_ADC * f_mv, color='gray', alpha=0.25, lw=0)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(f'{v} V' if VOLT[l] else f'nominal ({vn} V)',
                         fontsize=10,
                         color='black' if VOLT[l] else 'crimson')
            ax.text(0.97, 0.92, f'n={net.sum():,.0f}', transform=ax.transAxes,
                    ha='right', fontsize=8)
            ax.tick_params(labelsize=7)
            if k >= 5:
                ax.set_xlabel('plastic amplitude [mV]', fontsize=8)
            if k % 5 == 0:
                ax.set_ylabel('net triples / bin', fontsize=8)
        fig.suptitle(f'{RUN_STEM} {pmt}: triple-tagged plastic spectrum per HV '
                     f'step (linear axis, x-range 0..5x expected MIP; dashed = '
                     f'power-law-expected MIP, gray = acquisition threshold)',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f'mip_steps_{pmt}.png', dpi=140)
        plt.close(fig)
        print(f'{pmt} done')
print(f'-> {OUT}')

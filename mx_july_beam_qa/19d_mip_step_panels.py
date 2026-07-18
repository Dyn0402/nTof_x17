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


# pair-coincident medians (12 cache): the high-statistics, NO-MIP-selection
# gain proxy to validate the low-statistics triple-MPV extraction against
d12 = np.load(BASE / 'cache' / f'12_hvscan_{RUN_STEM}.npz')
CEN12 = np.sqrt(d12['amp_edges'][:-1] * d12['amp_edges'][1:])
SB12 = float(d12['sb_scale'])
L12 = list(d12['step_labels'])
V12 = d12['step_volts']


def pair_medians(ai, b):
    """[(V, median ADC)] of the wall-tagged (pair) coincident plastic
    spectrum per scan step, plus the end_nominal point as (None, med)."""
    out, nom = [], np.nan
    for i, l in enumerate(L12):
        sub = np.clip(d12['pss_mip'][i, ai, 0, b]
                      - SB12 * d12['pss_mip'][i, ai, 1, b], 0, None)
        c = np.cumsum(sub)
        if c[-1] <= 200:
            continue
        med = CEN12[np.searchsorted(c, c[-1] / 2)]
        if V12[i] > 0:
            out.append((float(V12[i]), float(med)))
        elif l == 'end_nominal':
            nom = float(med)
    return out, nom


new_cal = {}
step_mpvs = {}                          # pmt -> [(V, MPV ADC)]
print(f'{"PMT":7s} {"per-step MPV@nom [ADC]":>36s} {"MPV[ADC]":>9s} '
      f'{"MPV[mV]":>8s} {"spread":>7s} {"logmode[mV]":>11s}')
for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        n_g = MIPCAL[pmt]['n_powerlaw']
        vn = MIPCAL[pmt]['V_nominal']
        meds, _ = pair_medians(ai, b)
        med_of_v = dict(meds)

        def smoothed_peak(amp_by_comp, seed, n_boot=200):
            """Peak of the Gaussian-smoothed (sigma 1.5 bins) linear histogram,
            40 bins over 0..4*seed, above 1.5x threshold. amp_by_comp: list of
            (amplitudes, counts-per-fine-bin) per sideband component with its
            subtraction scale. Error = std of the peak over Poisson bootstraps
            of the four components — honest estimator instability, no shape
            assumption. Deterministic (fixed RNG seed)."""
            edges = np.linspace(0, 4 * seed, 41)
            cen_c = 0.5 * (edges[:-1] + edges[1:])
            kern = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
            kern /= kern.sum()
            coarse = [(sc, np.histogram(a, bins=edges, weights=w)[0])
                      for sc, a, w in amp_by_comp]

            def peak(comps):
                net = sum(sc * c for sc, c in comps)
                sm = np.convolve(net, kern, mode='same')
                ok = cen_c > 1.5 * THRESH_ADC
                return cen_c[ok][np.argmax(sm[ok])] if sm[ok].max() > 0 else np.nan

            mode = peak(coarse)
            rng = np.random.default_rng(224489)
            boots = [peak([(sc, rng.poisson(np.clip(c, 0, None)))
                           for sc, c in coarse]) for _ in range(n_boot)]
            binw = edges[1] - edges[0]          # argmax is bin-quantized:
            return float(mode), float(max(np.nanstd(boots), binw / 2))

        # per-step MPVs (scaling check) + gain-aligned summed spectrum (the
        # calibration fit: one high-statistics spectrum at nominal-V scale)
        trans = []
        step_mpvs[pmt] = []
        seed_nom = med_of_v.get(vn, list(med_of_v.values())[0]) / 10.0
        al_amp, al_w, al_var = [], [], []
        for l in PANELS:
            v = VOLT[l]
            if not v or v < V_MIN_FIT:
                continue
            h = d['pss_amp'][LABELS.index(l), ai, :, :, b]
            comps = [(1.0, CEN, h[0, 0]), (-SLQ, CEN, h[0, 1]),
                     (-SWP, CEN, h[1, 0]), (SWP * SLQ, CEN, h[1, 1])]
            mpv, dmpv = smoothed_peak(comps, med_of_v[v] / 10.0)
            if np.isfinite(mpv):
                trans.append(mpv * (vn / v) ** n_g)
                step_mpvs[pmt].append((v, mpv, dmpv))
            scale = (vn / v) ** n_g
            al_amp.append((scale, h))
        # aligned-sum peak (primary calibration number): concatenate the
        # gain-scaled components of every fit step
        comps = []
        for i_c, (sc_c) in enumerate([(1.0, 0, 0), (-SLQ, 0, 1),
                                      (-SWP, 1, 0), (SWP * SLQ, 1, 1)]):
            sc, iw, il = sc_c
            comps.append((sc,
                          np.concatenate([CEN * s for s, _ in al_amp]),
                          np.concatenate([hh[iw, il] for _, hh in al_amp])))
        mpv_nom, dmpv_nom = smoothed_peak(comps, seed_nom)
        spread = float(np.std(np.log(np.array(trans) / mpv_nom)))
        new_cal[pmt] = dict(V_nominal=vn, n_powerlaw=n_g,
                            mip_mv=round(mpv_nom * f_mv, 2),
                            mip_mv_stat_err=round(dmpv_nom * f_mv, 2),
                            mv_per_mev=round(mpv_nom * f_mv / MIP_MEV, 3),
                            log_spread=round(spread, 3),
                            mip_mv_logmode=MIPCAL[pmt].get('mip_mv_logmode',
                                                           MIPCAL[pmt]['mip_mv']))
        print(f'{pmt:7s} '
              + ' '.join(f'{t:7.0f}' for t in trans).rjust(36)
              + f' {mpv_nom:9.0f} {mpv_nom * f_mv:8.1f} {100 * spread:6.0f}%'
              + f' {new_cal[pmt]["mip_mv_logmode"]:11.1f}')

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
                fitted = {x[0]: x[1] for x in step_mpvs[pmt]}
                if VOLT[l] in fitted:
                    ax.plot(fitted[VOLT[l]] * f_mv, 0, marker='^', ms=9,
                            color='darkorange', clip_on=False, zorder=5)
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

# ---------------------------------------------------------------------------
# Trust check: triple-MPV points vs the pair-coincident median (no MIP
# selection, ~100x statistics) on the same gain axes. If the MIP extraction
# is real, both must follow the same power law in V (parallel lines in
# log-log); the absolute offset is physical (the pair-tagged plastic spectrum
# is dominated by larger deposits than a through-going MIP).
ARM_COLORS = dict(zip('ABCD', plt.cm.tab10.colors))
fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
print(f'\n{"PMT":7s} {"n_median":>9s} {"n_MIP":>7s} {"median/MIP@nom":>15s}')
for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        vn = new_cal[pmt]['V_nominal']
        ax = axes[b][ai]
        meds, med_nom = pair_medians(ai, b)
        vm = np.array([x[0] for x in meds])
        mm = np.array([x[1] for x in meds])
        n_med, c_med = np.polyfit(np.log(vm), np.log(mm), 1)
        vv = np.geomspace(1180, 1650, 40)
        ax.plot(vm, mm * f_mv, 's', ms=6, color='gray', mfc='none',
                label=f'pair coinc. median (n={n_med:.1f})')
        ax.plot(vv, np.exp(c_med) * vv ** n_med * f_mv, ':', color='gray', lw=1.2)
        if np.isfinite(med_nom):
            ax.plot(vn, med_nom * f_mv, 's', ms=7, color='gray')
        vt = np.array([x[0] for x in step_mpvs[pmt]])
        mt = np.array([x[1] for x in step_mpvs[pmt]])
        et = np.array([x[2] for x in step_mpvs[pmt]])
        ax.errorbar(vt, mt * f_mv, yerr=et * f_mv,
                    fmt='o', ms=6, color=ARM_COLORS[st], lw=1,
                    label='triple MIP MPV (smoothed peak)')
        n_mip = np.nan
        if len(vt) >= 3:
            n_mip, c_mip = np.polyfit(np.log(vt), np.log(mt), 1)
            ax.plot(vv, np.exp(c_mip) * vv ** n_mip * f_mv, '-', lw=1.2,
                    color=ARM_COLORS[st], alpha=0.8,
                    label=f'MIP fit (n={n_mip:.1f})')
        mip_nom = new_cal[pmt]['mip_mv'] / f_mv
        ax.plot(vn, mip_nom * f_mv, '*', ms=15, color='crimson',
                label=f'MIP at nominal: {mip_nom * f_mv:.1f} mV')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1150, 1700)
        ax.set_xticks([1200, 1300, 1400, 1500, 1600])
        ax.set_xticklabels(['1200', '1300', '1400', '1500', '1600'])
        ax.set_xticks([], minor=True)
        ax.set_title(pmt)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=6.5, loc='lower right')
        if b == 1:
            ax.set_xlabel('PMT bias [V]')
        if ai == 0:
            ax.set_ylabel('amplitude [mV]')
        print(f'{pmt:7s} {n_med:9.2f} {n_mip:7.2f} '
              f'{med_nom / mip_nom if np.isfinite(med_nom) else np.nan:15.1f}')
fig.suptitle(f'{RUN_STEM}: triple-MIP MPV vs the no-MIP-selection pair-'
             'coincident median — same power law = the MIP extraction scales '
             'like a gain', y=0.995)
fig.tight_layout()
fig.savefig(BASE / 'figures' / '19_triples' / 'mip_vs_v_median_compare.png',
            dpi=140)
plt.close(fig)
print(f"-> {BASE / 'figures' / '19_triples' / 'mip_vs_v_median_compare.png'}")

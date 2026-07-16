"""
12b_hv_scan_plots.py — Figures + HV equalization table from the 12 cache
(plastic PMT HV scan, run224466).

Figures (figures/12_hv_scan/):
  scan_timeline.png      HV staircase vs wall-clock, gating regions, protons/step
  gain_curves.png        coincident (wall-tagged, sideband-subtracted) plastic
                         median vs measured HV, log-log, per PMT, per pass,
                         with power-law fits and the end_nominal points
  rate_curves.png        late plastic hits per e10 protons vs HV (plateau check)
  wall_mip_stability.png wall MIP peak (ch-summed) vs scan step per arm
  coinc_spectra.png      coincident plastic spectra per voltage (pass 1), 8 PMTs,
                         log amplitude axis
  coinc_spectra_linear.png  same as densities on a linear amplitude axis, with
                         arrows marking each spectrum's median (the calibration
                         observable)
  equalization.png       how the equalization works: each PMT slides along its
                         own fitted power law to the fleet geometric-mean target

Also prints an HV equalization table: per-PMT voltage that would bring every
PMT's coincident median to the fleet geometric mean, using that PMT's own
fitted power-law index.

Usage: python 12b_hv_scan_plots.py [run_stem]
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224466'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '12_hv_scan'
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(BASE / 'cache' / f'12_hvscan_{RUN_STEM}.npz')
CEN = np.sqrt(d['amp_edges'][:-1] * d['amp_edges'][1:])
LABELS = list(d['step_labels'])
VOLTS = d['step_volts']
PASS = d['step_pass']
SB = float(d['sb_scale'])
N_STEP = len(LABELS)
NOMINAL_V = {('A', 0): 1325, ('A', 1): 1275, ('B', 0): 1325, ('B', 1): 1300,
             ('C', 0): 1300, ('C', 1): 1300, ('D', 0): 1300, ('D', 1): 1300}
PMT_NAME = {(ai, b): f'{st}{"LR"[b]}' for ai, st in enumerate('ABCD') for b in range(2)}
ARM_COLORS = dict(zip('ABCD', plt.get_cmap('tab10').colors))
_FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']
MV = {(ai, b): _FAC[f'PSS{st}'][str(b + 1)]          # mV per ADC count, per PMT
      for ai, st in enumerate('ABCD') for b in range(2)}
MV_MEAN = float(np.mean(list(MV.values())))


def coinc_median(i, ai, b):
    sub = np.clip(d['pss_mip'][i, ai, 0, b] - SB * d['pss_mip'][i, ai, 1, b], 0, None)
    c = np.cumsum(sub)
    return CEN[np.searchsorted(c, c[-1] / 2)] if c[-1] > 200 else np.nan


MED = np.array([[[coinc_median(i, ai, b) for i in range(N_STEP)]
                 for b in range(2)] for ai in range(4)])   # (arm, bar, step)


def fit_powerlaw(ai, b):
    """Combined-pass fit ln(med) = n ln(V) + c (passes agree; see per-pass table)."""
    ii = np.where((PASS == 1) | (PASS == 2))[0]
    v, m = VOLTS[ii], MED[ai, b, ii]
    ok = np.isfinite(m)
    return np.polyfit(np.log(v[ok]), np.log(m[ok]), 1)


def gain_curves():
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            ax = axes[b][ai]
            f_mv = MV[(ai, b)]
            for pas, mk, fill in ((1, 'o', 'none'), (2, 's', None)):
                ii = np.where(PASS == pas)[0]
                ax.plot(VOLTS[ii], MED[ai, b, ii] * f_mv, mk, ms=7, color=ARM_COLORS[st],
                        mfc=fill, label=f'pass {pas} (gating {"ON" if pas == 2 else "OFF"})')
            i_pre = LABELS.index('pre_1500V')
            ax.plot(1500, MED[ai, b, i_pre] * f_mv, 'x', ms=9, color='gray',
                    label='pre-scan 1500 V')
            i_end = LABELS.index('end_nominal')
            vn = NOMINAL_V[(st, b)]
            ax.plot(vn, MED[ai, b, i_end] * f_mv, '*', ms=13, color='crimson',
                    label='operating point')
            n_fit, c_fit = fit_powerlaw(ai, b)
            vv = np.geomspace(1150, 1650, 50)
            ax.plot(vv, np.exp(c_fit) * vv ** n_fit * f_mv, '-', lw=1, alpha=0.6,
                    color=ARM_COLORS[st])
            ax.text(0.05, 0.9, f'PSS{st}{b + 1} ({PMT_NAME[(ai, b)]})  n = {n_fit:.1f}',
                    transform=ax.transAxes, fontsize=11)
            ax.text(0.05, 0.82, f'operating: {vn} V',
                    transform=ax.transAxes, fontsize=9, color='crimson')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=0.3, which='both')
            if b == 1:
                ax.set_xlabel('PMT bias [V]')
            if ai == 0:
                ax.set_ylabel('coincident median amp [mV]')
            ax.set_xticks([1200, 1300, 1400, 1500, 1600],
                          ['1200', '1300', '1400', '1500', '1600'])
    axes[0][0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'{RUN_STEM}: plastic gain curves — wall-tagged coincident median '
                 f'(sideband-subtracted) vs measured HV; line = combined-pass power-law fit')
    fig.tight_layout()
    fig.savefig(OUT / 'gain_curves.png', dpi=140)
    plt.close(fig)


def rate_curves():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    rate = d['pss_n_late'] / (d['protons'] / 1e10)[:, None, None]
    for ai, st in enumerate('ABCD'):
        ax = axes[ai]
        for b, ls in ((0, '-'), (1, '--')):
            for pas, mk, fill in ((1, 'o', 'none'), (2, 's', None)):
                ii = np.where(PASS == pas)[0]
                o = np.argsort(VOLTS[ii])
                ax.plot(VOLTS[ii][o], rate[ii, ai, b][o], mk, ls=ls, ms=6,
                        color=ARM_COLORS[st], mfc=fill,
                        label=f'PSS{st}{b + 1} pass {pas}')
        ax.set_xlabel('PMT bias [V]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f'arm {st}')
    axes[0].set_ylabel('late hits / 1e10 protons')
    fig.suptitle(f'{RUN_STEM}: plastic late-hit rate vs HV — no plateau up to 1600 V '
                 '(threshold sits inside the spectrum)')
    fig.tight_layout()
    fig.savefig(OUT / 'rate_curves.png', dpi=140)
    plt.close(fig)


def wall_mip_stability():
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kern /= kern.sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(N_STEP)
    for ai, st in enumerate('ABCD'):
        pk = []
        for i in range(N_STEP):
            sub = (d['wal_mip'][i, ai, 0].sum(axis=0)
                   - SB * d['wal_mip'][i, ai, 1].sum(axis=0))
            sm = np.convolve(sub, kern, mode='same')
            m = CEN > 150
            pk.append(CEN[m][np.argmax(sm[m])] if sub.sum() > 500 else np.nan)
        pk = np.array(pk)
        ax.plot(x, pk / np.nanmean(pk), 'o-', color=ARM_COLORS[st],
                label=f'WAL{st} (mean {np.nanmean(pk):.0f} ADC)')
    bin_frac = CEN[1] / CEN[0] - 1
    ax.axhspan(1 - bin_frac, 1 + bin_frac, color='gray', alpha=0.15,
               label=f'±1 amp bin (±{100 * bin_frac:.1f}%)')
    ax.axvline(5.5, color='k', lw=1, ls=':')
    ax.text(5.6, ax.get_ylim()[0] + 0.005, 'SiPM flash gating ON →', fontsize=9)
    ax.set_xticks(x, LABELS, rotation=30, ha='right')
    ax.set_ylabel('wall MIP peak / scan mean')
    ax.set_ylim(0.9, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f'{RUN_STEM}: SiPM-wall MIP peak (ch-summed, sideband-subtracted) vs '
                 'plastic HV step — stable through the whole scan')
    fig.tight_layout()
    fig.savefig(OUT / 'wall_mip_stability.png', dpi=140)
    plt.close(fig)


def coinc_spectra():
    ii = [i for i in range(N_STEP) if PASS[i] == 1]
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            ax = axes[b][ai]
            for k, i in enumerate(sorted(ii, key=lambda j: VOLTS[j])):
                sub = np.clip(d['pss_mip'][i, ai, 0, b]
                              - SB * d['pss_mip'][i, ai, 1, b], 0, None)
                col = cmap(0.15 + 0.7 * k / (len(ii) - 1))
                ax.plot(CEN * MV[(ai, b)], sub / max(sub.sum(), 1), color=col, lw=1.4,
                        label=f'{VOLTS[i]:.0f} V')
            ax.set_xscale('log')
            ax.text(0.05, 0.9, f'PSS{st}{b + 1}', transform=ax.transAxes, fontsize=11)
            ax.grid(alpha=0.3)
            if b == 1:
                ax.set_xlabel('amp [mV]')
            if ai == 0:
                ax.set_ylabel('normalized counts / bin')
    axes[0][0].legend(fontsize=8, title='pass-1 HV')
    fig.suptitle(f'{RUN_STEM}: wall-tagged coincident plastic spectra vs HV '
                 '(pass 1, gating OFF; shape marches up with gain)')
    fig.tight_layout()
    fig.savefig(OUT / 'coinc_spectra.png', dpi=140)
    plt.close(fig)


def coinc_spectra_linear():
    """Pass-1 tagged spectra as densities on a LINEAR amplitude axis, with an
    arrow marking each spectrum's median — the observable the equalization
    calibrates on."""
    ii = sorted([i for i in range(N_STEP) if PASS[i] == 1], key=lambda j: VOLTS[j])
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            ax = axes[b][ai]
            f_mv = MV[(ai, b)]
            widths = np.diff(d['amp_edges']) * f_mv
            for k, i in enumerate(ii):
                sub = np.clip(d['pss_mip'][i, ai, 0, b]
                              - SB * d['pss_mip'][i, ai, 1, b], 0, None)
                dens = sub / max(sub.sum(), 1) / widths
                col = cmap(0.15 + 0.7 * k / (len(ii) - 1))
                ax.plot(CEN * f_mv, dens, color=col, lw=1.4, label=f'{VOLTS[i]:.0f} V')
                med = coinc_median(i, ai, b) * f_mv
                if np.isfinite(med):
                    y_med = np.interp(med, CEN * f_mv, dens)
                    y_top = 1.12 * dens.max()
                    ax.annotate('', xy=(med, y_med + 0.02 * y_top),
                                xytext=(med, y_med + 0.35 * y_top),
                                arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
            ax.set_xlim(0, 675)
            ax.set_ylim(bottom=0)
            ax.text(0.6, 0.9, f'PSS{st}{b + 1} ({PMT_NAME[(ai, b)]})',
                    transform=ax.transAxes, fontsize=11)
            ax.grid(alpha=0.3)
            if b == 1:
                ax.set_xlabel('amp [mV]')
            if ai == 0:
                ax.set_ylabel('probability density [1/mV]')
    axes[0][3].legend(fontsize=8, title='pass-1 HV')
    fig.suptitle(f'{RUN_STEM}: wall-tagged coincident plastic spectra vs HV, linear '
                 'amplitude axis (pass 1, gating OFF); arrows mark each spectrum\'s '
                 'median = the equalization observable')
    fig.tight_layout()
    fig.savefig(OUT / 'coinc_spectra_linear.png', dpi=140)
    plt.close(fig)


def equalization_diagram():
    """One panel showing the whole equalization: each PMT's measured operating
    point slides along its own fitted power law G(V) = med_now (V/V_now)^n to
    the horizontal target line (fleet geometric mean of current responses)."""
    i_end = LABELS.index('end_nominal')
    meds = {(ai, b): MED[ai, b, i_end] for ai in range(4) for b in range(2)}
    target = np.exp(np.mean([np.log(v) for v in meds.values()]))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    vv = np.geomspace(1120, 1660, 60)
    ends = []
    for (ai, b), m in meds.items():
        st = 'ABCD'[ai]
        f_mv = MV[(ai, b)]
        n = fit_powerlaw(ai, b)[0]
        v0 = NOMINAL_V[(st, b)]
        v_new = v0 * (target / m) ** (1 / n)
        curve = m * (vv / v0) ** n * f_mv
        ls = '-' if b == 0 else '--'
        ax.plot(vv, curve, ls, lw=1.2, alpha=0.7, color=ARM_COLORS[st])
        ax.plot(v0, m * f_mv, 'o', ms=8, color=ARM_COLORS[st])
        ax.annotate('', xy=(v_new, target * f_mv), xytext=(v0, m * f_mv),
                    arrowprops=dict(arrowstyle='-|>', color=ARM_COLORS[st],
                                    lw=1.8, shrinkA=5, shrinkB=2))
        ends.append((curve[-1], f'{st}{"LR"[b]}: n={n:.1f}, {v_new - v0:+.0f} V',
                     ARM_COLORS[st]))
    # right-edge labels, staggered to avoid collisions (log spacing >= 1.09x)
    ends.sort()
    y_lab = [ends[0][0]]
    for y, *_ in ends[1:]:
        y_lab.append(max(y, y_lab[-1] * 1.09))
    for (y, txt, col), yl in zip(ends, y_lab):
        ax.annotate(txt, xy=(vv[-1], y), xytext=(vv[-1] * 1.012, yl),
                    fontsize=9, color=col, va='center',
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.6, alpha=0.5))
    ax.axhline(target * MV_MEAN, color='k', lw=1.2, ls=':')
    ax.text(1125, target * MV_MEAN * 0.90,
            f'target = geometric mean of the 8 current responses = '
            f'{target * MV_MEAN:.1f} mV ({target:.0f} ADC)',
            fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1120, 1810)
    ax.set_xticks([1200, 1300, 1400, 1500, 1600],
                  ['1200', '1300', '1400', '1500', '1600'])
    ax.minorticks_off()
    ax.set_xlabel('PMT bias [V]')
    ax.set_ylabel('coincident median amp [mV]')
    ax.grid(alpha=0.3, which='both')
    ax.set_title(f'{RUN_STEM}: HV equalization — dots = current operating points, '
                 'lines = per-PMT fitted gain curves,\narrows = suggested moves to '
                 'the common target (solid = left bar, dashed = right bar)')
    fig.tight_layout()
    fig.savefig(OUT / 'equalization.png', dpi=140)
    plt.close(fig)


def scan_timeline():
    """Voltage staircase vs wall-clock with gating regions + protons per step."""
    spec12 = importlib.util.spec_from_file_location('hvscan', BASE / '12_plastic_hv_scan.py')
    hv = importlib.util.module_from_spec(spec12)
    spec12.loader.exec_module(hv)

    def hours(hms):
        parts = [int(x) for x in hms.split(':')] + [0]
        return parts[0] + parts[1] / 60 + parts[2] / 3600

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                  height_ratios=[2.2, 1])
    gate_col = {False: 'steelblue', True: 'darkorange'}
    for i, (label, v, pas, gate, t0, t1) in enumerate(hv.STEPS):
        h0, h1 = hours(t0), hours(t1)
        if v is None:                       # end_nominal: per-channel band
            ax.fill_between([h0, h1], 1275, 1325, color=gate_col[gate], alpha=0.5)
            ax.text((h0 + h1) / 2, 1345, 'per-channel\nnominal', ha='center',
                    fontsize=8)
        else:
            ax.plot([h0, h1], [v, v], lw=4, color=gate_col[gate],
                    solid_capstyle='butt')
            ax.text((h0 + h1) / 2, v + 12, label.replace('_', ' '), ha='center',
                    fontsize=8, rotation=0)
        axp.bar((h0 + h1) / 2, d['protons'][i] / 1e12, width=(h1 - h0) * 0.9,
                color=gate_col[gate], alpha=0.8)
        axp.text((h0 + h1) / 2, d['protons'][i] / 1e12 + 8,
                 f'{int(d["n_good_bunches"][i])}b', ha='center', fontsize=7)
    for gate, x, lbl in ((False, hours('13:45'), 'SiPM flash gating OFF'),
                         (True, hours('14:38'), 'gating ON')):
        ax.text(x, 1640, lbl, color=gate_col[gate], fontsize=11, ha='center')
    ax.axvline(hours('14:21:37'), color='k', lw=1, ls=':')
    ax.set_ylabel('common plastic HV [V]')
    ax.set_ylim(1150, 1690)
    ax.grid(alpha=0.3)
    axp.set_ylabel('protons / step [$10^{12}$]')
    axp.set_xlabel('2026-07-16 local time [h]')
    axp.grid(alpha=0.3)
    ax.set_title(f'{RUN_STEM}: plastic HV scan layout — proton-gated steps, two '
                 'interleaved passes; bar labels = good bunches per step')
    fig.tight_layout()
    fig.savefig(OUT / 'scan_timeline.png', dpi=140)
    plt.close(fig)


def trigger_threshold():
    """Common trigger threshold study at the POST-EQUALIZATION scale: tagged
    (wall-coincident, sideband-subtracted) efficiency and inclusive late rate
    vs a common threshold in mV. Amplitudes of each PMT are scaled by its
    equalization factor target/med_now, so one threshold applies to all 8."""
    i_end = LABELS.index('end_nominal')
    meds = {(ai, b): MED[ai, b, i_end] for ai in range(4) for b in range(2)}
    target = np.exp(np.mean([np.log(v) for v in meds.values()]))
    n_b = float(d['n_good_bunches'][i_end])
    thr = np.geomspace(1.5, 200, 250)
    eff = {}
    rate_tot = np.zeros_like(thr)
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            x = CEN * MV[(ai, b)] * (target / meds[(ai, b)])   # equalized mV
            sub = np.clip(d['pss_mip'][i_end, ai, 0, b]
                          - SB * d['pss_mip'][i_end, ai, 1, b], 0, None)
            inc = d['pss_amp'][i_end, ai, b]
            # cumulative-from-above via reversed cumsum on the sorted axis
            eff[(ai, b)] = np.interp(thr, x, np.cumsum(sub[::-1])[::-1] / max(sub.sum(), 1))
            rate_tot += np.interp(thr, x, np.cumsum(inc[::-1])[::-1] / n_b)
    min_eff = np.min([e for e in eff.values()], axis=0)
    t99 = thr[np.searchsorted(-min_eff, -0.99)]
    t95 = thr[np.searchsorted(-min_eff, -0.95)]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  height_ratios=[2, 1.2])
    for (ai, b), e in eff.items():
        st = 'ABCD'[ai]
        ax.plot(thr, e, '-' if b == 0 else '--', lw=1, alpha=0.6,
                color=ARM_COLORS[st], label=f'PSS{st}{b + 1}')
    ax.plot(thr, min_eff, 'k-', lw=2.2, label='worst PMT')
    for t, e_lab in ((t99, 0.99), (t95, 0.95)):
        ax.axvline(t, color='gray', lw=1, ls='--')
        ax.text(t * 1.03, 0.55, f'{e_lab:.0%} @ {t:.1f} mV', rotation=90, fontsize=9)
    ax.axvspan(thr[0], 2.2, color='red', alpha=0.08)
    ax.text(1.6, 0.45, 'below PSA acquisition\ncutoff (unobserved)', fontsize=8,
            color='darkred')
    ax.set_ylabel('tagged-coincidence efficiency above threshold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=3, loc='lower left')
    axr.plot(thr, rate_tot, 'k-', lw=1.8)
    for t in (t99, t95):
        axr.axvline(t, color='gray', lw=1, ls='--')
    axr.set_xlabel('common threshold, post-equalization scale [mV]')
    axr.set_ylabel('late hits / bunch (8 PMTs)')
    axr.grid(alpha=0.3, which='both')
    fig.suptitle(f'{RUN_STEM}: plastic threshold scan, equalized operating point '
                 '(no signal/noise valley: efficiency-driven choice)')
    fig.tight_layout()
    fig.savefig(OUT / 'threshold_scan.png', dpi=140)
    plt.close(fig)
    print(f'\nTrigger threshold (post-equalization, digitizer-equivalent mV): '
          f'99% tagged eff on every PMT down to {t99:.1f} mV, 95% to {t95:.1f} mV; '
          f'acquisition cutoff ~1.5-2 mV')


def equalization_table():
    i_end = LABELS.index('end_nominal')
    med_now = {}
    n_idx = {}
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            med_now[(st, b)] = MED[ai, b, i_end]
            n_idx[(st, b)] = fit_powerlaw(ai, b)[0]
    target = np.exp(np.mean([np.log(v) for v in med_now.values()]))
    print(f'\nHV equalization to fleet geometric-mean coincident median '
          f'({target:.0f} ADC = {target * MV_MEAN:.1f} mV), using per-PMT fitted n:')
    print(f'{"PMT":8s} {"V_now":>6s} {"med_now":>8s} {"med_mV":>7s} {"n":>5s} '
          f'{"V_sugg":>7s} {"dV":>5s}')
    for (st, b), m in med_now.items():
        n = n_idx[(st, b)]
        v0 = NOMINAL_V[(st, b)]
        ai = 'ABCD'.index(st)
        v_new = v0 * (target / m) ** (1 / n)
        print(f'PSS{st}{b + 1}    {v0:6d} {m:8.0f} {m * MV[(ai, b)]:7.1f} {n:5.2f} '
              f'{v_new:7.0f} {v_new - v0:+5.0f}')


if __name__ == '__main__':
    scan_timeline()
    gain_curves()
    rate_curves()
    wall_mip_stability()
    coinc_spectra()
    coinc_spectra_linear()
    equalization_diagram()
    trigger_threshold()
    equalization_table()
    print(f'Figures -> {OUT}')

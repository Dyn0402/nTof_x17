"""plot_threshold.py — From the top+bottom SUM cache (build_sum.py), find a
threshold on S = a_top + a_bot that keeps essentially all MIPs while rejecting
background, and make the supporting plots.

Method (per arm, per 4-bar group):
  MIP(S)  = signal(S) - sb_scale*sideband(S)   (Landau peak; the thing to keep)
  BKG(S)  = sb_scale*sideband(S)               (accidental top-bot coincidences)
  MIP efficiency  eps(Scut) = sum_{S>Scut} MIP / sum MIP
  Bkg acceptance  = sum_{S>Scut} BKG / sum BKG
Recommended cut = S where eps = 99% (keep ~all MIPs). We also report the cut at
the MIP-peak / valley and the resulting background numbers, in ADC and mV.

Usage: python plot_threshold.py [run_stem]     (default run224460)
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BASE = HERE.parent
STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224460'
# per-arm caches (build_sum.py writes one file per arm)
arm = {st: np.load(HERE / 'cache' / f'sum_{STEM}_{st}.npz') for st in 'ABCD'}
d0 = arm['A']
SUM_E = d0['sum_edges']
AMP_E = d0['amp_edges']
J_E = d0['j_edges']
SB = float(d0['sb_scale'])
NGOOD = int(d0['n_good'])
Sc = np.sqrt(SUM_E[:-1] * SUM_E[1:])           # sum bin centres (ADC)
Ac = np.sqrt(AMP_E[:-1] * AMP_E[1:])

# ADC -> mV (per-arm mean wall factor; ~0.0306, uniform)
cal = json.loads((BASE / 'calib' / f'adc_to_mv_{STEM}.json').read_text())['factors']
MV = {st: np.mean([cal[f'WAL{st}'][str(i + 1)] for i in range(8)]) for st in 'ABCD'}

kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
kern /= kern.sum()


def smooth(y):
    return np.convolve(y, kern, mode='same')


def eff_curve(mip):
    """Cumulative MIP efficiency above each bin edge (survivors / total)."""
    tot = mip.sum()
    if tot <= 0:
        return np.zeros_like(mip)
    return np.cumsum(mip[::-1])[::-1] / tot


def cut_at_eff(mip, target):
    e = eff_curve(np.clip(mip, 0, None))
    i = np.searchsorted(-e, -target)           # first bin where eff drops below target
    i = min(max(i, 0), len(Sc) - 1)
    return Sc[i], i


rows = []
fig, axes = plt.subplots(4, 4, figsize=(19, 15))
for r, st in enumerate('ABCD'):
    h_sum = arm[st]['h_sum']                    # (group, sig/side, sum)
    for g in range(4):
        sig = h_sum[g, 0]
        side = h_sum[g, 1]
        mip = smooth(sig - SB * side)
        bkg = SB * side
        mipc = np.clip(mip, 0, None)

        # peak (above 200 ADC to skip the low-amp junk) and 99% cut
        pk_i = 200 < Sc
        pk = Sc[pk_i][np.argmax(mipc[pk_i])] if mipc[pk_i].any() else np.nan
        s99, i99 = cut_at_eff(mip, 0.99)
        s995, _ = cut_at_eff(mip, 0.995)
        mv = MV[st]

        n_mip = mipc.sum()
        eff = eff_curve(mipc)
        bkg_cum = np.cumsum(bkg[::-1])[::-1]
        bkg_above = bkg_cum[i99]
        bkg_rej = 1 - bkg_above / bkg.sum() if bkg.sum() > 0 else np.nan
        # background rate per good bunch surviving the cut
        bkg_rate = bkg_above / NGOOD

        rows.append(dict(arm=st, grp=g + 1, mip_peak=pk, n_mip=n_mip,
                         cut99=s99, cut995=s995, cut99_mv=s99 * mv,
                         pk_mv=pk * mv, bkg_rej=bkg_rej, bkg_rate=bkg_rate,
                         mv=mv))

        ax = axes[r, g]
        ax.step(Sc, np.clip(sig, .1, None), where='mid', color='0.6', lw=.9,
                label='signal (raw)')
        ax.step(Sc, np.clip(bkg, .1, None), where='mid', color='tab:orange',
                lw=.9, label='accidental bkg')
        ax.step(Sc, np.clip(mipc, .1, None), where='mid', color='tab:blue',
                lw=1.4, label='MIP (sub)')
        ax.axvline(s99, color='tab:green', ls='--', lw=1.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(80, 1.5e5)
        ax.set_ylim(0.5, None)
        ax.set_title(f'WAL{st} grp{g + 1}  peak~{pk:.0f} ADC ({pk*mv:.0f} mV)\n'
                     f'99% cut {s99:.0f} ADC ({s99*mv:.0f} mV)', fontsize=9)
        if r == 3:
            ax.set_xlabel('top+bottom sum  [ADC]')
        if g == 0:
            ax.set_ylabel('coincidences')
        if r == 0 and g == 0:
            ax.legend(fontsize=7, loc='upper right')
fig.suptitle(f'{STEM}: top+bottom SiPM SUM spectra, MIP vs accidental background '
             f'(green = 99% MIP-efficiency cut)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(HERE / 'figures' / f'sum_spectra_{STEM}.png', dpi=110)
print(f'wrote figures/sum_spectra_{STEM}.png')

# ---- efficiency / background tradeoff, one panel per arm ----
fig2, axes2 = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
for r, st in enumerate('ABCD'):
    h_sum = arm[st]['h_sum']
    ax = axes2[r]
    mv = MV[st]
    for g in range(4):
        mip = np.clip(smooth(h_sum[g, 0] - SB * h_sum[g, 1]), 0, None)
        bkg = SB * h_sum[g, 1]
        e = eff_curve(mip)
        b = np.cumsum(bkg[::-1])[::-1]
        b = b / b.max() if b.max() > 0 else b
        ax.plot(Sc * mv, e, lw=1.4, label=f'grp{g+1} MIP eff')
        ax.plot(Sc * mv, b, lw=1.0, ls=':', color=ax.lines[-1].get_color())
    ax.axhline(0.99, color='k', lw=.6, ls='--')
    ax.axvspan(8, 10, color='green', alpha=0.12, lw=0)
    ax.set_xscale('log')
    ax.set_xlim(3, 5e3)
    ax.set_title(f'WAL{st}   (solid=MIP eff, dotted=bkg accept)')
    ax.set_xlabel('sum threshold  [mV]')
    if r == 0:
        ax.set_ylabel('fraction above threshold')
        ax.legend(fontsize=7)
fig2.suptitle(f'{STEM}: MIP efficiency & background acceptance vs top+bottom sum '
              f'threshold', fontsize=13)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(HERE / 'figures' / f'threshold_tradeoff_{STEM}.png', dpi=110)
print(f'wrote figures/threshold_tradeoff_{STEM}.png')

# ---- table ----
print(f'\n{STEM}: top+bottom SUM MIP threshold ({NGOOD} good bunches)')
print(f'{"grp":5s} {"MIP pk":>10s} {"n_MIP":>9s} {"99% cut":>16s} '
      f'{"99.5% cut":>10s} {"bkg rej":>8s} {"bkg/bunch":>10s}')
for x in rows:
    print(f'{x["arm"]}{x["grp"]:<4d} {x["mip_peak"]:6.0f} ADC {x["n_mip"]:9,.0f} '
          f'{x["cut99"]:6.0f} ADC/{x["cut99_mv"]:5.0f}mV {x["cut995"]:8.0f}A '
          f'{x["bkg_rej"]*100:6.1f}% {x["bkg_rate"]:10.2f}')

# global suggestion: a single mV cut that keeps >=99% on every group
valid = [x for x in rows if np.isfinite(x['cut99_mv']) and x['n_mip'] > 500]
if valid:
    gmax = max(x['cut99_mv'] for x in valid)
    print(f'\nSingle common cut to keep >=99% of MIPs on ALL populated groups: '
          f'{gmax:.0f} mV  (~{gmax/np.mean(list(MV.values())):.0f} ADC)')
    print('Per-arm 99% cut (max over that arm\'s 4 groups):')
    for st in 'ABCD':
        a = [x for x in valid if x['arm'] == st]
        if a:
            print(f'  WAL{st}: {max(x["cut99_mv"] for x in a):.0f} mV')

# ---- candidate fixed-mV thresholds: min MIP eff across groups + fake-trigger rate --
print(f'\nCandidate fixed thresholds (MIP eff from the clean top-bottom coincidence '
      f'tag; rates per beam bunch):')
print(f'{"thresh":>8s} {"minMIPeff":>10s} {"meanMIPeff":>11s} '
      f'{"accidental fake-trig/bunch":>27s}')
for mv_cut in (5, 8, 10, 15, 20, 30, 40):
    effs, acc = [], []
    for st in 'ABCD':
        f_mv = MV[st]
        i_s = min(np.searchsorted(Sc, mv_cut / f_mv), len(Sc) - 1)
        hsum = arm[st]['h_sum']
        for g in range(4):
            mip = np.clip(smooth(hsum[g, 0] - SB * hsum[g, 1]), 0, None)
            if mip.sum() < 500:
                continue
            effs.append(eff_curve(mip)[i_s])
            acc.append(np.cumsum((SB * hsum[g, 1])[::-1])[::-1][i_s] / NGOOD)
    print(f'{mv_cut:6d}mV {np.min(effs)*100:9.1f}% {np.mean(effs)*100:10.1f}% '
          f'{np.mean(acc):27.2f}')

# ---- sub-MIP noise/soft floor: single-channel rate below 5 mV (threshold-independent)
tot = sub5 = 0.0
for st in 'ABCD':
    f_mv = MV[st]
    i5 = np.searchsorted(Ac, 5.0 / f_mv)
    hsingle = arm[st]['h_single']
    tot += hsingle.sum() / NGOOD
    sub5 += hsingle[:, :i5].sum() / NGOOD
print(f'\nWall flux is MIP-DOMINATED: mean single-SiPM hit rate {tot/32:.0f}/bunch/ch; '
      f'only {100*sub5/tot:.1f}% below 5 mV (the sub-MIP soft/noise floor).')
print('=> No MIP-vs-background valley; the sum threshold sits above noise, not in a gap.')

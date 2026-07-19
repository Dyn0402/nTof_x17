"""23i_nonlinearity.py — visualize plastic PMT response NONLINEARITY: why
equalizing on the Y-88 699 keVee edge does not fully equalize the MIP.

Figure 1 (concept, schematic): amplitude-vs-energy for two PMTs pinned to the
same value at 699 keVee. If the response is linear (through the origin) they then
coincide everywhere -> MIP equal. If it bends per-PMT they diverge at higher
energy -> MIP differs.

Figure 2 (data): two independent lines of evidence that the response is not a
single per-PMT constant:
  (a) the TWO Y-88 edges (same run): amp(1612)/amp(699) should be the physics
      value 2.307 for every PMT if linear; measured spread ~1.8-2.6 (outer edge
      is the weaker fit).
  (b) the coincident MEDIAN (~MIP-scale deposit) / the 699 keVee edge: would be
      one constant for all PMTs if linear; measured 1.70-2.38 (1.4x), arm A/DL
      low -> those tubes compress more at high energy.

Outputs: figures/21_y88/nonlinearity_concept.png, nonlinearity_data.png
Usage: python 23i_nonlinearity.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
CALIB = BASE / 'calib'
OUT = BASE / 'figures' / '21_y88'
PMTS = ['PSSA1', 'PSSA2', 'PSSB1', 'PSSB2', 'PSSC1', 'PSSC2', 'PSSD1', 'PSSD2']
LAB = {'PSSA1': 'AL', 'PSSA2': 'AR', 'PSSB1': 'BL', 'PSSB2': 'BR',
       'PSSC1': 'CL', 'PSSC2': 'CR', 'PSSD1': 'DL', 'PSSD2': 'DR'}
E699, E1612 = 0.699, 1.612
E_MIP = 3.4


def concept_figure():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    E = np.linspace(0, 3.6, 200)
    E0 = 30.0                                  # both pinned here at 699 keVee
    # --- linear ---
    g = E0 / E699
    ax[0].plot(E, g * E, color='0.2', lw=3)
    ax[0].plot(E699, E0, 'o', color='green', ms=11, zorder=5)
    ax[0].plot(E_MIP, g * E_MIP, 'o', color='0.2', ms=10, zorder=5)
    ax[0].annotate('equalize here\n(699 keVee)', (E699, E0), (0.9, 55),
                   color='green', fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='green'))
    ax[0].text(E_MIP, g * E_MIP + 8, 'MIP:\nboth equal', ha='center', fontsize=10)
    ax[0].text(1.8, 20, 'PMT 1 = PMT 2\n(one straight line\nthrough the origin)',
               fontsize=10, color='0.2')
    ax[0].set_title('If the response were LINEAR', fontsize=13)
    # --- nonlinear ---
    for p, col, name in ((0.80, '#d62728', 'arm A / DL\n(compresses)'),
                         (1.06, '#1f77b4', 'B / C / D')):
        a = E0 / E699 ** p
        ax[1].plot(E, a * E ** p, color=col, lw=3, label=name)
        ax[1].plot(E_MIP, a * E_MIP ** p, 'o', color=col, ms=10, zorder=5)
    aA = E0 / E699 ** 0.80; aB = E0 / E699 ** 1.06
    yA, yB = aA * E_MIP ** 0.80, aB * E_MIP ** 1.06
    ax[1].plot([E_MIP, E_MIP], [yA, yB], color='k', lw=1.2, ls=':')
    ax[1].annotate(f'MIP differs\n$\\sim${100*(yB-yA)/((yA+yB)/2):.0f}%',
                   (E_MIP, (yA + yB) / 2), (2.35, 60), fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->'))
    ax[1].plot(E699, E0, 'o', color='green', ms=11, zorder=6)
    ax[1].text(0.9, 55, 'still equalized\nat 699 keVee', color='green',
               fontsize=10, ha='center')
    ax[1].set_title('If the response is NONLINEAR (reality)', fontsize=13)
    ax[1].legend(loc='upper left', fontsize=10)
    for a in ax:
        a.set_xlabel('energy deposited [MeV]'); a.set_xlim(0, 3.6); a.set_ylim(0, 175)
        a.axvline(E699, color='green', lw=0.6, ls=':'); a.grid(alpha=0.25)
    ax[0].set_ylabel('plastic signal [mV]')
    fig.suptitle('Equalizing at ONE energy only equalizes ALL energies if the '
                 'response is linear', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'nonlinearity_concept.png', dpi=140)
    plt.close(fig)


def coinc_medians(run):
    d = np.load(BASE / 'cache' / f'12_hvscan_{run}.npz')
    edges = d['amp_edges']; c = np.sqrt(edges[:-1] * edges[1:])
    sb = float(d['sb_scale']); volts = d['step_volts']
    fac = json.loads((CALIB / f'adc_to_mv_{run}.json').read_text())['factors']
    med = np.full((4, 2, len(volts)), np.nan)
    for i in range(len(volts)):
        for ai, st in enumerate('ABCD'):
            for b in range(2):
                sub = np.clip(d['pss_mip'][i, ai, 0, b]
                              - sb * d['pss_mip'][i, ai, 1, b], 0, None)
                cum = np.cumsum(sub)
                if cum[-1] > 200:
                    med[ai, b, i] = c[np.searchsorted(cum, cum[-1] / 2)] \
                        * fac[f'PSS{st}'][str(b + 1)]
    return volts, med


def data_figure():
    gain = json.loads((CALIB / 'plastic_hv_gain_absolute.json').read_text())['pmts']
    y = json.loads((CALIB / 'y88_energy_calib.json').read_text())['channels']
    v, m = coinc_medians('run224489')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    # (a) two Y-88 edges, normalized to edge699 = 1, on an energy axis
    ax[0].plot([0, E1612 + 0.2], [0, (E1612 + 0.2) / E699], 'k--', lw=1.5,
               label='linear (through origin)')
    for p, c in zip(PMTS, colors):
        r = y[p]['edge1612_mv'] / y[p]['edge699_mv']
        ax[0].plot([E699, E1612], [1, r], 'o-', color=c, ms=6, label=LAB[p])
    ax[0].axhline(E1612 / E699, color='gray', ls=':', lw=1)
    ax[0].text(1.62, E1612 / E699 + 0.05, 'physics: 2.307', fontsize=9, color='gray')
    ax[0].set_xlabel('energy [MeV]'); ax[0].set_ylabel('signal / (699 keVee signal)')
    ax[0].set_title('(a) The two Y-88 edges, same run\n(linear $\\Rightarrow$ all land on 2.307)')
    ax[0].legend(fontsize=7, ncol=3); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, 1.9)

    # (b) median/edge ratio per PMT (MIP-scale vs edge)
    ratios = []
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            p = f'PSS{st}{b + 1}'; Vn = gain[p]['V_equalized']
            vv = np.array([v[i] for i in range(len(v)) if v[i] > 0])
            mm = np.array([m[ai, b, i] for i in range(len(v)) if v[i] > 0])
            ok = np.isfinite(mm); n, lnA = np.polyfit(np.log(vv[ok]), np.log(mm[ok]), 1)
            ratios.append(np.exp(lnA) * Vn ** n / gain[p]['edge699_fifo_mv'])
    x = np.arange(8)
    bars = ax[1].bar(x, ratios, color=['#d62728' if LAB[PMTS[i]] in ('AL', 'AR', 'DL')
                                       else '#1f77b4' for i in range(8)])
    ax[1].axhline(np.mean(ratios), color='k', ls='--', lw=1,
                  label=f'fleet mean {np.mean(ratios):.2f}')
    ax[1].set_xticks(x); ax[1].set_xticklabels([LAB[p] for p in PMTS])
    ax[1].set_ylabel('MIP-scale median / 699 keVee edge')
    ax[1].set_title('(b) MIP-scale vs edge per PMT\n(linear $\\Rightarrow$ all bars equal; '
                    'red = arm A / DL, compress)')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis='y')
    fig.suptitle('Plastic response nonlinearity in the data: the high-energy / '
                 'low-energy signal ratio varies per PMT')
    fig.tight_layout()
    fig.savefig(OUT / 'nonlinearity_data.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    concept_figure()
    data_figure()
    print('-> figures/21_y88/nonlinearity_concept.png & nonlinearity_data.png')

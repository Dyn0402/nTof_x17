"""Render duplication-veto before/after coincident spectra."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224460'
BASE = Path('/home/dylan/PycharmProjects/nTof_x17/mx_july_beam_qa')
SP = Path(__file__).parent
OUT = BASE / 'figures' / RUN_STEM / 'autopsy'
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(BASE / 'cache' / f'17_dupveto_{RUN_STEM}.npz')
ae = d['amp_edges']
cen = np.sqrt(ae[:-1] * ae[1:])
wid = np.diff(ae)
sb = float(d['sb_scale'])
FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']

ARMS = ['A', 'D', 'B']
fig, axes = plt.subplots(3, 8, figsize=(22, 9.5), sharex=True)
for r, st in enumerate(ARMS):
    h = d[f'{st}_h']            # (variant, sig/side, ch, amp)
    ff = d[f'{st}_flagfrac_sig']
    for c in range(8):
        ax = axes[r, c]
        fmv = FAC[f'WAL{st}'][str(c + 1)]
        x = cen * fmv
        subs = [(h[v, 0, c] - sb * h[v, 1, c]) / (wid * fmv) for v in range(3)]
        ax.axhline(0, color='#9ca3af', lw=0.8)
        ax.step(x, subs[0], where='mid', color='#6b7280', lw=1.5,
                label='as-is' if (r, c) == (0, 0) else None)
        ax.fill_between(x, 0, subs[1], step='mid', color='#fb923c', alpha=0.65, lw=0)
        ax.step(x, subs[1], where='mid', color='#c2410c', lw=1.4,
                label='after duplication veto' if (r, c) == (0, 0) else None)
        ax.step(x, subs[2], where='mid', color='#1d4ed8', lw=1.1, ls='--',
                label='removed (flagged) component' if (r, c) == (0, 0) else None)
        ax.set_xlim(0, 90)
        ax.set_title(f'WAL{st}{c + 1}   ({100 * ff[c]:.0f}% flagged)', fontsize=10)
        ax.grid(alpha=0.2)
        if c == 0:
            ax.set_ylabel('true coinc. hits / mV')
        if r == 2:
            ax.set_xlabel('amplitude [mV]')
fig.legend(loc='upper right', fontsize=10, ncols=3, frameon=True,
           bbox_to_anchor=(0.995, 0.985))
fig.suptitle(f'{RUN_STEM}: sideband-subtracted coincident wall spectra — '
             'duplication veto (same-side neighbor within ±4 ns, amp ratio 1/3–3); '
             '% = flagged share of signal-window pairs; WALB = clean control',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / 'dup_veto_spectra.png', dpi=130)
print(OUT / 'dup_veto_spectra.png')

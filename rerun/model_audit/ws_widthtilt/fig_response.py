#!/usr/bin/env python3
"""fig_response.py — slide figures for the angle-response question (ws_widthtilt).

fig_f4_gate.png    : the F4 'implied v' curve (gated on the fitted slope, as 03_angles.py does) against the same
                     events ungated and sign-symmetrised with w0 removed: most of the F4 fall is the gate.
fig_response.png   : the corrected symmetric scale S(|tan|) per chamber and view, normalised to the bundle v.
Reads ws_widthtilt/gate_artefact.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_widthtilt')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
COL = {'det2': '#1f77b4', 'det3': '#2ca02c', 'det4': '#d62728', 'det6': '#9467bd', 'det7': '#ff7f0e'}


def main():
    G = json.load(open(f'{ROOT}/gate_artefact.json'))
    ctr = [0.5 * (lo + hi) for lo, hi in G['bins']]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for i, p in enumerate('xy'):
        ax = axs[i]
        for det in DETS:
            r = G[f'{det}_{p}']; v = r['v'] * r['kw']
            ax.plot(ctr, np.array(r['gated']) / v, '--', color=COL[det], alpha=0.6, lw=1.3)
            ax.errorbar(ctr, np.array(r['sym']) / v, yerr=np.array(r['sym_err']) / v, fmt='o-', color=COL[det], capsize=3,
                        lw=2, label=det)
        ax.axhline(1, color='k', lw=0.8)
        ax.set_xlabel('|tan θ| of the reference track (M3)'); ax.set_title(f'{p.upper()} view')
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('measured slope / true slope\n(normalised to the calibrated scale)')
    axs[0].legend(title='solid: ungated, w0 removed\ndashed: F4 (gated on fitted slope)', fontsize=8, title_fontsize=8)
    axs[0].set_ylim(0.85, 1.15)
    fig.suptitle('Angle response vs angle: the F4 "fall" is mostly a selection cut on the fitted slope', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{ROOT}/fig_f4_gate.png', dpi=130); plt.close(fig)
    print('wrote', f'{ROOT}/fig_f4_gate.png')


if __name__ == '__main__':
    main()

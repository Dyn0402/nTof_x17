"""32_hardware_trigger_compare.py — MEASURED hardware trigger rates vs emulation
and GEANT4, thermal gate (>1 ms).

Reads the N1081B M5.C time-tag capture (real sector-coincidence edges =
wall.AND.plastic per wall, streamed 300 s beam-on during run_64) and plots the
per-wall + total coincidence rate vs time-past-flash, in the same linear /
gate>1 ms format as the sim comparisons. Three curves per figure:

  * HARDWARE  — measured M5.C edges (this file), grouped by beam pulse
  * EMULATED  — software trigger emulation from ntof hits, run224524 (script 30)
  * GEANT4    — thermal-neutron sim (timedist_2cm.npz)

Pulse grouping: n_TOF PS supercycle, spills at 1.2 s multiples. A flash is the
onset of each burst (10 ms bins > 500 Hz, clusters split by >0.2 s gaps); t = 0
is the first edge of the burst = the gamma flash. Rates are per beam pulse per
ms (hardware: /n_flash; emulation: /n_bunch; GEANT4: /proton-pulse).

Channel -> wall: 1=A 2=B 4=C 5=D (M5.C panel map). Wall D plastic = D-R only.

Inputs:
  ~/x17/beam_july/tt_secC/edges.csv                              (hardware)
  cache/30_trigemul_run224524.npz                               (emulation)
  ~/CLionProjects/MX17_Full_Geant/analysis/thermal_2cm/timedist_2cm.npz
Output: figures/30_trigger/hw_*_run64.png
Usage:  python 32_hardware_trigger_compare.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import trigger_diagram

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
OUT = BASE / 'figures' / '30_trigger'
EDGES = Path.home() / 'x17/beam_july/tt_secC/edges.csv'
SIM_NPZ = Path.home() / 'CLionProjects/MX17_Full_Geant/analysis/thermal_2cm/timedist_2cm.npz'

GREEN, BLUE, ORANGE = '#2ca02c', '#1f77b4', '#ff7f0e'
HIT = '#ff9900'
CH2WALL = {1: 'A', 2: 'B', 4: 'C', 5: 'D'}


# ----------------------------- hardware: load + flashes -----------------------
def load_hardware():
    d = np.genfromtxt(EDGES, delimiter=',', names=True, dtype=None)
    ch = d['channel'].astype(int)
    tb = d['t_board_ns'].astype(np.float64)
    o = np.argsort(tb)
    ch, tb = ch[o], tb[o]
    t = (tb - tb.min()) / 1e9                       # elapsed seconds
    # flash = burst onset: hot 10 ms bins clustered, split by >0.2 s
    bw = 0.010
    h, be = np.histogram(t, bins=np.arange(0, t.max() + bw, bw))
    hot = be[np.flatnonzero(h / bw > 500)]
    starts = np.concatenate([[hot[0]], hot[np.flatnonzero(np.diff(hot) > 0.2) + 1]])
    ft = np.array([t[np.searchsorted(t, cs - 0.002)] for cs in starts])
    fi = np.searchsorted(ft, t, side='right') - 1
    trel_ms = (t - ft[np.clip(fi, 0, None)]) * 1e3
    ok = fi >= 0
    return ch[ok], trel_ms[ok], len(ft)


def main():
    ch_hw, trel_hw, n_flash = load_hardware()
    print(f'hardware: {len(trel_hw):,} in-pulse edges, {n_flash} flashes (beam pulses)')

    emu = np.load(CACHE / '30_trigemul_run224524.npz')
    n_ev = int(emu['n_ev']); flash_ms = float(emu['flash_ns']) / 1e6

    s = np.load(SIM_NPZ)
    tedges = s['tedges']; tc = s['tc']; dt = np.diff(tedges)
    w = float(s['n_pulse']) / float(s['n_events'])
    g4_leg = s['leg20'] * w / dt

    def hw_rate(mask):
        c = np.histogram(trel_hw[mask], bins=tedges)[0].astype(float)
        return c / n_flash / dt, np.sqrt(c) / n_flash / dt

    def emu_rate(keys):
        t = np.concatenate([emu[k] for k in keys]).astype(np.float64) / 1e6 - flash_ms
        return np.histogram(t, bins=tedges)[0].astype(float) / n_ev / dt

    def one(mask, emu_keys, g4, mode, wall, outname):
        hr, he = hw_rate(mask)
        er = emu_rate(emu_keys)
        tmax = float(tc[g4 > 1e-3 * g4.max()].max())
        m = tc <= tmax
        iH, iE, iG = (float((x[m] * dt[m]).sum()) for x in (hr, er, g4))
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(tc[m], g4[m], color=GREEN, lw=2.0, zorder=2,
                label=f'GEANT4 (int {iG:.0f}/pulse)')
        ax.fill_between(tc[m], g4[m], color=GREEN, alpha=0.12, zorder=1)
        ax.plot(tc[m], er[m], color=ORANGE, lw=1.6, marker='s', ms=3, zorder=3,
                label=f'nTof DAQ run224524 (int {iE:.0f}/pulse)')
        ax.plot(tc[m], hr[m], color=BLUE, lw=0.8, zorder=4)
        ax.errorbar(tc[m], hr[m], yerr=he[m], fmt='o', ms=4, color=BLUE, capsize=0,
                    elinewidth=0.9, zorder=5,
                    label=f'Trigger, M5.C run_64 (int {iH:.0f}/pulse)')
        ax.axvline(5.3, color='0.5', ls=':', lw=1.1)
        ax.annotate('thermal peak ~5.3 ms\n(E ~ 71 meV)', (5.5, 0.93 * hr[m].max()),
                    fontsize=9, color='0.35', va='top')
        ax.set_xlim(1.0, tmax); ax.set_ylim(bottom=0)
        ax.set_xlabel('time past gamma flash  t [ms]   (gate t > 1 ms)')
        ax.set_ylabel('coincidence rate  [ / pulse / ms ]')
        ax.set_title('Thermal-gate coincidence: Trigger vs nTof DAQ vs GEANT4  '
                     '(t > 1 ms)')
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
        trigger_diagram.draw(ax, mode, wall)
        fig.tight_layout(); fig.savefig(OUT / outname, dpi=140); plt.close(fig)
        print(f'  {outname:28s} HW {iH:6.1f}  EMU {iE:6.1f}  G4 {iG:6.1f}  '
              f'HW/G4 {iH/iG:.2f}  HW/EMU {iH/iE:.2f}')

    print('\nintegrals over [1, tmax] ms (per pulse):')
    # total (sum 4 sectors) vs GEANT4 legs (sum 4 arms)
    one(np.ones(len(ch_hw), bool), [f'singles_tof_{a}' for a in 'ABCD'], g4_leg,
        'sum', None, 'hw_total_run64.png')
    # per wall vs GEANT4 legs / 4
    for c, a in CH2WALL.items():
        one(ch_hw == c, [f'singles_tof_{a}'], g4_leg / 4, 'coinc', a, f'hw_coinc_WAL{a}_run64.png')
    print(f'\n-> figures in {OUT}/')


if __name__ == '__main__':
    main()

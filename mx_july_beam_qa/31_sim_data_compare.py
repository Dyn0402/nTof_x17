"""31_sim_data_compare.py — thermal-gate (>1 ms) DATA vs GEANT4, all rates.

Emulated run224524 trigger rates (script 30) vs the MX17_Full_Geant thermal-
neutron prediction, in the sim's IPC-spectrum format: LINEAR-LINEAR axes,
neutron arrival time in ms, gate t > 1 ms only. Data time = tof - gamma-flash;
both rates are per beam pulse (= per bunch) per ms. The trigger type is shown by
an in-plot detector diagram (thin box = SiPM wall, thick box = plastic; filled =
hit) with the wall letter, instead of in the title.

Produces, as SEPARATE figures:
  * total singles (coincidence, Sigma 4 walls)   vs GEANT4 legs (sum 4 arms)
  * per-wall coincidence  WALX                    vs GEANT4 legs / 4   (+ accidentals)
  * per-wall SiPM-wall singles (M1 wall-OR)       vs GEANT4 SiPM singles / 4
  * per-wall plastic singles   (M2 plastic-OR)    vs GEANT4 plastic singles / 4

Coincidence plots also overlay the accidental estimate (plastic leg delayed
+500 ns) and the accidental-subtracted data.

Inputs:  cache/30_trigemul_run224524.npz
         ~/CLionProjects/MX17_Full_Geant/analysis/thermal_2cm/timedist_2cm.npz
Output:  figures/30_trigger/*_datasim_<run>.png
Usage:   python 31_sim_data_compare.py [run_stem]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import trigger_diagram

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
OUT = BASE / 'figures' / '30_trigger'
SIM_NPZ = Path.home() / 'CLionProjects/MX17_Full_Geant/analysis/thermal_2cm/timedist_2cm.npz'

STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224524'
GATE_MS = 1.0
GREEN, BLUE, RED = '#2ca02c', '#1f77b4', '#d62728'
HIT, NOHIT = '#ff9900', 'white'

d = np.load(CACHE / f'30_trigemul_{STEM}.npz')
n_ev = int(d['n_ev'])
flash_ms = float(d['flash_ns']) / 1e6
HAS_ACC = 'singles_acc_tof_A' in d.files

s = np.load(SIM_NPZ)
tedges = s['tedges']; tc = s['tc']; dt = np.diff(tedges)
w = float(s['n_pulse']) / float(s['n_events'])
sim_rate = {'leg': s['leg20'] * w / dt, 'sipm': s['sipm'] * w / dt,
            'plas': s['plas20'] * w / dt}


def data_times(keys):
    t = np.concatenate([d[k] for k in keys]) if keys else np.empty(0)
    return t.astype(np.float64) / 1e6 - flash_ms


def compare(t_data, sim, outname, title, ylabel, data_lab, sim_lab,
            mode, wall, t_acc=None):
    cnt = np.histogram(t_data, bins=tedges)[0].astype(float)
    rate = cnt / n_ev / dt
    rerr = np.sqrt(cnt) / n_ev / dt
    tmax = float(tc[sim > 1e-3 * sim.max()].max()) if sim.max() > 0 else tc[-1]
    m = tc <= tmax
    id_, is_ = float((rate[m] * dt[m]).sum()), float((sim[m] * dt[m]).sum())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(tc[m], sim[m], color=GREEN, lw=2.0, zorder=2,
            label=f'{sim_lab} (int {is_:.0f}/pulse)')
    ax.fill_between(tc[m], sim[m], color=GREEN, alpha=0.12, zorder=1)
    ax.plot(tc[m], rate[m], color=BLUE, lw=0.8, zorder=3)
    ax.errorbar(tc[m], rate[m], yerr=rerr[m], fmt='o', ms=4, color=BLUE,
                capsize=0, elinewidth=0.9, zorder=4,
                label=f'{data_lab} (int {id_:.0f}/pulse)')
    if t_acc is not None:
        acnt = np.histogram(t_acc, bins=tedges)[0].astype(float)
        arate = acnt / n_ev / dt
        ia = float((arate[m] * dt[m]).sum())
        corr = rate - arate
        icorr = float((corr[m] * dt[m]).sum())
        ax.plot(tc[m], arate[m], color=RED, lw=1.6, ls='--', zorder=3,
                label=f'accidental estimate (int {ia:.0f}/pulse)')
        ax.plot(tc[m], corr[m], color=BLUE, lw=0.8, ls=':', zorder=3)
        ax.plot(tc[m], corr[m], 'o', ms=4, mfc='white', mec=BLUE, mew=1.1, zorder=3,
                label=f'nTof DAQ - accidental (int {icorr:.0f}/pulse)')
        print(f'  {outname:36s} data {id_:7.1f}  acc {ia:6.1f}  data-acc {icorr:6.1f}'
              f'  g4 {is_:6.1f}  (data-acc)/g4 {icorr/is_:.2f}')
    else:
        print(f'  {outname:36s} data {id_:7.1f}  g4 {is_:7.1f}  data/g4 {id_/is_:.2f}')

    ax.axvline(5.3, color='0.5', ls=':', lw=1.1)
    ax.annotate('thermal peak ~5.3 ms\n(E ~ 71 meV)', (5.5, 0.93 * rate[m].max()),
                fontsize=9, color='0.35', va='top')
    ax.set_xlim(GATE_MS, tmax); ax.set_ylim(bottom=0)
    ax.set_xlabel('neutron arrival time  t [ms]   (time past gamma flash; gate t > 1 ms)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    trigger_diagram.draw(ax, mode, wall)
    fig.tight_layout()
    fig.savefig(OUT / outname, dpi=140); plt.close(fig)


def main():
    print(f'{STEM}: thermal-gate DATA vs GEANT4 (per pulse per ms)'
          f'{"  [+accidentals]" if HAS_ACC else ""}\n')
    T = f'Thermal-gate rate: nTof DAQ ({STEM}) vs GEANT4   (t > 1 ms)'

    # 1) total singles (coincidence), summed over 4 walls
    compare(data_times([f'singles_tof_{a}' for a in 'ABCD']), sim_rate['leg'],
            f'singles_vs_time_datasim_{STEM}.png', T,
            'total singles  [ / pulse / ms ]',
            f'nTof DAQ: total singles, Sigma 4 walls', 'GEANT4 legs, 2.0 cm plastic',
            'sum', None,
            t_acc=data_times([f'singles_acc_tof_{a}' for a in 'ABCD']) if HAS_ACC else None)

    # 2) per-wall coincidence vs GEANT4 legs / 4  (+ accidentals)
    print()
    for a in 'ABCD':
        compare(data_times([f'singles_tof_{a}']), sim_rate['leg'] / 4,
                f'coinc_WAL{a}_datasim_{STEM}.png', T,
                f'WAL{a} coincidence  [ / pulse / ms ]',
                f'nTof DAQ: WAL{a} coincidence', 'GEANT4 legs / 4 (per arm)',
                'coinc', a,
                t_acc=data_times([f'singles_acc_tof_{a}']) if HAS_ACC else None)

    # 3) per-wall SiPM-wall singles (M1 wall-OR) vs GEANT4 SiPM / 4
    print()
    for a in 'ABCD':
        compare(data_times([f'wallor_tof_{a}']), sim_rate['sipm'] / 4,
                f'sipm_WAL{a}_datasim_{STEM}.png', T,
                f'WAL{a} SiPM-wall singles  [ / pulse / ms ]',
                f'nTof DAQ: WAL{a} SiPM-wall singles', 'GEANT4 SiPM singles / 4',
                'sipm', a)

    # 4) per-wall plastic singles (M2 plastic-OR) vs GEANT4 plastic / 4
    print()
    for a in 'ABCD':
        compare(data_times([f'plasor_tof_{a}']), sim_rate['plas'] / 4,
                f'plastic_WAL{a}_datasim_{STEM}.png', T,
                f'WAL{a} plastic singles  [ / pulse / ms ]',
                f'nTof DAQ: WAL{a} plastic singles', 'GEANT4 plastic singles / 4',
                'plastic', a)

    print(f'\n-> figures in {OUT}/')


if __name__ == '__main__':
    main()

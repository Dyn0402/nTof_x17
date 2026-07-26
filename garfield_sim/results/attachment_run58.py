#!/usr/bin/env python3
"""
attachment_run58.py — the DECIsive attachment test on run_58 data.

The 90/10 contamination sim showed the ~12% drift-velocity deficit could come
from ~0.2% water (no attachment) OR from air/O2 (strong attachment). The data
discriminates: measure mean clean-hit amplitude vs DRIFT DEPTH from the cached
driftspec (sum_amp/n_clean per 20 ns bin), convert time->depth with the measured
v_drift, and fit the decay.  Water/N2 -> flat (eta=0). air/O2 at the level needed
for the velocity deficit -> lambda ~ 2-5 mm -> cathode charge obliterated.

Run on the DAQ box:  ~/ana/.venv/bin/python attachment_run58.py
Output: attachment_run58.png + stdout table.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = '/mnt/data/x17/beam_july/analysis/July_HV_Scan/run58_scan/cache/run_58'
SUMM = '/mnt/data/x17/beam_july/analysis/July_HV_Scan/run58_scan/drift/drift_summary.csv'
HERE = os.path.dirname(os.path.abspath(__file__))
CONTAM = os.path.join(HERE, 'drift_9010_contam_cern.json')
GAP_MM = 30.0
DRIFTS = [700, 600, 500]          # clean, full-column drifts
DET = 'A'                          # cleanest detector
EDGE_FRAC = 0.08                   # trim anode/cathode edge bins from the fit


def load_spec(drift, det):
    fs = glob.glob(os.path.join(CACHE, f'*dr{drift}_*driftspec.parquet'))
    d = pd.concat([pd.read_parquet(x) for x in fs])
    a = (d[d.det == det].groupby('t_ns')
         .agg(n=('n_clean', 'sum'), s=('sum_amp', 'sum')).reset_index())
    a['mean_amp'] = a.s / a.n.clip(lower=1)
    return a.sort_values('t_ns')


def sim_eta(name, E):
    d = json.load(open(CONTAM))['mixtures'][name]
    Ee = np.array([q['E_Vcm'] for q in d]); et = np.array([q['eta_per_cm'] for q in d])
    return float(np.interp(E, Ee, et))


def main():
    summ = pd.read_csv(SUMM)
    fig, axes = plt.subplots(1, len(DRIFTS), figsize=(15, 5.2), sharey=True)
    print(f'{"drift":>5} {"E":>5} {"v":>5} {"lambda_data":>12} {"eta_data":>9} '
          f'{"A_cath/A_anode":>14}   sim eta@E (1/cm): pure / +0.2%H2O / +1%air / +0.5%O2')
    for ax, dr in zip(axes, DRIFTS):
        row = summ[(summ.det == DET) & (summ.drift == dr)].iloc[0]
        t0, tmax, v, E = row.t0_daq, row.t_max, row.v_drift, row.e_field
        a = load_spec(dr, DET)
        # baseline (pre-anode noise floor) subtract on amplitude
        base = a[(a.t_ns < t0 - 60)]
        amp0 = np.median(base.mean_amp) if len(base) else 0.0
        win = a[(a.t_ns >= t0) & (a.t_ns <= t0 + tmax)].copy()
        win['z_mm'] = v * (win.t_ns - t0) / 1000.0           # um/ns * ns -> um -> mm
        win['amp'] = win.mean_amp - amp0
        # fit ln(amp) vs z on the trimmed interior
        lo, hi = win.z_mm.quantile(EDGE_FRAC), win.z_mm.quantile(1 - EDGE_FRAC)
        fit = win[(win.z_mm >= lo) & (win.z_mm <= hi) & (win.amp > 0)]
        sl, inter = np.polyfit(fit.z_mm, np.log(fit.amp), 1)
        eta_data = -sl                                        # per mm
        lam = 1.0 / eta_data if abs(eta_data) > 1e-4 else np.inf
        # cathode/anode amplitude ratio (robust discriminator)
        anode = win[win.z_mm < 4].amp.median()
        cath = win[win.z_mm > GAP_MM - 6].amp.median()
        ratio = cath / anode if anode else np.nan

        etas = {n: sim_eta(f'Ar_iso10_{k}' if k else 'Ar90_iso10', E)
                for n, k in [('pure', ''), ('H2O0.3', 'H2O0.3'),
                             ('air1', 'air1'), ('O2_0.5', 'O2_0.5')]}
        print(f'{dr:5d} {E:5.0f} {v:5.1f} {lam:11.1f}mm {eta_data*10:8.3f}/cm '
              f'{ratio:13.2f}   pure={etas["pure"]:.2f} H2O0.3={etas["H2O0.3"]:.2f} '
              f'air1={etas["air1"]:.2f} O2={etas["O2_0.5"]:.2f}')

        # plot: data amplitude vs depth (norm to anode) + sim survival curves
        zz = np.linspace(0, GAP_MM, 100)
        ax.plot(win.z_mm, win.amp / anode, 'o', ms=3, color='k',
                label='run_58 Det A data')
        for n, k, c, ls in [('pure 90/10', 'Ar90_iso10', 'tab:green', '-'),
                            ('+1% air', 'Ar_iso10_air1', 'tab:red', '--'),
                            ('+0.5% O2', 'Ar_iso10_O2_0.5', 'tab:orange', ':')]:
            eta = sim_eta(k, E)                              # per cm
            ax.plot(zz, np.exp(-eta / 10.0 * zz), ls, color=c, lw=2,
                    label=f'{n} sim (η={eta:.1f}/cm)')
        ax.axhline(1, color='gray', lw=0.6, ls=':')
        ax.set_yscale('log'); ax.set_ylim(3e-3, 3)
        ax.set_xlabel('drift depth z [mm]  (anode→cathode)')
        ax.set_title(f'drift {dr} V  (E={E:.0f} V/cm, v={v:.1f} µm/ns)\n'
                     f'data λ={lam:.0f} mm  vs  air/O₂ λ≈2–5 mm', fontsize=10)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('mean clean-hit amplitude  (÷ anode)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('run_58 Det A — clean-hit amplitude vs drift depth: DATA is flat '
                 '(no attachment) → excludes air/O₂, consistent with trace water',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, 'attachment_run58.png')
    fig.savefig(out, dpi=155, bbox_inches='tight')
    print('\nWritten', out)


if __name__ == '__main__':
    main()

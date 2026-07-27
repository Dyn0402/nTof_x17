#!/usr/bin/env python3
"""
analyze_9010_contam.py — compare the run_58 measured July-beam drift velocity
(Ar/iso 90/10, CERN) against the pure-90/10 Magboltz curve and the contaminated
candidates from mm_drift_9010_contam_cern.py.

Two questions:
  1) v(E): what contaminant FRACTION reproduces the ~12-16% velocity deficit on
     the clean Det A?  -> RMS fit over the reliable high-field points.
  2) eta(E): does the candidate ALSO attach (amplitude decay with drift depth)?
     water attaches weakly, O2/air strongly -> the discriminator.  We report the
     surviving-charge fraction over the 30 mm gap, exp(-eta*3cm).

Output: results/drift_9010_contam_cern.png (+ _attachment.png) and a stdout table.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
GAP_CM = 3.0                      # nominal 30 mm drift gap
FIT_EMIN, FIT_EMAX = 145.0, 240.0  # reliable band (below ~150 V/cm the column is
                                   # window-truncated on the 3.84 us readout)
DETS = ['A', 'C', 'D']           # B has pathological t0_daq (negative) -> skip in fit
FIT_DET = 'A'                    # cleanest detector, used_pure True, monotonic


def load_meas():
    df = pd.read_csv(os.path.join(RES, 'run58_drift_summary.csv'))
    return df


def load_curves():
    curves = {}
    # pure 90/10 CERN reference
    p = os.path.join(RES, 'drift_velocity_Ar_iC4H10_90_10_CERN.json')
    if os.path.exists(p):
        d = json.load(open(p))
        curves['Ar90_iso10 (pure ref)'] = (
            np.array([q['E_Vcm'] for q in d['points']]),
            np.array([q['v_um_per_ns'] for q in d['points']]),
            None)
    # contamination suite
    p = os.path.join(RES, 'drift_9010_contam_cern.json')
    d = json.load(open(p))
    for name, pts in d['mixtures'].items():
        E = np.array([q['E_Vcm'] for q in pts])
        V = np.array([q['v_um_per_ns'] for q in pts])
        eta = np.array([q.get('eta_per_cm', 0.0) for q in pts])
        curves[name] = (E, V, eta)
    return curves


LABELS = {
    'Ar90_iso10': 'pure 90/10',
    'Ar_iso10_H2O0.3': '+0.3% H2O', 'Ar_iso10_H2O0.5': '+0.5% H2O',
    'Ar_iso10_H2O1.0': '+1.0% H2O', 'Ar_iso10_H2O1.5': '+1.5% H2O',
    'Ar_iso10_H2O2.0': '+2.0% H2O', 'Ar_iso10_H2O3.0': '+3.0% H2O',
    'Ar_iso10_air1': '+1% air', 'Ar_iso10_air2': '+2% air', 'Ar_iso10_air3': '+3% air',
    'Ar_iso10_O2_0.5': '+0.5% O2', 'Ar_iso10_O2_1.0': '+1.0% O2',
    'Ar_iso10_N2_1': '+1% N2', 'Ar_iso10_N2_2': '+2% N2', 'Ar_iso10_N2_5': '+5% N2',
}
FAMILY_COLOR = {'H2O': 'tab:blue', 'air': 'tab:red', 'O2': 'tab:orange',
                'N2': 'tab:green', 'pure': 'black'}


def family(name):
    for k in ('H2O', 'air', 'O2', 'N2'):
        if k in name:
            return k
    return 'pure'


def main():
    df = load_meas()
    curves = load_curves()

    # ---- v(E) figure ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    detcol = {'A': 'k', 'C': 'dimgray', 'D': 'darkgray'}
    for det in DETS:
        d = df[df['det'] == det].sort_values('e_field')
        mk = 'o' if det == 'A' else ('s' if det == 'C' else '^')
        ax.plot(d['e_field'], d['v_drift'], mk, color=detcol[det], ms=8 if det == 'A' else 6,
                zorder=6, label=f'measured Det {det}'
                + (' (clean, fit)' if det == 'A' else ''))

    shade = {'H2O': np.linspace(0.35, 1.0, 6), 'air': np.linspace(0.5, 1.0, 3),
             'O2': np.linspace(0.55, 1.0, 2), 'N2': np.linspace(0.5, 1.0, 3)}
    fam_idx = {k: 0 for k in shade}
    for name, (E, V, eta) in curves.items():
        base = name.replace(' (pure ref)', '')
        fam = family(base)
        if fam == 'pure':
            ax.plot(E, V, '-', color='purple', lw=2.6, label='Magboltz pure 90/10', zorder=5)
            continue
        alpha = shade[fam][min(fam_idx[fam], len(shade[fam]) - 1)]
        fam_idx[fam] += 1
        ax.plot(E, V, '-', color=FAMILY_COLOR[fam], lw=1.5, alpha=alpha,
                label=LABELS.get(base, base))
    ax.axvspan(FIT_EMIN, FIT_EMAX, color='gold', alpha=0.12, zorder=0)
    ax.set_xlabel('drift field  E = HV / 3 cm   [V/cm]')
    ax.set_ylabel('drift velocity [µm/ns]')
    ax.set_title('run_58 July-beam measured v(E) vs Ar/iso 90/10 + contamination (CERN 720.8 Torr)')
    ax.set_xlim(40, 320)
    ax.set_ylim(10, 47)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    fig.tight_layout()
    out = os.path.join(RES, 'drift_9010_contam_cern.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('Written', out)

    # ---- RMS fit table (Det A, reliable band) ----
    d = df[df['det'] == FIT_DET]
    m = (d['e_field'] >= FIT_EMIN) & (d['e_field'] <= FIT_EMAX)
    Em, Vm = d['e_field'].values[m], d['v_drift'].values[m]
    print(f'\nFit band E in [{FIT_EMIN:.0f},{FIT_EMAX:.0f}] V/cm, Det {FIT_DET}, '
          f'{m.sum()} points: v_meas = {np.array2string(Vm, precision=1)}')
    print(f'\n{"candidate":22s} {"RMS dev":>9s}  {"surv.30mm":>10s}')
    rows = []
    for name, (E, V, eta) in curves.items():
        base = name.replace(' (pure ref)', '')
        pred = np.interp(Em, E, V)
        rms = float(np.sqrt(np.mean((pred - Vm) ** 2)))
        surv = None
        if eta is not None:
            eta_m = np.interp(Em.mean(), E, eta)
            surv = float(np.exp(-max(eta_m, 0.0) * GAP_CM))
        rows.append((base, rms, surv))
    for base, rms, surv in sorted(rows, key=lambda r: r[1]):
        s = f'{surv*100:8.1f} %' if surv is not None else '     n/a'
        print(f'{LABELS.get(base, base):22s} {rms:8.2f}  {s:>10s}')

    # ---- attachment discriminator figure ----
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for name, (E, V, eta) in curves.items():
        base = name.replace(' (pure ref)', '')
        if eta is None:
            continue
        fam = family(base)
        if fam in ('pure',):
            continue
        surv = np.exp(-np.clip(eta, 0, None) * GAP_CM)
        ax2.plot(E, surv * 100, '-', color=FAMILY_COLOR[fam], lw=1.6,
                 alpha=0.85, label=LABELS.get(base, base))
    ax2.axvspan(FIT_EMIN, FIT_EMAX, color='gold', alpha=0.12)
    ax2.set_xlabel('drift field [V/cm]')
    ax2.set_ylabel('charge surviving 30 mm drift  exp(-η·3cm)  [%]')
    ax2.set_title('Attachment discriminator: water attaches weakly, O2/air strongly')
    ax2.set_xlim(40, 320)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    out2 = os.path.join(RES, 'drift_9010_contam_attachment.png')
    fig2.savefig(out2, dpi=160, bbox_inches='tight')
    print('\nWritten', out2)


if __name__ == '__main__':
    main()

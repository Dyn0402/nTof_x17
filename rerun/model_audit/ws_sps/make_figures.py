#!/usr/bin/env python3
"""make_figures.py [--dir DIR] -- slide-ready PNGs for ws_sps (rerun/model_audit/ws_sps/figures/).

Reads headon_summary.json + headon_keep.pkl (analyse_headon.py), zs_treatment_*.pkl (zs_treatment.py collect),
known-angle npz (make_sps_dag.py arms base/nozs/zsb4) and res_*_{cens,zsb4}.npz + the res caches (resolution_cache.py).
Each figure is drawn only if its inputs exist.
"""
import argparse, glob, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, 'figures')
E = {'raw700': 243, 'raw450': 156, 'raw275': 96}
PROD = {'x': 0.238, 'y': 0.343}
PROF = {'x': 0.42, 'y': 0.50}
C = {'raw700': '#1f5fa8', 'raw450': '#3a9d5d', 'raw275': '#c9792b', 'raw700_zs4': '#7a3fa8', 'raw700_zs5': '#b03a6a',
     'raw450_zs4': '#7a3fa8', 'raw275_zs4': '#7a3fa8', 'raw700_nocens': '#888888'}
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def fig_profiles(S, K, S0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, v in zip(axs, 'xy'):
        for tag, lab in (('raw700', 'RAW 243 V/cm'), ('raw450', 'RAW 156 V/cm'), ('raw275', 'RAW 96 V/cm'),
                         ('raw700_zs4', 'ZS 4σ emulated, 243'), ('raw700_zs5', 'ZS 5σ emulated, 243')):
            if tag in S and v in S[tag]:
                p = np.array(S[tag][v]['prof_s0']); ls = '--' if 'zs' in tag else '-'
                ax.plot(S0, p / max(S[tag][v]['n'], 1), ls, color=C[tag], lw=2, label=f'{lab}: {S[tag][v]["s0"]:.2f} mm')
        ax.axvline(PROD[v], color='k', lw=1, ls=':'); ax.text(PROD[v], ax.get_ylim()[1] * 0.02 if False else 0, ' production', rotation=90,
                                                                 va='bottom', fontsize=10)
        ax.axvline(PROF[v], color='0.5', lw=1, ls=':'); ax.text(PROF[v], 0, ' profile "wide"', rotation=90, va='bottom', fontsize=10, color='0.4')
        ax.set_xlabel('prompt footprint σ0 [mm]'); ax.set_title(f'{v.upper()} view (strips {"across" if v == "x" else "along"} the resistive strips)')
        ax.set_ylim(0, None); ax.legend(fontsize=9, frameon=False)
    axs[0].set_ylabel('Δχ² per event, ÷ χ²/dof')
    fig.suptitle('det4 at SPS, head-on beam: which footprint do the waveforms want?  (Dp, D_rc profiled)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sigma0_profiles.png'), dpi=130); plt.close(fig)


def fig_dp(S):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    f = np.linspace(80, 260, 50)
    mb = S.get('magboltz_dp', {})
    xs = [E[t] for t in ('raw700', 'raw450', 'raw275') if t in mb]
    ax.plot(xs, [mb[t] for t in ('raw700', 'raw450', 'raw275') if t in mb], 'k--', lw=1.5, label='Magboltz, run_71 beam gas (wet CF4 mix)')
    for v, mk in (('x', 'o'), ('y', 's')):
        pts = [(E[t], S[t][v]['dp'], S[t][v]['dp_err']) for t in ('raw700', 'raw450', 'raw275') if t in S and v in S[t]]
        if pts:
            a = np.array(pts); ax.errorbar(a[:, 0] + (2 if v == 'y' else -2), a[:, 1], a[:, 2], fmt=mk, ms=8, capsize=3, label=f'fit, {v.upper()} view')
    ax.set_xlabel('drift field [V/cm]'); ax.set_ylabel('drift diffusion Dp [mm/√ns]')
    ax.set_title('Drift diffusion from head-on SPS tracks vs Magboltz'); ax.legend(frameon=False, fontsize=10); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'dp_vs_field.png'), dpi=130); plt.close(fig)


def fig_amp(S):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, v in zip(axs, 'xy'):
        for tag in ('raw700', 'raw450', 'raw275'):
            if tag in S and v in S[tag]:
                T = S[tag][v]['terciles']
                ax.plot([1, 2, 3], [t['s0_at_global'] for t in T], 'o-', color=C[tag], lw=2, ms=8, label=f'{E[tag]} V/cm')
        ax.axhline(PROD[v], color='k', ls=':', lw=1); ax.axhline(PROF[v], color='0.5', ls=':', lw=1)
        ax.text(3.05, PROD[v], 'production', va='center', fontsize=9); ax.text(3.05, PROF[v], 'profile "wide"', va='center', fontsize=9, color='0.4')
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['faint third', 'middle third', 'bright third'])
        ax.set_title(f'{v.upper()} view'); ax.set_xlim(0.7, 3.6)
    axs[0].set_ylabel('best σ0 [mm]'); axs[0].legend(frameon=False)
    fig.suptitle('Does the footprint grow with pulse height?  (head-on, no tilt to trade against)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sigma0_vs_amplitude.png'), dpi=130); plt.close(fig)


def fig_zs(d):
    fs = sorted(glob.glob(os.path.join(d, 'zs_treatment_*_zs*.pkl')))
    fs = [f for f in fs if '_c' not in os.path.basename(f)[-8:]]
    if not fs:
        return
    f = [x for x in fs if 'raw700_zs4' in x] or fs
    R = pickle.load(open(f[0], 'rb'))
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    lab = {'zero': 'zeros = measured 0', 'cens': 'zeros censored (F16/F20)', 'bound': 'zeros = below threshold (new)'}
    col = {'zero': '#888888', 'cens': '#c9392b', 'bound': '#1f5fa8'}
    for j, v in enumerate('xy'):
        rows = [r[v] for r in R if v in r and all(a in r[v] for a in ('raw', 'zero', 'cens', 'bound'))]
        for i, (q, unit, rng) in enumerate((('t0', 'ns', 300), ('p0', 'µm', 600))):
            ax = axs[i, j]
            for a in ('zero', 'cens', 'bound'):
                dd = np.array([r[a][q] - r['raw'][q] for r in rows]) * (1000 if q == 'p0' else 1)
                ax.hist(np.clip(dd, -rng, rng), bins=60, range=(-rng, rng), histtype='step', lw=2, color=col[a],
                        label=f'{lab[a]}: rσ {rsig(dd):.0f} {unit}')
            ax.set_xlabel(f'{q} (ZS 4σ) − {q} (RAW) [{unit}]'); ax.legend(fontsize=9, frameon=False)
            ax.set_title(f'{v.upper()} view, {"track time" if q == "t0" else "position at the mesh"}')
    fig.suptitle('Same events, RAW vs zero-suppressed: how should a suppressed sample enter the fit?', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'zs_treatment.png'), dpi=130); plt.close(fig)


def fig_known(d):
    arms = {'base': ('zeros censored', '#c9392b'), 'nozs': ('zeros = 0', '#888888'), 'zsb4': ('zeros = below threshold', '#1f5fa8')}
    pls = [p for p in ('d675', 'd575', 'd475') if os.path.exists(os.path.join(d, f'run63_rot25_operating_00_{p}_base.npz'))]
    if not pls:
        return
    fig, axs = plt.subplots(1, len(pls), figsize=(5 * len(pls), 4.4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, pls):
        for a, (lab, c) in arms.items():
            f = os.path.join(d, f'run63_rot25_operating_00_{p}_{a}.npz')
            if not os.path.exists(f):
                continue
            per = np.load(f, allow_pickle=True)['per']
            y = per[per[:, 1] == 1]; w = y[:, 3]; uend = y[:, 14]
            r = w / np.median(w)
            ax.hist(np.clip(r, 0, 2), bins=60, range=(0, 2), histtype='step', lw=2, color=c,
                    label=f'{lab}: rσ {rsig(r) * 100:.1f} %, |r−1|>0.3 {np.mean(np.abs(r - 1) > 0.3) * 100:.0f} %')
        ax.set_title(f'25.64° mount, plateau {p}'); ax.set_xlabel('per-event slope / median slope'); ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Real ZS data at a known angle: spread of the fitted slope (inclined view)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'known_angle_zs.png'), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps'))
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    sp = os.path.join(a.dir, 'headon_summary.json')
    if os.path.exists(sp):
        S = json.load(open(sp))
        S0 = np.round(np.arange(0.10, 0.601, 0.05), 3)
        fig_profiles(S, None, S0); fig_dp(S); fig_amp(S)
    fig_zs(a.dir); fig_known(a.dir)
    print('figures in', FIG, sorted(os.listdir(FIG)))


if __name__ == '__main__':
    main()

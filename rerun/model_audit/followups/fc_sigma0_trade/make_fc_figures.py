#!/usr/bin/env python3
"""make_fc_figures.py -- figures for FC (items 5, 6) from results/fc_headon.json and results/fc_trade.json.

figures/fc_sigma0_xy.png      sigma0_X vs sigma0_Y per dataset, production X model and with the bounded X RC term
figures/fc_sigma0_profiles.png chi2 profiles in sigma0, X and Y, SPS 243/156/96 V/cm
figures/fc_trade_response.png  symmetric angle response S(|tan|) per chamber/view: before (production) vs all corrections
figures/fc_trade_sigma.png     compression C and brightness B against the fixed sigma0 (the trade), all chambers
figures/fc_trade_arms.png      C, B, s68 per arm (prod -> pin -> fixed sigma0 -> +map -> 20 ns)
Captions: figures/captions.txt
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, 'figures'); RES = os.path.join(HERE, 'results')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
CAP = {}


def headon():
    f = os.path.join(RES, 'fc_headon.json')
    if not os.path.exists(f):
        return
    H = json.load(open(f))
    tags = [t for t in H if not t.endswith('_asym') and not (t.startswith('bench_') and f'bench05_{t[6:]}' in H)]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for i, t in enumerate(tags):
        r = H[t]; M, E = r['models'], r['boot_err']
        lab = t.replace('sps_raw700', 'SPS 243 V/cm').replace('sps_raw450', 'SPS 156 V/cm').replace('sps_raw275', 'SPS 96 V/cm').replace('bench05_', 'bench ').replace('bench_', 'bench ')
        c = f'C{i}'
        ax.errorbar(M['prod']['s0_x'], M['prod']['s0_y'], xerr=E['prod_s0_x'], yerr=E['prod_s0_y'], fmt='o', color=c, label=lab)
        ax.errorbar(M['free']['s0_x'], M['free']['s0_y'], xerr=E['free_s0_x'], yerr=E['free_s0_y'], fmt='s', mfc='none', color=c)
        ax.annotate('', (M['free']['s0_x'], M['free']['s0_y']), (M['prod']['s0_x'], M['prod']['s0_y']),
                    arrowprops=dict(arrowstyle='->', color=c, lw=0.7))
    ax.plot([0.1, 0.5], [0.1, 0.5], 'k--', lw=0.8); ax.set_xlim(0.1, 0.45); ax.set_ylim(0.1, 0.45)
    ax.set_xlabel('σ0 X [mm]'); ax.set_ylabel('σ0 Y [mm]')
    ax.set_title('Prompt footprint per view (● production X model, □ + bounded X RC)', fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fc_sigma0_xy.png'), dpi=130); plt.close(fig)
    CAP['fc_sigma0_xy.png'] = 'σ0 X vs Y on head-on tracks: SPS det4 RAW (3 fields) and five bench chambers; dashed = one common σ0; arrows show what modelling the X across-strip RC does.'
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for a, t in zip(ax, ['sps_raw700', 'sps_raw450', 'sps_raw275']):
        if t not in H:
            continue
        r = H[t]; S0 = np.array(r['grid']['S0']); s = r['chi2dof']
        a.plot(S0, np.array(r['prof']['x_prod']) / s, 'o-', label='X, production (no X RC)')
        a.plot(S0, np.array(r['prof']['x']) / s, 's-', label='X, + bounded across-strip RC')
        a.plot(S0, np.array(r['prof']['y']) / s, '^-', label='Y, RC free')
        a.set_ylim(0, 400); a.set_xlabel('σ0 [mm]'); a.set_title(t.replace('sps_', 'SPS '), fontsize=9)
    ax[0].set_ylabel('Δχ² / (χ²/dof)'); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fc_sigma0_profiles.png'), dpi=130); plt.close(fig)
    CAP['fc_sigma0_profiles.png'] = 'SPS det4 head-on RAW: σ0 profile of the summed χ² (other parameters minimised) per view and field.'


def trade():
    f = os.path.join(RES, 'fc_trade.json')
    if not os.path.exists(f):
        return
    T = json.load(open(f))
    mids = [0.055, 0.11, 0.17, 0.24, 0.36]
    fig, axs = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True, sharey=True)
    for j, det in enumerate(DETS):
        for i, v in enumerate('xy'):
            a = axs[i, j]
            for arm, mp, st, lab in (('prod', 'nomap', 'o-', 'production (free t0, σ0 choice, no map)'),
                                      ('s31', 'map', 's-', 'pinned t0 + σ0 0.31 + map'), ('s31g20', 'map', '^-', '... + 20 ns grid')):
                k = f'{det}|{v}|{arm}|{mp}'
                if k in T:
                    S = np.array(T[k]['S'], float); e = np.array(T[k]['err']['S'], float)
                    ref = np.nanmean(S[1:3])
                    a.errorbar(mids, S / ref, yerr=e / ref, fmt=st, ms=4, label=lab)
            a.axhline(1, color='k', lw=0.5); a.set_ylim(0.85, 1.12); a.set_title(f'{det} {v.upper()}', fontsize=9)
            if i == 1:
                a.set_xlabel('|tan θ_ref|')
    axs[0, 0].set_ylabel('angle scale / shallow'); axs[1, 0].set_ylabel('angle scale / shallow'); axs[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fc_trade_response.png'), dpi=130); plt.close(fig)
    CAP['fc_trade_response.png'] = 'Symmetric angle response vs |tan| (normalised to 0.08–0.20), held-out bench events: production vs all corrections (pinned t0, σ0 0.31 fixed, cross-validated bow map) and with the 20 ns grid.'
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sig = [0.24, 0.31, 0.38, 0.45]
    for j, det in enumerate(DETS):
        for v, mk in (('x', 'o-'), ('y', 's--')):
            ks = [f'{det}|{v}|s{int(s*100)}|map' for s in sig]
            if not all(k in T for k in ks):
                continue
            ax[0].plot(sig, [T[k]['ladder'] / T[ks[1]]['ladder'] for k in ks], mk, color=f'C{j}', label=f'{det} {v.upper()}')
            ax[1].errorbar(sig, [100 * T[k]['C'] for k in ks], yerr=[100 * T[k]['err']['C'] for k in ks], fmt=mk, color=f'C{j}')
            ax[2].errorbar(sig, [100 * T[k]['B'] for k in ks], yerr=[100 * T[k]['err']['B'] for k in ks], fmt=mk, color=f'C{j}')
    ax[0].set_ylabel('angle scale (ladder) / at σ0 0.31'); ax[1].set_ylabel('compression C [%] (steep / shallow − 1)')
    ax[2].set_ylabel('brightness B [%] (bright / faint − 1)')
    for a in ax:
        a.set_xlabel('fixed σ0 (both views) [mm]'); a.axhline(0 if a is not ax[0] else 1, color='k', lw=0.5)
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fc_trade_sigma.png'), dpi=130); plt.close(fig)
    CAP['fc_trade_sigma.png'] = 'The width–tilt trade with everything else on: global angle scale, steep-track compression and bright-track reading vs the fixed σ0, five chambers × two views.'
    arms = [('prod', 'nomap', 'production'), ('prod', 'map', '+ map'), ('lat', 'map', 'pin t0 (σ0 choice)'),
            ('s31', 'map', 'pin + σ0 0.31'), ('split', 'map', 'pin + 0.27/0.33'), ('s31g20', 'map', '+ 20 ns')]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(arms))
    for j, det in enumerate(DETS):
        for v, mk, off in (('x', 'o', -0.15), ('y', 's', 0.15)):
            for q, a in (('C', ax[0]), ('B', ax[1]), ('s68', ax[2])):
                ys = []; es = []
                for arm, mp, _ in arms:
                    k = f'{det}|{v}|{arm}|{mp}'
                    ys.append(T[k][q] * (100 if q != 's68' else 1) if k in T else np.nan)
                    es.append(T[k]['err'][q] * (100 if q != 's68' else 1) if k in T else np.nan)
                a.errorbar(x + off + 0.05 * (j - 2), ys, yerr=es, fmt=mk, color=f'C{j}', ms=4, label=f'{det} {v.upper()}' if q == 'C' else None)
    for a, yl in zip(ax, ['compression C [%]', 'brightness B [%]', 'angle s68 [°]']):
        a.set_xticks(x); a.set_xticklabels([l for *_, l in arms], rotation=30, ha='right', fontsize=8); a.set_ylabel(yl)
        if yl != 'angle s68 [°]':
            a.axhline(0, color='k', lw=0.5)
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fc_trade_arms.png'), dpi=130); plt.close(fig)
    CAP['fc_trade_arms.png'] = 'Compression, brightness dependence and angle s68 as the corrections are added one by one (same 1320 held-out events per chamber).'


def main():
    os.makedirs(FIG, exist_ok=True)
    headon(); trade()
    CAP.setdefault('stackup_mechanisms.png', 'Left: static induction through the 70 µm insulator is 0.07 mm and identical for both views (same pad plane). Right: RC spreading in the resistive sheet — same D, but across the strips (X) it stops at w²/12 within 20–60 ns.')
    old = {}
    cf = os.path.join(FIG, 'captions.txt')
    if os.path.exists(cf):
        for l in open(cf):
            if ': ' in l:
                k, v = l.rstrip('\n').split(': ', 1); old[k] = v
    old.update(CAP)
    open(cf, 'w').write(''.join(f'{k}: {v}\n' for k, v in sorted(old.items())))


if __name__ == '__main__':
    main()

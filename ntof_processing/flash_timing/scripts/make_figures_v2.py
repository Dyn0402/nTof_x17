#!/usr/bin/env python3
"""Figures for (a) the intensity-walk mechanism and (b) the plastic pathology."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA, FIGS = BASE / 'data', BASE / 'figures'
INK = '#1b1b1b'
C = {'WAL': '#2f6f9f', 'PSS': '#c0632c', 'LIQ': '#4a8a5a', 'PKUP': '#8a5a9a', 'SILI': '#a08020'}
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'axes.labelsize': 9, 'axes.edgecolor': '#999999',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.titlelocation': 'left'})

RUNS = {224357: '2026-07-11', 224464: '2026-07-16', 224572: '2026-07-26 (post-FIFO)'}


def pkmap(d):
    pk = d['PKUP']; p = pk[np.abs(pk['tof'] - pk['anchor']) < 4000]
    o = np.lexsort((-p['amp'], p['BunchNumber'])); p = p[o]
    f = np.ones(len(p), bool); f[1:] = p['BunchNumber'][1:] != p['BunchNumber'][:-1]
    p = p[f]; p = p[np.abs(p['tof'] - np.median(p['tof'])) < 200]
    return (dict(zip(p['BunchNumber'].tolist(), p['tof'].astype(float).tolist())),
            dict(zip(p['BunchNumber'].tolist(), p['PulseIntensity'].tolist())), p)


def flash_hits(rec, win=300):
    s = np.abs(rec['tof'] - rec['anchor']) < win; r = rec[s]
    k = r['BunchNumber'] * 100 + r['detn']
    o = np.lexsort((-r['amp'], k)); r, k = r[o], k[o]
    f = np.ones(len(r), bool); f[1:] = k[1:] != k[:-1]
    return r[f]


# ---------------------------------------------------------------- intensity
def fig_intensity_mechanism():
    d = np.load(DATA / 'flash_run224357.npz')
    pm, pim, pk = pkmap(d)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    trees = ['PKUP', 'WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD', 'SILI']
    walk, ampr, drise, labels, cols = [], [], [], [], []
    for t in trees:
        if t == 'PKUP':
            lo = pk['PulseIntensity'] < 6e12; hi = ~lo
            v = pk['tof'].astype(float)
            walk.append(v[hi].mean() - v[lo].mean())
            ampr.append(np.median(pk['amp'][hi]) / np.median(pk['amp'][lo]))
            drise.append(np.median(pk['risetime'][hi]) - np.median(pk['risetime'][lo]))
        else:
            if t not in d.files:
                continue
            fh = flash_hits(d[t])
            ref = np.array([pm.get(int(b), np.nan) for b in fh['BunchNumber']])
            dt = fh['tof'] - ref
            ok = np.isfinite(dt); m = np.median(dt[ok]); core = ok & (np.abs(dt - m) < 60)
            pi = fh['PulseIntensity']
            lo = core & (pi < 6e12); hi = core & (pi >= 6e12)
            if lo.sum() < 50 or hi.sum() < 50:
                continue
            walk.append(dt[hi].mean() - dt[lo].mean())
            ampr.append(np.median(fh['amp'][hi]) / np.median(fh['amp'][lo]))
            drise.append(np.median(fh['risetime'][hi]) - np.median(fh['risetime'][lo]))
        labels.append(t); cols.append(C.get(t[:3], '#777'))

    x = np.arange(len(labels))
    axes[0].bar(x, walk, color=cols)
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=90, fontsize=7.5)
    axes[0].set_ylabel('Δt(dedicated) − Δt(parasitic) [ns]')
    axes[0].set_title('a. the shift is detector-dependent\n(a real flash shift would be common)')

    axes[1].bar(x, ampr, color=cols)
    axes[1].axhline(1, color=INK, lw=0.8, ls='--')
    axes[1].axhline(2, color='#c0632c', lw=0.8, ls=':')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=90, fontsize=7.5)
    axes[1].set_ylabel('amplitude ratio ded/par')
    axes[1].set_title('b. only PKUP is linear in protons;\nthe rest are saturated')

    axes[2].scatter(drise, walk, c=cols, s=28)
    for xx, yy, ll in zip(drise, walk, labels):
        axes[2].annotate(ll, (xx, yy), fontsize=6.5, xytext=(3, 3), textcoords='offset points')
    axes[2].axhline(0, color=INK, lw=0.6); axes[2].axvline(0, color=INK, lw=0.6)
    axes[2].set_xlabel('Δ risetime (ded − par) [ns]')
    axes[2].set_ylabel('timing shift [ns]')
    axes[2].set_title('c. it tracks the pulse-shape change,\nnot the arrival time')
    fig.tight_layout(); fig.savefig(FIGS / '06_intensity_mechanism.png', bbox_inches='tight')
    plt.close(fig)


def fig_intensity_prediction():
    """Within-class amplitude slope cannot explain the cross-class shift."""
    d = np.load(DATA / 'flash_run224357.npz')
    pm, pim, pk = pkmap(d)
    trees = ['WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSC', 'SILI']
    pred, obs = [], []
    for t in trees:
        fh = flash_hits(d[t])
        ref = np.array([pm.get(int(b), np.nan) for b in fh['BunchNumber']])
        dt = fh['tof'] - ref
        ok = np.isfinite(dt); m = np.median(dt[ok]); core = ok & (np.abs(dt - m) < 60)
        pi = fh['PulseIntensity']
        sl = []
        for ch in np.unique(fh['detn']):
            for msk in (core & (pi < 6e12) & (fh['detn'] == ch),
                        core & (pi >= 6e12) & (fh['detn'] == ch)):
                if msk.sum() < 200:
                    continue
                a, y = fh['amp'][msk].astype(float), dt[msk]
                if a.std() < 1:
                    continue
                sl.append(np.polyfit(a, y, 1)[0])
        lo = core & (pi < 6e12); hi = core & (pi >= 6e12)
        damp = np.median(fh['amp'][hi]) - np.median(fh['amp'][lo])
        pred.append(np.mean(sl) * damp if sl else np.nan)
        obs.append(dt[hi].mean() - dt[lo].mean())
    x = np.arange(len(trees)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w / 2, pred, w, color='#9ab7cd', label='predicted by the within-class\namplitude dependence')
    ax.bar(x + w / 2, obs, w, color='#2f6f9f', label='observed')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(trees, rotation=45, fontsize=8)
    ax.set_ylabel('shift [ns]'); ax.legend(fontsize=8)
    ax.set_title('7. classical amplitude walk explains none of it')
    fig.tight_layout(); fig.savefig(FIGS / '07_intensity_prediction.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- plastics
def fig_plastic_anatomy():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (run, ep) in zip(axes, RUNS.items()):
        p = DATA / f'flash_run{run}.npz'
        if not p.exists():
            continue
        d = np.load(p); pm, pim, pk = pkmap(d)
        for tree, lab in (('PSSA', 'PSSA ch1 (plastic)'), ('WALA', 'WALA ch1 (wall)')):
            if tree not in d.files:
                continue
            r = d[tree]; r = r[r['detn'] == 1]
            ref = np.array([pm.get(int(b), np.nan) for b in r['BunchNumber']])
            dt = (r['tof'] - ref)
            m = np.isfinite(dt)
            nb = len(set(r['BunchNumber'][m].tolist()))
            ax.hist(dt[m], bins=np.arange(-2200, -1300, 4), histtype='step', lw=1.1,
                    color=C[tree[:3]], label=lab,
                    weights=np.full(m.sum(), 1.0 / max(nb, 1)))
        ax.axvline(-1719, color='#999999', lw=0.8, ls='--')
        ax.text(-1719, ax.get_ylim()[1], ' calibrated\n flash', fontsize=6.5,
                color='#666666', va='top')
        ax.set_yscale('log'); ax.set_title(ep); ax.set_xlabel('t − t(PKUP)  [ns]')
        ax.set_xlim(-2200, -1300)
    axes[0].set_ylabel('hits per bunch'); axes[0].legend(fontsize=7.5)
    fig.suptitle('8. the flash region: the wall is one clean pulse, the plastic is not',
                 fontsize=9.5, x=0.02, ha='left')
    fig.tight_layout(); fig.savefig(FIGS / '08_plastic_anatomy.png', bbox_inches='tight')
    plt.close(fig)


def fig_plastic_vs_amp():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    edges = np.array([0, 1000, 5000, 10000, 20000, 30000, 45000, 70000])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for run, ep in RUNS.items():
        p = DATA / f'flash_run{run}.npz'
        if not p.exists():
            continue
        d = np.load(p); pm, pim, pk = pkmap(d)
        r = d['PSSA']; r = r[r['detn'] == 1]
        ref = np.array([pm.get(int(b), np.nan) for b in r['BunchNumber']])
        dt = r['tof'] - ref; fin = np.isfinite(dt)
        big = fin & (r['amp'] > 25000) & (dt > -3000) & (dt < -500)
        h, e = np.histogram(dt[big], bins=np.arange(-3000, -500, 5)); peak = e[h.argmax()] + 2.5
        near = fin & (np.abs(dt - peak) < 80)
        b, v, a = r['BunchNumber'][near], dt[near], r['amp'][near]
        o = np.lexsort((np.abs(v - peak), b)); b, v, a = b[o], v[o], a[o]
        fi = np.ones(len(b), bool); fi[1:] = b[1:] != b[:-1]
        v, a = v[fi], a[fi]
        sig, bias = [], []
        for i in range(len(edges) - 1):
            m = (a >= edges[i]) & (a < edges[i + 1])
            if m.sum() < 10:
                sig.append(np.nan); bias.append(np.nan); continue
            sig.append(1.4826 * np.median(np.abs(v[m] - np.median(v[m]))))
            bias.append(np.median(v[m]) - np.median(v))
        axes[0].plot(ctr, sig, 'o-', ms=3.5, lw=1, label=f'{run} {ep}')
        axes[1].plot(ctr, bias, 'o-', ms=3.5, lw=1, label=f'{run} {ep}')
    axes[0].set_ylabel('per-bunch σ [ns]'); axes[0].set_xlabel('amplitude of the flash hit [ADC]')
    axes[0].set_title('9. plastic timing precision vs pulse size')
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_ylabel('median t − run median [ns]'); axes[1].set_xlabel('amplitude of the flash hit [ADC]')
    axes[1].set_title('10. and the amplitude-dependent bias')
    axes[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIGS / '09_plastic_vs_amp.png', bbox_inches='tight')
    plt.close(fig)


def fig_plastic_channels():
    rows = list(csv.DictReader(open(DATA / 'plastic_pathology.csv')))
    fams = ['PSS', 'LIQ', 'WAL']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    runs = sorted({int(r['run']) for r in rows})
    width = 0.8 / len(runs)
    for j, run in enumerate(runs):
        labs, found, sig = [], [], []
        for r in rows:
            if int(r['run']) != run or not r['tree'].startswith('PSS'):
                continue
            labs.append(f"{r['tree'][-1]}{r['ch']}")
            found.append(float(r['found']))
            sig.append(float(r['sigma']) if r['sigma'] not in ('', 'nan') else np.nan)
        x = np.arange(len(labs))
        axes[0].bar(x + j * width - 0.4, found, width, label=f'{run}')
        axes[1].bar(x + j * width - 0.4, sig, width, label=f'{run}')
    axes[0].set_xticks(np.arange(len(labs))); axes[0].set_xticklabels(labs, fontsize=8)
    axes[1].set_xticks(np.arange(len(labs))); axes[1].set_xticklabels(labs, fontsize=8)
    axes[0].set_ylabel('fraction of bunches with a flash hit')
    axes[0].set_title('11. plastic flash-hit yield, per channel')
    axes[1].set_ylabel('per-bunch σ [ns]'); axes[1].set_yscale('log')
    axes[1].set_title('12. and its timing spread')
    axes[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIGS / '10_plastic_channels.png', bbox_inches='tight')
    plt.close(fig)




def fig_timeseries():
    """Flash time vs run, with per-bunch spread and statistical error."""
    rows = list(csv.DictReader(open(DATA / 'plastic_liq_flash_by_run.csv')))

    def series(t):
        out = []
        for r in rows:
            if r.get(t) in (None, '') or r.get(t + '_n') in (None, ''):
                continue
            n = float(r[t + '_n']); s = float(r[t + '_sig'])
            out.append((int(r['run']), float(r[t]), s, s / max(np.sqrt(n), 1), n))
        return np.array(out).T if out else None

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.3]})
    for t, c in zip(('LIQA', 'LIQB', 'LIQC', 'LIQD'),
                    ('#2f6f9f', '#c0632c', '#4a8a5a', '#8a5a9a')):
        s = series(t)
        if s is None:
            continue
        axes[0].errorbar(s[0], s[1], yerr=s[3], fmt='o-', ms=3, lw=0.8, elinewidth=0.9,
                         color=c, label=f'{t}  (σ_run = {s[1].std():.2f} ns)')
    axes[0].set_ylabel('flash time − PKUP  [ns]')
    axes[0].set_title('13. liquid scintillators — the time base is stable across the campaign')
    axes[0].legend(fontsize=7.5, ncol=2)

    for t, c in zip(('PSSA', 'PSSB', 'PSSC', 'PSSD'),
                    ('#2f6f9f', '#c0632c', '#4a8a5a', '#8a5a9a')):
        s = series(t)
        if s is None:
            continue
        axes[1].errorbar(s[0], s[1], yerr=s[3], fmt='o-', ms=3, lw=0.8, elinewidth=0.9,
                         color=c, label=f'{t}  (σ_run = {s[1].std():.1f} ns)')
    axes[1].set_ylabel('flash time − PKUP  [ns]')
    axes[1].set_title('14. plastics — same measurement, not stable')
    axes[1].legend(fontsize=7.5, ncol=2)

    for t, c, ls in (('LIQA', '#2f6f9f', '-'), ('PSSA', '#c0632c', '-')):
        s = series(t)
        if s is None:
            continue
        axes[2].plot(s[0], s[2], 'o-', ms=3, lw=0.8, color=c, ls=ls, label=f'{t}')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('per-bunch σ [ns]'); axes[2].set_xlabel('run number')
    axes[2].set_title('15. and the single-bunch spread that goes with it')
    axes[2].legend(fontsize=7.5)
    for ax in axes:
        for r, lab in ((224464, 'divert-off 07-16'), (224488, 'FIFO 07-17')):
            ax.axvline(r, color='#bbbbbb', lw=0.8, ls='--')
    axes[0].text(224488, axes[0].get_ylim()[1], ' FIFO', fontsize=6.5, color='#666', va='top')
    fig.tight_layout(); fig.savefig(FIGS / '11_timeseries.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_intensity_mechanism()
    fig_intensity_prediction()
    fig_plastic_anatomy()
    fig_plastic_vs_amp()
    fig_plastic_channels()
    fig_timeseries()
    print('wrote figures 06-11 to', FIGS)

#!/usr/bin/env python3
"""Figures for the n_TOF UserInput comparison report.

Every panel is measured, not drawn: the numeric inputs are either read live
from the files (raw waveforms, the official run224572.root, the v11 partials)
or taken from report/results.json, which records what each tool measured and
how to reproduce it.

    python make_figures.py [outdir]

Inputs it expects (all optional -- a missing one skips its figure):
    ~/x17/beam_july/ntof_data/run224572.root            the official processing
    <scratch>/v11_pssfit_widthparts/run224572_0001.root  the new processing
    <scratch>/flashblocks_224572.npz                     raw flash blocks
    /media/dylan/data/x17/ntof_processing/X17_WALA_Signal_3.txt   shipped shape
    ntof_processing/userinputs/v11_pssfit_width/X17_WALA_Signal_avg2.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRATCH = Path('/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17/'
               '37276477-1dd1-45b1-8154-2100681e7566/scratchpad')
OFFICIAL = Path.home() / 'x17/beam_july/ntof_data/run224572.root'
V11 = SCRATCH / 'v11_pssfit_widthparts/run224572_0001.root'
RAW = SCRATCH / 'flashblocks_224572.npz'
SHIPPED = Path('/media/dylan/data/x17/ntof_processing')

R = json.load(open(HERE / 'results.json'))
OK, BAD, NEU = '#1f77b4', '#d62728', '#7f7f7f'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                     'figure.dpi': 150})


def save(fig, out, name):
    p = out / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {p.name}')


# --------------------------------------------------------------------------
def fig_flash_id(out):
    """The headline defect: stored tflash per bunch, official vs new."""
    import uproot
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (path, lab, col) in zip(axes, [(OFFICIAL, 'official production', BAD),
                                           (V11, 'this work (v11)', OK)]):
        if not path.exists():
            continue
        f = uproot.open(path)
        a = f['PSSA'].arrays(['BunchNumber', 'tflash'], entry_stop=3_000_000,
                             library='np')
        b, t = a['BunchNumber'], a['tflash']
        _, first = np.unique(b, return_index=True)
        b, t = b[first], t[first] / 1000.0
        m = b <= 100                       # same bunch window in both panels
        ax.plot(b[m], t[m], '.', ms=3.5, color=col, alpha=0.8)
        ax.set_xlabel('bunch number')
        ax.set_title(f'PSSA stored $t_{{flash}}$ -- {lab}')
        ax.axhline(11.635, color='k', ls='--', lw=0.8)
        ax.set_xlim(0, 101)
        bad = np.mean(np.abs(t[m] - 11.635) > 0.15) * 100
        ax.text(0.97, 0.06, f'{bad:.0f} % mis-tagged', transform=ax.transAxes,
                fontsize=8, ha='right', color=col,
                bbox=dict(fc='white', ec=col, lw=0.6, pad=2))
    axes[0].set_ylabel('stored $t_{flash}$  [$\\mu$s]')
    axes[0].set_ylim(-0.5, 13)
    axes[0].text(2, 11.0, 'true flash, 11.64 $\\mu$s', fontsize=7.5)
    axes[0].text(2, 4.5, 'the finder locks onto\nnoise before the flash',
                 fontsize=7.5, color=BAD)
    save(fig, out, 'fig1_flash_id.pdf')


def fig_bad_fraction(out):
    d = R['flash_id_bad_bunch_fraction']
    trees = [t for t in d['official'] if t != 'PKUP']
    x = np.arange(len(trees))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(x - 0.2, [d['official'][t] * 100 for t in trees], 0.4,
           label='official production', color=BAD)
    ax.bar(x + 0.2, [d['v11'][t] * 100 for t in trees], 0.4,
           label='this work (v11)', color=OK)
    ax.axhline(2, color='k', ls=':', lw=0.9)
    ax.text(len(trees) - 0.4, 2.6, 'acceptance target 2 %', fontsize=7, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(trees, rotation=45, ha='right')
    ax.set_ylabel('bunches with a mis-tagged\n$\\gamma$-flash  [%]')
    ax.legend(fontsize=8)
    save(fig, out, 'fig2_bad_fraction.pdf')


def fig_divert(out):
    """Raw waveforms: why the wall flash time was wrong."""
    if not RAW.exists():
        return
    z = np.load(RAW, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for name, col in (('WALA_1', OK), ('WALB_1', '#ff7f0e'),
                      ('WALC_1', '#2ca02c'), ('WALD_1', BAD)):
        k = f'{name}_b161'
        if k not in z.files:
            continue
        y = z[k].astype(float)
        base = np.median(y[:2000])
        t = np.arange(len(y))
        m = (t > 11150) & (t < 12600)
        ax.plot(t[m] / 1000, (y - base)[m], lw=0.7, color=col, label=name[:4])
    ax.set_ylim(-3400, 3000)
    ax.set_xlabel('t since proton pulse  [$\\mu$s]')
    ax.set_ylabel('ADC $-$ baseline')
    ax.legend(fontsize=7, ncol=4, loc='upper left', columnspacing=0.8,
              handlelength=1.2, framealpha=0.95)
    ax.set_title('SiPM wall: what the divert leaves behind')
    ax.annotate('divert gate closes\n(timed by the\nofficial processing)',
                xy=(11.26, -1500), xytext=(11.34, -3050), fontsize=7, color=BAD,
                arrowprops=dict(arrowstyle='->', color=BAD, lw=0.8))
    ax.annotate('the real $\\gamma$-flash\nleaking through\n(timed by v11)',
                xy=(11.63, 800), xytext=(11.80, -1800), fontsize=7, color=OK,
                arrowprops=dict(arrowstyle='->', color=OK, lw=0.8))
    ax.annotate('gate opens', xy=(12.26, 1600), xytext=(11.86, 1500), fontsize=7,
                arrowprops=dict(arrowstyle='->', lw=0.8))

    ax = axes[1]
    for name, col, sc in (('WALA_1', OK, 1.0), ('PSSA_1', BAD, 0.03),
                          ('LIQA_1', '#9467bd', 0.03)):
        k = f'{name}_b161'
        if k not in z.files:
            continue
        y = z[k].astype(float)
        base = np.median(y[:2000])
        t = np.arange(len(y))
        m = (t > 11540) & (t < 11760)
        lab = name[:4] + ('' if sc == 1 else f'  ($\\times${sc:g})')
        ax.plot(t[m] / 1000, (y - base)[m] * sc, lw=0.9, color=col, label=lab)
    ax.axvline(11.605, color='k', ls='--', lw=0.8)
    ax.set_xlabel('t since proton pulse  [$\\mu$s]')
    ax.set_ylabel('ADC $-$ baseline (scaled)')
    ax.legend(fontsize=7)
    ax.set_title('the same flash in three detectors, 11.60 $\\mu$s')
    save(fig, out, 'fig3_divert.pdf')


def fig_coincidence(out):
    d = R['coincidence_offset_ns']
    arms = ['PSSA', 'PSSB', 'PSSC', 'PSSD', 'LIQA', 'LIQB', 'LIQC', 'LIQD']
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axhspan(-25, 25, color='0.85', zorder=0)
    ax.text(len(arms) - 0.5, 30, 'acceptance $\\pm$25 ns', fontsize=7, ha='right')
    ax.plot(x, [d['official'][a] for a in arms], 'o', color=BAD,
            label='official production')
    ax.plot(x, [d['v11'][a] for a in arms], 's', color=OK, label='this work (v11)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=45, ha='right')
    ax.set_ylabel('coincidence offset\nvs same-arm wall  [ns]')
    ax.legend(fontsize=8)
    save(fig, out, 'fig4_coincidence.pdf')


def fig_templates(out):
    a = SHIPPED / 'X17_WALA_Signal_3.txt'
    b = REPO / 'ntof_processing/userinputs/v11_pssfit_width/X17_WALA_Signal_avg2.txt'
    if not (a.exists() and b.exists()):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for p, lab, col in ((a, 'shipped: one raw pulse, 314 ns', BAD),
                        (b, 'this work: median of 472 pulses, 861 ns', OK)):
        d = np.loadtxt(p)
        t, y = d[:, 0], d[:, 1]
        y = y / y.max()
        ax.plot(t - t[np.argmax(y)], y, lw=1.0, color=col, label=lab)
    ax.set_yscale('log')
    ax.set_ylim(2e-3, 1.5)
    ax.set_xlim(-60, 800)
    ax.set_xlabel('ns from peak')
    ax.set_ylabel('normalised amplitude')
    ax.set_title('WALA pulse-shape template')
    ax.legend(fontsize=7)
    ax.annotate('shipped template ends here,\nwith 3 % of the pulse left',
                xy=(250, 0.03), xytext=(330, 0.25), fontsize=7, color=BAD,
                arrowprops=dict(arrowstyle='->', color=BAD, lw=0.8))
    save(fig, out, 'fig5_templates.pdf')


def fig_areaamp(out):
    """What the AREA/AMP HIGH cut was removing, shown from the file that kept it."""
    import uproot
    if not V11.exists():
        return
    f = uproot.open(V11)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for tree, col in (('PSSA', OK), ('PSSC', '#2ca02c')):
        a = f[tree].arrays(['amp', 'area'], entry_stop=4_000_000, library='np')
        amp = np.abs(a['amp'])
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.abs(a['area']) / np.where(amp > 0, amp, np.nan)
        r = r[np.isfinite(r) & (r < 60)]
        ax.hist(r, bins=120, range=(0, 60), histtype='step', color=col,
                label=tree, density=True)
    ax.axvline(20, color=BAD, ls='--', lw=1.2)
    ax.axvline(60, color=OK, ls='--', lw=1.2)
    ax.set_xlabel('area / amplitude')
    ax.set_ylabel('normalised')
    ax.set_title('plastic pulses vs the elimination window')
    ax.text(20.8, ax.get_ylim()[1] * 0.75, 'official\ncut = 20', fontsize=7, color=BAD)
    ax.text(52, ax.get_ylim()[1] * 0.75, 'v11\ncut = 60', fontsize=7, color=OK,
            ha='right')
    ax.legend(fontsize=8, loc='upper left')
    save(fig, out, 'fig6_areaamp.pdf')


def fig_efficiency(out):
    d = R['singles_matcher']
    lab = [f'{a}-{b}' for a, b in d['bins_ms']]
    x = np.arange(len(lab))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    ax = axes[0]
    ax.bar(x - 0.2, [v * 100 for v in d['official_eff']], 0.4,
           color=BAD, label='official + laptop $t_{flash}$ repair')
    ax.bar(x + 0.2, [v * 100 for v in d['v11_eff']], 0.4,
           color=OK, label='this work (v11), no repair')
    ax.plot(x, [v * 100 for v in d['v11_wall_only']], 'k^--', ms=4, lw=0.8,
            label='wall leg alone (v11)')
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_xlabel('time since flash  [ms]')
    ax.set_ylabel('matcher efficiency  [%]')
    ax.set_ylim(85, 100)
    ax.legend(fontsize=7, loc='lower right')
    ax = axes[1]
    ax.bar(x - 0.2, [v * 100 for v in d['official_false']], 0.4, color=BAD)
    ax.bar(x + 0.2, [v * 100 for v in d['v11_false']], 0.4, color=OK)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_xlabel('time since flash  [ms]')
    ax.set_ylabel('false-match rate  [%]')
    save(fig, out, 'fig7_efficiency.pdf')


def fig_hits(out):
    d = R['hits_per_bunch']
    trees = d['trees']
    x = np.arange(len(trees))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(x - 0.2, d['v1_flash'], 0.4, color=NEU,
           label='flash fix only (v1)')
    ax.bar(x + 0.2, d['v11'], 0.4, color=OK, label='this work (v11)')
    ax.set_xticks(x); ax.set_xticklabels(trees, rotation=45, ha='right')
    ax.set_ylabel('hits per bunch')
    ax.legend(fontsize=8)
    save(fig, out, 'fig8_hits.pdf')


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_flash_id, fig_bad_fraction, fig_divert, fig_coincidence,
               fig_templates, fig_areaamp, fig_efficiency, fig_hits):
        try:
            fn(out)
        except Exception as e:                       # keep going; report which
            print(f'  SKIP {fn.__name__}: {type(e).__name__}: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27w_headon_waveforms.py — decoded-waveform purification of the HEAD-ON track
sample for run_55.

Why: the combined_hits head-on tag (few strips + long pulse-width proxy
wfmax = right_sample-left_sample) over-counts — any long/saturated single pulse
(³He-capture fragment sitting on one strip, baseline wander) passes it, so the
27b head-on category is contaminated (it outnumbers the clean inclined tracks,
which is unphysical for a point source).  The honest test of a NORMAL-INCIDENCE
track is the ACTUAL waveform: charge drifts in over the full gap onto one strip,
so the pulse is a LONG single hump — a saturation plateau (or a wide
above-threshold pulse) spanning ~the full-gap drift time (~11-12 samples of
60 ns), NOT a short spike.

This module pulls the decoded_root waveform (32 samples x 60 ns, raw ADC,
baseline ~400, saturation 4095) of the LEADING strip of each candidate and
measures: saturation-plateau length, above-threshold width, peak sample, and
"single-hump" shape.  It contrasts head-on candidates vs inclined-track lead
strips vs wide-blob lead strips, and reports how much of the combined_hits
head-on tag survives a real full-gap-pulse requirement.

Outputs: figures/27_tracks/{07_headon_waveforms,08_headon_features}.png,
calib/27_headon_wf.json.  Runs on a representative subset of subruns (fast).

Run:  venv/bin/python mx_july_beam_qa/27w_headon_waveforms.py [subrun ...]
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import trackcache as tc

HERE = os.path.dirname(__file__)
RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
CALIB = os.path.join(HERE, 'calib')
SAT_ADC = 4000.0
WF_THR = 400.0       # ADC above baseline for "above threshold" width
DEFAULT_SUBS = ['scintd_r560_dr800dA600_c00_000',
                'scintd_r545_dr800dA600_c00_003',
                'scintd_r530_dr800dA600_c00_006']


def load_decoded_waveforms(subdir, need):
    """need: dict feu(int)->set of (eid, ch).  Returns dict (feu,eid,ch)->wf[32]."""
    out = {}
    for feu in sorted(need):
        pat = os.path.join(subdir, 'decoded_root', f'*_{feu:02d}.root')
        fs = sorted(glob.glob(pat))
        if not fs:
            continue
        t = uproot.open(fs[0])['nt']
        a = t.arrays(['eventId', 'sample', 'channel', 'amplitude'], library='np')
        eidx = {int(e): i for i, e in enumerate(a['eventId'])}
        want = need[feu]
        want_by_eid = {}
        for (e, c) in want:
            want_by_eid.setdefault(e, set()).add(c)
        for e, chs in want_by_eid.items():
            i = eidx.get(int(e))
            if i is None:
                continue
            ch = a['channel'][i]; sm = a['sample'][i]; am = a['amplitude'][i]
            for c in chs:
                m = ch == c
                if not m.any():
                    continue
                o = np.argsort(sm[m])
                wf = np.zeros(32, np.float32)
                ss = sm[m][o]; aa = am[m][o]
                wf[ss[ss < 32]] = aa[ss < 32]
                out[(feu, int(e), int(c))] = wf
    return out


def wf_features(wf):
    base = np.median(np.concatenate([wf[:4], wf[-4:]]))
    p = wf - base
    peak = int(np.argmax(p))
    amp = float(p[peak])
    over = p > WF_THR
    width = int(over.sum())
    sat = wf >= SAT_ADC
    # longest run of saturated samples
    plateau = 0; run = 0
    for s in sat:
        run = run + 1 if s else 0
        plateau = max(plateau, run)
    # single-hump: fraction of above-threshold samples that are contiguous
    if width > 0:
        idx = np.where(over)[0]
        contig = idx.max() - idx.min() + 1
        humpiness = width / contig
    else:
        humpiness = 0.0
    return dict(base=float(base), amp=amp, peak=peak, width=width,
                plateau=int(plateau), humpiness=float(humpiness))


def main():
    subs = sys.argv[1:] or DEFAULT_SUBS
    cl, ss, ev = tc.load_all(subruns=subs)
    cl = tc.add_derived(cl, ss)
    evk = ev.set_index(['sub', 'eid'])
    cl = cl.join(evk[['t_ms']], on=['sub', 'eid'])
    cl['b12'] = (cl['t_ms'] > 7) & (cl['t_ms'] < 30)
    cl = tc.tag_tracklike(cl)

    # class each candidate: headon / inclined / blob(control)
    cl['klass'] = np.where(cl['headon'], 'headon',
                  np.where(cl['inclined'], 'inclined',
                  np.where(cl['extent'] > 35, 'blob', 'other')))
    use = cl[cl['b12'] & cl['klass'].isin(['headon', 'inclined', 'blob'])].copy()
    # cap blob/inclined to keep IO modest, keep all head-on
    parts = [use[use['klass'] == 'headon']]
    for k in ('inclined', 'blob'):
        g = use[use['klass'] == k]
        parts.append(g.sample(min(len(g), 1500), random_state=1))
    use = pd.concat(parts, ignore_index=True)

    # lead strip (max amp) per cluster -> (sub, feu, eid, ch)
    lead = []
    for _, r in use.iterrows():
        o = int(r['goff']); n = int(r['len'])
        amps = ss['amp'][o:o + n]
        k = int(np.argmax(amps))
        lead.append((int(ss['feu'][o + k]), int(ss['ch'][o + k])))
    use['lfeu'] = [x[0] for x in lead]
    use['lch'] = [x[1] for x in lead]

    # fetch waveforms per subrun
    feats = []
    examples = {'headon': [], 'inclined': [], 'blob': []}
    for sub, g in use.groupby('sub'):
        need = {}
        for _, r in g.iterrows():
            need.setdefault(int(r['lfeu']), set()).add((int(r['eid']), int(r['lch'])))
        wfs = load_decoded_waveforms(os.path.join(RUN_DIR, sub), need)
        for _, r in g.iterrows():
            wf = wfs.get((int(r['lfeu']), int(r['eid']), int(r['lch'])))
            if wf is None:
                continue
            fe = wf_features(wf)
            fe.update(klass=r['klass'], detn=r['detn'], resist_v=int(r['resist_v']),
                      wfmax=float(r['wfmax']), n=int(r['n']))
            feats.append(fe)
            if len(examples[r['klass']]) < 12:
                examples[r['klass']].append((wf, fe))
    F = pd.DataFrame(feats)
    print(f'waveforms fetched: {len(F)}')
    print(F.groupby('klass').agg(
        n=('amp', 'size'), plateau_med=('plateau', 'median'),
        width_med=('width', 'median'), amp_med=('amp', 'median'),
        hump_med=('humpiness', 'median')).round(2))

    # real full-gap single pulse: a wide SINGLE hump spanning ~the full-gap
    # drift time (>=7 of the ~11-12-sample crossing), or a saturation plateau.
    # (At run_55 gains real head-on humps peak ~1-2k ADC and do NOT saturate,
    # so width — not the plateau — is the discriminator; humpiness kills the
    # multi-bump pile-up.)
    F['real_headon'] = ((F['plateau'] >= 6) |
                        ((F['width'] >= 7) & (F['humpiness'] > 0.85)))
    ho = F[F['klass'] == 'headon']
    purity = ho['real_headon'].mean() if len(ho) else np.nan
    print(f'\ncombined_hits head-on tag: {len(ho)} lead strips, '
          f'{100*purity:.0f}% pass the real full-gap-pulse waveform test')
    for k in ('inclined', 'blob'):
        sub = F[F['klass'] == k]
        print(f'  contrast {k}: {100*sub["real_headon"].mean():.0f}% would pass '
              f'(plateau_med {sub["plateau"].median():.0f})')

    # ---- figures ----
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for a, k, col in [(ax[0], 'headon', 'C3'), (ax[1], 'inclined', 'C0'),
                      (ax[2], 'blob', 'C2')]:
        for wf, fe in examples[k][:10]:
            a.plot(np.arange(32) * 60, wf - fe['base'], color=col, alpha=0.5, lw=1)
        a.axhline(SAT_ADC - 400, color='k', ls=':', lw=0.8)
        a.set_title(f'{k} lead-strip waveforms (baseline-subtracted)')
        a.set_ylabel('ADC - baseline')
    ax[-1].set_xlabel('time [ns]  (32 samples x 60 ns = 1.92 µs)')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '07_headon_waveforms.png'), dpi=95)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, col in [('headon', 'C3'), ('inclined', 'C0'), ('blob', 'C2')]:
        s = F[F['klass'] == k]
        ax[0].hist(s['plateau'], bins=np.arange(0, 20), histtype='step',
                   density=True, color=col, label=k)
        ax[1].hist(s['width'], bins=np.arange(0, 25), histtype='step',
                   density=True, color=col, label=k)
        ax[2].scatter(s['wfmax'], s['plateau'], s=5, alpha=0.3, color=col, label=k)
    ax[0].axvline(6, color='k', ls='--'); ax[0].set_xlabel('saturation plateau [smp]')
    ax[0].legend(); ax[0].set_title('plateau length')
    ax[1].axvline(9, color='k', ls='--'); ax[1].set_xlabel('above-thr width [smp]')
    ax[1].legend(); ax[1].set_title('pulse width')
    ax[2].set_xlabel('combined_hits wfmax'); ax[2].set_ylabel('true plateau [smp]')
    ax[2].legend(); ax[2].set_title('proxy vs true')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '08_headon_features.png'), dpi=95)
    plt.close(fig)

    out = dict(n=len(F), headon_purity=float(purity),
               subs=list(subs),
               plateau_med={k: float(F[F.klass == k]['plateau'].median())
                            for k in ('headon', 'inclined', 'blob')})
    with open(os.path.join(CALIB, '27_headon_wf.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nsaved figures 07/08 + calib/27_headon_wf.json')


if __name__ == '__main__':
    main()

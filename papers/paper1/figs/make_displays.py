#!/usr/bin/env python3
"""make_displays.py -- every Paper I event display of chamber A, in the one style of evd.py (2026-11-03).

    .venv/bin/python papers/paper1/figs/make_displays.py figures [--only ID,...]   # the paper's displays
    .venv/bin/python papers/paper1/figs/make_displays.py gallery --out DIR         # style candidates (deck appendix)
    .venv/bin/python papers/paper1/figs/make_displays.py sheet --ids 1,2,3 --out DIR  # contact sheet, to pick events
    common: --root <paper1_eff copy> --ana <Analysis_pass1 copy> --eos-run-dir <june_tests run dir copy>

Data (Run 1 = sat_det3, pass 1): the reference rays of the efficiency budget (<root>/budget/<rel>/rays_det3.parquet),
the pass's fits (<ana>/<rel>/wft/events.parquet and events.candidates.parquet), rr1 waveforms (pedestal + common mode
as the reconstruction reads them; a1_displays.read_waveforms).  Fits drawn: the pass's tracks (track_id 0 'first',
>0 'second') and, where a figure asks, the candidates it did not keep (track_id -1).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'papers'), os.path.join(REPO, 'rerun', 'paper1_figs')]
import a1_displays as a1  # noqa: E402
import evd  # noqa: E402

KEY = 'sat_det3'
LOCAL = '/media/ucla/work/claude-jobs/17c92c46-displays'
DEF_ROOT = '/media/ucla/work/paper1_figs'
DEF_ANA = os.path.join(LOCAL, 'Analysis_pass1')
DEF_EOS = os.path.join(LOCAL, 'june_tests', 'mx17_det3_saturday_scan_6-27-26')

#: the gallery: events every style is tried on (a clean muon, two separate particles, a delta ray, merged in Y)
GALLERY = [(7884, 'tracks', 'clean muon'), (1063, 'tracks', 'two particles'), (28537, 'tracks', r'$\delta$-ray'),
           (17907, 'all', 'merged in Y')]


class Data:
    def __init__(self, root, ana, eos):
        row = dict(a1.inputs.row_for(KEY)); row['key'] = KEY
        row['eos_run_dir'] = eos
        self.row = row
        rel = os.path.join(row['run'], row['subrun'], row['det'])
        self.R = pd.read_parquet(os.path.join(root, 'budget', rel, 'rays_det3.parquet')).drop_duplicates('event_id').set_index('event_id')
        wdir = os.path.join(ana, rel, 'wft')
        E = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
        self.E = E[~E.event_id.duplicated(keep=False)].set_index('event_id')
        self.K = pd.read_parquet(os.path.join(wdir, 'events.candidates.parquet'))
        self.v_mm = float(json.load(open(os.path.join(wdir, 'calib_bundle', 'bundle.json')))['v_drift']) * 1e-3
        self.PM, _ = a1.posmap(row)
        self._wf = {}

    def waveforms(self, ids):
        need = [e for e in ids if e not in self._wf]
        if need:
            sys.path.insert(0, HERE)
            import make_two_particle_displays as m2
            self._wf.update(m2.waveforms(self.row, self.E, need))
        return {e: self._wf[e] for e in ids}

    def event(self, e, cands='tracks', title='', split=30.0):
        w = self.waveforms([e])[e]
        r = self.R.loc[e] if e in self.R.index else None
        f = self.E.loc[e] if e in self.E.index else None
        ref, t0 = {}, {}
        for v in 'xy':
            if r is not None:
                ref[v] = (float(r[f'mesh_{v}']), float(r[f'tan_{v}']))
            t0[v] = float(f[f'{v}_t0']) if f is not None and np.isfinite(f[f'{v}_t0']) else (257.5 if v == 'x' else 302.5)
        k = self.K[self.K.event_id == e]
        if f is not None:
            k = k[k.acq_tag == f['acq_tag']]
        if cands == 'tracks':
            k = k[k.track_id >= 0]
        elif cands == 'none':
            k = k.iloc[:0]
        fits = [dict(view=c.plane, p0=c.p0, w=c.w, t0=c.t0,
                     role='first' if c.track_id == 0 else ('second' if c.track_id > 0 else 'candidate'))
                for c in k.itertuples()]
        return evd.Event(eid=e, W={v: w[v] for v in 'xy'}, P={v: self.PM[v].astype(float) for v in 'xy'},
                         v_mm=self.v_mm, t0=t0, ref=ref, fits=fits, title=title, split=split)


def single(ev, path, height=2.6, title=None, **kw):
    import matplotlib.pyplot as plt
    import paper_style as ps
    fig = plt.figure(figsize=(ps.FULL[0], height))
    gs = fig.add_gridspec(1, 1, left=0.09, right=0.84, bottom=0.17, top=0.86)
    cax = fig.add_axes([0.925, 0.17, 0.014, 0.69])
    out = evd.draw(fig, gs[0], ev, cbar_ax=cax, **kw)
    evd.legend(fig, out['handles'], ax=out['axes'][0], bbox_to_anchor=(0.0, 1.0), loc='lower left')
    if title:
        fig.text(0.5, 0.995, title, ha='center', va='top', fontsize=8)
    ps.save(fig, path)


def gallery(D, out):
    """One page per style: the four GALLERY events; plus reference-mode, gamma and spacing comparisons."""
    import matplotlib.pyplot as plt
    import paper_style as ps
    evs = [D.event(e, c, t) for e, c, t in GALLERY]
    os.makedirs(out, exist_ok=True)

    def page(name, setup, note):
        setup()
        fig = plt.figure(figsize=(ps.FULL[0], 5.4))
        gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, bottom=0.08, top=0.86, hspace=0.42, wspace=0.42)
        hs = {}
        for i, ev in enumerate(evs):
            o = evd.draw(fig, gs[i // 2, i % 2], ev, zaxis=False, ylabel=(i % 2 == 0), colorbar=True)
            o['axes'][0].set_title(f'{ev.title} (event {ev.eid})', fontsize=7.5, loc='left', pad=2)
            hs.update(o['handles'])
        evd.legend(fig, hs, bbox_to_anchor=(0.5, 0.905), loc='lower center')
        fig.text(0.5, 0.995, note, ha='center', va='top', fontsize=9, fontweight='bold')
        ps.save(fig, os.path.join(out, name), png=True)
        print('gallery', name, flush=True)

    for s in evd.STYLES:
        page(f'style_{s}', lambda s=s: evd.set_style(s, ref_mode='dashed', gamma=1.0, abut=True), evd.STYLES[s]['label'])
    for m in evd.REF_MODES:
        page(f'ref_{m}', lambda m=m: evd.set_style(evd.DEFAULT, ref_mode=m, gamma=1.0, abut=True),
             f'reference drawn as: {m}')
    page('gamma_0p5', lambda: evd.set_style(evd.DEFAULT, ref_mode='dashed', gamma=0.5, abut=True),
         'colour scale sqrt(ADC): faint charge brighter')
    page('spaced', lambda: evd.set_style(evd.DEFAULT, ref_mode='dashed', gamma=1.0, abut=False),
         'views separated by a gap (for comparison with abutting)')
    page('zoomed', lambda: evd.set_style(evd.DEFAULT, ref_mode='dashed', gamma=1.0, abut=True, equal=False),
         'zoomed to the charge (depth not to the strip scale)')
    evd.set_style(evd.DEFAULT)


#: single displays of section 5 (figure id -> event, fits drawn, height in)
SINGLES = {
    'display_two_separate': (1063, 'tracks'),
    'display_merged_projection': (17907, 'all'),
    'display_muon_deposit': (5423, 'none'),       # 11-03: was 36603 (Dylan: looked like two particles); a muon and a
                                                  # curling low-energy electron, vetoed by the 50-strip rule
}
#: the two-particle figure: (label, class row of two_particle.csv, event, fits)
TWO = [('two particles, separate in both views', 'two, separate in both views', 1063, 'tracks'),
       ('two particles, merged in one projection', 'two, merged in one projection', 17907, 'all'),
       ('muon + separate deposit (> 50 strips: vetoed)', 'muon + deposit (> 50 strips, veto)', 5423, 'none')]
#: what a TPC sees: (title, event, fits)
SHOWCASE = [('a cosmic muon', 7884, 'tracks'), (r'a muon and its $\delta$-ray', 28537, 'none'),
            ('a muon and a curling low-energy electron', 5423, 'none'), ('two particles', 1063, 'tracks')]
#: main-text loss gallery (A1 class keys of a1_displays.PAPER_DET3)
LOSS_MAIN = ['multi:other_picked', 'multi:m3_not_found', 'far:DELTA', 'far:FIT_SAT', 'spark:discharge', 'spark:shower']
TWO_CSV = os.path.join(REPO, 'rerun', 'effbudget', 'single', 'figures', 'two_particle.csv')


def _row_display(fig, cell, ev, **kw):
    o = evd.draw(fig, cell, ev, **kw)
    return o


def two_particle(D, out):
    """Efficiency per two-particle configuration next to one real event of each (Dylan 11-03: 'show these
    efficiencies alongside event displays')."""
    import matplotlib.pyplot as plt
    import paper_style as ps
    T = pd.read_csv(TWO_CSV).set_index('cls')
    one = T.loc['one particle']
    fig = plt.figure(figsize=(ps.FULL[0], 5.7))
    gs = fig.add_gridspec(3, 2, width_ratios=[4.25, 1.0], left=0.085, right=0.985, bottom=0.075, top=0.94,
                          hspace=0.42, wspace=0.12)
    hs = {}
    vmax = None
    for i, (lab, key, e, cands) in enumerate(TWO):
        ev = D.event(e, cands)
        o = evd.draw(fig, gs[i, 0], ev, vmax=2000.0, zaxis=False, colorbar=False, xlabel=True)
        hs.update(o['handles'])
        pc_last = [o['pc']]
        o['axes'][0].set_title(f'({chr(97 + i)}) {lab}', loc='left', fontsize=8, pad=3)
        r = T.loc[key]
        ax = fig.add_subplot(gs[i, 1])
        ax.axvline(one['eff'], color=ps.MUTED, lw=0.8, ls=':', zorder=1)
        ax.errorbar([r['eff']], [0], xerr=[r['err']], fmt='o', color=ps.DET_COLOR['det3'], ms=5, capsize=0, lw=1.2)
        ax.set_xlim(0, 100); ax.set_ylim(-1, 1.2)
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.grid(axis='y', visible=False)
        ax.text(0.5, 0.97, f"{r['eff']:.1f} $\\pm$ {r['err']:.1f} %", transform=ax.transAxes, ha='center', va='top',
                fontsize=9, fontweight='bold', color=ps.DET_COLOR['det3'])
        ax.text(0.5, 0.74, f"{int(r['k'])} of {int(r['n'])} events", transform=ax.transAxes, ha='center', va='top',
                fontsize=7, color=ps.MUTED)
        ax.set_xticks([0, 50, 100])
        if i == len(TWO) - 1:
            ax.set_xlabel('muon found [%]')
    evd.legend(fig, hs, bbox_to_anchor=(0.0, 0.995), loc='upper left')
    cax = fig.add_axes([0.80, 0.972, 0.17, 0.011])
    cb = fig.colorbar(pc_last[0], cax=cax, orientation='horizontal')
    cb.set_label('amplitude [ADC]', fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.ax.xaxis.set_ticks_position('bottom')
    ps.save(fig, os.path.join(out, 'two_particle'))
    print('wrote two_particle', flush=True)


def showcase(D, out):
    import matplotlib.pyplot as plt
    import paper_style as ps
    fig = plt.figure(figsize=(ps.FULL[0], 4.9))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.9, bottom=0.085, top=0.92, hspace=0.4, wspace=0.16)
    cax = fig.add_axes([0.93, 0.085, 0.014, 0.835])
    hs = {}
    for i, (t, e, c) in enumerate(SHOWCASE):
        o = evd.draw(fig, gs[i // 2, i % 2], D.event(e, c), vmax=2000.0, zaxis=False, ylabel=(i % 2 == 0),
                     colorbar=(i == 0), cbar_ax=cax)
        o['axes'][0].set_title(f'({chr(97 + i)}) {t}', loc='left', fontsize=8, pad=3)
        hs.update(o['handles'])
    evd.legend(fig, hs, bbox_to_anchor=(0.5, 0.995), loc='upper center')
    ps.save(fig, os.path.join(out, 'evd_showcase'))
    print('wrote evd_showcase', flush=True)


def loss_displays(D, out, keys=None):
    """The A1 appendix displays (one figure per class) and the main-text gallery of six classes."""
    import matplotlib.pyplot as plt
    import paper_style as ps
    picks = {c: (fid, title, e) for c, fid, title, e in a1.PAPER_DET3}
    for c, (fid, title, e) in picks.items():
        if keys and fid not in keys:
            continue
        cands = 'none' if c.startswith('spark') else 'tracks'
        evd.OPT['mark_sat'] = c in ('far:FIT_SAT',)
        single(D.event(e, cands, split=1e4 if c == 'spark:discharge' else 30.0), os.path.join(out, 'a1', fid),
               height=2.5, title=None)
        evd.OPT['mark_sat'] = False
        print('wrote', fid, flush=True)
    if keys and 'loss_gallery' not in keys:
        return
    fig = plt.figure(figsize=(ps.FULL[0], 6.6))
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.9, bottom=0.07, top=0.935, hspace=0.45, wspace=0.16)
    cax = fig.add_axes([0.93, 0.07, 0.014, 0.865])
    hs = {}
    for i, c in enumerate(LOSS_MAIN):
        fid, title, e = picks[c]
        cands = 'none' if c.startswith('spark') else 'tracks'
        o = evd.draw(fig, gs[i // 2, i % 2], D.event(e, cands, split=1e4 if c == 'spark:discharge' else 30.0), vmax=3550.0, zaxis=False, ylabel=(i % 2 == 0),
                     colorbar=(i == 0), cbar_ax=cax)
        t = title.replace('M3 one', 'reference one').replace('"Spark": ', '')
        o['axes'][0].set_title(f'({chr(97 + i)}) {t}', loc='left', fontsize=7.5, pad=3)
        hs.update(o['handles'])
    evd.legend(fig, hs, bbox_to_anchor=(0.5, 0.997), loc='upper center')
    ps.save(fig, os.path.join(out, 'loss_gallery'))
    print('wrote loss_gallery', flush=True)


def figures(D, out, only):
    want = lambda k: not only or k in only
    for fid, (e, c) in SINGLES.items():
        if want(fid):
            single(D.event(e, c), os.path.join(out, fid))
            print('wrote', fid, flush=True)
    if want('two_particle'):
        two_particle(D, out)
    if want('evd_showcase'):
        showcase(D, out)
    lk = [k for k in (only or []) if k.startswith('loss_')]
    if not only or lk:
        loss_displays(D, out, lk or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['figures', 'gallery', 'sheet'])
    ap.add_argument('--root', default=DEF_ROOT); ap.add_argument('--ana', default=DEF_ANA)
    ap.add_argument('--eos-run-dir', default=DEF_EOS)
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--only', default='')
    ap.add_argument('--ids', default='')
    ap.add_argument('--style', default=None)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import paper_style as ps
    ps.use()
    if a.style:
        evd.set_style(a.style)
    D = Data(a.root, a.ana, a.eos_run_dir)
    if a.mode == 'gallery':
        gallery(D, a.out)
    elif a.mode == 'sheet':
        for e in [int(x) for x in a.ids.split(',')]:
            single(D.event(e, 'all'), os.path.join(a.out, f'evt_{e}'), title=f'event {e}')
    else:
        figures(D, a.out, [x for x in a.only.split(',') if x])


if __name__ == '__main__':
    main()

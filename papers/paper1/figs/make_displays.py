#!/usr/bin/env python3
"""make_displays.py -- every Paper I event display of chamber A, in the one style of evd.py (2026-11-03).

    .venv/bin/python papers/paper1/figs/make_displays.py figures [--only ID,...]   # the paper's displays
    .venv/bin/python papers/paper1/figs/make_displays.py gallery --out DIR         # colour map x normalisation (deck appendix)
    options: --style reds|greens|fire|jet (default evd.DEFAULT), --norm full|event|adc (default evd.OPT)
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
LOCAL = os.environ.get('WORK_DATA', '/media/ucla/work') + '/claude-jobs/17c92c46-displays'
DEF_ROOT = os.environ.get('WORK_DATA', '/media/ucla/work') + '/paper1_figs'
DEF_ANA = os.path.join(LOCAL, 'Analysis_pass1')
DEF_EOS = os.path.join(LOCAL, 'june_tests', 'mx17_det3_saturday_scan_6-27-26')

#: the gallery (deck appendix): the same events in every colour map x normalisation
GALLERY = [(7884, 'tracks', 'a cosmic muon'), (47235, 'tracks', 'two muons'), (28537, 'none', r'a muon and its $\delta$-ray'),
           (17907, 'all', 'two particles, merged in Y')]
NOBRK = 1e4          # Event.split for a figure drawn without broken axes

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


def stack(path, rows, row_h=2.05, top_in=0.42, bot_in=0.42, hspace_in=0.5, legend=True, axis_labels='all',
          title_fs=8.0):
    """One display per row, full width, one colour bar on the right.  rows: list of (title, Event).
    axis_labels: 'all' (x labels under every row) or 'last' (only under the bottom row)."""
    import matplotlib.pyplot as plt
    import paper_style as ps
    n = len(rows)
    H = top_in + bot_in + n * row_h + (n - 1) * hspace_in
    fig = plt.figure(figsize=(ps.FULL[0], H))
    top, bot = 1 - top_in / H, bot_in / H
    gs = fig.add_gridspec(n, 1, left=0.085, right=0.865, bottom=bot, top=top, hspace=hspace_in / row_h)
    cax = fig.add_axes([0.945, bot, 0.013, top - bot])
    hs = {}
    for i, (t, ev) in enumerate(rows):
        o = evd.draw(fig, gs[i, 0], ev, colorbar=(i == 0), cbar_ax=cax,
                     xlabel=(axis_labels == 'all' or i == n - 1))
        if t:
            o['axes'][0].set_title(t, loc='left', fontsize=title_fs, pad=3)
        if axis_labels == 'last':          # compact rows: one time and one depth label for the whole column
            o['axes'][0].set_ylabel('')
            o['sec'].set_ylabel('')
        hs.update(o['handles'])
    if axis_labels == 'last':
        fig.text(0.012, 0.5 * (top + bot), 'drift time [ns]', rotation=90, ha='left', va='center', fontsize=ps.BASE_PT)
        fig.text(0.915, 0.5 * (top + bot), 'distance from mesh [mm]', rotation=90, ha='center', va='center',
                 fontsize=ps.BASE_PT)
    if legend:
        evd.legend(fig, hs, bbox_to_anchor=(0.5, 1 - 0.02 / H), loc='upper center')
    ps.save(fig, path)
    return fig


def single(ev, path, height=2.45, title=None, **kw):
    """One event, full width (the A1 appendix displays and the contact sheets)."""
    stack(path, [(title or '', ev)], row_h=height - 0.84, top_in=0.42, bot_in=0.42)


def gallery(D, out):
    """The deck appendix: the GALLERY events, one per row, in every colour map x normalisation."""
    os.makedirs(out, exist_ok=True)
    evs = [(t, e, c) for e, c, t in GALLERY]
    for s in evd.STYLES:
        for norm in ('full', 'event'):
            evd.set_style(s, norm=norm)
            rows = [(f'({chr(97 + i)}) {t} (event {e})', D.event(e, c, split=NOBRK)) for i, (t, e, c) in enumerate(evs)]
            stack(os.path.join(out, f'style_{s}_{norm}'), rows, row_h=1.75)
            print('gallery', s, norm, flush=True)
    evd.set_style(evd.DEFAULT)


#: the separate section-5 displays: no longer in the paper (11-03: replaced by two_particle); kept so they can be
#: split out again (figure id -> event, fits drawn)
SINGLES = {
    'display_two_separate': (1063, 'tracks'),
    'display_merged_projection': (17907, 'all'),  # 11-03 round 4: 17907 restored (45001 replaced it by mistake)
    'display_muon_deposit': (33974, 'none'),      # 11-03 round 4: was 5423, 36603 before (both read as 'two particles');
                                                  # a straight muon plus a wide U-shaped curl, vetoed by the 50-strip rule
}
#: the two-particle figure: (label, key of TWO_JSON, event, fits).  Events (Run 1): 1063 two muons ~105 / 70 mm apart;
#: 45001 two columns 20 mm apart in Y, one in X (the pass fits the other particle in Y, oneview class 'merged in
#: projection', recat multi:other_picked) was only a trial and is NOT used: (b) is 17907 again (two columns 63 mm apart in
#: X, one broad column in Y; drawn WITHOUT the axis break, its X window is ~90 mm); 33974 a straight muon plus a wide
#: U-shaped non-track deposit (~70 strips wide in X), > 50 strips, vetoed.
TWO = [('two particles, separate in both views', 'two, separate in both views', 1063, 'tracks'),
       ('two particles, merged in one projection (Y)', 'two, merged in one projection', 17907, 'all'),
       ('a muon and a separate deposit (> 50 strips: vetoed)', 'muon + deposit (> 50 strips, veto)', 33974, 'none')]
#: the efficiencies of two_particle: ONE file, so a re-pass updates the figure by editing it.  Values 11-03: pass 1
#: before the merged-projection / veto fixes (rerun/effbudget/single/figures/two_particle.csv, Runs 1+2, full area).
TWO_JSON = os.path.join(HERE, 'data', 'two_particle_eff.json')
#: what a TPC sees (evd_showcase): (title, event, fits).  11-03: just the muon and two muons, full width; the curling
#: electron (5423) stays in the gallery and in two_particle (c).
SHOWCASE = [('(a) a single cosmic muon', 7884, 'tracks'), ('(b) two cosmic muons', 47235, 'tracks')]
#: main-text loss gallery (A1 class keys of a1_displays.PAPER_DET3)
LOSS_MAIN = ['multi:other_picked', 'multi:m3_not_found', 'far:DELTA', 'far:FIT_SAT', 'spark:discharge', 'spark:shower']
#: A1 classes drawn with a broken axis (particles far apart); every other display has none
A1_BREAK = {'multi:m3_picked', 'multi:other_picked', 'spark:muon+other'}


def two_particle(D, out):
    """One real event per two-particle configuration, one per row, with the muon efficiency in each row's title
    (Dylan 11-03: no side plots, no broken axes)."""
    T = json.load(open(TWO_JSON))['classes']
    rows = []
    for i, (lab, key, e, cands) in enumerate(TWO):
        r = T[key]
        t = (f'({chr(97 + i)}) {lab}:  muon found in {r["eff"]:.1f} $\\pm$ {r["err"]:.1f} % '
             f'({r["k"]} of {r["n"]})')
        rows.append((t, D.event(e, cands, split=NOBRK)))
    stack(os.path.join(out, 'two_particle'), rows, row_h=1.85, title_fs=7.8)
    print('wrote two_particle', flush=True)


def showcase(D, out):
    rows = [(t, D.event(e, c, split=NOBRK)) for t, e, c in SHOWCASE]
    stack(os.path.join(out, 'evd_showcase'), rows, row_h=2.0)
    print('wrote evd_showcase', flush=True)


def _loss_title(title):
    return title.replace('M3 one', 'reference one').replace('"Spark": ', '')


def loss_displays(D, out, keys=None):
    """The A1 appendix displays (one figure per class) and the main-text gallery of six classes."""
    picks = {c: (fid, title, e) for c, fid, title, e in a1.PAPER_DET3}
    for c, (fid, title, e) in picks.items():
        if keys and fid not in keys:
            continue
        cands = 'none' if c.startswith('spark') else 'tracks'
        evd.OPT['mark_sat'] = c in ('far:FIT_SAT',)
        single(D.event(e, cands, split=30.0 if c in A1_BREAK else NOBRK), os.path.join(out, 'a1', fid))
        evd.OPT['mark_sat'] = False
        print('wrote', fid, flush=True)
    if keys and 'loss_gallery' not in keys:
        return
    rows = []
    for i, c in enumerate(LOSS_MAIN):
        fid, title, e = picks[c]
        cands = 'none' if c.startswith('spark') else 'tracks'
        rows.append((f'({chr(97 + i)}) {_loss_title(title)}',
                     D.event(e, cands, split=30.0 if c in A1_BREAK else NOBRK)))
    stack(os.path.join(out, 'loss_gallery'), rows, row_h=1.0, hspace_in=0.36, top_in=0.36, bot_in=0.4,
          axis_labels='last', title_fs=7.8)   # 8.2 in tall: a full page with its caption
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
    ap.add_argument('--norm', default=None, choices=list(evd.NORMS))
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg')
    import paper_style as ps
    ps.use()
    if a.style or a.norm:
        evd.set_style(a.style, **({'norm': a.norm} if a.norm else {}))
    D = Data(a.root, a.ana, a.eos_run_dir)
    if a.mode == 'gallery':
        gallery(D, a.out)
    elif a.mode == 'sheet':
        for e in [int(x) for x in a.ids.split(',')]:
            single(D.event(e, 'all', split=NOBRK), os.path.join(a.out, f'evt_{e}'), title=f'event {e}')
    else:
        figures(D, a.out, [x for x in a.only.split(',') if x])


if __name__ == '__main__':
    main()

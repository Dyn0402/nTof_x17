#!/usr/bin/env python3
"""make_displays_v2.py -- Paper I v2 event displays, recombined from the existing producers (2026-11-10).

    WORK_DATA=~/work_data .venv/bin/python papers/paper1_v2/figs/make_displays_v2.py [--only muon_data_model,two_particle_ab,
                                                                                    not_straight,loss_gallery]
Writes papers/paper1_v2/figs/out/{muon_data_model,two_particle_ab,not_straight,loss_gallery_1,loss_gallery_2}.{pdf,png}
and displays_v2.json (run / event of every displayed event).  Event selection and display style are those of
papers/paper1/figs (make_displays.py, evd.py, make_tpc_principle.py); nothing is recomputed.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OLD = os.path.join(REPO, 'papers', 'paper1', 'figs')
sys.path[:0] = [OLD, os.path.join(REPO, 'papers')]
import make_displays as md  # noqa: E402
import evd  # noqa: E402
import paper_style as ps  # noqa: E402

OUT = os.path.join(HERE, 'out')
CACHE = md.LOCAL + '/principle'
INFO = {}

# the fourteen classes in the order of papers/paper1/sections/a1_table_rows.tex: (class key, public title)
LOSS_ORDER = [
    ('band', 'noise band on a strip patch'),
    ('multi:m3_picked', 'second particle, reference particle fitted, view cut'),
    ('multi:other_picked', 'second particle, the other one fitted'),
    ('multi:m3_not_found', 'second particle, merged in projection'),
    ('spark:discharge', 'spark: discharge'),
    ('spark:muon+other', 'spark: muon and an unrelated deposit'),
    ('spark:shower', 'spark: shower at the crossing'),
    ('far:DELTA', r'$\delta$-ray on the track'),
    ('far:FIT_SAT', r'saturated $\delta$ blob, fit pulled off'),
    ('far:FIT', 'fit off, no other flag'),
    ('fit_failed', 'fit failed in one view'),
    ('far:POS_TAIL', 'position tail, mostly bent reference tracks'),
    ('far:EDGE', 'edge of the other coordinate'),
    ('far:MULTI', 'second deposit, not particle-like'),
]
NOT_STRAIGHT = [('far:DELTA', r'a $\delta$-ray along the track'),
                ('far:FIT_SAT', r'a saturated $\delta$ blob'),
                ('spark:shower', 'a shower at the crossing'),
                ('far:MULTI', 'a second deposit, not particle-like')]


def muon_data_model(D=None):
    import matplotlib.pyplot as plt
    ev = 7884
    r = pickle.load(open(os.path.join(CACHE, f'principle_{ev}.pkl'), 'rb'))
    v_mm = float(r['v_drift']) * 1e-3
    g = evd.GAP
    t0m = np.mean([r['views'][v]['t0'] for v in 'xy'])       # as evd.draw: both views on the mean mesh time
    tlim = (t0m - 7.0 / v_mm, t0m + (g + 7.0) / v_mm)
    half = 9.0
    allW = np.concatenate([np.asarray(r['views'][v]['W'], float).ravel() for v in 'xy'])
    vmax = max(float(np.percentile(allW, 99.5)), 60.0)       # one colour scale for the four panels
    fig, axs = plt.subplots(2, 2, figsize=(ps.FULL[0], 5.3), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.80, bottom=0.09, top=0.97, wspace=0.07, hspace=0.34)
    labs = iter('abcd')
    pc = None
    for i, v in enumerate('xy'):
        d = r['views'][v]
        dt = t0m - float(d['t0'])
        cen = float(d['p0'] + d['w'] * 0.5 * g / v_mm)
        P = np.asarray(d['pos'], float)
        for j, (key, name) in enumerate((('W', 'data'), ('M', 'model at the fit'))):
            ax = axs[i, j]
            pc = evd.cells(ax, np.asarray(d[key], float), P, cen, half, vmax, dt=dt)
            evd.gap_lines(ax, t0m, v_mm, label=False)
            ax.set_xlim(cen - half, cen + half)
            ax.set_ylim(*tlim)
            evd.reference(ax, float(r['m3'][f'mesh_{v}']), float(r['m3'][f'tan_{v}']), t0m, v_mm)
            evd.fit(ax, dict(view=v, p0=d['p0'], w=d['w'], t0=float(d['t0']), role='first'), v_mm, dt=dt)
            ax.set_title(f'({next(labs)}) {v.upper()} view, {name}', loc='left', fontsize=8, pad=3)
            ax.tick_params(direction='in', top=True, right=(j == 0), length=2.5)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color('#555555')
            ax.set_xlabel(f'${v}$ [mm]')
            if j == 0:
                ax.set_ylabel('drift time [ns]')
            else:
                ax.tick_params(labelleft=False)
                sec = ax.secondary_yaxis('right', functions=(lambda t: (t - t0m) * v_mm, lambda z: z / v_mm + t0m))
                sec.set_ylabel('distance from mesh [mm]')
                sec.tick_params(direction='in', length=2.5)
    cax = fig.add_axes([0.915, 0.09, 0.016, 0.88])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label('amplitude [ADC]', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    h = {}
    for ax in axs.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    axs[0, 0].legend([h['reference track'], h['fit']], ['reference track', 'fit'], loc='lower right', fontsize=6.0,
                     frameon=True, framealpha=0.9, edgecolor='none', handlelength=1.5, borderpad=0.25, labelspacing=0.25)
    ps.save(fig, os.path.join(OUT, 'muon_data_model'))
    INFO['muon_data_model'] = dict(run='Run 1', event=ev, views='X and Y, data and model both', vmax_adc=vmax,
                                   fit_tan={v: r['views'][v]['tan'] for v in 'xy'},
                                   ref_tan={v: float(r['m3'][f'tan_{v}']) for v in 'xy'})
    print('wrote muon_data_model', flush=True)


def two_particle_ab(D):
    T = json.load(open(md.TWO_JSON))['classes']
    rows, ids = [], []
    for i, (lab, key, e, cands) in enumerate(md.TWO[:2]):
        r = T[key]
        t = (f'({chr(97 + i)}) {lab}:  muon found in {r["eff"]:.1f} $\\pm$ {r["err"]:.1f} % ({r["k"]} of {r["n"]})')
        rows.append((t, D.event(e, cands, split=md.NOBRK)))
        ids.append(e)
    md.stack(os.path.join(OUT, 'two_particle_ab'), rows, row_h=1.85, title_fs=7.8)
    INFO['two_particle_ab'] = dict(run='Run 1', events=ids)
    print('wrote two_particle_ab', flush=True)


def _rows(D, spec, picks, ids):
    rows = []
    for i, (c, title) in enumerate(spec):
        fid, _, e = picks[c]
        if D.recat is not None and D.recat.get(e) != c:
            print(f'WARNING {fid}: event {e} is {D.recat.get(e)!r} in this tree, not {c!r}', flush=True)
        cands = 'none' if c.startswith('spark') else 'tracks'
        rows.append((f'({chr(97 + i)}) {title}', D.event(e, cands, split=30.0 if c in md.A1_BREAK else md.NOBRK)))
        ids.append(dict(panel=chr(97 + i), event=e, cls=fid))
    return rows


def not_straight(D, picks):
    ids = []
    rows = _rows(D, NOT_STRAIGHT, picks, ids)
    md.stack(os.path.join(OUT, 'not_straight'), rows, row_h=1.5, hspace_in=0.36, top_in=0.36, bot_in=0.4,
             axis_labels='last', title_fs=7.8)
    INFO['not_straight'] = dict(run='Run 1', panels=ids)
    print('wrote not_straight', flush=True)


def loss_gallery(D, picks):
    for k, (lo, hi) in enumerate(((0, 7), (7, 14)), 1):
        ids = []
        rows = _rows(D, LOSS_ORDER[lo:hi], picks, ids)
        md.stack(os.path.join(OUT, f'loss_gallery_{k}'), rows, row_h=0.8, hspace_in=0.3, top_in=0.36, bot_in=0.4,
                 axis_labels='last', title_fs=7.8)
        INFO[f'loss_gallery_{k}'] = dict(run='Run 1', panels=ids)
        print(f'wrote loss_gallery_{k}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    only = [x for x in a.only.split(',') if x]
    import matplotlib; matplotlib.use('Agg')
    ps.use()
    os.makedirs(OUT, exist_ok=True)
    want = lambda k: not only or k in only
    evd.OPT['mark_sat'] = False
    if want('muon_data_model'):
        muon_data_model()
    if want('two_particle_ab') or want('not_straight') or want('loss_gallery'):
        D = md.Data(*md.TREES['tt'], md.DEF_EOS)
        picks = {c: (fid, title, e) for c, fid, title, e in md.a1.PAPER_DET3}
        if want('two_particle_ab'):
            two_particle_ab(D)
        if want('not_straight'):
            not_straight(D, picks)
        if want('loss_gallery'):
            loss_gallery(D, picks)
    p = os.path.join(OUT, 'displays_v2.json')
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(INFO)
    json.dump(old, open(p, 'w'), indent=1)


if __name__ == '__main__':
    main()

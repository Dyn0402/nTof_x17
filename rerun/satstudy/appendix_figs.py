#!/usr/bin/env python3
"""appendix_figs.py -- Paper I appendix A1 (loss categories): one real det3 event display per loss class, as PDF, plus
the class table (fractions) as a LaTeX fragment.

    ../../.venv/bin/python appendix_figs.py [--out DIR]

-> <DIR>/loss_<class>.pdf  (default ~/x17/cosmic_bench/satstudy/appendix/), read by papers/paper1/figures.toml
-> <DIR>/loss_table.json   (class, n, %, example event) for the appendix table

Each display is make_deck.py's evd() (strip position x drift time, M3 extrapolation dashed, production fit solid), redrawn
in matplotlib for print: same data access (make_deck.row / wf / posmap, imported, not copied), same zoom per class.
Classes and fractions: summarize.recat (recat_det3.parquet) -- the re-categorised budget on the PRE-PASS final_phys table
of the golden row sat_det3 (the re-processing pass tables of RERUN_PLAN §11 did not exist on 2026-10-26).
Example events: the deck's hand-checked examples where the deck has one for the class; otherwise the first event of the
class (fixed seed) whose waveforms are cached.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_deck as md  # noqa: E402  (evd's data access; make_deck itself is not modified)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = os.path.join(md.OUT, 'appendix')

#: class -> (figure id, label, example (deck's or chosen by eye from 6 random ones of the class) or None, zoom [mm], draw the production fit, mark clipped samples)
CLASSES = [
    ('band', 'loss_band', 'noise band (baseline offset on a strip patch)', 29808, 60, True, False),
    ('multi:m3_picked', 'loss_multi_m3picked', 'second particle: M3 one fitted, view cut', 30375, 60, True, False),
    ('multi:other_picked', 'loss_multi_otherpicked', 'second particle: the other one fitted', 28078, 45, True, False),
    ('multi:m3_not_found', 'loss_multi_merged', 'second particle: merged in projection', 46350, 22, True, False),
    ('spark:discharge', 'loss_spark_discharge', '"spark": discharge', 18645, 80, False, False),
    ('spark:muon+other', 'loss_spark_muonother', '"spark": muon + unrelated deposit', 36694, 110, False, False),
    ('spark:shower', 'loss_spark_shower', '"spark": shower at the crossing', 41538, 45, False, False),
    ('far:DELTA', 'loss_delta', 'delta-ray on the track', 28537, 18, True, False),
    ('far:FIT_SAT', 'loss_satblob', 'saturated delta blob, fit pulled off', 41373, 14, True, True),
    ('far:FIT', 'loss_fit', 'clean track, fit off', 4490, 16, True, False),
    ('head_on', 'loss_headon', 'head-on: 2-strip cluster, never seeded', 43536, 8, True, False),
    ('far:EDGE', 'loss_edge', 'edge of the other coordinate', None, 20, True, False),
    ('far:POS_TAIL', 'loss_postail', 'position tail (mostly bent M3 tracks)', None, 16, True, False),
    ('far:MULTI', 'loss_multi_deposit', 'second deposit (not particle-like)', None, 30, True, False),
]
GROUP_OTHER = ('cut:other', 'other')


def pick(R, cls, deck, have):
    if deck is not None and R.recat.get(deck) == cls:
        return int(deck)
    ids = np.array(sorted(e for e in R.index[R.recat == cls] if e in have))
    if not len(ids):
        return None
    return int(np.random.default_rng(17).choice(ids))


def evd_mpl(ax, eid, v, zoom, fit=True, mark=False, tmax=1920):
    """make_deck.evd in matplotlib: same cells (|ADC| >= 4 % of the 99.5 % quantile), lines and zoom."""
    r = md.row(eid); ev = md.wf(eid)
    W = ev[v].astype(np.float32); pos = md.posmap(v); ok = np.isfinite(pos)
    o = np.argsort(np.where(ok, pos, np.inf))[:ok.sum()]; P, Wv = pos[o], W[o]
    pref = r.get(f'ref_mesh_{v}', np.nan); tref = r.get(f'ref_tan_{v}', 0.0)
    cen = pref + 15 * tref if np.isfinite(pref) else P[np.argmax(Wv.max(1))]
    sel = np.abs(P - cen) < zoom
    vmax = max(float(np.percentile(Wv[sel], 99.5)) if sel.any() else 100, 60.0)
    ax.set_facecolor(md.sd.seq_color(0))
    pitch = 0.778; rects, vals, clip, bot = [], [], [], []
    for i in np.flatnonzero(sel):
        for k in range(Wv.shape[1]):
            val = float(Wv[i, k])
            if mark and val >= md.S_SAT:
                clip.append(Rectangle((P[i] - pitch / 2, k * md.TS), pitch, md.TS))
            elif mark and val <= -200 and Wv[i].max() >= md.S_SAT:
                bot.append(Rectangle((P[i] - pitch / 2, k * md.TS), pitch, md.TS))
            if val < 0.04 * vmax:
                continue
            rects.append(Rectangle((P[i] - pitch / 2, k * md.TS), pitch, md.TS)); vals.append(min(val / vmax, 1.0))
    pc = PatchCollection(rects, cmap='viridis', linewidths=0); pc.set_array(np.array(vals)); pc.set_clim(0, 1)
    ax.add_collection(pc)
    for lst, col in ((clip, md.CLIPC), (bot, md.BOTC)):
        if lst:
            ax.add_collection(PatchCollection(lst, facecolor='none', edgecolor=col, linewidths=0.7))
    t0v = r.get(f'{v}_t0', np.nan)
    if not np.isfinite(t0v):
        t0v = 257.5 if v == 'x' else 302.5
    ts = np.array([t0v, t0v + 30.0 / md.V_D])
    if np.isfinite(pref):
        ax.plot(pref + tref * md.V_D * (ts - t0v), ts, color=md.M3C, lw=1.4, ls=(0, (5, 3)), label='M3 reference')
    if fit and r.get(f'{v}_ok', False):
        tf = np.array([r[f'{v}_t0'], r[f'{v}_t0'] + 30.0 / md.V_D])
        ax.plot(r[f'{v}_p0'] + r[f'{v}_w'] * (tf - tf[0]), tf, color='white', lw=1.2, label='production fit')
    ax.set_xlim(cen - zoom, cen + zoom); ax.set_ylim(tmax, 0)
    ax.set_xlabel(f'{v} strip position [mm]'); ax.tick_params(labelsize=8)
    return pc


def figure(cls, fid, label, eid, zoom, fit, mark, frac, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True, constrained_layout=True)
    for ax, v in zip(axs, 'xy'):
        pc = evd_mpl(ax, eid, v, zoom, fit=fit, mark=mark)
    axs[0].set_ylabel('drift time [ns]')
    fig.suptitle(f'{label}: {frac:.2f} % of interior M3 tracks  (det3 event {eid})', fontsize=9, x=0.02, ha='left')
    cb = fig.colorbar(pc, ax=axs, shrink=0.9, pad=0.01); cb.set_label('ADC / view max', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    h, lab = axs[0].get_legend_handles_labels()
    if h:
        axs[0].legend(h, lab, fontsize=6.5, loc='lower left', framealpha=0.6)
    p = os.path.join(out, f'{fid}.pdf'); fig.savefig(p); fig.savefig(p.replace('.pdf', '.png'), dpi=110)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=OUT)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = pd.read_parquet(os.path.join(md.OUT, 'recat_det3.parquet'))
    md.row(int(R.index[0]))
    md.wf(28537)                     # a cached event: loads the waveform caches into make_deck._WF (no ROOT read)
    have = set(md._WF)
    n = len(R); cnt = R.recat.value_counts()
    rows = []
    for cls, fid, label, deck, zoom, fit, mark in CLASSES:
        eid = pick(R, cls, deck, have)
        k = int(cnt.get(cls, 0))
        if eid is None:
            print('no example for', cls); continue
        p = figure(cls, fid, label, eid, zoom, fit, mark, 100 * k / n, a.out)
        rows.append(dict(cls=cls, fig=fid, label=label, n=k, pct=100 * k / n, event=eid)); print(p)
    ko = int(sum(cnt.get(c, 0) for c in GROUP_OTHER))
    rows.append(dict(cls='other', fig=None, label='other (cuts, charge elsewhere)', n=ko, pct=100 * ko / n, event=None))
    json.dump(dict(n=n, detected=int(cnt.get('detected', 0)), pct_detected=100 * cnt.get('detected', 0) / n,
                   table='pre-pass final_phys (sat_det3), summarize.recat', rows=rows),
              open(os.path.join(a.out, 'loss_table.json'), 'w'), indent=1)
    print('detected', cnt.get('detected', 0), 'of', n)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""excursions.py -- the local baseline excursions before ~2 ms (open item 2 of README.md, 2026-11-23).

    python -m rerun.ntof.ringing.excursions [--tags 000,001,002,003]

On the char_all.py cache (run_145 stat090_0000), after the ringing correction (deflect, v1), we look for groups of
neighbouring channels whose whole waveform sits below the baseline.  Per channel and event:
    level   the median of the corrected waveform over the window [ADC]
    slope   mean of the last 5 samples minus mean of the first 5 [ADC]
A channel is "low" if level < -NSIG x its single-sample noise; an excursion is a run of >= MINRUN adjacent low channels
in one plane (block edges allowed).  We report, per FEU and ToF bin: the fraction of events with an excursion, their
number, width and depth, the slope (a recovering baseline rises towards the end of the window), where they sit
(block, channel in block, readout connector = 64-channel group), whether the same channels repeat from event to event,
and the dependence on the flash: the flash trigger of the same bunch (is_flash) gives a per-bunch flash size (rms of
its plain-CM waveform), compared with the depth of the excursions in that bunch's early events.
Out: $WORK_DATA/ntof_pass1/ana/ringing/exc_groups.parquet, exc_summary.csv, exc_*.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.ringing import deflect as DF                         # noqa: E402
from rerun.ntof.ringing.char_all import load, FEU_OF, OUT, trigtime, cache_path   # noqa: E402

NSIG = 3.0
MINRUN = 5
BINS = [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 8.0, 80.0]


def groups_of(low):
    k = np.flatnonzero(low)
    if not len(k):
        return []
    runs = np.split(k, np.flatnonzero(np.diff(k) > 1) + 1)
    return [r for r in runs if len(r) >= MINRUN]


def scan(run, sub, tags):
    rows, evrows = [], []
    for tag in tags:
        tt = trigtime(run, sub)
        for feu, name in FEU_OF.items():
            if not cache_path(run, sub, tag, feu).exists():
                continue
            A, ms, ev, nz = load(run, sub, tag, feu)
            bunch = tt.reindex(ev).bunch.to_numpy()
            isf = tt.reindex(ev).is_flash.fillna(0).to_numpy().astype(bool)
            for i in range(len(A)):
                P = DF.plain_cm(A[i])
                if isf[i]:
                    evrows.append(dict(tag=tag, feu=feu, view=name, event=int(ev[i]), bunch=bunch[i], flash=True,
                                       ms=ms[i], flash_rms=float(np.sqrt((P ** 2).mean())),
                                       flash_cm_rms=float(np.median(A[i].reshape(A.shape[1], 8, 64), axis=2).std())))
                    continue
                if not (ms[i] < 3.0 or (ms[i] > 30 and i % 10 == 0)):
                    continue
                W = DF.deflect(A[i], nz)
                lev = np.median(W, axis=0)
                low = lev < -NSIG * nz
                g = groups_of(low)
                evrows.append(dict(tag=tag, feu=feu, view=name, event=int(ev[i]), bunch=bunch[i], flash=False, ms=ms[i],
                                   n_groups=len(g), n_low=int(low.sum()), cm_rms=float(np.median(
                                       A[i].reshape(A.shape[1], 8, 64), axis=2).std())))
                for r in g:
                    seg = W[:, r]
                    rows.append(dict(tag=tag, feu=feu, view=name, event=int(ev[i]), bunch=bunch[i], ms=ms[i],
                                     ch_lo=int(r[0]), ch_hi=int(r[-1]), width=len(r), depth=float(np.median(lev[r])),
                                     depth_min=float(lev[r].min()), depth_sig=float(np.median(lev[r] / nz[r])),
                                     slope=float(np.median(seg[-5:].mean(0) - seg[:5].mean(0))),
                                     shape=np.median(seg, axis=1).astype(np.float32).tolist(),
                                     prof=lev[max(0, r[0] - 10):r[-1] + 11].astype(np.float32).tolist()))
            print(f'tag {tag} {name}: {sum(1 for r in rows if r["tag"] == tag and r["feu"] == feu)} groups', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(evrows)


def summarise(G, E):
    out = []
    ne = E[~E.flash]
    for (feu, name), e in ne.groupby(['feu', 'view']):
        g = G[G.feu == feu]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (e.ms >= lo) & (e.ms < hi)
            if m.sum() < 5:
                continue
            gg = g[(g.ms >= lo) & (g.ms < hi)]
            out.append(dict(feu=feu, view=name, lo=lo, hi=hi, n_events=int(m.sum()),
                            frac_with=float((e.n_groups[m] > 0).mean()), groups_per_event=float(e.n_groups[m].mean()),
                            width_med=float(gg.width.median()) if len(gg) else np.nan,
                            depth_med=float(gg.depth.median()) if len(gg) else np.nan,
                            depth_sig_med=float(gg.depth_sig.median()) if len(gg) else np.nan,
                            slope_med=float(gg.slope.median()) if len(gg) else np.nan))
    return pd.DataFrame(out)


def repeat(G, E):
    """Same channels event to event?  Jaccard overlap of the low-channel sets of consecutive early events (1-2.5 ms) of
    one FEU against random pairs of such events."""
    rng = np.random.default_rng(0)
    out = []
    for feu, g in G[(G.ms >= 1) & (G.ms < 2.5)].groupby('feu'):
        sets = {}
        for r in g.itertuples():
            sets.setdefault((r.tag, r.event), set()).update(range(r.ch_lo, r.ch_hi + 1))
        keys = sorted(sets)
        if len(keys) < 10:
            continue
        jac = lambda a, b: len(a & b) / max(len(a | b), 1)
        cons = [jac(sets[a], sets[b]) for a, b in zip(keys[:-1], keys[1:])]
        rnd = [jac(sets[keys[i]], sets[keys[j]]) for i, j in rng.integers(0, len(keys), (500, 2)) if i != j]
        occ = np.zeros(512)
        for s in sets.values():
            occ[list(s)] += 1
        occ /= len(sets)
        conn = np.bincount(np.arange(512) // 64, weights=occ, minlength=8) / 64
        out.append(dict(feu=feu, view=FEU_OF[feu], n_events=len(keys), jaccard_consecutive=float(np.median(cons)),
                        jaccard_random=float(np.median(rnd)), top_channel_occupancy=float(occ.max()),
                        frac_ch_over_50pct=float((occ > 0.5).mean()),
                        occupancy_by_block=' '.join(f'{v:.2f}' for v in conn)))
    return pd.DataFrame(out)


def flash_dep(G, E):
    """Per bunch: the flash size (rms of the flash trigger's plain-CM waveform) against the summed depth of the
    excursions in that bunch's 1-2 ms events."""
    out = []
    for feu in E.feu.unique():
        f = E[(E.feu == feu) & E.flash].groupby(['tag', 'bunch']).flash_rms.first()
        e = E[(E.feu == feu) & ~E.flash & (E.ms >= 1) & (E.ms < 2)]
        g = G[(G.feu == feu) & (G.ms >= 1) & (G.ms < 2)]
        gs = (g.assign(w=g.width * g.depth).groupby(['tag', 'event']).w.sum())
        e = e.set_index(['tag', 'event'])
        e['exc'] = gs.reindex(e.index).fillna(0).to_numpy()
        e = e.reset_index()
        e['flash_rms'] = [f.get((t, b), np.nan) for t, b in zip(e.tag, e.bunch)]
        e = e.dropna(subset=['flash_rms'])
        if len(e) < 20:
            continue
        from scipy.stats import spearmanr
        r1 = spearmanr(e.flash_rms, e.exc)
        r2 = spearmanr(e.flash_rms, e.n_low)
        r3 = spearmanr(e.cm_rms, e.n_low)
        q = np.quantile(e.flash_rms, [0, 1 / 3, 2 / 3, 1])
        terc = [float((e.n_groups[(e.flash_rms >= a) & (e.flash_rms <= b)] > 0).mean()) for a, b in zip(q[:-1], q[1:])]
        out.append(dict(feu=feu, view=FEU_OF[feu], n=len(e), n_bunches=int(e.bunch.nunique()),
                        spearman_flash_vs_excursion=float(r1.correlation), spearman_flash_vs_nlow=float(r2.correlation),
                        spearman_own_cm_vs_nlow=float(r3.correlation),
                        frac_with_by_flash_tercile=' '.join(f'{v:.2f}' for v in terc),
                        flash_rms_terciles=' '.join(f'{v:.0f}' for v in q)))
    return pd.DataFrame(out)


def figures(G):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    for k, feu in enumerate((3, 4, 7, 8, 1, 2)):
        a = ax.flat[k]
        g = G[(G.feu == feu) & (G.ms >= 1) & (G.ms < 2.5)]
        for s in g['shape'].head(40):
            a.plot(s, color='0.6', lw=0.6)
        if len(g):
            a.plot(np.median(np.stack(g['shape'].to_numpy()), axis=0), 'k', lw=2)
        a.set_title(f'{FEU_OF[feu]}: {len(g)} excursions, 1-2.5 ms', fontsize=9)
        a.set_xlabel('sample (60 ns)')
        a.set_ylabel('median over the group [ADC]')
    fig.tight_layout()
    fig.savefig(OUT / 'exc_shapes.png', dpi=110)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    for k, feu in enumerate((3, 4, 7, 8, 1, 2)):
        a = ax.flat[k]
        g = G[(G.feu == feu) & (G.ms >= 1) & (G.ms < 2.5)]
        if len(g):
            a.hist(np.concatenate([np.arange(r.ch_lo, r.ch_hi + 1) for r in g.itertuples()]), bins=np.arange(0, 513, 4))
        for b in range(1, 8):
            a.axvline(64 * b, color='r', lw=0.5)
        a.set_title(f'{FEU_OF[feu]}: channels in excursions, 1-2.5 ms', fontsize=9)
        a.set_xlabel('channel')
    fig.tight_layout()
    fig.savefig(OUT / 'exc_channels.png', dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub', default='stat090_0000')
    ap.add_argument('--tags', default='000,001,002,003')
    ap.add_argument('--saved', action='store_true', help='summarise the saved scan')
    a = ap.parse_args()
    if a.saved:
        G, E = pd.read_parquet(OUT / 'exc_groups.parquet'), pd.read_parquet(OUT / 'exc_events.parquet')
    else:
        G, E = scan(a.run, a.sub, a.tags.split(','))
        G.to_parquet(OUT / 'exc_groups.parquet', index=False)
        E.to_parquet(OUT / 'exc_events.parquet', index=False)
    S = summarise(G, E)
    S.to_csv(OUT / 'exc_summary.csv', index=False)
    R = repeat(G, E)
    R.to_csv(OUT / 'exc_repeat.csv', index=False)
    F = flash_dep(G, E)
    F.to_csv(OUT / 'exc_flash.csv', index=False)
    figures(G)
    pd.set_option('display.width', 250)
    print(S.round(2).to_string(index=False))
    print(R.round(3).to_string(index=False))
    print(F.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

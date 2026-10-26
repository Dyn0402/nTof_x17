#!/usr/bin/env python3
"""bench.py -- the two-track benchmark for the X17 pair reconstruction (question 3).

    ../../.venv/bin/python bench.py        -> multiparticle/bench.json, bench_pairs.parquet

Two samples, both det3 golden row, forward-fit geometry only:

1. Overlay with truth on BOTH tracks (rerun/satstudy/overlay.py; arms re-run by overlay_arm.py): two clean single
   tracks with t0 within 30 ns summed in the waveforms. Truth for each donor = its M3 ray (absolute) and its own
   single-track fit (isolates what the second track does). A donor is found in a view when a distinct candidate lies
   within MATCH = 2 mm of its single fit (overlay.score's rule). Per separation bin (min of the two views' mid-gap
   separations): both-found efficiency, and for both-found pairs the resolution (robust sigma) of
     separation    reco minus truth, per view
     opening angle 3-D angle between the two reco lines minus the same for the truth lines
     closest approach (DCA) and its height Z_v; Z_v only where the truth opening angle exceeds 10 deg
   The two donors are unrelated muons, so their 'vertex' is geometric only: what is measured is how well det3 fixes
   two lines' relative geometry, which is what a vertex fit uses.
2. The real multi-particle events (legs.py): truth on the M3 leg only. The M3 leg's residual and angle error against
   M3, binned by the separation to the other particle, against clean singles; and the real opening angle's error from
   that leg (open_reco - open, the second line common to both).
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402

MATCH = 2.0
SEP_BINS = [0, 6, 12, 18, 24, 40, 80, 400]
ARMS = {'prod (10-26)': os.path.join(M.S.OUT, 'overlay'), 'off': os.path.join(M.OUT, 'overlay_off'),
        'tt': os.path.join(M.OUT, 'overlay_tt')}


def score_arm(od, L, E, R):
    Mt = pd.read_parquet(os.path.join(od, 'overlays.parquet')); C = pd.read_parquet(os.path.join(od, 'candidates.parquet'))
    C = C[C['rank'] >= 0].copy()
    if 'split_replaced' in C:
        C = C[~C.split_replaced.fillna(False).astype(bool)]
    C['tl'] = np.where(C.plane == 'x', M.tl(C.w, 'x', L), M.tl(C.w, 'y', L))
    rows = []
    for m in Mt.itertuples():
        g = C[C.oid == m.oid]; r = dict(oid=m.oid, mode=m.mode, sep=getattr(m, 'sep', np.nan), a=m.a, b=m.b,
                                        n_x=int((g.plane == 'x').sum()), n_y=int((g.plane == 'y').sum()))
        pick = {}
        for p in 'xy':
            gp = g[g.plane == p]
            for k, d in (('a', m.a), ('b', m.b)):
                if d < 0:
                    continue
                dist = (gp.p0 - E.loc[d, f'{p}_p0']).abs()
                pick[k + p] = dist.idxmin() if len(gp) and dist.min() <= MATCH else None
        if m.mode == 'single':
            r['ok'] = pick['ax'] is not None and pick['ay'] is not None; r['split'] = r['n_x'] > 1 or r['n_y'] > 1
            rows.append(r); continue
        found = all(pick[k] is not None for k in ('ax', 'ay', 'bx', 'by')) and pick['ax'] != pick['bx'] and pick['ay'] != pick['by']
        r['both_found'] = float(bool(found))
        r['one_found'] = float((pick['ax'] is not None and pick['ay'] is not None)
                               or (pick['bx'] is not None and pick['by'] is not None))
        if found:
            reco, sing, tru = {}, {}, {}
            for k, d in (('a', m.a), ('b', m.b)):
                cx, cy = C.loc[pick[k + 'x']], C.loc[pick[k + 'y']]
                reco[k] = (cx.p0, cy.p0, cx.tl, cy.tl)
                sing[k] = (E.loc[d, 'x_p0'], E.loc[d, 'y_p0'], M.tl(E.loc[d, 'x_w'], 'x', L), M.tl(E.loc[d, 'y_w'], 'y', L))
                tru[k] = (R.loc[d, 'mesh_x'], R.loc[d, 'mesh_y'], R.loc[d, 'tan_x'], R.loc[d, 'tan_y'])
            for nm, S_ in (('reco', reco), ('single', sing), ('truth', tru)):
                a_, b_ = S_['a'], S_['b']
                r[f'{nm}_sepx'] = (a_[0] + 15 * a_[2]) - (b_[0] + 15 * b_[2])
                r[f'{nm}_sepy'] = (a_[1] + 15 * a_[3]) - (b_[1] + 15 * b_[3])
                p1, d1 = M.line(*a_); p2, d2 = M.line(*b_)
                r[f'{nm}_dca'], r[f'{nm}_zv'], r[f'{nm}_open'] = (float(v) for v in M.dca(p1, d1, p2, d2))
            for k in 'ab':
                for i, p in enumerate('xy'):
                    r[f'res_{k}{p}'] = reco[k][i] - tru[k][i]; r[f'dtan_{k}{p}'] = reco[k][2 + i] - tru[k][2 + i]
                    r[f'shift_{k}{p}'] = reco[k][i] - sing[k][i]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(T):
    out = dict(n_single=int((T['mode'] == 'single').sum()))
    sg = T[T['mode'] == 'single']
    out['single_ok'] = float(sg.ok.mean()); out['single_split'] = float(sg.split.mean())
    ov = T[T['mode'] == 'overlay'].copy(); ov['sbin'] = np.digitize(ov.sep, SEP_BINS) - 1
    bins = []
    for k, g in ov.groupby('sbin'):
        f = g[g.both_found.astype(bool)]
        b = dict(lo=SEP_BINS[k], hi=SEP_BINS[k + 1], n=int(len(g)), both_found=float(g.both_found.mean()),
                 one_found=float(g.one_found.mean()), n_found=int(len(f)))
        if len(f) >= 5:
            b.update(sep_res=M.rsig(np.r_[f.reco_sepx - f.truth_sepx, f.reco_sepy - f.truth_sepy]),
                     sep_res_vs_single=M.rsig(np.r_[f.reco_sepx - f.single_sepx, f.reco_sepy - f.single_sepy]),
                     sep_bias=float(np.median(np.r_[np.abs(f.reco_sepx) - np.abs(f.truth_sepx),
                                                     np.abs(f.reco_sepy) - np.abs(f.truth_sepy)])),
                     open_res=M.rsig(f.reco_open - f.truth_open), open_res_vs_single=M.rsig(f.reco_open - f.single_open),
                     dca_res=M.rsig(f.reco_dca - f.truth_dca),
                     pos_res=M.rsig(np.r_[f.res_ax, f.res_bx, f.res_ay, f.res_by]),
                     ang_res_deg=float(np.degrees(M.rsig(np.r_[f.dtan_ax, f.dtan_bx, f.dtan_ay, f.dtan_by]))),
                     shift_vs_single=M.rsig(np.r_[f.shift_ax, f.shift_bx, f.shift_ay, f.shift_by]))
            w = f[f.truth_open > 10]
            if len(w) >= 5:
                b['zv_res_open_gt10'] = M.rsig(w.reco_zv - w.truth_zv); b['n_open_gt10'] = int(len(w))
        bins.append(b)
    out['bins'] = bins
    f = ov[ov.both_found.astype(bool) & (ov.sep >= 24)]
    if len(f):
        rel = (f.reco_zv - f.truth_zv); w = f.truth_open > 10
        out['ge24'] = dict(n=int(len(f)), sep_res=M.rsig(np.r_[f.reco_sepx - f.truth_sepx, f.reco_sepy - f.truth_sepy]),
                           open_res=M.rsig(f.reco_open - f.truth_open), dca_res=M.rsig(f.reco_dca - f.truth_dca),
                           zv_res_open_gt10=M.rsig(rel[w]), n_open_gt10=int(w.sum()),
                           # vertex height error scales as separation error / tan(opening): check
                           zv_res_scaled=M.rsig(rel[w] * np.tan(np.radians(f.truth_open[w]))))
    return out


def real(L):
    """Real multi-particle events: the M3 leg against M3, binned by separation to the other particle."""
    Lg = pd.read_parquet(os.path.join(M.OUT, 'legs.parquet'))
    m3 = Lg[Lg.is_m3].copy()
    oth = Lg[~Lg.is_m3 & Lg.m3_found].assign(asep=lambda d: d.sep.abs()).groupby(['event_id', 'plane']).asep.min()
    m3['asep'] = oth.reindex(pd.MultiIndex.from_frame(m3[['event_id', 'plane']])).to_numpy()
    m3['dth'] = np.degrees(np.arctan(m3.tl) - np.arctan(m3.tref))
    out = dict(bins=[])
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 1000)):
        g = m3[(m3.asep >= lo) & (m3.asep < hi)]
        out['bins'].append(dict(lo=lo, hi=hi, n=int(len(g)), res=M.rsig(g.res), dth=M.rsig(g.dth),
                                agree=float(((g.res.abs() <= 3) & (g.dth.abs() <= 5)).mean()) if len(g) else None))
    E = pd.read_parquet(os.path.join(M.OUT, 'events.parquet')); F = E[E.full]
    out['open_err_from_m3_leg'] = M.rsig(F.open_reco - F.open); out['n_full'] = int(len(F))
    out['sep_hist'] = np.histogram(np.minimum(F.sepx.abs(), F.sepy.abs()), bins=SEP_BINS)[0].tolist()
    return out


def main():
    L = M.ladder(); R = M.rays()
    E = pd.read_parquet(MT_EV()).drop_duplicates('event_id').set_index('event_id')
    out = dict(match_mm=MATCH, sep_bins=SEP_BINS, arms={})
    allT = []
    for arm, od in ARMS.items():
        if not os.path.exists(os.path.join(od, 'candidates.parquet')):
            out['arms'][arm] = None; continue
        T = score_arm(od, L, E, R); T['arm'] = arm; allT.append(T)
        out['arms'][arm] = summarize(T)
    pd.concat(allT).to_parquet(os.path.join(M.OUT, 'bench_pairs.parquet'))
    out['real'] = real(L)
    json.dump(out, open(os.path.join(M.OUT, 'bench.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


def MT_EV():
    return M.MT.CAND.replace('.candidates', '')


if __name__ == '__main__':
    main()

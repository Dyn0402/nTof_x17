#!/usr/bin/env python3
"""satblob.py -- saturated delta-electron blobs: the per-channel ADC rail, the bottom rail, and a track + compact-blob
forward model, on the same 350 saturated views + 150 controls as satfit.py (Dylan 10-26 next steps).

Analysis-side prototype: nothing in wft/ is changed (the unified reconstruction rewrite takes the result as its spec).

    satblob.py rails                 per event and view, the pedestal (per channel) and the block common mode (per
                                     sample) the waveforms in wf_det3.pkl were built with -> rails_det3.pkl. The top
                                     rail of a sample is 4095 - ped - CM, the bottom rail 0 - ped - CM (raw ADC 0).
    satblob.py pack [--chunks 60]    payload chunks (window, t0 prior, rails, starts) -> satblob_chunks/chunk_*.pkl
    satblob_core.py <chunk> <out>    the fits, per view (run on lxplus condor: 60 jobs x ~30 min; ~100 s per saturated
                                     view on the laptop).  Track-only production fit (wft.reco.fit_plane) under
          rail      censor at the channel's own top rail instead of SAT = 3550
          bot       SAT = 3550 + the bottom rail censored (one-sided: the true value is <= the rail)
          railbot   both
                                     then the own optimiser (below), track-only and track + blob, on the masks prod,
                                     railbot and railbot@0.05 (+ a 5 % model-error term in quadrature)
    satblob.py pack_pinned           saturated views with the M3 line in the raw frame -> satblob_core.py --pinned
                                     (the same model with the track fixed to M3: optimiser vs model)
    satblob.py merge --dir D         collect the condor outputs -> satblob_{track,blob,pinned}.parquet
    satblob.py score                 -> satblob.json, satblob_scored.parquet;  make_satblob_report.py -> report.html

Track + blob model.  W ~ M_track(p0, w, t0) q + M_blob(pb, kb, t0) b,  q, b >= 0.  M_blob is wft.model.build_matrix
of a w = 0 "track" at pb, restricted to the three depth bins kb-1..kb+1: a point-like deposit at (pb, depth kb) with
its own amplitudes, drifted and read out through the same diffusion, RC kernel and impulse response as the track.
Same NNLS, the same noise weighting, the same t0 prior; censored samples carry the one-sided penalty of production
(top: model >= rail; bottom: model <= rail).

Optimiser (per view): starts = every track fit of that view in fits_det3.parquet / satblob_track.parquet (prod,
tail, strip, low, both, rail, bot, railbot); from each, a track-only Nelder-Mead (p0, w, t0) under the arm's mask,
for the bundle's sigma_p0 and its profile value (as fit_plane).  The best is the track-only fit `T` (reported as arm
mt_<mask>: what the multi-start alone does).  Then a (pb, kb) grid on the window's strips x depth bins at the T track,
the 3 best cells refined by Nelder-Mead over (p0, w, t0, pb) at fixed kb, then kb +- 1: the fit `TB`.
Model selection (Gaussian likelihood, unknown sigma -- the bench chi2/dof is model-error dominated, ~35 on clean
tracks, so a fixed-sigma Delta chi2 always adds a blob):
    dBIC = n ln(chi2_T / chi2_TB) - k ln n,   k = 5 (pb, kb, 3 bin amplitudes), n = uncensored samples
the blob is kept when dBIC > 0.  Run with WFT_PIVOT_START=1 (as satfit.py).  Run heavy steps under the flock wrapper.
"""
import argparse, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402  (sets sys.path for wft / qa_config / olib)
from olib import load  # noqa: E402

RAILS = os.path.join(S.OUT, 'rails_det3.pkl')
TRACK = os.path.join(S.OUT, 'satblob_track.parquet')
BLOB = os.path.join(S.OUT, 'satblob_blob.parquet')
PINNED = os.path.join(S.OUT, 'satblob_pinned.parquet')


# ------------------------------------------------------------------------------------------------ rails
def rails():
    from qa_config import get_config
    from wft import io as wio
    D = pickle.load(open(os.path.join(S.OUT, 'wf_det3.pkl'), 'rb'))
    want = set(D['wf']); cfg = get_config(S.KEY); out = {}; chk = []; rawmax = []
    nblk = 512 // wio.CNS_BLOCK
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f)
            idx = np.flatnonzero(np.isin(r.event_ids, list(want)))
            if not len(idx):
                continue
            for i in idx:
                arr = r.tree.arrays(['eventId', 'amplitude'], entry_start=int(i), entry_stop=int(i) + 1, library='np')
                a = arr['amplitude'][0]
                if len(a) != r.n_sample * 512:
                    continue
                raw = a.reshape(-1, 512).astype(np.float32)
                sub = raw - r.ped
                cms = np.median(sub.reshape(raw.shape[0], nblk, wio.CNS_BLOCK), axis=2)     # (nsamp, nblk)
                eid = int(arr['eventId'][0])
                out.setdefault(eid, {})[v] = dict(ped=r.ped.astype(np.float32), cms=cms.astype(np.float32))
                W = D['wf'][eid].get(v)
                if W is not None:
                    rec = (sub - np.repeat(cms, wio.CNS_BLOCK, axis=1)).T
                    chk.append(float(np.abs(rec - W.astype(np.float32)).max()))
                m = raw.max(0); rawmax += list(m[m > 3800])
    pickle.dump(out, open(RAILS, 'wb'))
    h = np.bincount(np.clip(np.array(rawmax, int) - 3800, 0, 300))
    print(f'{len(out)} events; max |rebuilt - stored W| = {max(chk):.2f} ADC (median {np.median(chk):.2f})')
    print('raw strip maxima > 3800: top values', {3800 + i: int(c) for i, c in enumerate(h) if c > 50})


def rail_arrays(R, eid, v, ch):
    """(top, bottom) rail of every (channel, sample) of a window, pedestal + CM subtracted like W."""
    r = R[eid][v]
    cm = r['cms'][:, np.asarray(ch) // 64].T             # (n, nsamp)
    ped = r['ped'][np.asarray(ch)][:, None]
    return (4095.0 - ped - cm).astype(np.float32), (-ped - cm).astype(np.float32)


# ------------------------------------------------------------------------------------------------ jobs
def jobs():
    """The same 350 saturated + 150 control event-views as satfit.fit (same windows, same random subset)."""
    Sd = pickle.load(open(os.path.join(S.OUT, 'wf_det3.pkl'), 'rb')); wf = Sd['wf']; sat = set(Sd['sat'])
    d = load(S.DET).set_index('event_id')
    from wft.io import strip_position_map
    from qa_config import get_config
    cfg = get_config(S.KEY); pm = strip_position_map(cfg); pmv = {'x': pm[cfg.MX17_FEU_X], 'y': pm[cfg.MX17_FEU_Y]}
    J = []
    for eid, ev in wf.items():
        r = d.loc[eid]
        for v in 'xy':
            if v not in ev or not r[f'{v}_ok']:
                continue
            P = S.window(ev[v].astype(np.float32), ev[f'noise_{v}'].astype(np.float32), pmv[v], r[f'{v}_p0'],
                         r[f'{v}_w'] * 1e3 / 35.8)
            if P is not None:
                J.append((int(eid), v, P, r.get(f'{v}_ftst', np.nan)))
    rng = np.random.default_rng(7)
    js = [j for j in J if j[0] in sat and d.loc[j[0], f'h{j[1]}_nsat'] >= S.NSAT_MIN]
    jc = [j for j in J if j[0] not in sat]
    js = [js[i] for i in sorted(rng.choice(len(js), min(S.N_SAT_VIEWS, len(js)), replace=False))]
    jc = [jc[i] for i in sorted(rng.choice(len(jc), min(S.N_CTRL_VIEWS, len(jc)), replace=False))]
    return js + jc, sat


def pack(nchunk, outdir):
    """Payload chunks for satblob_core.py (local or condor): window, prior, rails, starts, track arms."""
    from wft import reco
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(S.BUNDLE)
    R = pickle.load(open(RAILS, 'rb'))
    J, sat = jobs()
    F = pd.read_parquet(os.path.join(S.OUT, 'fits_det3.parquet')); F = F[F.ok]
    st = {k: g[['p0', 'w', 't0']].to_numpy() for k, g in F.groupby(['event_id', 'view'])}
    out = []
    for e, v, P, ftst in J:
        top, bot = rail_arrays(R, e, v, P['ch'])
        pr = reco.t0_prior_for(cal, v, ftst) if ftst is not None and np.isfinite(ftst) else None
        out.append(dict(event_id=e, view=v, group='sat' if e in sat else 'ctrl',
                        P={k: np.asarray(P[k]) for k in ('W', 'pos', 'noise', 'ch')}, prior=pr, top=top, bot=bot,
                        starts=st.get((e, v), np.zeros((0, 3))),
                        arms=list(TRACK_ARMS) if e in sat else ['rail']))   # controls never clip: only rail can move
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0); order = rng.permutation(len(out))     # mix slow saturated and fast controls
    for c in range(nchunk):
        pickle.dump([out[i] for i in order[c::nchunk]], open(os.path.join(outdir, f'chunk_{c:03d}.pkl'), 'wb'))
    print(len(out), 'views in', nchunk, 'chunks ->', outdir)


TRACK_ARMS = ('rail', 'bot', 'railbot')


def pack_pinned(nchunk, outdir):
    """Payload for the pinned diagnostic: the saturated views, with the M3 track in the fit's raw frame
    (p0 = ref_mesh, w from the ladder: w = (tan * s + c) / 1e3 mm/ns)."""
    import glob
    d = load(S.DET).set_index('event_id'); L = d.attrs['ladder']
    Js = []
    for f in sorted(glob.glob(os.path.join(S.OUT, 'satblob_chunks', 'chunk_*.pkl'))):
        Js += [j for j in pickle.load(open(f, 'rb')) if j['group'] == 'sat']
    for j in Js:
        r = d.loc[j['event_id']]; v = j['view']; s_, c_ = L[v]
        j['ref'] = (float(r[f'ref_mesh_{v}']), float((r[f'ref_tan_{v}'] * s_ + c_) / 1e3))
    os.makedirs(outdir, exist_ok=True)
    for c in range(nchunk):
        pickle.dump(Js[c::nchunk], open(os.path.join(outdir, f'pin_{c:03d}.pkl'), 'wb'))
    print(len(Js), 'saturated views in', nchunk, 'chunks ->', outdir)


def merge(indir):
    import glob
    for kind, path in (('track', TRACK), ('blob', BLOB), ('pinned', PINNED)):
        fs = sorted(glob.glob(os.path.join(indir, f'*_{kind}.parquet')))
        if not fs:
            continue
        F = pd.concat([pd.read_parquet(f) for f in fs if os.path.getsize(f) > 0], ignore_index=True)
        F.to_parquet(path); print(kind, len(F), 'rows from', len(fs), 'files ->', path)


# ------------------------------------------------------------------------------------------------ score
def score():
    import json
    sys.path.insert(0, HERE)
    from summarize import view_agree
    d = load(S.DET).set_index('event_id'); L = d.attrs['ladder']
    F0 = pd.read_parquet(os.path.join(S.OUT, 'fits_det3.parquet'))
    F0 = F0[F0.arm == 'prod'][['event_id', 'view', 'arm', 'ok', 'p0', 'w', 'chi2', 'dof', 'group']]
    T = pd.read_parquet(TRACK) if os.path.exists(TRACK) else pd.DataFrame()
    B = pd.read_parquet(BLOB) if os.path.exists(BLOB) else pd.DataFrame()
    grp = F0.drop_duplicates(['event_id', 'view']).set_index(['event_id', 'view']).group
    rows = [F0]
    if len(T):
        T = T.copy(); T['group'] = [grp.get((e, v)) for e, v in zip(T.event_id, T.view)]
        rows.append(T[['event_id', 'view', 'arm', 'ok', 'p0', 'w', 'chi2', 'dof', 'group']])
    if len(B):
        B = B.copy(); B['group'] = [grp.get((e, v)) for e, v in zip(B.event_id, B.view)]
        mt = B.assign(arm='mt_' + B['mask'], ok=True, p0=B.p0_T, w=B.w_T, chi2=B.chi2_T, dof=B.n)
        rows.append(mt[['event_id', 'view', 'arm', 'ok', 'p0', 'w', 'chi2', 'dof', 'group']])
        for thr in (0.0,):
            take = B.dbic > thr
            bb = B.assign(arm=B.arm, ok=True, p0=np.where(take, B.p0_TB, B.p0_T), w=np.where(take, B.w_TB, B.w_T),
                          chi2=np.where(take, B.chi2_TB, B.chi2_T), dof=B.n)
            rows.append(bb[['event_id', 'view', 'arm', 'ok', 'p0', 'w', 'chi2', 'dof', 'group']])
    F = pd.concat(rows, ignore_index=True)
    F = F[F.event_id.isin(d.index)].reset_index(drop=True)
    ref = d.loc[F.event_id]
    for v in 'xy':
        m = (F.view == v).to_numpy()
        a, dl, tl = view_agree(F.p0[m], F.w[m], ref[f'ref_mesh_{v}'].to_numpy()[m],
                               ref[f'ref_tan_{v}'].to_numpy()[m], L[v])
        F.loc[m, 'agree'] = a; F.loc[m, 'dl'] = dl; F.loc[m, 'tl'] = tl
        F.loc[m, 'rtan'] = ref[f'ref_tan_{v}'].to_numpy()[m]
        F.loc[m, 'res'] = F.p0[m].to_numpy() - ref[f'ref_mesh_{v}'].to_numpy()[m]
    F['agree'] = F.agree.fillna(False).astype(bool) & F.ok.astype(bool)
    F['steeper'] = np.abs(F.tl) > np.abs(F.rtan)
    out = {}
    for (g, arm), h in F.groupby(['group', 'arm']):
        far = h[~h.agree]
        out[f'{g}|{arm}'] = dict(n=int(len(h)), agree=float(h.agree.mean()),
                                 dl_med=float(h.dl.median()), dl_absmed=float(h.dl.abs().median()),
                                 dl_sigma68=float(np.percentile(np.abs(h.dl.dropna()), 68.27)),
                                 dtan_med=float((np.abs(h.tl) - np.abs(h.rtan)).median()),
                                 steeper_frac=float(h.steeper.mean()),
                                 far_steeper=float(far.steeper.mean()) if len(far) else None,
                                 res_med=float(h.res.median()))
    piv = F.pivot_table(index=['event_id', 'view', 'group'], columns='arm', values='agree', aggfunc='first')
    for arm in piv.columns:
        if arm == 'prod':
            continue
        for g in ('sat', 'ctrl'):
            p = piv.xs(g, level='group')[['prod', arm]].dropna()
            if len(p):
                out[f'switch|{g}|{arm}'] = dict(gained=int((~p['prod'].astype(bool) & p[arm].astype(bool)).sum()),
                                                lost=int((p['prod'].astype(bool) & ~p[arm].astype(bool)).sum()),
                                                n=int(len(p)))
    if len(B):
        for mk, h in B.groupby('mask'):
            for g in ('sat', 'ctrl'):
                hh = h[h.group == g]
                if not len(hh):
                    continue
                out[f'blobrate|{g}|{mk}'] = dict(n=int(len(hh)), added=float((hh.dbic > 0).mean()),
                                                 dbic_q={str(p): float(np.percentile(hh.dbic, p))
                                                         for p in (10, 50, 90)},
                                                 qblob_frac_med=float((hh.q_blob / (hh.q_blob + hh.q_track))
                                                                      [hh.dbic > 0].median())
                                                 if (hh.dbic > 0).any() else None)
            # threshold scan: blob kept if dBIC > t
            scan = []
            for t in (-20, 0, 20, 50, 100, 200, 400, 800):
                take = h.dbic > t
                hh = h.assign(p0=np.where(take, h.p0_TB, h.p0_T), w=np.where(take, h.w_TB, h.w_T))
                rr = d.loc[hh.event_id]; ag = np.zeros(len(hh), bool)
                for v in 'xy':
                    m = (hh.view == v).to_numpy()
                    ag[m] = view_agree(hh.p0[m], hh.w[m], rr[f'ref_mesh_{v}'].to_numpy()[m],
                                       rr[f'ref_tan_{v}'].to_numpy()[m], L[v])[0]
                scan.append(dict(thr=t, sat_agree=float(ag[(hh.group == 'sat').to_numpy()].mean()),
                                 ctrl_agree=float(ag[(hh.group == 'ctrl').to_numpy()].mean())
                                 if (hh.group == 'ctrl').any() else None,
                                 sat_added=float(take[hh.group == 'sat'].mean()),
                                 ctrl_added=float(take[hh.group == 'ctrl'].mean()) if (hh.group == 'ctrl').any()
                                 else None))
            out[f'scan|{mk}'] = scan
    # the rail facts on the fitted windows (from the payload chunks)
    import glob
    from satblob_core import masks as cmasks
    a_only = b_only = both_ = nbot = nview_bot = nview = 0
    for f in sorted(glob.glob(os.path.join(S.OUT, 'satblob_chunks', 'chunk_*.pkl'))):
        for j in pickle.load(open(f, 'rb')):
            if j['group'] != 'sat':
                continue
            Wr = np.asarray(j['P']['W'], float)
            hp, _ = cmasks(Wr, j['top'], j['bot'], 'prod'); hr, lo = cmasks(Wr, j['top'], j['bot'], 'railbot')
            a_only += int((hp & ~hr).sum()); b_only += int((hr & ~hp).sum()); both_ += int((hp & hr).sum())
            nbot += int(lo.sum()); nview_bot += int(lo.any()); nview += 1
    out['railfacts'] = dict(n_views=nview, prod_only=a_only, rail_only=b_only, both=both_, n_bot=nbot,
                            views_with_bot=nview_bot)
    if os.path.exists(PINNED) and len(B):
        Pn = pd.read_parquet(PINNED)
        ag = F[F.arm.str.startswith('blob_')][['event_id', 'view', 'arm', 'agree']]
        for mk, h in Pn.groupby('mask'):
            j = h.merge(B[B['mask'] == mk][['event_id', 'view', 'chi2_T', 'chi2_TB']], on=['event_id', 'view'])
            j = j.merge(ag[ag.arm == f'blob_{mk}'][['event_id', 'view', 'agree']], on=['event_id', 'view'])
            for nm, g in (('fail', j[~j.agree]), ('agree', j[j.agree])):
                if not len(g):
                    continue
                out[f'pinned|{mk}|{nm}'] = dict(
                    n=int(len(g)), ref_worse_TB=float((g.chi2_TBref > g.chi2_TB * 1.0001).mean()),
                    ratio_TB_med=float((g.chi2_TBref / g.chi2_TB).median()),
                    ratio_T_med=float((g.chi2_Tref / g.chi2_T).median()),
                    blob_gain_ref_med=float((1 - g.chi2_TBref / g.chi2_Tref).median()),
                    blob_gain_free_med=float((1 - g.chi2_TB / g.chi2_T).median()))
    F.to_parquet(os.path.join(S.OUT, 'satblob_scored.parquet'))
    json.dump(out, open(os.path.join(S.OUT, 'satblob.json'), 'w'), indent=1)
    for k, v in out.items():
        if not k.startswith('scan'):
            print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != 'dbic_q'})
        else:
            print(k); [print('  ', s) for s in v]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['rails', 'pack', 'pack_pinned', 'merge', 'score'])
    ap.add_argument('--chunks', type=int, default=50)
    ap.add_argument('--dir', default=os.path.join(S.OUT, 'satblob_chunks'))
    a = ap.parse_args()
    if a.cmd == 'rails':
        rails()
    elif a.cmd == 'pack':
        pack(a.chunks, a.dir)
    elif a.cmd == 'pack_pinned':
        pack_pinned(a.chunks, a.dir)
    elif a.cmd == 'merge':
        merge(a.dir)
    else:
        score()


if __name__ == '__main__':
    main()

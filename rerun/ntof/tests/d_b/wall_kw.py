#!/usr/bin/env python3
"""wall_kw.py -- D's x angle scale on run_149 cosmics from its own SiPM wall, on the pass-1 v1_D fits.

The estimator is ntof_cosmics/cosmic_wall_scale.py's, unchanged: ntof_scint_stack.ana.fit_wall_u, the scale s in
u + L s tan set by how sharp the wall's three internal group boundaries are (L = w_wall - w_strip = 97.4 mm), on
single x tracks (x quality + plausible), single-group events.  It is valid on cosmics only (on beam s is degenerate
with the boundary offsets).  On the old chain it read A 1.15 against the A-C line's 1.11 (+3.6 %), which is carried
here as the method's systematic.

Wall hits are the ones cosmic_wall_scale.py build already extracted (wall_hits_D.parquet: clock_match triggers,
+-1 us around the matched singles time; window +-50 ns around the per-triggering-arm peak), joined to the new D fits
(d_b/reco, events.py lists).  s = true / fitted, so kw_D,x = kw_v1 / s.

    wall_kw.py [--reco-glob ...] [--label v1]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]

W = '/media/ucla/work/ntof_pass1'
OLD = '/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale'
OUT = f'{W}/d_b/wall_kw'


def plastic_hits(arm, subs):
    """the arm's plastic (PSS<arm>, two bars) hits within +-1 us of every clock-matched trigger of these sub-runs, as
    cosmic_wall_scale.wall_hits reads the wall."""
    import uproot
    from ntof_cosmics import clock_match as CM
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_scint_stack.extract import MV_PER_ADC
    cache = {}
    out = []
    for sub in subs:
        trig = CWS.matched_triggers(sub)
        for ntof, g in trig.groupby('ntof'):
            if ntof not in cache:
                want = ['BunchNumber', 'detn', 'amp', 'tof']
                parts = [uproot.open(f)[f'PSS{arm}'].arrays(want, library='np') for f in CM._parts(int(ntof))]
                d = {k: np.concatenate([p[k] for p in parts]) for k in want}
                o = np.lexsort((d['tof'], d['BunchNumber']))
                cache[ntof] = {k: v[o] for k, v in d.items()}
            P = cache[ntof]
            key = P['BunchNumber'].astype(np.float64) * 1e9 + P['tof']
            k0 = g.bunch.to_numpy(np.float64) * 1e9 + g.tof.to_numpy()
            lo, hi = np.searchsorted(key, k0 - 1000), np.searchsorted(key, k0 + 1000)
            n = hi - lo
            i_t = np.repeat(np.arange(len(g)), n)
            i_h = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
            out.append(pd.DataFrame(dict(subrun=sub, event_id=g.event_id.to_numpy()[i_t],
                                         trig_arm=g.ntof_arm.to_numpy()[i_t], detn=P['detn'][i_h].astype(int),
                                         amp=P['amp'][i_h] * MV_PER_ADC, dt=P['tof'][i_h] - g.tof.to_numpy()[i_t])))
    return pd.concat(out, ignore_index=True)


def plastic(T, arm, label):
    """D's x scale from its two plastic bars: the L/R gap (one edge, lever L_plas = 192.6 mm, twice the wall's),
    ntof_scint_stack.ana.fit_plas_u, single x tracks with exactly one bar lit."""
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_scint_stack import ana as SA
    g = SA.geometry(arm)
    cp = f'{OUT}/plas_hits_{arm}.parquet'
    subs = sorted(T.subrun.unique())
    if os.path.exists(cp):
        H = pd.read_parquet(cp)
        if set(subs) - set(H.subrun.unique()):
            H = None
    else:
        H = None
    if H is None:
        H = plastic_hits(arm, subs)
        H.to_parquet(cp)
    pk = CWS.peaks(H)
    c = H.trig_arm.map({a: v[0] for a, v in pk.items()})
    x = H[(H.dt - c).abs() <= CWS.WIN_NS].sort_values('amp', ascending=False).drop_duplicates(
        ['subrun', 'event_id', 'detn'])
    P = x.pivot_table(index=['subrun', 'event_id'], columns='detn', values='amp', aggfunc='first')
    P.columns = [f'p{int(c)}_amp_on' for c in P.columns]
    for n in (1, 2):
        if f'p{n}_amp_on' not in P:
            P[f'p{n}_amp_on'] = np.nan
    P = P[['p1_amp_on', 'p2_amp_on']].reset_index()
    t = T.drop(columns=[c for c in ('p1_amp_on', 'p2_amp_on') if c in T]).merge(P, on=['subrun', 'event_id'],
                                                                                 how='left')
    sel = t.x_quality_ok.astype(bool) & t.x_plausible.astype(bool)
    x = CWS._cosmic_singles(t, sel)
    r = SA.fit_plas_u(x, g, g['L_plas'], 1.0)
    r = SA.fit_plas_u(x, g, g['L_plas'], r['s'])
    rng = np.random.default_rng(2)
    subs = x.subrun.unique()
    boot = []
    for _ in range(200):
        xb = pd.concat([x[x.subrun == s_] for s_ in rng.choice(subs, len(subs))])
        try:
            boot.append(SA.fit_plas_u(xb, g, g['L_plas'], r['s'])['s'])
        except Exception:      # noqa: BLE001
            pass
    res = dict(layer='plastic L/R gap', lever=g['L_plas'], plas_gap=g['plas_gap'], n_single_x=int(len(x)),
               n_one_bar=int((x.p1_amp_on.notna() ^ x.p2_amp_on.notna()).sum()), **{k: r[k] for k in ('s', 's_err', 'sigma', 'n')},
               boot_lo=float(np.percentile(boot, 16)), boot_hi=float(np.percentile(boot, 84)),
               boot_median=float(np.median(boot)), peaks={k: v[0] for k, v in pk.items()})
    print('plastic:', json.dumps(res, default=float))
    json.dump(res, open(f'{OUT}/plastic_{arm}_{label}.json', 'w'), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='D')
    ap.add_argument('--reco-glob', default=f'{W}/d_b/reco/r149_*_D_v1_[a-z].parquet')
    ap.add_argument('--label', default='v1')
    a = ap.parse_args()
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_scint_stack import ana as SA
    from sept26_prelim_analysis.det_a_scint import layer_geometry
    import line_kw as LK
    arm = a.arm
    CWS.ARM = arm
    os.makedirs(OUT, exist_ok=True)
    geo = layer_geometry('run_149', arm)
    L = geo['w_wall'] - geo['w_strip']
    old_dir = OLD if arm == 'A' else f'{OLD}/arm_{arm}'
    H = pd.read_parquet(f'{old_dir}/wall_hits_{arm}.parquet')
    pk = CWS.peaks(H)
    Wd = pd.concat([CWS.wide(g, pk).assign(subrun=s) for s, g in H.groupby('subrun')], ignore_index=True)
    trig = H[['subrun', 'event_id', 'trig_arm']].drop_duplicates(['subrun', 'event_id'])
    trs = LK.transforms()
    T = []
    for f in sorted(glob.glob(a.reco_glob)):
        sub = 'cosbounce_' + os.path.basename(f).split('_cosbounce_')[1].split(f'_{arm}_')[0]
        t = LK.fitted(f, arm, trs[f'mx17_{arm}'], single=False).assign(subrun=sub)
        T.append(t)
    T = pd.concat(T, ignore_index=True)
    # matched triggers only (the wall hits exist only for them)
    T = T.merge(trig, on=['subrun', 'event_id'], how='inner')
    T = T.merge(Wd, on=['subrun', 'event_id'], how='left')
    T['u_mm'] = T.x_local + geo['u_mm']
    T['tan_raw_x'] = T.tanx
    # frame check against the old chain's tables (same events): x_local must agree, tans correlate
    old = pd.read_parquet(f'{old_dir}/tracks_{arm}.parquet')
    j = T.merge(old[['subrun', 'event_id', 'x_local', 'tan_raw_x']], on=['subrun', 'event_id'], suffixes=('', '_old'))
    j = j.groupby(['subrun', 'event_id']).filter(lambda g: len(g) == 1)
    frame = dict(n_common=int(len(j)),
                 dx_local_med=float(np.median(j.x_local - j.x_local_old)) if len(j) else np.nan,
                 dx_local_within2mm=float(np.mean(np.abs(j.x_local - j.x_local_old) < 2)) if len(j) else np.nan,
                 tan_new_over_old_med=float(np.median((j.tan_raw_x / j.tan_raw_x_old)[j.tan_raw_x_old.abs() > 0.1]))
                 if len(j) else np.nan)
    # the old chain's wall result chained onto the new fits: true = s_old x tan_old and tan_new = r x tan_old on the
    # same tracks, so s_new = s_old / r (s_old = 1.525 from 3,391 tracks over 35 sub-runs, cosmic_wall_scale arm_D)
    old_s = json.load(open(f'{old_dir}/summary.json'))
    w = j.tan_raw_x_old.abs().between(0.1, 0.6) & np.isfinite(j.tan_raw_x)
    rr = (j.tan_raw_x / j.tan_raw_x_old)[w]
    if len(rr) >= 10:
        rng = np.random.default_rng(1)
        b = [np.median(rr.to_numpy()[rng.integers(0, len(rr), len(rr))]) for _ in range(300)]
        frame.update(n_chain=int(len(rr)), r_new_over_old=float(np.median(rr)), r_err=float(np.std(b)),
                     s_old=float(old_s['s_best']), s_chain=float(old_s['s_best'] / np.median(rr)),
                     s_chain_err=float(old_s['s_best'] / np.median(rr) ** 2 * np.std(b)))
    print('frame check vs old chain:', frame)
    sel_x = T.x_quality_ok.astype(bool) & T.x_plausible.astype(bool)
    rows = []
    for lab, m in (('gated', T.gated.astype(bool)), ('x plane only (quality + plausible)', sel_x)):
        x = CWS._cosmic_singles(T, m)
        for samp, mm in (('all', np.ones(len(x), bool)), (f'{arm}-triggered', x.trig_arm == arm),
                         ('other-arm triggered', x.trig_arm != arm),
                         ('|tan| < 0.15', x.tan_raw_x.abs() < 0.15),
                         ('|tan| 0.15-0.3', x.tan_raw_x.abs().between(0.15, 0.3)),
                         ('|tan| 0.3-0.6', x.tan_raw_x.abs().between(0.3, 0.6))):
            rows.append(CWS.scales(x[mm], L, 'tan_raw_x', samp) | dict(selection=lab, sample=samp))
    R = pd.DataFrame(rows)
    x = CWS._cosmic_singles(T, sel_x)
    P = CWS.profile(x, L, np.round(np.arange(0.70, 1.501, 0.0125), 4))
    subs = x.subrun.unique()
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):
        xb = pd.concat([x[x.subrun == s] for s in rng.choice(subs, len(subs))])
        try:
            boot.append(SA.fit_wall_u(xb, L, 1.0)['s'])
        except Exception:      # noqa: BLE001
            pass
    # the profile's 1-sigma interval (delta -log L = 0.5)
    Pm = P.s[P.dnll.idxmin()]
    inside = P[P.dnll <= 0.5]
    kw = json.load(open(f'{W}/bundles/v1_{arm}/bundle.json'))['kw']
    s_best = float(Pm)
    summ = dict(arm=arm, label=a.label, L=L, n_tracks=int(len(T)), n_single_x=int(len(x)), frame_check=frame,
                s_best=s_best, s_prof_lo=float(inside.s.min()), s_prof_hi=float(inside.s.max()),
                boot_median=float(np.median(boot)), boot_lo=float(np.percentile(boot, 16)),
                boot_hi=float(np.percentile(boot, 84)), n_boot=len(boot),
                kw_bundle_x=kw['x'], kw_x_from_wall=kw['x'] / s_best,
                method_systematic='A on the old chain: wall 1.15 vs A-C line 1.11 (+3.6 %)')
    print(json.dumps(summ, indent=1))
    with pd.option_context('display.width', 250):
        print(R[['selection', 'sample', 'n', 's', 's_err', 'sigma']].round(3).to_string(index=False))
    R.to_csv(f'{OUT}/scales_{arm}_{a.label}.csv', index=False)
    P.to_csv(f'{OUT}/profile_{arm}_{a.label}.csv', index=False)
    json.dump(summ, open(f'{OUT}/summary_{arm}_{a.label}.json', 'w'), indent=1)
    T.drop(columns=[c for c in T.columns if c.startswith('d_') or c.startswith('p0_')]).to_parquet(
        f'{OUT}/tracks_{arm}_{a.label}.parquet')
    plastic(T, arm, a.label)


if __name__ == '__main__':
    main()

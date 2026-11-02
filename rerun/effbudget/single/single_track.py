#!/usr/bin/env python3
"""single_track.py [--root R] [--out DIR] -- Paper I: is det3's single-track efficiency measurable without bias?

The A1 loss budget calls an event "multi-particle" when det3 itself sees >= 2 particle-like candidates in a view.
That flag is set by the chamber under test, so a chamber failure that looks like two tracks (a delta ray, a split
or broken track, a discharge read as a shower) would be moved out of a "single-track" sample by its own failure:
a selection biased towards efficiency. Here the event is tagged with the M3 telescope instead, which det3 cannot
influence:

  M3 2nd line   another straight 4-layer cluster combination (chi2 < 1 mm^2, the production recipe) in a view,
                >= SEP_MM from the chosen ray at det3 -> a second through-going particle in the telescope;
  M3 clean      no such line in either view (extra clusters that do not form a line are allowed: M3 planes carry
                1-2 noise/split clusters on ~30-40 % of triggers).

Efficiency on the M3-clean sample is the single-track efficiency with an external selection. A second particle
that misses M3 (a shower or delta produced near det3, a steep secondary) stays in that sample and counts against
det3, so the number is conservative, never inflated. Comparing with the det3-tagged split shows how large the
self-selection bias of the det3 flag is.

Inputs (same relative layout as EOS cosmic_bench/, so --root is either EOS on lxplus or a local mirror):
  x17rerun/paper1_eff/a1/<rel>/recat.parquet, cand_feat_det3.parquet   (A1 classes, candidate time extent)
  x17rerun/paper1_eff/budget/<rel>/rays_det3.parquet                   (M3 rays at det3, cat_xy_5)
  Analysis_pass1/<rel>/wft/events.candidates.parquet, alignment/alignment.json
  x17rerun/m3clus/<run>__<subrun>__NNN_{analyse,rays}.root             (M3 cluster trees, rerun/outliers/m3clus)
Writes <out>/single_track.json and per-event <out>/single_track_events.parquet.
"""
import argparse, glob, itertools, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': ('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V', 'mx17_3'),
        'p2_det3': ('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check', 'mx17_3')}
PLANES = {1: ('x', 0), 3: ('x', 1), 5: ('x', 2), 6: ('x', 3), 0: ('y', 0), 2: ('y', 1), 4: ('y', 2), 7: ('y', 3)}
ZL = np.array([1302.0, 1185.0, 144.0, 24.0])
MAXCL, CHI2_ALT, SEP_MM, EXT_MAX, R_MM = 6, float(os.environ.get("CHI2_ALT", 1.0)), 5.0, 24, 5.0
CREDIT = ('detected', 'multi:m3_picked', 'multi:other_picked')
DET3_MULTI = ('multi:m3_picked', 'multi:other_picked', 'multi:m3_not_found', 'far:MULTI', 'spark:shower',
              'spark:muon+other')


def fit_line(z, x):
    A = np.c_[z, np.ones_like(z)]
    (s, b), *_ = np.linalg.lstsq(A, x, rcond=None)
    return s, b, float(((x - (s * z + b)) ** 2).sum())


def m3_tags(root, run, sub, zdet):
    """Per event: M3 second straight line per view (count, max separation at det3), extra clusters."""
    import uproot, awkward as ak
    rows = []
    for fa in sorted(glob.glob(os.path.join(root, 'x17rerun', 'm3clus', f'{run}__{sub}__*_analyse.root'))):
        an = uproot.open(fa)['T'].arrays(['evn', 'MGv2_NClus', 'MGv2_ClusPos', 'MGv2_ClusAmpl'], library='np')
        r = uproot.open(fa.replace('_analyse', '_rays'))['T'].arrays(
            ['evn'] + [f'm{c}{l}' for c in 'XY' for l in range(4)], library='ak')
        ry = pd.DataFrame({'evn': ak.to_numpy(r.evn), **{k: ak.to_numpy(ak.fill_none(ak.firsts(r[k]), np.nan))
                                                          for k in r.fields[1:]}}).set_index('evn')
        ie = {int(e): i for i, e in enumerate(an['evn'])}
        # strip -> mm per plane (a, b, c), from single-cluster planes against the tracker's measured positions
        C = {}
        for p, (v, l) in PLANES.items():
            o = 'Y' if v == 'x' else 'X'
            m, mo = ry[f'm{v.upper()}{l}'].to_numpy(), ry[f'm{o}{l}'].to_numpy()
            ok = np.isfinite(m) & np.isfinite(mo)
            idx = np.array([ie.get(int(e), -1) for e in ry.index])
            ok &= idx >= 0
            ok[ok] &= an['MGv2_NClus'][idx[ok], p] == 1
            A = np.c_[an['MGv2_ClusPos'][idx[ok], p, 0], np.ones(ok.sum()), mo[ok]]
            C[p], *_ = np.linalg.lstsq(A, m[ok], rcond=None)
        for i, e in enumerate(an['evn']):
            e = int(e); ncl = an['MGv2_NClus'][i]
            out = {'event_id': e, 'm3_extra': int(sum(max(int(n) - 1, 0) for n in ncl))}
            if e not in ry.index:
                rows.append(out); continue
            rr = ry.loc[e]
            ch = {v: np.array([rr[f'm{v.upper()}{l}'] for l in range(4)], float) for v in 'xy'}
            for v in 'xy':
                o = 'y' if v == 'x' else 'x'
                cand = [[] for _ in range(4)]
                for p, (vv, l) in PLANES.items():
                    if vv != v:
                        continue
                    oth = ch[o][l] if np.isfinite(ch[o][l]) else 0.0
                    n = min(int(ncl[p]), 300)
                    k = np.argsort(-an['MGv2_ClusAmpl'][i, p, :n])[:MAXCL]
                    cand[l] = list(C[p][0] * an['MGv2_ClusPos'][i, p, k] + C[p][1] + C[p][2] * oth)
                nalt, sep = 0, 0.0
                if np.all(np.isfinite(ch[v])) and all(len(c) for c in cand):
                    s0, b0, _ = fit_line(ZL, ch[v]); x0 = s0 * zdet[v] + b0
                    for combo in itertools.product(*cand):
                        xx = np.array(combo)
                        if np.allclose(xx, ch[v], atol=0.05):
                            continue
                        s, b, c2 = fit_line(ZL, xx)
                        d = abs(s * zdet[v] + b - x0)
                        if c2 < CHI2_ALT and d >= SEP_MM:
                            nalt += 1; sep = max(sep, d)
                out[f'm3_nalt_{v}'] = nalt; out[f'm3_altsep_{v}'] = sep
            rows.append(out)
    if not rows:                                       # no M3 cluster trees for this row (m3clus not run)
        return pd.DataFrame(columns=['m3_extra', 'm3_nalt_x', 'm3_nalt_y', 'm3_altsep_x', 'm3_altsep_y'],
                            index=pd.Index([], name='event_id'), dtype=float)
    return pd.DataFrame(rows).set_index('event_id')


GAP_MM, ATTACH_MM, LQ_NSIG = 28.0, 3.0, 5.0


def seg_dist(a, b):
    """Smallest distance between two straight segments x = p0 + tan*z, z in [0, GAP] (one view)."""
    d0 = a.p0 - b.p0; d1 = (a.p0 + a.tan_theta * GAP_MM) - (b.p0 + b.tan_theta * GAP_MM)
    return 0.0 if d0 * d1 <= 0 else min(abs(d0), abs(d1))


def second_particle(root, rel, B, events):
    """For det3-tagged multi events: is the extra candidate a real second particle (both views, charge-paired)
    and is it attached to the M3 track inside the gas (a delta ray from the muon)?"""
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
    C = C[(C['rank'] >= 0) & C.event_id.isin(B.index)].copy()
    C['mesh'] = np.where(C.plane == 'x', B.mesh_x.reindex(C.event_id).to_numpy(), B.mesh_y.reindex(C.event_id).to_numpy())
    C['d'] = (C.p0 - C.mesh).abs()
    n = C.groupby(['event_id', 'plane']).size().unstack(fill_value=0)
    one = n.index[(n.x == 1) & (n.y == 1)]
    S = C[C.event_id.isin(one)].pivot(index='event_id', columns='plane')
    S = S[(S['d']['x'] < R_MM) & (S['d']['y'] < R_MM)]
    lq1 = np.log(S['q_sum']['x'].clip(lower=1) / S['q_sum']['y'].clip(lower=1))
    med = float(lq1.median()); q = np.percentile(lq1, [16, 84]); rs = 0.5 * (q[1] - q[0])
    rows = []
    for e in events:
        g = C[C.event_id == e]; gx = g[g.plane == 'x'].sort_values('d'); gy = g[g.plane == 'y'].sort_values('d')
        r = dict(event_id=e, n_x=len(gx), n_y=len(gy), second_3d=False, second_paired=False, attached=False,
                 one_view_only=False)
        if len(gx) >= 2 and len(gy) >= 2:
            r['second_3d'] = True
            mx, my = gx.iloc[0], gy.iloc[0]
            best = min(((abs(np.log(max(u.q_sum, 1) / max(v.q_sum, 1)) - med) / rs, u, v)
                        for _, u in gx.iloc[1:].iterrows() for _, v in gy.iloc[1:].iterrows()), key=lambda t: t[0])
            r['second_paired'] = bool(best[0] < LQ_NSIG)
            r['attached'] = bool(seg_dist(mx, best[1]) < ATTACH_MM and seg_dist(my, best[2]) < ATTACH_MM)
        elif (len(gx) >= 2) != (len(gy) >= 2):
            r['one_view_only'] = True
        rows.append(r)
    return pd.DataFrame(rows).set_index('event_id') if rows else pd.DataFrame(), dict(lq_med=med, lq_rsig=rs)


def eff(mask_det, mask_sel):
    n = int(mask_sel.sum()); k = int((mask_det & mask_sel).sum())
    p = k / n if n else np.nan
    return dict(n=n, k=k, eff=100 * p, err=100 * np.sqrt(p * (1 - p) / n) if n else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {}
    for key, (run, sub, ch) in ROWS.items():
        rel = f'{run}/{sub}/{ch}'
        a1 = os.path.join(a.root, 'x17rerun/paper1_eff/a1', rel)
        R = pd.read_parquet(os.path.join(a1, 'recat.parquet'))
        R = R[R.good_x]
        B = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet'))
        B = B.drop_duplicates('event_id').set_index('event_id').reindex(R.index)
        al = json.load(open(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/alignment/alignment.json')))
        M = m3_tags(a.root, run, sub, {'x': al['z_x'], 'y': al['z_y']}).reindex(R.index)
        D = R.join(M)
        D['credit'] = D.recat.isin(CREDIT)
        D['strict'] = D.recat.eq('detected')
        D['det3_multi'] = D.recat.isin(DET3_MULTI)
        D['m3_2nd_x'] = D.m3_nalt_x.fillna(0) > 0
        D['m3_2nd_y'] = D.m3_nalt_y.fillna(0) > 0
        D['m3_2nd'] = D.m3_2nd_x & D.m3_2nd_y          # a second particle seen in both M3 views
        D['m3_2nd_any'] = D.m3_2nd_x | D.m3_2nd_y
        D['m3_clean'] = ~D.m3_2nd_any
        drop = D.recat.isin(['far:POS_TAIL', 'far:EDGE'])  # reference / geometry, out of the denominator
        D['m3_tags'] = D.m3_extra.notna()
        P, pref = second_particle(a.root, rel, B, list(D.index[D.det3_multi]))
        for c in ('second_3d', 'second_paired', 'attached', 'one_view_only'):
            D[c] = P[c].reindex(D.index).astype('boolean').fillna(False).astype(bool) if len(P) else False
        # a real, independent second particle: both views, charge-paired, not a delta attached to the muon in the gas
        D['indep_2nd'] = D.second_3d & D.second_paired & ~D.attached
        sel = {
            'all': np.ones(len(D), bool),
            'all, ref/geom dropped': ~drop,
            'det3-tagged single (biased)': ~D.det3_multi & ~drop,
            'det3-tagged multi': D.det3_multi & ~drop,
            'M3 clean (external single)': D.m3_tags & D.m3_clean & ~drop,
            'M3 clean, m3_extra == 0 (strictest)': D.m3_tags & D.m3_clean & (D.m3_extra == 0) & ~drop,
            'M3 second line, both views': D.m3_2nd & ~drop,
            'M3 second line, one view only': (D.m3_2nd_any & ~D.m3_2nd) & ~drop,
            'SKEPTICAL single: all but an independent 2nd particle': ~D.indep_2nd & ~drop,
            'independent 2nd particle (both views, paired, detached)': D.indep_2nd & ~drop,
            'det3 multi, 2nd particle attached to the muon (delta)': D.det3_multi & D.attached & ~drop,
            'det3 multi, 2nd candidate in one view only': D.det3_multi & D.one_view_only & ~drop,
            'det3 multi, no candidates (spark-vetoed: shower / muon+other)': D.recat.isin(['spark:shower', 'spark:muon+other']) & ~drop,
        }
        out = {'n_interior_good': int(len(D)), 'pairing_ref': pref, 'selections': {}}
        for nm, m in sel.items():
            m = np.asarray(m, bool)
            out['selections'][nm] = dict(credit=eff(D.credit.to_numpy(), m), strict=eff(D.strict.to_numpy(), m),
                                         classes={k: int(v) for k, v in D[m].recat.value_counts().items()})
        # cross-tab: how do the two tags agree?
        out['crosstab'] = {f'det3_multi={i}, m3_2nd_any={j}': int(((D.det3_multi == i) & (D.m3_2nd_any == j)).sum())
                           for i in (False, True) for j in (False, True)}
        res[key] = out
        D.to_parquet(os.path.join(a.out, f'single_track_events_{key}.parquet'))
        print(f'== {key}  n={len(D)}')
        for nm, s in out['selections'].items():
            c, st = s['credit'], s['strict']
            print(f"  {nm:42s} n={c['n']:5d}  credit {c['eff']:6.2f} ± {c['err']:.2f}   strict {st['eff']:6.2f}")
        print('  ', out['crosstab'])
    json.dump(res, open(os.path.join(a.out, 'single_track.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

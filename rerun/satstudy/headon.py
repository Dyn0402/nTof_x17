#!/usr/bin/env python3
"""headon.py -- det3 "fit failed" = head-on x tracks, and an adaptive seeding rule that recovers them without fakes
(Dylan 10-26: "can we work on this with a more relaxed or adaptive algorithm to check for head-ons? ... lots of
head-ons we need to recover [at n_TOF]").

    WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage \\
      reco_variant.py --events ids_minstrips.csv --out reco_ho20/events.parquet --headon-sig 20      (+ --min-strips 2/3)
      reco_variant.py --events ids_outside.csv   --out reco_out_{min2,min3,ho20}/events.parquet ...
    ../../.venv/bin/python headon.py          -> ~/x17/cosmic_bench/satstudy/headon.json

Fake test: ids_outside.csv = 742 interior-trigger events whose M3 ray passes > 20 mm outside det3's active area (742 of
the 1 339 'outside' rays; 800 asked, seed 1). Nothing from the triggering muon reaches det3, so a fit that a seeding
rule adds there, and production (MIN_STRIPS = 3) does not have, is noise or an unrelated deposit.
Features of a rule's added cluster (from its full-plane waveforms, strips within max(4, |15 tan| + 3) mm of mid-gap):
the per-strip peak SNR (peak ADC / pedestal-run noise), sorted; the time extent as multitrack.py.
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
import summarize as Z  # noqa: E402

OUT = S.OUT


def _wf():
    import make_deck as M
    return M


def feats(C, keys):
    M = _wf(); out = []
    for e, v in keys:
        c = C[(C.event_id == e) & (C.plane == v) & (C['rank'] == 0)].iloc[0]
        ev = M.wf(e); W = ev[v].astype(float); n = ev[f'noise_{v}'].astype(float); p = M.posmap(v)
        sel = np.isfinite(p) & (np.abs(p - (c.p0 + 15 * c.tan_theta)) < max(4, abs(15 * c.tan_theta) + 3))
        prof = W[sel].sum(0); snr = np.sort(W[sel].max(1) / np.maximum(n[sel], 1))[::-1]
        out.append(dict(event_id=int(e), plane=v, p0=float(c.p0), ext=int((prof > 0.2 * prof.max()).sum()),
                        snr1=float(snr[0]), snr2=float(snr[1]), snr3=float(snr[2]), n10=int((snr > 10).sum())))
    return out


def added(A, B):
    """(event, plane) fitted under rule A and not under B."""
    ka = set(map(tuple, A[['event_id', 'plane']].drop_duplicates().to_numpy().tolist()))
    kb = set(map(tuple, B[['event_id', 'plane']].drop_duplicates().to_numpy().tolist()))
    return sorted(ka - kb)


def main():
    R = Z.rays(); res = {}
    ids = pickle.load(open(os.path.join(OUT, 'wf_examples.pkl'), 'rb'))['picks']['fit_failed']
    # where the failures sit in angle
    Ra = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet'))
    Ra = Ra[(Ra.region == 'interior') & ~Ra.not_measured]
    edges = [0, .01, .02, .04, .08, .2, 1.0]
    b = pd.cut(Ra.tan_x.abs(), edges)
    res['by_tanx'] = [dict(lo=float(k.left), hi=float(k.right), n=int(len(g)), detected=float((g.cat_xy_5 == 'detected').mean()),
                           fit_failed=float((g.cat_xy_5 == 'fit_failed').mean()))
                      for k, g in Ra.groupby(b, observed=True)]
    ff = Ra[Ra.cat_xy_5 == 'fit_failed']
    res['frac_tanx_lt_003'] = dict(failed=float((ff.tan_x.abs() < .03).mean()),
                                   detected=float((Ra[Ra.cat_xy_5 == 'detected'].tan_x.abs() < .03).mean()))
    # recovery + controls per rule
    base = pd.read_parquet(os.path.join(OUT, 'reco_min3', 'events.parquet')); DB = Z.detected(base, R)
    for rule in ('min2', 'ho20'):
        p = os.path.join(OUT, f'reco_{rule}', 'events.parquet')
        if not os.path.exists(p):
            continue
        A = pd.read_parquet(p); DA = Z.detected(A, R)
        f = [i for i in ids if i in DA.index]
        ctl = sorted(set(DA.index) & set(DB.index) - set(ids))
        a = A.drop_duplicates('event_id').set_index('event_id').loc[ctl]
        bb = base.drop_duplicates('event_id').set_index('event_id').loc[ctl]
        res[rule] = dict(n=len(ids), detected=int(DA.reindex(f).detected.fillna(False).sum()),
                         controls=dict(n=len(ctl), det=int(DA.loc[ctl].detected.sum()), det_base=int(DB.loc[ctl].detected.sum()),
                                       views_moved=sum(int(((a[f'{v}_p0'] - bb[f'{v}_p0']).abs() > 0.05).sum()) for v in 'xy')))
    # fakes on the outside rays
    O = {k: os.path.join(OUT, f'reco_out_{k}', 'events.candidates.parquet') for k in ('min2', 'min3', 'ho20')}
    if all(os.path.exists(p) for p in O.values()):
        C = {k: pd.read_parquet(p) for k, p in O.items()}
        E = {k: pd.read_parquet(p.replace('.candidates', '')) for k, p in O.items()}
        n_out = len(pd.read_csv(os.path.join(OUT, 'ids_outside.csv')))
        res['outside'] = dict(n=n_out, **{k: dict(events=int(len(E[k])), both=int((E[k].x_ok & E[k].y_ok).sum()),
                                                  added_views=len(added(C[k], C['min3'])))
                                          for k in ('min2', 'ho20', 'min3')})
        res['outside']['fake_feats'] = feats(C['min2'], added(C['min2'], C['min3']))
        A = pd.read_parquet(os.path.join(OUT, 'reco_min2', 'events.candidates.parquet'))
        res['headon_feats'] = feats(A, [(e, 'x') for e in ids])
    json.dump(res, open(os.path.join(OUT, 'headon.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('feats')}, indent=1, default=float))


if __name__ == '__main__':
    main()

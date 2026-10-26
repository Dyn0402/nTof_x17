#!/usr/bin/env python3
"""choose.py -- a reference-free rule to pick the triggering particle among det3's particle candidates (question 2).

    ../../.venv/bin/python choose.py       -> multiparticle/choose.json

Sample: every (event, view) of the multi-particle events where the M3 leg is among >= 2 particle candidates
(legs.py). Truth = the M3 leg. A rule sees only det3 (and the fixed telescope geometry), never the M3 hits.

Features per candidate (all from the forward fit and the candidate's own waveforms):
  lq       log charge per unit path in the view, sqrt(1 + tan^2), relative to the clean-single median
  atan     |ladder tan|                        (muons that fire the telescope are steep-limited; electrons are not)
  dt0      |t0 - median t0 of clean singles|  (prompt)
  ext      time extent of the summed signal [samples]
  uend     depth at which the charge ends (q_uend) -- a through-going column reaches the cathode
  chi2     chi2 / dof of the fit
  nstr     strips in the fit
  acc      the candidate's projection, extrapolated to the four M3 planes, stays inside the region recipe rays cross
           (telescope geometry only: a triggering particle must cross all four planes)
  rank     production's rank (what production picks today)
Single-feature rules are evaluated on the whole sample (they have no free parameter). The combined rule is a
conditional logit (softmax over the candidates of one view, L2-regularised), trained on even event ids and tested on
odd ones and vice versa (2-fold by event, both folds reported and pooled).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402

FEATS = ['lq', 'atan', 'dt0', 'ext', 'uend', 'chi2', 'nstr', 'acc']


def table():
    Lg = pd.read_parquet(os.path.join(M.OUT, 'legs.parquet')); ref = json.load(open(os.path.join(M.OUT, 'singles_ref.json')))
    G = Lg[Lg.m3_found & (Lg.n_in_view >= 2)].copy()
    G['lq'] = np.log(np.where(G.plane == 'x', G.q / ref['qx_med'], G.q / ref['qy_med']) / np.sqrt(1 + G.tl ** 2))
    G['atan'] = G.tl.abs(); G['dt0'] = (G.t0 - ref['t0_med']).abs(); G['uend'] = G.q_uend
    G['chi2'] = G.chi2dof; G['nstr'] = G.n_strips
    # telescope acceptance in the candidate's own projection. theta ~ 90 deg: raw x <-> M3 y, raw y <-> M3 x.
    a = M.align(); boxes = M.m3_boxes(M.rays_all(), a); R = M.rays()
    acc = np.ones(len(G), bool)
    for z in M.ZM3:
        bx, by = boxes[z]
        # put the candidate in the M3 frame with the other coordinate taken from the M3-free det3 centre (200 mm)
        px = np.where(G.plane == 'x', G.p0 + G.tl * (z - M.Z0), 200.0)
        py = np.where(G.plane == 'y', G.p0 + G.tl * (z - M.Z0), 200.0)
        X, Y = M.raw_to_m3(px, py, a)
        acc &= np.where(G.plane == 'x', (Y >= by[0]) & (Y <= by[1]), (X >= bx[0]) & (X <= bx[1]))
    G['acc'] = acc.astype(float)
    G['gid'] = G.event_id.astype(str) + G.plane
    return G


def pick(G, score):
    """For each group, True if the highest-score candidate is the M3 leg (ties broken by production rank)."""
    G = G.assign(_s=score, _r=-G['rank'])
    top = G.sort_values(['gid', '_s', '_r'], ascending=[True, False, False]).groupby('gid').head(1)
    return top.set_index('gid').is_m3


def fit_logit(G, feats, lam=1.0):
    X = G[feats].to_numpy(float); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    gi = pd.factorize(G.gid)[0]; y = G.is_m3.to_numpy()
    def nll(w):
        s = Z @ w; mx = pd.Series(s).groupby(gi).transform('max').to_numpy(); ex = np.exp(s - mx)
        lse = np.log(pd.Series(ex).groupby(gi).transform('sum').to_numpy()) + mx
        return -(s - lse)[y].sum() + lam * (w ** 2).sum()
    w = minimize(nll, np.zeros(len(feats)), method='L-BFGS-B').x
    return lambda H: ((H[feats].to_numpy(float) - mu) / sd) @ w, dict(zip(feats, w.round(3).tolist()))


def ci(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(0.16, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.84, k + 1, n - k)) if k < n else 1.0]


def main():
    G = table(); n = G.gid.nunique()
    out = dict(n_groups=int(n), n_cands=int(len(G)), chance=float((1 / G.groupby('gid').size()).mean()), rules={})
    single = dict(production_rank=-G['rank'], largest_q=G.lq, smallest_q=-G.lq, most_vertical=-G.atan,
                  most_prompt=-G.dt0, longest_column=G.uend, best_chi2=-G.chi2, most_strips=G.nstr,
                  in_acceptance_then_rank=G.acc * 10 - G['rank'], in_acceptance_then_vertical=G.acc * 10 - G.atan)
    for k, sc in single.items():
        r = pick(G, sc.to_numpy()); out['rules'][k] = dict(acc=float(r.mean()), k=int(r.sum()), ci68=ci(int(r.sum()), len(r)))
    ev = G.event_id.to_numpy() % 2 == 0
    held = {}
    for nm, feats in (('logit_all', FEATS), ('logit_5', ['atan', 'dt0', 'ext', 'chi2', 'acc']),
                      ('logit_q_tan_acc', ['lq', 'atan', 'acc']), ('logit_noacc', ['lq', 'atan', 'dt0', 'ext', 'chi2']), ('logit_tan_acc', ['atan', 'acc'])):
        res, ws = [], {}
        for fold, (tr, te) in enumerate(((ev, ~ev), (~ev, ev))):
            f, w = fit_logit(G[tr], feats); r = pick(G[te], f(G[te])); res.append(r); ws[f'fold{fold}'] = w
            out['rules'].setdefault(nm, {})[f'fold{fold}'] = float(r.mean())
        r = pd.concat(res); held[nm] = r
        out['rules'][nm].update(acc=float(r.mean()), k=int(r.sum()), ci68=ci(int(r.sum()), len(r)), weights=ws)
        f, _ = fit_logit(G, feats); out['rules'][nm]['train_on_all_acc'] = float(pick(G, f(G)).mean())
    # paired comparison with the best parameter-free rule (McNemar, held-out picks)
    base = pick(G, single['in_acceptance_then_rank'].to_numpy()); prod = pick(G, single['production_rank'].to_numpy())
    from scipy.stats import binomtest
    for nm, r in held.items():
        for bn, b in (('in_acceptance_then_rank', base), ('production_rank', prod)):
            b = b.reindex(r.index); w, l = int((r & ~b).sum()), int((~r & b).sum())
            out['rules'][nm][f'vs_{bn}'] = dict(wins=w, losses=l, p=float(binomtest(w, w + l).pvalue) if w + l else 1.0)
    # event level over all 187 multi-particle events: every view must end on the M3 leg
    Lg = pd.read_parquet(os.path.join(M.OUT, 'legs.parquet'))
    vw = Lg.groupby(['event_id', 'plane']).agg(found=('m3_found', 'first'), n=('n_in_view', 'first'))
    rules_ev = dict(production_rank=prod, in_acceptance_then_rank=base, most_vertical=pick(G, single['most_vertical'].to_numpy()), logit_noacc=held['logit_noacc'], logit_5=held['logit_5'], logit_all=held['logit_all'])
    out['event_level'] = {}
    nev = Lg.event_id.nunique()
    for nm, r in rules_ev.items():
        ok = []
        for (e, v), row in vw.iterrows():
            if not row.found:
                ok.append((e, False))
            elif row.n < 2:
                ok.append((e, True))
            else:
                ok.append((e, bool(r.get(f'{e}{v}', False))))
        o = pd.DataFrame(ok, columns=['e', 'ok']).groupby('e').ok.all()
        # a view with no particle candidate at all is absent from legs: treat as not found
        o = o & (Lg.groupby('event_id').plane.nunique() == 2)
        out['event_level'][nm] = dict(n_events=int(nev), both_views_right=int(o.sum()), frac=float(o.sum() / nev),
                                      of_interior=float(o.sum() / 6451))
    # how the M3 leg differs from the others (medians)
    out['medians'] = {f: dict(m3=float(G[G.is_m3][f].median()), other=float(G[~G.is_m3][f].median())) for f in FEATS}
    json.dump(out, open(os.path.join(M.OUT, 'choose.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

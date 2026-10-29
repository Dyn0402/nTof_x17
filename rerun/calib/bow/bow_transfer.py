"""Bow correction across datasets of the same chamber (Dylan 10-29: a correction must be fixed per detector and hold
across that detector's datasets, not fit one run; det3 first).

For every chamber and every one of its datasets as the TARGET, the drift-tilt surface A(U, W) (FB, set_bow.py) comes
from somewhere else and only the per-run rigid tilt c0 and slope B (= kw v) are refit on the target, 2-fold by event
parity (fit on one half, score the other, swap).  Arms:
  const     no bow: w = c0 + B t                                           (today's per-run w0/kw only)
  gold_um   golden row's A in um/ns, as the bundle block stores it (FB static_plus_tilt)
  gold_tan  golden row's A converted to a geometric tan offset a = A / B_gold, applied as w = c0 + B (t + a(U, W)):
            the tilt is v * r_f * dh/du, so in tan units it does not depend on the drift field / v of the run
  loo_tan   a(U, W) pooled (n-weighted) over all OTHER datasets of the chamber, golden included; never the target
  loo_amp   loo_tan's shape with one per-run amplitude (like kw) refit on the target
  own       the target's own full 2-D map, 2-fold (the ceiling)
Inputs: FB's per-event tables (fbcommon.DATASETS; finalize-era reco, promoted bundles).  Rows with broken angles
(det3_o23 ladder 4.4 um/ns, det2_o17 half-dead X) are excluded as in FB.  Paired bootstrap errors vs const.
Usage: python -I bow_transfer.py  -> results/bow_transfer.json + a markdown table on stdout
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.abspath(os.path.join(HERE, '../../model_audit/followups/fb_bow_maps'))
sys.path.insert(0, FB)
from fbcommon import DATASETS, DETS, load, huber, surf_terms          # noqa: E402
from gate_bow import prep, fit_bow2d, tan_bow2d                       # noqa: E402

EXCLUDE = {'det3_o23', 'det2_o17'}


def s68(d):
    d = d[np.isfinite(d)]
    return float(0.5 * np.diff(np.percentile(d, [16, 84]))[0])


def deg(res, t):
    return np.degrees(np.arctan(t + res)) - np.degrees(np.arctan(t))


def shape_tan(D):
    """golden-style 2-D fit on all of D; returns (A coefficients without constant [um/ns], B, a = A / B in tan)."""
    b = fit_bow2d(D, D['m'])
    return b[1:6], b[6], b[1:6] / b[6]


def apply_um(D, k, tr, A):
    a = surf_terms(D['U'][k], D['W'][k])[:, 1:] @ A
    t, wv = D['t'][k], D['wv'][k]
    b = huber(np.c_[np.ones(tr.sum()), t[tr]], wv[tr] - a[tr], 0.03 * D['v'])
    return (wv[~tr] - a[~tr] - b[0]) / b[1] - t[~tr]


def apply_tan(D, k, tr, atan):
    # w = c0 + B (t + a)  ->  linear in (c0, B) with regressor (t + a); tan = (w - c0) / B - a
    a = surf_terms(D['U'][k], D['W'][k])[:, 1:] @ atan
    t, wv = D['t'][k], D['wv'][k]
    b = huber(np.c_[np.ones(tr.sum()), (t + a)[tr]], wv[tr], 0.03 * D['v'])
    return (wv[~tr] - b[0]) / b[1] - a[~tr] - t[~tr]


def apply_tan_amp(D, k, tr, atan):
    # per-run amplitude of the fixed shape: w = c0 + B t + C a  (C = lambda B), tan = (w - c0 - C a) / B
    a = surf_terms(D['U'][k], D['W'][k])[:, 1:] @ atan
    t, wv = D['t'][k], D['wv'][k]
    b = huber(np.c_[np.ones(tr.sum()), t[tr], a[tr]], wv[tr], 0.03 * D['v'])
    return (wv[~tr] - b[0] - b[2] * a[~tr]) / b[1] - t[~tr]


def apply_const(D, k, tr):
    return apply_um(D, k, tr, np.zeros(5))


def apply_own(D, k, tr):
    kk = np.where(k)[0]
    ktr = np.zeros_like(k); ktr[kk[tr]] = True
    kte = np.zeros_like(k); kte[kk[~tr]] = True
    return tan_bow2d(D, fit_bow2d(D, ktr), kte) - D['t'][kte]


def main():
    by_det = {d: [n for n, v in DATASETS.items() if v[0] == d and n not in EXCLUDE] for d in DETS}
    cache = {n: {p: prep(load(n), p) for p in 'xy'} for ns in by_det.values() for n in ns}
    shapes = {n: {p: shape_tan(cache[n][p]) for p in 'xy'} for n in cache}
    rng = np.random.default_rng(7); out = {}
    for det, names in by_det.items():
        for tgt in names:
            for p in 'xy':
                D = cache[tgt][p]; k = D['m']
                par = (D['eid'] % 2 == 0)[k]
                others = [n for n in names if n != tgt]
                Agold, Bgold, agold = shapes[det][p]
                loo = (np.average([shapes[n][p][2] for n in others], axis=0, weights=[cache[n][p]['m'].sum() for n in others])
                       if others else None)
                arms = {'const': lambda tr: apply_const(D, k, tr), 'gold_um': lambda tr: apply_um(D, k, tr, Agold),
                        'gold_tan': lambda tr: apply_tan(D, k, tr, agold), 'own': lambda tr: apply_own(D, k, tr)}
                if loo is not None:
                    arms['loo_tan'] = lambda tr: apply_tan(D, k, tr, loo)
                    arms['loo_amp'] = lambda tr: apply_tan_amp(D, k, tr, loo)
                t = np.concatenate([D['t'][k][~par], D['t'][k][par]])
                R = {a: deg(np.concatenate([f(par), f(~par)]), t) for a, f in arms.items()}
                base = R['const']; row = dict(n=int(k.sum()), golden=(tgt == det), note=DATASETS[tgt][3],
                                              B_ratio=float(huber(np.c_[np.ones(k.sum()), D['t'][k]], D['wv'][k], 0.03 * D['v'])[1] / Bgold))
                for a, d in R.items():
                    bs = [s68(d[j]) - s68(base[j]) for j in (rng.integers(0, len(d), len(d)) for _ in range(200))]
                    row[a] = dict(s68=s68(d), d=float(s68(d) - s68(base)), err=float(np.std(bs)))
                out[f'{tgt}_{p}'] = row
    json.dump(out, open(f'{HERE}/results/bow_transfer.json', 'w'), indent=1)
    print('| target dataset | view | n | B/B_gold | const s68 | gold_um Δ | gold_tan Δ | loo_tan Δ | loo_amp Δ | own Δ |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for key, r in out.items():
        tgt, p = key.rsplit('_', 1)
        cell = lambda a: '—' if a not in r else f"{r[a]['d']:+.3f} ± {r[a]['err']:.3f}"
        print(f"| {tgt}{' (golden)' if r['golden'] else ''} — {r['note']} | {p.upper()} | {r['n']} | {r['B_ratio']:.2f} | "
              f"{r['const']['s68']:.3f} | {cell('gold_um')} | {cell('gold_tan')} | {cell('loo_tan')} | {cell('loo_amp')} | {cell('own')} |")


if __name__ == '__main__':
    main()

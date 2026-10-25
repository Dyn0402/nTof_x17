#!/usr/bin/env python3
"""nearest.py -- the test-beam convention on the finalize candidate lists: take the plausible candidate whose position at
mid-gap lies nearest the reference, within a WIDE window (|Δ| < WIN mm, wide against the 0.5-0.6 mm resolution, so the
choice cannot shape the core), instead of the production rank-0 winner. Per chamber and view, interior region
(fiducial.py), all fitted events (no n_dropped cut: the choice replaces the competing-cluster cut):
  production   rank 0, accepted (n_dropped <= 2)   -- the current number
  nearest      nearest plausible candidate within WIN, every fitted event
s68 / far share of the angle, |res| > 3 mm share, and how many event-views each keeps.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from olib import BASE, DETS, load  # noqa: E402
from fiducial import regions  # noqa: E402
from m3feat import ALIGN  # noqa: E402

WIN = 10.0


def stats(dl, rr):
    far = np.abs(dl) > 5
    return dict(n=int(len(dl)), s68=float(np.percentile(np.abs(dl[~far]), 68)), far=float(100 * far.mean()),
                pos_far=float(100 * np.mean(np.abs(rr) > 3)))


def main():
    out = {}
    for det in DETS:
        d = load(det, quality=False); lad = d.attrs['ladder']
        C = pd.read_parquet(os.path.join(BASE, 'final', ALIGN[det], 'wft/final_phys/events.candidates.parquet'))
        C = C[C.plausible | (C['rank'] == 0)]
        for v in 'xy':
            reg, inward, dist, hw = regions(d, det, v)
            keep = (reg == 'interior') & ~hw & np.isfinite(d[f'ref_tan_{v}'].to_numpy())
            D = d[keep].set_index('event_id')
            s, c = lad[v]
            prod = D[D[f'{v}_ok'] & (D[f'{v}_n_dropped'] <= 2)]
            out[f'{det}|{v}|production'] = stats(prod[f'dl_{v}'].to_numpy(), prod[f'rres_{v}'].to_numpy())
            g = C[(C.plane == v) & C.event_id.isin(D.index)].copy()
            ref = D.loc[g.event_id]
            tl = (g.w.to_numpy() * 1e3 - c) / s
            g['dmid'] = np.abs(g.p0.to_numpy() + 15 * tl - (ref[f'ref_mesh_{v}'].to_numpy() + 15 * ref[f'ref_tan_{v}'].to_numpy()))
            g['dl'] = np.degrees(np.arctan(tl) - np.arctan(ref[f'ref_tan_{v}'].to_numpy()))
            g['rr'] = g.p0.to_numpy() - ref[f'ref_mesh_{v}'].to_numpy()
            n = g[g.dmid < WIN].sort_values('dmid').drop_duplicates('event_id')
            out[f'{det}|{v}|nearest'] = stats(n.dl.to_numpy(), n.rr.to_numpy())
            out[f'{det}|{v}|nearest']['n_changed'] = int((n['rank'] > 0).sum())
            # production events with no plausible candidate inside the window: a match failure, counted as far
            nomatch = int((~prod.index.isin(n.event_id)).sum())
            q = out[f'{det}|{v}|nearest']; q['n_nomatch_of_production'] = nomatch
            q['far_incl_nomatch'] = float(100 * (q['far'] / 100 * q['n'] + nomatch) / (q['n'] + nomatch))
            p, q = out[f'{det}|{v}|production'], out[f'{det}|{v}|nearest']
            print(f"{det} {v} interior: production n {p['n']} s68 {p['s68']:.3f} far {p['far']:.2f} pos>3 {p['pos_far']:.2f} | "
                  f"nearest<{WIN:.0f}mm n {q['n']} (+{q['n'] - p['n']}, {q['n_changed']} not rank 0) s68 {q['s68']:.3f} far {q['far']:.2f} (incl. {q['n_nomatch_of_production']} no-match: {q['far_incl_nomatch']:.2f}) pos>3 {q['pos_far']:.2f}")
    json.dump(out, open(os.path.join(BASE, 'report', 'nearest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

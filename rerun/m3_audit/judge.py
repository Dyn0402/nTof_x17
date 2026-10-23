#!/usr/bin/env python3
"""judge.py -- judge M3 tracking variants and hit common-mode versions against the detector under test.

For one golden subrun and a few of its files: every clean M3 ray (production recipe: NClus >= 3 in both coordinates,
0 < chi2 < 5, one clean ray in the event) is projected to the detector plane and compared with the detector's own
cluster in each view (largest-charge contiguous cluster of the combined hits, amplitude-weighted centroid in mm from
the strip map).  The detector frame is tied to the M3 frame by an affine fit per view (position on ray x, ray y and
both slopes: absorbs offset, rotation and a z error), fitted on the baseline rays with the plain-CM hits and then
frozen, so every variant is read on the same geometry.

Read, per (rays variant, hits version), inside the detector footprint:
  * match = a detector cluster within R mm in BOTH views (an efficiency proxy, hits are fine for this);
  * residual width per view (1.4826 x MAD of the matched residuals);
and the rays a variant adds (events clean in the variant, not in the baseline) separately: if those are real
tracks they match and resolve like the baseline rays.

    judge.py --run-config <run_config.json> --det mx17_7 --sub <subrun dir> --tags 260626_06H03_000,260626_06H03_005
             --rays base=<dir>,out5=<dir>,... --hits cm0,cm5 --out <json>
"""
import argparse
import glob
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from common.Mx17StripMap import RunConfig  # noqa: E402

R_MATCH = 5.0


def pos_table(det):
    """(feu, channel) -> (view, mm) as lookup arrays."""
    tab = {}
    for feu in det.feu_map:
        xs, ys = np.full(512, np.nan), np.full(512, np.nan)
        for ch in range(512):
            r = det.map_hit(feu, ch)
            if r is None:
                continue
            x, y = r
            if x is not None:
                xs[ch] = x
            if y is not None:
                ys[ch] = y
        tab[feu] = (xs, ys)
    return tab


def det_clusters(paths, tab):
    """eventId -> (x_mm, y_mm), largest-charge contiguous cluster per view (NaN if none)."""
    out = {}
    for p in paths:
        h = uproot.open(p)['hits'].arrays(['eventId', 'feu', 'channel', 'amplitude'], library='np')
        ev, feu, ch, amp = h['eventId'].astype(np.int64), h['feu'].astype(int), h['channel'].astype(int), h['amplitude']
        xs = np.full(len(ev), np.nan); ys = np.full(len(ev), np.nan)
        nev = len(np.unique(ev))
        for f, (tx, ty) in tab.items():
            m = feu == f
            # hot channels (firing in > 20 % of events, e.g. det4's edge strip at y ~ 370 mm) would win the
            # largest-charge choice: drop them
            occ = np.bincount(np.unique(ev[m] * 512 + ch[m]) % 512, minlength=512) / max(nev, 1)
            hot = occ > 0.2
            xs[m] = np.where(hot[ch[m]], np.nan, tx[ch[m]]); ys[m] = np.where(hot[ch[m]], np.nan, ty[ch[m]])
        for view, pos in (('x', xs), ('y', ys)):
            ok = np.isfinite(pos) & (amp > 0)
            e, q, a = ev[ok], pos[ok], amp[ok]
            o = np.lexsort((q, e))
            e, q, a = e[o], q[o], a[o]
            # several hits per strip (multi-peak): fine, the centroid weights them
            brk = np.r_[True, (e[1:] != e[:-1]) | (np.diff(q) > 1.6)]   # new cluster at a gap > ~2 strips (0.78 mm pitch)
            cid = np.cumsum(brk) - 1
            cq = np.bincount(cid, weights=a)
            cx = np.bincount(cid, weights=a * q) / np.maximum(cq, 1e-9)
            ce = e[brk]
            # largest-charge cluster per event
            o2 = np.lexsort((-cq, ce))
            first = np.r_[True, ce[o2][1:] != ce[o2][:-1]]
            for evn, x in zip(ce[o2][first], cx[o2][first]):
                out.setdefault(int(evn), [np.nan, np.nan])['xy'.index(view)] = x
    return out


def clean_rays(paths, z):
    """evn -> (x, y, tx, ty) at z for events with exactly one clean ray; also raw counts."""
    res = {}
    for p in paths:
        t = uproot.open(p)['T'].arrays(['evn', 'NClusX', 'NClusY', 'Chi2X', 'Chi2Y', 'X_Up', 'X_Down', 'Y_Up',
                                        'Y_Down', 'Z_Up', 'Z_Down'])
        c = (t['NClusX'] >= 3) & (t['NClusY'] >= 3) & (t['Chi2X'] > 0) & (t['Chi2Y'] > 0) & (t['Chi2X'] < 5) \
            & (t['Chi2Y'] < 5)
        one = ak.to_numpy(ak.sum(c, axis=1) == 1)
        sel = t[one]
        cs = c[one]
        g = {k: ak.to_numpy(ak.flatten(sel[k][cs])) for k in ('X_Up', 'X_Down', 'Y_Up', 'Y_Down')}
        zu, zd = ak.to_numpy(sel['Z_Up']), ak.to_numpy(sel['Z_Down'])
        zu = zu if zu.ndim == 1 else ak.to_numpy(ak.flatten(sel['Z_Up'][cs]))
        zd = zd if zd.ndim == 1 else ak.to_numpy(ak.flatten(sel['Z_Down'][cs]))
        f = (z - zd) / (zu - zd)
        x = g['X_Down'] + (g['X_Up'] - g['X_Down']) * f
        y = g['Y_Down'] + (g['Y_Up'] - g['Y_Down']) * f
        tx = (g['X_Up'] - g['X_Down']) / (zu - zd)
        ty = (g['Y_Up'] - g['Y_Down']) / (zu - zd)
        for e, a, b, c_, d in zip(ak.to_numpy(sel['evn']), x, y, tx, ty):
            res[int(e)] = (a, b, c_, d)
    return res


def design(r):
    return np.column_stack([np.ones(len(r)), r[:, 0], r[:, 1], r[:, 2], r[:, 3]])


def fit_align(rays, det):
    """Affine map per view, robust (iterative 3-sigma trim)."""
    ev = [e for e in rays if e in det and np.all(np.isfinite(det[e]))]
    R = np.array([rays[e] for e in ev]); D = np.array([det[e] for e in ev])
    coef = []
    for v in range(2):
        keep = np.ones(len(ev), bool)
        for _ in range(6):
            c, *_ = np.linalg.lstsq(design(R[keep]), D[keep, v], rcond=None)
            res = D[:, v] - design(R) @ c
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            keep = np.abs(res) < max(3 * s, 0.5)
        coef.append(c)
    return coef


def score(rays, det, coef, box, events=None):
    ev = [e for e in (rays if events is None else events) if e in rays]
    if not ev:
        return dict(n=0)
    R = np.array([rays[e] for e in ev])
    P = np.column_stack([design(R) @ coef[0], design(R) @ coef[1]])
    inside = (P[:, 0] > box[0]) & (P[:, 0] < box[1]) & (P[:, 1] > box[2]) & (P[:, 1] < box[3])
    D = np.array([det.get(e, [np.nan, np.nan]) for e in ev])
    res = D - P
    match = inside & (np.abs(res[:, 0]) < R_MATCH) & (np.abs(res[:, 1]) < R_MATCH)
    out = dict(n=int(len(ev)), inside=int(inside.sum()), match=int(match.sum()),
               eff=float(match.sum() / max(inside.sum(), 1)))
    for v, lab in ((0, 'x'), (1, 'y')):
        r = res[match, v]
        out[f'sig_{lab}'] = float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) > 20 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-config', required=True); ap.add_argument('--det', required=True)
    ap.add_argument('--sub', required=True); ap.add_argument('--tags', required=True)
    ap.add_argument('--rays', required=True); ap.add_argument('--hits', default='cm0,cm5')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rc = RunConfig(a.run_config, os.path.join(REPO, 'mx17_m1_map.csv'))
    det = rc.get_detector(a.det)
    z = json.load(open(a.run_config))
    z = [d for d in z['detectors'] if d['name'] == a.det][0]['det_center_coords']['z']
    tags = a.tags.split(',')
    tab = pos_table(det)
    hits = {}
    for g in a.hits.split(','):
        paths = [glob.glob(os.path.join(a.sub, f'combined_hits_root_rr1_{g}', f'*_{t}_feu-combined_hits.root'))[0]
                 for t in tags]
        hits[g] = det_clusters(paths, tab)
    variants = dict(kv.split('=') for kv in a.rays.split(','))
    rays = {}
    for v, d in variants.items():
        run, sub = os.path.basename(os.path.dirname(a.sub)), os.path.basename(a.sub)
        paths = []
        for t in tags:
            num = t.split('_')[-1]
            p = os.path.join(d, f'{run}__{sub}__{num}.root')
            if os.path.exists(p):
                paths.append(p)
        rays[v] = clean_rays(paths, z) if len(paths) == len(tags) else None
    base, h0 = rays['base'], hits['cm0']
    coef = fit_align(base, h0)
    pts = np.array([h0[e] for e in base if e in h0 and np.all(np.isfinite(h0[e]))])
    box = (*np.percentile(pts[:, 0], [2, 98]), *np.percentile(pts[:, 1], [2, 98]))
    # footprint in the projected frame: use the detector-cluster range (same frame after the fit)
    out = dict(det=a.det, sub=a.sub, tags=tags, z=z, box=[float(b) for b in box], results={})
    for v, r in rays.items():
        if r is None:
            out['results'][v] = 'missing'
            continue
        for g, h in hits.items():
            res = dict(all=score(r, h, coef, box))
            if v != 'base':
                gained = sorted(set(r) - set(base))
                lost = sorted(set(base) - set(r))
                res['gained'] = score(r, h, coef, box, gained)
                res['lost_in_base'] = score(base, h, coef, box, lost)
            out['results'][f'{v}/{g}'] = res
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"{a.det} {os.path.basename(a.sub)} {tags}")
    for k, r in out['results'].items():
        if isinstance(r, str):
            print(f'  {k:14s} {r}'); continue
        s = r['all']
        line = (f"  {k:14s} clean rays {s['n']:6d} in-footprint {s['inside']:6d} matched {s['match']:6d} "
                f"eff {100*s['eff']:5.1f} %  sigma x/y {s['sig_x'] or 0:.3f}/{s['sig_y'] or 0:.3f} mm")
        if 'gained' in r and r['gained'].get('n'):
            gsc = r['gained']
            line += (f" | gained {gsc['n']} eff {100*gsc['eff']:5.1f} % sig {gsc['sig_x'] or 0:.3f}/"
                     f"{gsc['sig_y'] or 0:.3f}")
        if 'lost_in_base' in r and r['lost_in_base'].get('n'):
            line += f" | lost {r['lost_in_base']['n']} (base eff {100*r['lost_in_base']['eff']:.1f} %)"
        print(line)


if __name__ == '__main__':
    main()

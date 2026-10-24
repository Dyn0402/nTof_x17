#!/usr/bin/env python3
"""ladder_summary.py <dir with <det>[_r06]_<mode>.json> [out.json] -- the reconstruction's drift velocity (ladder:
robust slope of the fitted transverse speed w against the M3 tan_ref) per chamber, plane, kernel and fit mode, and
the paired event-by-event change of w between modes, against the timing v (gap / head-on drift duration)."""
import glob, json, os, sys
import numpy as np

TIMING = dict(det2=(39.3, 1.2), det3=(35.8, 0.6), det4=(33.0, 0.7), det6=(25.9, 0.8), det7=(36.8, 1.1))


def ladder(w, tr):
    m = np.isfinite(w) & (np.abs(w / np.where(tr == 0, np.nan, tr)) < 200) | (np.abs(tr) < 0.05)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1)
        r = w - (s * tr + c)
        m = np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27)
    # bootstrap error
    rng = np.random.default_rng(1)
    idx = np.flatnonzero(m)
    bs = [np.polyfit(tr[j], w[j], 1)[0] for j in (rng.choice(idx, len(idx)) for _ in range(200))]
    return float(s), float(np.std(bs)), float(c), int(m.sum())


def main():
    d = sys.argv[1]
    R = {}
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        tag = os.path.basename(f)[:-5]
        R[tag] = json.load(open(f))['events']
    out = {}
    for det in sorted(TIMING):
        vt, ve = TIMING[det]
        print(f'--- {det}   timing v {vt} ± {ve}')
        base = R.get(f'{det}_prod')
        for tag in sorted(t for t in R if t.startswith(det + '_')):
            ev = R[tag]
            row = {}
            for p in ('x', 'y'):
                ids = [k for k, e in ev.items() if p in e]
                w = np.array([ev[k][p]['w'] for k in ids]); tr = np.array([ev[k][p]['tr'] for k in ids])
                s, se, c, n = ladder(w, tr)
                row[p] = dict(v=s, err=se, icpt=c, n=n)
                if base is not None and tag != f'{det}_prod':
                    com = [k for k in ids if p in base.get(k, {}) and abs(ev[k][p]['tr']) > 0.15]
                    dw = np.array([(ev[k][p]['w'] - base[k][p]['w']) / ev[k][p]['tr'] for k in com])
                    row[p]['paired_dv_vs_prod'] = float(np.median(dw)) if len(dw) else None
            out[tag] = row
            pv = lambda p: (f"{row[p]['v']:5.2f}±{row[p]['err']:.2f} ({100*(row[p]['v']/vt-1):+5.1f}%)"
                            + (f" paired {row[p]['paired_dv_vs_prod']:+5.2f}" if row[p].get('paired_dv_vs_prod') is not None else ''))
            print(f"  {tag[len(det)+1:]:14s} X {pv('x')}   Y {pv('y')}")
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()

"""Clean-track check of the saturation study (Dylan 10-29: "clipping at 1500 of ~4000 ADC costs nothing can't be right
-- is it an artefact of saturation correlating with delta blobs / non-track events?  Charge matters for x-y matching").

Event classes from the RAW waveforms and M3 only (no fit), per view, on the 8000-event det3 cache:
  strips above 5 sigma (peak) -> clusters (gap <= 1 strip); Q = summed positive ADC of the main cluster;
  expected width n_exp = |tan_ref| * 30 mm / 0.78 mm + 2 strips.
  clean  : one cluster, width <= n_exp + 3, Q <= 2 x the median Q of its |tan| bin       (both views)
  blob   : any second cluster with > 20 % of the main charge, or width > n_exp + 6, or Q > 3 x the bin median (either view)
  other  : the rest
Arms (heldout_arm on events 1500-6318, det3 base bundle, SAT_RAIL on, rr1 cache): `rail` = the reference (true clips
only); `cap<C>` / `emu<C>` = every window also clipped at C ADC (the same operation, --emulate-rail C).
Metrics per class and arm, event-paired against `rail`:
  angle  s68 / far of the angle error per view (t5_compare.angerr: each arm's own ladder)
  charge (a) median q_sum ratio arm / rail per view; (b) x-y charge consistency: robust sigma of log(qx / qy) on events
         with both views; (c) x-y PAIRING power: P(|log qx_i/qy_i| < |log qx_i/qy_j|) for a random other event j of the
         same class (0.5 = charge useless for matching, 1 = perfect)
Usage: python -I clean_check.py CACHE OUTDIR [arms ...] [--tag NAME]  -> results/<NAME>.json + tables on stdout
"""
import glob, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'templates'))
from t5_compare import angerr, s68                                # noqa: E402


def clusters(W, noise):
    pk = W.max(axis=1); sig = pk > 5 * np.maximum(noise, 3.0)
    idx = np.flatnonzero(sig); out = []
    if not len(idx):
        return out
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > 2:
            out.append((start, prev)); start = i
        prev = i
    out.append((start, prev))
    return [(a, b, float(np.clip(W[a:b + 1], 0, None).sum())) for a, b in out]


def classify(C, ids):
    feats = {}
    for e in ids:
        ev = C[e]; f = {}
        for p in 'xy':
            if p not in ev:
                continue
            W = np.asarray(ev[p]['W'], float); cl = clusters(W, np.asarray(ev[p]['noise'], float))
            if not cl:
                continue
            cl.sort(key=lambda c: -c[2]); a, b, q = cl[0]
            f[p] = dict(width=b - a + 1, Q=q, second=(cl[1][2] / q if len(cl) > 1 else 0.0), ncl=len(cl),
                        n_exp=abs(float(ev[f'tan_{p}'])) * 30 / 0.78 + 2, tan=abs(float(ev[f'tan_{p}'])))
        feats[e] = f
    # median Q per |tan| bin and view
    med = {}
    for p in 'xy':
        T = np.array([[f[p]['tan'], f[p]['Q']] for f in feats.values() if p in f])
        edges = np.quantile(T[:, 0], np.linspace(0, 1, 9))
        med[p] = (edges, [np.median(T[(T[:, 0] >= lo) & (T[:, 0] <= hi), 1]) for lo, hi in zip(edges[:-1], edges[1:])])
    cls = {}
    for e, f in feats.items():
        if set(f) != {'x', 'y'}:
            cls[e] = 'other'; continue
        clean, blob = True, False
        for p in 'xy':
            g = f[p]; edges, m = med[p]
            mq = m[min(max(np.searchsorted(edges, g['tan']) - 1, 0), len(m) - 1)]
            clean &= (g['ncl'] == 1) and g['width'] <= g['n_exp'] + 3 and g['Q'] <= 2 * mq
            blob |= g['second'] > 0.2 or g['width'] > g['n_exp'] + 6 or g['Q'] > 3 * mq
        cls[e] = 'blob' if blob else ('clean' if clean else 'other')
    return cls


def ladder(R):
    """the arm's angle scale and zero from ALL its events of a view (as kw/w0 are measured per row on all events):
    scoring a class subset with a scale re-fitted on that subset alone is noisy (det3 clean Y, 338 events: its zero
    sat +1.5 deg off and turned a -0.09 deg gain into a +0.15 loss)."""
    tr, w = R[:, 2], R[:, 3]
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    c = np.median(w[np.abs(tr) < 0.05])
    return c, np.median((w[m] - c) / tr[m])


def err_with(R, cs):
    c, sc = cs
    return np.degrees(np.arctan((R[:, 3] - c) / sc) - np.arctan(R[:, 2]))


def load_arm(d, arm):
    fs = sorted(glob.glob(f'{d}/det3big_{arm}_c*.npz'))
    return np.concatenate([np.load(f)['per'][:, :16] for f in fs]) if fs else None


def pairing_power(lq_pairs, qx, qy, rng):
    n = len(qx)
    if n < 50:
        return np.nan
    j = rng.permutation(n)
    j = np.where(j == np.arange(n), (j + 1) % n, j)
    right = np.abs(np.log(qx / qy)); wrong = np.abs(np.log(qx / qy[j]))
    return float(np.mean(right < wrong))


def main(cache, d, *arms_in, tag='clean_check'):
    C = pickle.load(open(cache, 'rb'))
    ref = load_arm(d, 'rail')
    ids = sorted(set(ref[:, 0].astype(int)))
    cls = classify(C, ids)
    arms = ['rail'] + [a for a in (arms_in or ('emu300', 'emu600', 'cap800', 'cap1000', 'cap1200', 'cap1500', 'emu1500',
                                               'cap2000', 'emu2500')) if load_arm(d, a) is not None]
    rng = np.random.default_rng(1); out = {'counts': {c: sum(1 for v in cls.values() if v == c) for c in ('clean', 'blob', 'other')}}
    R = {a: load_arm(d, a) for a in arms}
    LAD = {(a, vi): ladder(R[a][R[a][:, 1] == vi]) for a in arms for vi in (0, 1)}
    for c in ('clean', 'blob', 'all'):
        out[c] = {}
        for a in arms:
            row = {}
            for vi, v in enumerate('xy'):
                A = {int(r[0]): r for r in R[a][R[a][:, 1] == vi]}; B = {int(r[0]): r for r in ref[ref[:, 1] == vi]}
                ev = sorted(e for e in set(A) & set(B) if c == 'all' or cls.get(e) == c)
                RA = np.array([A[e] for e in ev]); RB = np.array([B[e] for e in ev])
                dA = err_with(RA, LAD[(a, vi)]); dB = err_with(RB, LAD[('rail', vi)])
                bs = [s68(dA[j]) - s68(dB[j]) for j in (rng.integers(0, len(ev), len(ev)) for _ in range(300))]
                qr = RA[:, 15] / np.where(RB[:, 15] > 0, RB[:, 15], np.nan)
                def pres(R):                     # position residual at the mesh, offset + tan * dz removed per arm
                    r = R[:, 9] - R[:, 10]; t = R[:, 2]; k = np.isfinite(r) & (np.abs(r - np.nanmedian(r)) < 5)
                    X = np.c_[np.ones(k.sum()), t[k]]; b = np.linalg.lstsq(X, r[k], rcond=None)[0]
                    return r - (b[0] + b[1] * t)
                pa, pb = pres(RA), pres(RB)
                okp = np.isfinite(pa) & np.isfinite(pb) & (np.abs(pb) < 5)
                pbs = [s68(pa[okp][j]) - s68(pb[okp][j]) for j in (rng.integers(0, okp.sum(), okp.sum()) for _ in range(200))]
                row[v] = dict(n=len(ev), s68=float(s68(dA)), ds68=float(s68(dA) - s68(dB)), err=float(np.std(bs)),
                              far=float(100 * np.mean(np.abs(dA) > 5)), q_ratio=float(np.nanmedian(qr)),
                              pos_s68=float(s68(pa[okp])), dpos=float(s68(pa[okp]) - s68(pb[okp])), dpos_err=float(np.std(pbs)))
            # x-y charge consistency and pairing power on events with both views
            X = {int(r[0]): r[15] for r in R[a][R[a][:, 1] == 0]}; Y = {int(r[0]): r[15] for r in R[a][R[a][:, 1] == 1]}
            ev = sorted(e for e in set(X) & set(Y) if (c == 'all' or cls.get(e) == c) and X[e] > 0 and Y[e] > 0)
            qx = np.array([X[e] for e in ev]); qy = np.array([Y[e] for e in ev]); lq = np.log(qx / qy)
            row['xy'] = dict(n=len(ev), lq_rsig=float(1.4826 * np.median(np.abs(lq - np.median(lq)))),
                             pairing=float(np.mean([pairing_power(None, qx, qy, rng) for _ in range(20)])))
            out[c][a] = row
    json.dump(out, open(f'{HERE}/results/{tag}.json', 'w'), indent=1)
    print('classes (events):', out['counts'])
    for c in ('clean', 'blob', 'all'):
        print(f'\n**{c}**\n\n| arm | n X | angle Δs68 X | angle Δs68 Y | far X / Y % | position Δs68 X / Y [mm] | q ratio X / Y | x-y log-charge σ | x-y pairing power |')
        print('|---|---|---|---|---|---|---|---|---|')
        for a, r in out[c].items():
            x, y, xy = r['x'], r['y'], r['xy']
            print(f"| {a} | {x['n']} | {x['ds68']:+.3f} ± {x['err']:.3f} (s68 {x['s68']:.3f}) | {y['ds68']:+.3f} ± {y['err']:.3f} (s68 {y['s68']:.3f}) | "
                  f"{x['far']:.1f} / {y['far']:.1f} | {x['dpos']:+.3f} ± {x['dpos_err']:.3f} / {y['dpos']:+.3f} ± {y['dpos_err']:.3f} (s68 {x['pos_s68']:.2f} / {y['pos_s68']:.2f}) | "
                  f"{x['q_ratio']:.3f} / {y['q_ratio']:.3f} | {xy['lq_rsig']:.3f} | {xy['pairing']:.3f} |")


if __name__ == '__main__':
    a = sys.argv[1:]
    tag = 'clean_check'
    if '--tag' in a:
        i = a.index('--tag'); tag = a[i + 1]; a = a[:i] + a[i + 2:]
    main(*a, tag=tag)

#!/usr/bin/env python3
"""pin_compare.py [--base-dir heldout --base t0i] [--arm-dir ws_t0/heldout --arm lat] -- the position half of the
held-out comparison (heldout_compare.py does the angles): event-paired robust sigma of (p0 - M3 at the mesh) after a
per-arm linear re-fit against tan_ref (each arm gets its own mesh z), |r| < 5 mm, with a bootstrap over events; and the
x-y coincidence of the column time t0 + u50 cannot be formed from these rows, so only p0 is compared.
"""
import argparse, json, os
import numpy as np

M = '/home/dylan/x17/cosmic_bench/model_audit'
HERE = os.path.dirname(os.path.abspath(__file__))


def rsig(a):
    q = np.percentile(a, [16, 84]); return 0.5 * (q[1] - q[0])


def pos_res(R):
    tr, p0, ref = R[:, 2], R[:, 9], R[:, 10]
    res = p0 - ref
    ok = np.isfinite(res) & (np.abs(res - np.median(res[np.isfinite(res)])) < 5)
    cf = np.polyfit(tr[ok], res[ok], 1)
    r = res - np.polyval(cf, tr)
    return r, ok, cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default=f'{M}/heldout'); ap.add_argument('--base', default='t0i')
    ap.add_argument('--arm-dir', default=f'{M}/ws_t0/heldout'); ap.add_argument('--arm', default='lat')
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    out = {}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        fb, fa = f'{a.base_dir}/{det}_{a.base}.npz', f'{a.arm_dir}/{det}_{a.arm}.npz'
        if not (os.path.exists(fb) and os.path.exists(fa)):
            continue
        B, A = np.load(fb)['per'], np.load(fa)['per']
        out[det] = {}
        for vi, view in enumerate('xy'):
            b = {int(r[0]): r for r in B[B[:, 1] == vi]}; x = {int(r[0]): r for r in A[A[:, 1] == vi]}
            ids = sorted(set(b) & set(x))
            Rb, Ra = np.array([b[i] for i in ids]), np.array([x[i] for i in ids])
            rb, okb, cfb = pos_res(Rb); ra, oka, cfa = pos_res(Ra)
            ok = okb & oka
            sb, sa = rsig(rb[ok]), rsig(ra[ok])
            idx = np.where(ok)[0]
            bs = [rsig(ra[j]) - rsig(rb[j]) for j in (rng.choice(idx, len(idx)) for _ in range(300))]
            out[det][view] = dict(n=int(ok.sum()), base=float(sb), arm=float(sa), delta=float(sa - sb), err=float(np.std(bs)),
                                  slope_base=float(cfb[0]), slope_arm=float(cfa[0]))
            o = out[det][view]
            print(f'{det} {view} n={o["n"]:4d}  position rsig {a.base} {sb:.3f} -> {a.arm} {sa:.3f} mm  '
                  f'({100 * (sa / sb - 1):+.1f} %, {o["delta"]:+.3f} +- {o["err"]:.3f})  tan slope {cfb[0]:+.2f} -> {cfa[0]:+.2f} mm')
    json.dump(out, open(os.path.join(HERE, f'pin_compare_{a.arm}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

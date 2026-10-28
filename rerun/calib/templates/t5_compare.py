"""T5 comparison: event-paired held-out metrics of the template arms against cur_d+0.
Per view: s68 of the angle error (far > 5 deg excluded, the arm's own ladder), far %, chi2/dof median, ladder,
charge (q_sum) median ratio; paired bootstrap errors on the s68 difference.
Usage: python -I t5_compare.py DIR det3 [BASE_ARM [OUT_TAG]]   (default base cur_d+0, out results/t5_<det>.json)"""
import glob, json, os, sys
import numpy as np


def angerr(R):
    tr, w = R[:, 2], R[:, 3]
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    c = np.median(w[np.abs(tr) < 0.05])
    s = np.median((w[m] - c) / tr[m])
    return np.degrees(np.arctan((w - c) / s) - np.arctan(tr)), s


def s68(d):
    k = np.abs(d) <= 5
    return np.percentile(np.abs(d[k] - np.median(d[k])), 68)


def main(d, det, base_name='cur_d+0', tag='t5'):
    arms = {os.path.basename(f)[len(det) + 1:-4]: np.load(f)['per'] for f in sorted(glob.glob(f'{d}/{det}_*.npz'))}
    base = arms.pop(base_name); rng = np.random.default_rng(3); out = {}
    for vi, view in enumerate('xy'):
        B = {int(r[0]): r for r in base[base[:, 1] == vi]}
        print(f'== {det} {view}  (vs {base_name})')
        for name, per in sorted(arms.items()):
            A = {int(r[0]): r for r in per[per[:, 1] == vi]}
            ids = sorted(set(A) & set(B))
            RA = np.array([A[i] for i in ids]); RB = np.array([B[i] for i in ids])
            dA, sA = angerr(RA); dB, sB = angerr(RB)
            boots = []
            for _ in range(300):
                j = rng.integers(0, len(ids), len(ids)); boots.append(s68(dA[j]) - s68(dB[j]))
            x2A = np.median(RA[:, 4] / np.maximum(RA[:, 5], 1)); x2B = np.median(RB[:, 4] / np.maximum(RB[:, 5], 1))
            q = np.median(RA[:, 15] / np.where(RB[:, 15] > 0, RB[:, 15], np.nan)[:, None].ravel()) if RA.shape[1] > 15 else np.nan
            row = dict(n=len(ids), ds68=float(s68(dA) - s68(dB)), ds68_err=float(np.std(boots)), s68=float(s68(dA)),
                       far=float(100 * np.mean(np.abs(dA) > 5)), far_base=float(100 * np.mean(np.abs(dB) > 5)),
                       chi2=float(x2A), chi2_base=float(x2B), ladder=float(sA), ladder_base=float(sB), q_ratio=float(q),
                       t0_shift=float(np.median(RA[:, 8] - RB[:, 8])))
            out[f'{view}_{name}'] = row
            print(f"  {name:10s} n={row['n']} s68 {row['s68']:.3f} (d {row['ds68']:+.3f}±{row['ds68_err']:.3f}) far {row['far']:.1f}/{row['far_base']:.1f} "
                  f"chi2 {x2A:.1f}/{x2B:.1f} ladder {sA:.2f}/{sB:.2f} q {q:.3f}")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/{tag}_{det}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])

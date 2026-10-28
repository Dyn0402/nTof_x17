#!/usr/bin/env python3
"""ga_analyse.py -- GA tasks 1 and 3 from ga_gainfit.py's output: the in-fit per-channel gain maps (split-half
reproducibility, scale decomposition, connector steps, weak channels) and the cross-validated effect of applying them
on angles (s68 against M3), far fraction and chi2/dof, paired by event with a bootstrap.

    ga_analyse.py [--dets det4,det3]  -> results/ga_eval.json, results/ga_eval.md, figures/infit_map_<det>.png
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_INFO = 50.0
ARMS = ['base', 'cv', 'cv_smooth', 'cv_fine', 'self']


def per_arm(rows, vi):
    R = rows[rows[:, 1] == vi]
    return {int(r[0]): r for r in R}


def ladder_c(R):
    tr, w = R[:, 2], R[:, 3]
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    c = np.median(w[np.abs(tr) < 0.05]) if (np.abs(tr) < 0.05).sum() > 20 else 0.0
    s = float(np.median((w[m] - c) / tr[m]))
    return s, c


def stats(R, s, c, cls):
    tr, w = R[:, 2], R[:, 3]
    d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
    m = dict(all=np.ones(len(tr), bool), headon=np.abs(tr) < 0.1, inclined=np.abs(tr) >= 0.1)[cls]
    k = m & ~far
    return np.array([np.percentile(np.abs(d[k] - np.median(d[k])), 68) if k.sum() > 10 else np.nan,
                     100 * far[m].mean(), np.median(R[m, 4] / np.maximum(R[m, 5], 1))])


def compare(rows_b, rows_a, vi, nboot=400):
    B, A = per_arm(rows_b, vi), per_arm(rows_a, vi)
    ids = sorted(set(B) & set(A))
    Rb = np.array([B[e] for e in ids]); Ra = np.array([A[e] for e in ids])
    sb, cb = ladder_c(Rb); sa, ca = ladder_c(Ra)
    out = dict(n=len(ids), ladder_base=sb, ladder_arm=sa)
    rs = np.random.default_rng(3)
    for cls in ('all', 'headon', 'inclined'):
        b, a = stats(Rb, sb, cb, cls), stats(Ra, sa, ca, cls)
        bs = []
        for _ in range(nboot):
            i = rs.integers(0, len(ids), len(ids))
            bs.append(stats(Ra[i], sa, ca, cls) - stats(Rb[i], sb, cb, cls))
        e = np.nanstd(np.array(bs), axis=0)
        out[cls] = dict(s68=[float(b[0]), float(a[0]), float(a[0] - b[0]), float(e[0])],
                        far=[float(b[1]), float(a[1]), float(a[1] - b[1]), float(e[1])],
                        chi2dof=[float(b[2]), float(a[2]), float(a[2] - b[2]), float(e[2])])
    # median per-event chi2/dof ratio (paired)
    out['chi2dof_ratio_med'] = float(np.median((Ra[:, 4] / Ra[:, 5]) / (Rb[:, 4] / Rb[:, 5])))
    return out


def sig(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    c = float(np.mean(a * b))
    return dict(signal=float(np.sign(c) * np.sqrt(abs(c))), noise=float(np.std(a - b) / 2),
                corr=float(np.corrcoef(a, b)[0, 1]), n=int(m.sum()))


def map_stats(z, p):
    gA, gB = z[f'map_A_{p}'], z[f'map_B_{p}']
    m = (z[f'info_A_{p}'] > MIN_INFO) & (z[f'info_B_{p}'] > MIN_INFO)
    lA, lB = np.log(np.clip(gA, 1e-3, None)), np.log(np.clip(gB, 1e-3, None))
    sA, sB = np.log(z[f'smooth_A_{p}']), np.log(z[f'smooth_B_{p}'])
    fA, fB = lA - sA, lB - sB
    # nearest-neighbour difference (strip-to-strip) in position order
    pos = z[f'pos_{p}']
    o = np.argsort(np.where(np.isfinite(pos), pos, 1e9))
    o = o[m[o]]
    dA, dB = np.diff(lA[o]), np.diff(lB[o])
    adj = np.diff(pos[o]) < 1.2          # true neighbours (no gap / masked strip between)
    conn = {}
    for h, l in (('A', lA), ('B', lB)):
        conn[h] = [float(np.median(l[m & (np.arange(512) // 64 == k)])) if (m & (np.arange(512) // 64 == k)).sum() > 8 else np.nan
                   for k in range(8)]
    cA, cB = np.array(conn['A']), np.array(conn['B'])
    ok = np.isfinite(cA) & np.isfinite(cB)
    weak = int(np.sum(m & (np.sqrt(gA * gB) < 0.4)))
    return dict(n_ch=int(m.sum()), rms_A=float(np.std(lA[m])), rms_B=float(np.std(lB[m])),
                total=sig(lA, lB, m), smooth=sig(sA, sB, m), fine=sig(fA, fB, m),
                neighbour_diff=sig(dA[adj], dB[adj], np.ones(adj.sum(), bool)) if adj.sum() > 10 else None,
                connector_A=conn['A'], connector_B=conn['B'],
                connector=sig(cA, cB, ok) if ok.sum() > 3 else None, n_weak_lt04=weak)


def fig(det, z, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(2, 1, figsize=(14, 7))
    for ax, p in zip(axs, 'xy'):
        pos = z[f'pos_{p}']
        for h, c in (('A', 'C0'), ('B', 'C1')):
            m = z[f'info_{h}_{p}'] > MIN_INFO
            ax.plot(pos[m], np.log(z[f'map_{h}_{p}'][m]), '.', ms=3, color=c, label=f'half {h} (per channel)')
            o = np.argsort(pos[m])
            ax.plot(pos[m][o], np.log(z[f'smooth_{h}_{p}'][m][o]), '-', color=c, lw=1.2)
        ax.set_ylabel(f'{p.upper()}: ln g (in-fit)'); ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-2, 2)
        ax.set_xlabel('strip position [mm]')
    axs[0].legend(fontsize=8); axs[0].set_title(f'{det}: in-fit per-channel gain, two disjoint event halves (lines: 4 mm smooth)')
    f.tight_layout(); f.savefig(path, dpi=110); plt.close(f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dets', default='det4,det3'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    R, md = {}, []
    for det in a.dets.split(','):
        fz = os.path.join(HERE, 'results', f'gainfit_{det}{a.tag}.npz')
        if not os.path.exists(fz):
            print('missing', fz); continue
        z = np.load(fz)
        fig(det, z, os.path.join(HERE, 'figures', f'infit_map_{det}{a.tag}.png'))
        R[det] = dict(maps={p: map_stats(z, p) for p in 'xy'}, eval={})
        for arm in ARMS[1:]:
            if f'rows_{arm}' not in z.files:
                continue
            R[det]['eval'][arm] = {p: compare(z['rows_base'], z[f'rows_{arm}'], vi) for vi, p in enumerate('xy')}
        md.append(f'## {det}\n\nmaps (ln g, channels with info in both halves):\n')
        md.append('| view | n ch | rms A / B | signal total (noise) | smooth 4 mm | fine | neighbour diff | connector | weak < 0.4 |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for p in 'xy':
            s = R[det]['maps'][p]
            f = lambda d: f"{d['signal']:+.3f} ({d['noise']:.3f})" if d else '-'
            md.append(f"| {p} | {s['n_ch']} | {s['rms_A']:.3f} / {s['rms_B']:.3f} | {f(s['total'])} | {f(s['smooth'])} | "
                      f"{f(s['fine'])} | {f(s['neighbour_diff'])} | {f(s['connector'])} | {s['n_weak_lt04']} |")
        md.append('\nangles vs M3, paired, arm − base (bootstrap σ):\n')
        md.append('| arm | view | n | s68 all base → arm (Δ ± σ) | head-on Δ | inclined Δ | far Δ pp | χ²/dof med base → arm |')
        md.append('|---|---|---|---|---|---|---|---|')
        for arm, E in R[det]['eval'].items():
            for p in 'xy':
                e = E[p]
                md.append(f"| {arm} | {p} | {e['n']} | {e['all']['s68'][0]:.3f} → {e['all']['s68'][1]:.3f} "
                          f"({e['all']['s68'][2]:+.3f} ± {e['all']['s68'][3]:.3f}) | {e['headon']['s68'][2]:+.3f} ± {e['headon']['s68'][3]:.3f} | "
                          f"{e['inclined']['s68'][2]:+.3f} ± {e['inclined']['s68'][3]:.3f} | {e['all']['far'][2]:+.2f} ± {e['all']['far'][3]:.2f} | "
                          f"{e['all']['chi2dof'][0]:.1f} → {e['all']['chi2dof'][1]:.1f} |")
        md.append('')
    json.dump(R, open(os.path.join(HERE, 'results', f'ga_eval{a.tag}.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', f'ga_eval{a.tag}.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

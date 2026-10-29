#!/usr/bin/env python3
"""resid.py -- how the model's prediction compares with a clipped DREAM pulse and the samples around it.

    resid.py run DET OUT.pkl [--n N] [--start 180] [--jobs 8]     fit (heldout_arm's fitter: pass bundle, t0 pinned,
                                                                  LM, the arm's WFT_DIAG switches) and keep, per plane,
                                                                  W, the model, the censor mask, raw, noise, the fit
    resid.py stack OUT.pkl [...]                                  -> results/resid_<tag>.json + figures/resid_<tag>.png

Stacks (gain-corrected ADC, data - model; the model is the fit's own prediction, so where a sample is censored the
model is free and the residual shows what the fit would have had to explain):
  A  clipped strips, aligned on the LAST clipped sample (k = 0), split by plateau length L <= 2 / L >= 4, and by
     whether the strip falls to the ADC floor;   also aligned on the FIRST clipped sample (the rising edge)
  B  the +-1 / +-2 neighbours of clipped strips (not clipped themselves), same alignment
  C  bright UNCLIPPED strips, aligned on their peak, in bins of peak amplitude (does the shape change with amplitude:
     compression below the rail, deeper undershoot?)
  D  the chi2 share of each sample class in the base fit (clipped-strip tail, floor samples, neighbours, rest)
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit')]
TMP = os.path.join(os.environ.get('CLAUDE_JOB_DIR', os.path.expanduser('~/.claude/jobs/c68d6c6a')), 'tmp')
KS = np.arange(-6, 15)
AMP_BINS = [(500, 1000), (1000, 2000), (2000, 3000), (3000, 3550), (3550, 4100)]


def _one(eid):
    import heldout_arm as H
    from wft import model as wm, reco as wr
    cal = wm.CAL; ev = H._EV[eid]; out = []
    for vi, plane in enumerate('xy'):
        if plane not in ev:
            continue
        P = ev[plane]; W0 = np.asarray(P['W'])
        rr = H._RAILS.get(int(eid), {}).get(plane) if H._RAILS is not None else None
        if rr is None or rr['raw'].shape != W0.shape:
            continue
        raw = rr['raw'].astype(np.float64)
        P = dict(P, rail_hi=W0 + (4095.0 - raw), rail_lo=W0 - raw)
        if W0.shape[1] != wm.NSAMP:
            wm.set_nsamp(W0.shape[1])
        ft = ev.get(f'ftst_{plane}'); t0a = (cal.t0_abs or {}).get(plane, {})
        pred = t0a.get(int(ft)) if ft is not None else None
        prior = (float(pred), 0.05) if pred is not None else None
        try:
            f = wr.fit_plane(P, plane, cal, t0_prior=prior)
        except Exception:
            f = None
        if f is None:
            continue
        W, noise, pos, sat = wm.prep_plane(P, plane)
        M = wm.model_waveforms(plane, pos, f.p0, f.w, f.t0, f._q, cal.hyper)
        out.append(dict(eid=int(eid), view=vi, tan=float(ev[f'tan_{plane}']), w=f.w * 1e3, chi2=f.chi2, dof=f.dof,
                        W=W.astype(np.float32), M=M.astype(np.float32), sat=sat, raw=raw.astype(np.int16),
                        noise=noise.astype(np.float32), pos=pos.astype(np.float32), q_sum=f.q_sum))
    return out


def run(det, outp, n=1320, start=180, jobs=8, bundle=None, cache=None, rails=None):
    import heldout_arm as H
    bundle = bundle or f'{TMP}/t15/bundles/{det}/base'
    cache = cache or f'{TMP}/t15/caches/{det}.pkl'
    rails = rails or f'{TMP}/satur/rails_{det}.pkl'
    ids = sorted(pickle.load(open(cache, 'rb')))[start:start + n]
    with ProcessPoolExecutor(jobs, initializer=H._init,
                             initargs=(cache, bundle, 24, {}, None, None, False, True, False, None, None, rails)) as pool:
        rows = [r for rs in pool.map(_one, ids, chunksize=2) for r in rs]
    pickle.dump(dict(det=det, rows=rows, env={k: v for k, v in os.environ.items() if k.startswith('WFT_')}),
                open(outp, 'wb'), protocol=4)
    print(outp, len(rows))


def _acc(D, key, k, r, m, s):
    d = D.setdefault(key, {'r': [[] for _ in KS], 'm': [[] for _ in KS], 's': [[] for _ in KS]})
    i = k - KS[0]
    if 0 <= i < len(KS):
        d['r'][i].append(r); d['m'][i].append(m); d['s'][i].append(s)


def stack(path):
    D = pickle.load(open(path, 'rb')); rows = D['rows']; S = {}
    chi = dict(clip_plateau=0.0, clip_post=0.0, clip_rise=0.0, floor=0.0, nbr=0.0, rest=0.0)
    chi_planes_sat = 0
    for r in rows:
        W, M, raw, sat, nz = r['W'].astype(float), r['M'].astype(float), r['raw'], r['sat'], r['noise'].astype(float)
        clip = raw >= 4095; floor = raw <= 0; ns = W.shape[1]
        crow = clip.any(1)
        Z2 = ((W - M) / nz[:, None]) ** 2
        cls = np.full(W.shape, 'rest', dtype=object)
        for i in np.flatnonzero(crow):
            k = np.flatnonzero(clip[i]); a, b = k.min(), k.max(); L = b - a + 1
            reach = bool(floor[i, b + 1:].any())
            cls[i, :a] = 'clip_rise'; cls[i, a:b + 1] = 'clip_plateau'; cls[i, b + 1:] = 'clip_post'
            for j in range(ns):
                tag = 'L<=2' if L <= 2 else ('L>=4' if L >= 4 else 'L3')
                _acc(S, f'A_last_{tag}_{"floor" if reach else "nofloor"}', j - b, W[i, j] - M[i, j], M[i, j], sat[i, j])
                _acc(S, f'A_first_{tag}', j - a, W[i, j] - M[i, j], M[i, j], sat[i, j])
            for o in (-2, -1, 1, 2):
                n_ = i + o
                if 0 <= n_ < len(W) and not crow[n_]:
                    if cls[n_, 0] == 'rest':
                        cls[n_, :] = 'nbr'
                    for j in range(ns):
                        _acc(S, f'B_pm{abs(o)}', j - b, W[n_, j] - M[n_, j], M[n_, j], sat[n_, j])
        cls[floor & (cls == 'rest')] = 'floor'
        cls[floor & (cls == 'nbr')] = 'floor'
        for i in np.flatnonzero(~crow):
            pk = W[i].max(); j0 = int(np.argmax(W[i]))
            for lo, hi in AMP_BINS:
                if lo <= pk < hi:
                    for j in range(ns):
                        _acc(S, f'C_{lo}_{hi}', j - j0, (W[i, j] - M[i, j]) / pk, M[i, j] / pk, sat[i, j])
        if crow.any():
            chi_planes_sat += 1
            for c in chi:
                chi[c] += float(Z2[(cls == c) & ~sat].sum())
    out = dict(path=path, env=D.get('env'), n_planes=len(rows), planes_with_clip=chi_planes_sat,
               chi2_share_on_clipped_planes={c: v / max(sum(chi.values()), 1e-9) for c, v in chi.items()}, stacks={})
    for key, d in sorted(S.items()):
        out['stacks'][key] = dict(k=KS.tolist(), n=[len(x) for x in d['r']],
                                  resid_med=[float(np.median(x)) if len(x) > 5 else None for x in d['r']],
                                  model_med=[float(np.median(x)) if len(x) > 5 else None for x in d['m']],
                                  censored_frac=[float(np.mean(x)) if len(x) > 5 else None for x in d['s']],
                                  resid_p16=[float(np.percentile(x, 16)) if len(x) > 5 else None for x in d['r']],
                                  resid_p84=[float(np.percentile(x, 84)) if len(x) > 5 else None for x in d['r']])
    tag = os.path.basename(path).replace('.pkl', '')
    json.dump(out, open(f'{HERE}/results/resid_{tag}.json', 'w'), indent=1)
    plot(out, tag)
    return out


def plot(out, tag):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    st = out['stacks']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for key, c in (('A_last_L<=2_nofloor', 'C0'), ('A_last_L<=2_floor', 'C1'), ('A_last_L>=4_nofloor', 'C2'),
                   ('A_last_L>=4_floor', 'C3')):
        if key in st:
            s = st[key]; k = np.array(s['k']); r = np.array([np.nan if v is None else v for v in s['resid_med']])
            m = np.array([np.nan if v is None else v for v in s['model_med']])
            ax[0].plot(k, r, '-o', ms=3, c=c, label=f"{key[7:]} (n {max(s['n'])})")
            ax[0].plot(k, m, ':', c=c, lw=1)
    ax[0].axhline(0, c='k', lw=.5); ax[0].axvline(0, c='k', lw=.5)
    ax[0].set(xlabel='sample - last clipped sample', ylabel='data - model [ADC] (dotted: model)', title='A: clipped strips')
    ax[0].legend(fontsize=7)
    for key, c in (('B_pm1', 'C0'), ('B_pm2', 'C1')):
        if key in st:
            s = st[key]; k = np.array(s['k']); r = np.array([np.nan if v is None else v for v in s['resid_med']])
            ax[1].plot(k, r, '-o', ms=3, c=c, label=f"{key} (n {max(s['n'])})")
    ax[1].axhline(0, c='k', lw=.5); ax[1].axvline(0, c='k', lw=.5)
    ax[1].set(xlabel='sample - last clipped sample of the neighbour', ylabel='data - model [ADC]', title='B: neighbours of clipped strips')
    ax[1].legend(fontsize=7)
    for (lo, hi), c in zip(AMP_BINS, ('C0', 'C1', 'C2', 'C3', 'C4')):
        key = f'C_{lo}_{hi}'
        if key in st:
            s = st[key]; k = np.array(s['k']); r = np.array([np.nan if v is None else v for v in s['resid_med']])
            ax[2].plot(k, r, '-o', ms=3, c=c, label=f"peak {lo}-{hi} (n {max(s['n'])})")
    ax[2].axhline(0, c='k', lw=.5); ax[2].axvline(0, c='k', lw=.5)
    ax[2].set(xlabel='sample - peak sample', ylabel='(data - model) / peak', title='C: unclipped strips by peak amplitude')
    ax[2].legend(fontsize=7)
    fig.suptitle(f'{tag}: stacked residuals of the fit (medians)')
    fig.tight_layout(); os.makedirs(f'{HERE}/figures', exist_ok=True)
    fig.savefig(f'{HERE}/figures/resid_{tag}.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('args', nargs='+')
    ap.add_argument('--n', type=int, default=1320); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--bundle'); ap.add_argument('--cache'); ap.add_argument('--rails')
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.args[0], a.args[1], a.n, a.start, a.jobs, a.bundle, a.cache, a.rails)
    else:
        for p in a.args:
            o = stack(p)
            print(p, o['n_planes'], o['planes_with_clip'], {k: round(v, 3) for k, v in o['chi2_share_on_clipped_planes'].items()})

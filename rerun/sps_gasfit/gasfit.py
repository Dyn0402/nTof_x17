#!/usr/bin/env python3
"""gasfit.py --stacks DIR [--out results/gasfit.json] [--gap 28.8] [arm options]

The SPS gas composition (water, air) per epoch from the final-pass stacks (gas_stacks.py), with the one physics model
of cloud_basics FINDINGS §25 (predict.current_field: uniform ionisation over the gap, Magboltz v / eta / D_L on the
(water, air) grid of gasmodel.GasGrid 'air_hs', eta smoothed in log E) (x) a parametric shaper (x) the 60 ns sampling.
No free loss rate: the late-charge loss is the composition's attachment.

1. Shaper: fitted on the three RAW run71 fields (X and Y), jointly with run71's composition (as beam_comp_fit.py),
   iterated three times.  It is then FIXED for every other group (same DREAM registers on every SPS run).
2. ZS rows: the model gets the additive ZS distortion (RAW-emulated minus RAW, gas_stacks.py) of run71 at the nearest
   field, same view and strip sum, at the row's ZS threshold (registry zs_sigma).
3. Groups (one composition each; per stack amplitude and t0 free): see GROUPS.  Inclined views of rotated mounts are
   left out (their strip sum truncates with depth inside the +-16-strip window: a fake gradient); head-on views only.
4. chi2 on the (water, air) grid; 68 % region = chi2 <= min + 2.30 x max(1, chi2/ndf) (errors scaled when the model
   does not describe the stacks to their statistical error).

Arms: --gap (28.8 registry / 30), --zs-scale (ZS distortion x0.5 / x1.5), --k (linear field profile), --sat (keep
saturated events), --hx/--hy (strip sums), --raw-eta (unsmoothed Magboltz eta), --source (grid).
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CB = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
sys.path[:0] = [CB]
import gasmodel as M                      # noqa: E402
import predict as P                       # noqa: E402

T = np.arange(64) * 60.0
WIN = (500.0, 3800.0)
NORM = (1080.0, 1260.0)
REG = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
#: one composition per group.  CF4 epoch (Ar/CF4/iso 88/10/2, >= 26 h after the switch, GAS_FLUSH_TIMELINE §2);
#: CO2 epoch (Ar/CO2/iso 95/3/2, 16-22 h after the flush start).  run63 flat700 (one hour) is also split in time.
GROUPS = {
    'cf4_run62_ladder': dict(gas='beam', keys=['run62_rot25_ladder__d700', 'run62_rot25_ladder__d600', 'run62_rot25_ladder__d500',
                                               'run62_rot25_ladder__d400', 'run62_rot25_ladder__d300', 'run62_rot25_ladder__d200']),
    'cf4_run63_ladder': dict(gas='beam', keys=['run63_rot25__d675', 'run63_rot25__d625', 'run63_rot25__d575', 'run63_rot25__d525',
                                               'run63_rot25__d475', 'run63_rot25__d425', 'run63_rot25__d325']),
    'cf4_run63_flat': dict(gas='beam', keys=['run63_flat__flat700']),
    'cf4_run71_raw': dict(gas='beam', keys=['run71_raw__raw700', 'run71_raw__raw450', 'run71_raw__raw275']),
    'co2_run55': dict(gas='co2', keys=['run55_flatdrift__d700', 'run55_flatdrift__d600', 'run55_flatdrift__d500', 'run55_flatdrift__d400']),
    'co2_run56': dict(gas='co2', keys=['run56_m70V__590V', 'run56_m70V__625V']),
    'co2_run57': dict(gas='co2', keys=['run57_rot25_co2__d700', 'run57_rot25_co2__d650', 'run57_rot25_co2__d600', 'run57_rot25_co2__d550']),
    'co2_run59': dict(gas='co2', keys=['run59_co2__co2']),
}
#: the CO2 -> CF4 flush (run60, 0.6-8.1 h after the switch): the base gas itself changes, so a (water, air) fit in
#: either pure base is INDICATIVE only (the grid has no mixed base); each sub-run is fitted in both and both are reported.
for _i in range(15):
    for _g, _tag in (('beam', 'cf4'), ('co2', 'co2')):
        GROUPS[f'trans_ov{_i:02d}_{_tag}'] = dict(gas=_g, keys=[f'run60_flush__ov{_i:02d}'], transition=True)
RAW_KEYS = ('run71_raw__raw700', 'run71_raw__raw450', 'run71_raw__raw275')


def load(sdir, key):
    p = os.path.join(sdir, f'{key}.npz')
    if not os.path.exists(p):
        return None
    return dict(np.load(p)), json.load(open(os.path.join(sdir, f'{key}.json')))


def head_on_views(Z, meta):
    """flat mount: both views; rotated: the view whose charge profile is narrower (the rotation axis)."""
    if meta['mount'] == 'flat':
        return ['x', 'y']
    w = {}
    for v in 'xy':
        p = np.asarray(Z[f'prof_{v}'], float); o = np.arange(len(p)) - len(p) // 2
        w[v] = np.sqrt((p * o ** 2).sum() / max(p.sum(), 1e-9))
    return [min(w, key=w.get)], w


def stack(Z, v, arm, h, sat):
    """normalised strip-sum stack over |o| <= h, jackknife error over blocks."""
    def SN(s):
        return Z[f'{v}_{arm}_{s}_S'].astype(float), Z[f'{v}_{arm}_{s}_N'].astype(float)
    S, N = SN('ok')
    if sat:
        S2, N2 = SN('sat'); S, N = S + S2, N + N2
    c = S.shape[1] // 2; sl = slice(c - h, c + h + 1)
    S, N = S[:, sl], N[:, sl]

    def curve(Sb, Nb):
        m = Sb / np.where(Nb > 0, Nb, np.nan)
        y = np.nansum(m, 0)
        return y / np.nanmean(y[(T >= NORM[0]) & (T <= NORM[1])])
    y = curve(S.sum(0), N.sum(0)); nb = S.shape[0]
    jk = np.array([curve(S.sum(0) - S[b], N.sum(0) - N[b]) for b in range(nb)])
    e = np.sqrt((nb - 1) / nb * ((jk - jk.mean(0)) ** 2).sum(0)) if nb > 1 else np.full(64, 0.02)
    return y, np.maximum(e, 0.005)


class Model:
    def __init__(self, gas, source, gap, k, smooth=True):
        self.G = M.GasGrid(gas, source, smooth=smooth); self.gap = gap; self.k = k; self.C = {}

    def E(self, d):
        """the drift field of a stack under this model's gap (HV / gap; whether the mesh voltage adds is in no record)"""
        return d['V'] / (self.gap / 10.0)

    def current(self, comp, E):
        key = (round(comp[0], 5), round(comp[1], 5), round(E, 3))
        if key not in self.C:
            # a field profile k reaches past the grid's last field (275 V/cm) at the short gap: hold the transport at the
            # grid edge instead of extrapolating it (the linear extrapolation of D_L / eta goes non-physical)
            lo, hi = float(self.G.E.min()), float(self.G.E.max())
            self.C[key] = P.current_field(lambda e: self.G(comp[0], comp[1], min(max(e, lo), hi)), E, self.gap, self.k)
        return self.C[key]


def fit_stack(t, y, e, I, par, zsd=None):
    """chi2, n, model, (t0, amp) of one normalised stack against the current I (t0: coarse scan, then least squares;
    a single start at 650 ns falls into a wrong minimum at low field, closure.py)."""
    m = (t >= WIN[0]) & (t <= WIN[1]) & np.isfinite(y)
    s, off = P.shaped(I, par)
    add = np.zeros_like(t) if zsd is None else zsd
    nm = (t >= NORM[0]) & (t <= NORM[1])

    # q = (t0, amplitude).  The ZS distortion is defined on the normalised stack: the model is normalised the same
    # way (plateau window), scaled by the free amplitude, then the distortion is added.
    def full(q):
        c = np.interp(t, P.FINE + off + q[0], s)
        mn = float(np.mean(c[nm]))
        return c / (mn if np.isfinite(mn) and mn > 1e-12 else 1e-12) * q[1] + add
    res = lambda q: ((full(q) - y) / e)[m]
    t0s = np.arange(-700.0, 1400.0, 25.0)            # the shaper carries its own delay (n tau): t0 is relative to it
    c2 = np.array([np.sum(res((x, 1.0)) ** 2) for x in t0s]); c2[~np.isfinite(c2)] = np.inf   # NaN: model empty in NORM
    if not np.isfinite(c2).any():
        return 1e12, int(m.sum()), np.full_like(t, np.nan), np.array([np.nan, np.nan])
    t00 = t0s[int(np.argmin(c2))]
    sol = least_squares(res, [t00, 1.0], x_scale=[20, 0.05])
    return float(np.sum(sol.fun ** 2)), int(m.sum()), full(sol.x), sol.x


def field(key, gap):
    return float(REG[key]['drift_V']) / (gap / 10.0)


def group_data(sdir, keys, a, zsd_tab):
    out = []
    for key in keys:
        L = load(sdir, key)
        if L is None:
            continue
        Z, meta = L
        r = head_on_views(Z, meta); views = r if meta['mount'] == 'flat' else r[0]
        raw = meta['raw']; E = field(key, a.gap)
        for v in views:
            h = a.hx if v == 'x' else a.hy
            y, e = stack(Z, v, 'all', h, a.sat)
            zsd = None
            if not raw:
                ks = int(round(float(meta['zs_sigma'] or 4)))
                near = min(zsd_tab, key=lambda kk: abs(kk[0] - E) if kk[1] == v and kk[2] == ks else 1e9)
                zsd = a.zs_scale * zsd_tab[near]
            if not raw and a.zs_win_hi:
                e = np.where(T > a.zs_win_hi, 1e6, e)       # arm: ZS rows only up to the drift end
            out.append(dict(key=key, view=v, V=float(REG[key]['drift_V']), E=E, t=T, y=y, e=e, zsd=zsd, raw=raw, n=meta['n'],
                            n_sat=meta['n_sat']))
    return out


def zs_table(sdir, a):
    tab = {}
    for key in RAW_KEYS:
        L = load(sdir, key)
        if L is None:
            continue
        Z, meta = L; E = field(key, a.gap)
        for v in 'xy':
            h = a.hx if v == 'x' else a.hy
            y0, _ = stack(Z, v, 'all', h, a.sat)
            for ks in (4, 5):
                yz, _ = stack(Z, v, f'zs{ks}', h, a.sat)
                tab[(E, v, ks)] = np.nan_to_num(yz - y0)
    return tab


def grid_scan(Mo, data, par, W, A):
    chi = np.full((len(W), len(A)), np.nan)
    for i, w in enumerate(W):
        for j, x in enumerate(A):
            chi[i, j] = sum(fit_stack(d['t'], d['y'], d['e'], Mo.current((w, x), Mo.E(d)), par, d['zsd'])[0] for d in data)
    return chi


def summarise(chi, W, A, ndf):
    i, j = np.unravel_index(np.nanargmin(chi), chi.shape); c0 = float(chi[i, j]); s = max(1.0, c0 / max(ndf, 1))
    ok = chi <= c0 + 2.30 * s
    wr = [float(W[ok.any(1)].min()), float(W[ok.any(1)].max())]; ar = [float(A[ok.any(0)].min()), float(A[ok.any(0)].max())]
    edge = dict(water=bool(i in (0, len(W) - 1)), air=bool(j in (0, len(A) - 1) and A[j] > 0))
    return dict(water=float(W[i]), air=float(A[j]), o2_ppm=float(A[j]) * 2095, water_68=wr, air_68=ar,
                o2_68=[x * 2095 for x in ar], chi2=c0, ndf=int(ndf), scale=s, edge=edge)


def fit_shaper(Mo, data, comp, par0):
    def fsh(p):
        if p[0] < 0.5 or p[1] < 5 or p[3] < 50:
            return np.full(sum(int(((d['t'] >= WIN[0]) & (d['t'] <= WIN[1])).sum()) for d in data), 1e3)
        out = []
        for d in data:
            _, _, mc, _ = fit_stack(d['t'], d['y'], d['e'], Mo.current(comp, Mo.E(d)), p, d['zsd'])
            m = (d['t'] >= WIN[0]) & (d['t'] <= WIN[1])
            out.append(((mc - d['y']) / d['e'])[m])
        r = np.concatenate(out)
        return np.where(np.isfinite(r), r, 1e3)     # a shaper/transport combination with an empty plateau window
    # physical bounds: CR-RC order and time, a non-negative undershoot, its time constant within the 3.8 us window
    # scale (unbounded, the k = +0.1 arm ran tu to 2e9 ns with u < 0 and a non-finite stack fit)
    lo, hi = np.array([0.5, 5.0, 0.0, 50.0]), np.array([60.0, 400.0, 1.5, 2e4])
    sol = least_squares(fsh, np.clip(par0, lo + 1e-6, hi - 1e-6), x_scale=[0.3, 10, 0.01, 100], diff_step=[1e-2] * 4,
                        bounds=(lo, hi))
    return sol.x


ARGS = ZTAB = RAWD = None


def shaper_for_gap(gp):
    a = ARGS
    if a.bench_template:
        return None          # arm: the measured bench X template (cloud_basics template_det3_x.json), no free shaper
    Mb = Model('beam', a.source, gp, a.k, not a.raw_eta)
    par = np.array([8.886, 40.73, 0.062, 2438.0]); comp = (1.55, 0.07)
    Wb = np.linspace(Mb.G.W.min(), Mb.G.W.max(), a.nw); Ab = np.linspace(0, Mb.G.A.max(), a.na)
    for it in range(3):
        par = fit_shaper(Mb, RAWD, comp, par)
        chi = grid_scan(Mb, RAWD, par, Wb, Ab)
        i, j = np.unravel_index(np.nanargmin(chi), chi.shape); comp = (Wb[i], Ab[j])
        print(f'gap {gp}: shaper it {it}: {np.round(par, 3)}  water {comp[0]:.3f} air {comp[1]:.3f}  chi2 {np.nanmin(chi):.0f}',
              flush=True)
    return par


def scan_task(g, gp, par):
    a = ARGS; spec = GROUPS[g]
    data = group_data(a.stacks, spec['keys'], a, ZTAB)
    Mo = Model(spec['gas'], a.source, gp, a.k, not a.raw_eta)
    W = np.linspace(Mo.G.W.min(), Mo.G.W.max(), a.nw); A = np.linspace(0, Mo.G.A.max(), a.na)
    if not data:
        return None, W, A
    chi = grid_scan(Mo, data, par, W, A)
    print(f'  scanned {g} gap {gp}: chi2 min {np.nanmin(chi):.0f}', flush=True)
    return chi, W, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stacks', required=True); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'gasfit.json'))
    ap.add_argument('--gap', type=float, default=28.8); ap.add_argument('--k', type=float, default=0.0)
    ap.add_argument('--zs-scale', type=float, default=1.0); ap.add_argument('--sat', action='store_true')
    ap.add_argument('--hx', type=int, default=2); ap.add_argument('--hy', type=int, default=8)
    ap.add_argument('--raw-eta', action='store_true'); ap.add_argument('--source', default='air_hs')
    ap.add_argument('--groups', nargs='*', default=None); ap.add_argument('--nw', type=int, default=33)
    ap.add_argument('--na', type=int, default=14); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--zs-win-hi', type=float, default=None, help='arm: fit ZS stacks only up to this time [ns]')
    ap.add_argument('--bench-template', action='store_true', help='arm: the measured bench template instead of the fitted shaper')
    ap.add_argument('--gaps', type=float, nargs='*', default=[26.2, 27.0, 27.9, 28.8, 30.0],
                    help='drift gaps profiled [mm] (ws_spsv: slope x duration 26.2-27.0, registry 28.8, nominal 30)')
    a = ap.parse_args()
    global ARGS, ZTAB, RAWD
    ARGS = a; ZTAB = zs_table(a.stacks, a); RAWD = group_data(a.stacks, list(RAW_KEYS), a, ZTAB)
    res = dict(args=vars(a), groups={}, shaper={})
    gaps = a.gaps or [a.gap]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.jobs) as pool:
        # 1. per gap: the shaper on run71 RAW, iterated with its composition
        pars = dict(zip(gaps, pool.map(shaper_for_gap, gaps)))
        for gp, par in pars.items():
            res['shaper'][str(gp)] = None if par is None else [float(x) for x in par]
        # 2. every group x gap with that gap's fixed shaper; the gap is then profiled
        todo = [(g, gp, pars[gp]) for g in GROUPS if not a.groups or g in a.groups for gp in gaps]
        scans = dict(zip([(g, gp) for g, gp, _ in todo], pool.starmap(scan_task, todo)))
    for g, spec in GROUPS.items():
        if a.groups and g not in a.groups:
            continue
        data = group_data(a.stacks, spec['keys'], a, ZTAB)
        if not data:
            continue
        ndf = sum(int(((d['t'] >= WIN[0]) & (d['t'] <= WIN[1]) & (d['e'] < 1e5)).sum()) - 2 for d in data) - 3
        per_gap = {}
        for gp in gaps:
            chi, W, A = scans[(g, gp)]
            per_gap[gp] = (chi, W, A, Model(spec['gas'], a.source, gp, a.k, not a.raw_eta))
        cmin = min(np.nanmin(v[0]) for v in per_gap.values()); gbest = min(per_gap, key=lambda k: np.nanmin(per_gap[k][0]))
        chi, W, A, Mo = per_gap[gbest]
        S = summarise(chi, W, A, ndf)
        sc = S['scale']
        # profile over the gap: the 68 % region across every gap's grid
        wl, wh, al, ah, gl = [], [], [], [], []
        for gp, (c, Wg, Ag, _) in per_gap.items():
            ok = c <= cmin + 2.30 * sc
            if ok.any():
                wl.append(Wg[ok.any(1)].min()); wh.append(Wg[ok.any(1)].max()); al.append(Ag[ok.any(0)].min()); ah.append(Ag[ok.any(0)].max())
                gl.append(gp)
        S.update(gap=gbest, gap_68=[min(gl), max(gl)] if gl else None, water_68_gap=[float(min(wl)), float(max(wh))] if wl else None,
                 air_68_gap=[float(min(al)), float(max(ah))] if al else None,
                 chi2_by_gap={str(k): float(np.nanmin(v[0])) for k, v in per_gap.items()},
                 best_by_gap={str(k): summarise(v[0], v[1], v[2], ndf) for k, v in per_gap.items()})
        S['o2_68_gap'] = [x * 2095 for x in S['air_68_gap']] if S['air_68_gap'] else None
        best = (S['water'], S['air']); par = pars[gbest]; Mo = per_gap[gbest][3]
        per = []
        for d in data:
            c, n, mc, q = fit_stack(d['t'], d['y'], d['e'], Mo.current(best, Mo.E(d)), par, d['zsd'])
            c0, _, m0, _ = fit_stack(d['t'], d['y'], d['e'], Mo.current((best[0], 0.0), Mo.E(d)), par, d['zsd'])
            gq = Mo.G(best[0], best[1], Mo.E(d))
            per.append(dict(key=d['key'], view=d['view'], E=Mo.E(d), raw=d['raw'], n=d['n'], n_sat=d['n_sat'], chi2=c, npt=n,
                            chi2_noair=c0, v=gq['v'], etav=gq['etav'], t0=float(q[0]), y=d['y'].tolist(), e=d['e'].tolist(),
                            model=mc.tolist(), model_noair=m0.tolist()))
        for k in ('best_by_gap',):
            for kk in S[k].values():
                kk.pop('edge', None)
        S.update(gas=spec['gas'], transition=bool(spec.get('transition')), keys=spec['keys'], per_stack=per, grid_w=W.tolist(),
                 grid_a=A.tolist(), chi=chi.tolist())
        res['groups'][g] = S
        print(f'{g:18s} gap {gbest} water {S["water"]:.3f} {S["water_68_gap"]}  air {S["air"]:.3f} ({S["o2_ppm"]:.0f} ppm O2, '
              f'{S["o2_68_gap"]})  chi2 {S["chi2"]:.0f}/{S["ndf"]}  by gap {S["chi2_by_gap"]}  edge {S["edge"]}', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()

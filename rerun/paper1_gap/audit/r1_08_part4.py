"""r1_08_part4.py: what limits the prediction: split b (plane / long / short wavelength), velocity exponent, noise attenuation (IV), residual maps."""
import sys
from r1_lib2 import *
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import b_lib as BL  # noqa  (iv_slope, boot_cells use only numpy)
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
rng = np.random.default_rng(91)
NB = 100
J = {}
# --- projection of a K=12 surface onto the K=6 spline space (long wavelengths)
SPL6 = Spl(6)
gu, gw = np.linspace(*UD, 40), np.linspace(*WD, 36)
GU_, GW_ = np.meshgrid(gu, gw, indexing='ij')
B12g = SPL.basis(GU_.ravel(), GW_.ravel()); B6g = SPL6.basis(GU_.ravel(), GW_.ravel())
PROJ = np.linalg.pinv(B6g) @ B12g          # 36 x 144

def multifit(v, cols, extra=None):
    X = np.vstack([np.ones(len(v['w'])), v['tr']] + cols + (extra or [])).T
    c = CL.lsq_clip(X, v['w'])[0]
    return c[2:2 + len(cols)] / c[1], c

def boot_multi(v, make, names, nb=NB, extra=None):
    """make(rep or None, idx) -> list of regressor arrays at events idx"""
    n = len(v['w'])
    c0, _ = multifit(v, make(None, np.arange(n)), extra)
    bs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n); rep = int(rng.integers(0, 150))
        vv = {k: v[k][idx] for k in ('U', 'W', 'w', 'tr')}
        bs.append(multifit(vv, make(rep, idx))[0])
    bs = np.array(bs)
    return {nm: [float(c0[i]), float(bs[:, i].std())] for i, nm in enumerate(names)}

CASES = [('T', 'run1', 'run2'), ('T', 'run2', 'run1'), ('q', 'run1', 'run2'), ('q', 'run2', 'run1'), ('q', 'r625', 'r625')]
for kind, src, dst in CASES:
    name = f'{kind}_{src}'; fac = G0 / (1 + FLD[name]['beta']); beta = FLD[name]['beta']
    for p in 'xy':
        v = V[(dst, p)]; i = 0 if p == 'x' else 1
        Bd = SPL.basis(v['U'], v['W'], 1 - i, i) / 100
        Bl = (SPL6.basis(v['U'], v['W'], 1 - i, i) / 100) @ PROJ
        Bs = Bd - Bl
        Bv = SPL.basis(v['U'], v['W'])
        bun = BUNDLE625 if dst == 'r625' else BUNDLE
        for hw in ('h', 'perview'):
            gh = hgrad(v['U'], v['W'], hw, bun)[i]
            cf = lambda rep: (FLD[name]['c'] if rep is None else FLD[name]['cb'][rep])
            # (1) plane / rest of the gap gradient
            def mk1(rep, idx, gh=gh, Bd=Bd, cf=cf):
                gg = fac * Bd @ cf(rep)
                pl = plane_removed(v['U'], v['W'], gg)
                return [gh[idx], (gg - pl)[idx], pl[idx]]
            J[f'{name}->{dst}:{p}:{hw}:plane_split'] = boot_multi(v, mk1, ['a', 'b_plane', 'b_fine'])
            # (2) long (K=6 projection) / short wavelength part
            def mk2(rep, idx, gh=gh, Bl=Bl, Bs=Bs, cf=cf):
                return [gh[idx], (fac * Bl @ cf(rep))[idx], (fac * Bs @ cf(rep))[idx]]
            J[f'{name}->{dst}:{p}:{hw}:wl_split'] = boot_multi(v, mk2, ['a', 'b_long', 'b_short'])
            # (3) a, b and the velocity exponent kappa:  w = c0 + c1 tr (1 - kappa L) + c1 (a gh + b gg), L = delta g / g0
            def mk3(rep, idx, gh=gh, Bd=Bd, Bv=Bv, cf=cf):
                gg = fac * Bd @ cf(rep); L = fac * Bv @ cf(rep) / G0
                return [gh[idx], gg[idx], (v['tr'] * L)[idx]]
            # the third regressor multiplies c1 differently: tr L has coefficient -c1 kappa
            r3 = boot_multi(v, mk3, ['a', 'b', 'minus_kappa'])
            r3['kappa'] = [-r3['minus_kappa'][0], r3['minus_kappa'][1]]
            r3['beta_assumed'] = beta
            J[f'{name}->{dst}:{p}:{hw}:abkappa'] = r3
        r = J[f'{name}->{dst}:{p}:perview:plane_split']; r2 = J[f'{name}->{dst}:{p}:perview:wl_split']; r3 = J[f'{name}->{dst}:{p}:perview:abkappa']
        print(name, dst, p, 'plane/fine', {k: [round(x, 2) for x in a] for k, a in r.items()}, 'long/short', {k: [round(x, 2) for x in a] for k, a in r2.items()},
              'kappa', [round(x, 2) for x in r3['kappa']], 'beta', beta, flush=True)

# ------------------------------------------------ IV / noise attenuation at cell level (golden)
def cellpred(r, p, n, field, hw):
    v = V[(r, p)]; i = 0 if p == 'x' else 1
    gh = hgrad(v['U'], v['W'], hw)[i]
    gg = field.grad(v['U'], v['W'])[i]
    return cellmean(v['U'], v['W'], gh, n), cellmean(v['U'], v['W'], gg, n)

for kind in ('T', 'q'):
    for n in (6, 8):
        f1, f2 = Fld(f'{kind}_run1'), Fld(f'{kind}_run2')
        for p in 'xy':
            M1, A1, Ae1, _ = None, None, None, None
            res = {}
            data = {}
            for r in ('run1', 'run2'):
                v = V[(r, p)]; vg = CO.line(v['tr'], v['w'])[1]
                M = CL.cellmap(v['U'], v['W'], v['w'], v['tr'], n)
                data[r] = (M, M['w0'] / vg, M['w0e'] / vg)
            h1, g1 = cellpred('run1', p, n, f1, 'perview'); h2, g2 = cellpred('run2', p, n, f2, 'perview')
            # gap-gradient cell maps of each field evaluated on the SAME events (run 1's events for both) so that cells agree
            v = V[('run1', p)]; i = 0 if p == 'x' else 1
            ggA = cellmean(v['U'], v['W'], f1.grad(v['U'], v['W'])[i], n); ggB = cellmean(v['U'], v['W'], f2.grad(v['U'], v['W'])[i], n)
            ghA = cellmean(v['U'], v['W'], hgrad(v['U'], v['W'], 'perview')[i], n)
            M, A1, Ae1 = data['run1']; _, A2, Ae2 = data['run2']
            okc = np.isfinite(A1) & np.isfinite(A2) & np.isfinite(ggA) & np.isfinite(ggB)
            Uc, Wc = M['Uc'], M['Wc']
            def fine(x, w=None):
                X = np.vstack([np.ones(okc.sum()), Uc[okc], Wc[okc]]).T
                c = np.linalg.lstsq(X, x[okc], rcond=None)[0]
                return x[okc] - X @ c
            def stat(ix):
                a1, a2, gA, gB, hh = [fine(x)[ix] for x in (A1, A2, ggA, ggB, ghA)]
                # tilt of run 2 on gap of run 1, instrument = gap of run 2 (noise independent); and the reverse
                num = (np.cov(a2, gA)[0, 1] + np.cov(a1, gB)[0, 1]) / 2
                den = np.cov(gA, gB)[0, 1]
                ols = (np.cov(a2, gA)[0, 1] / np.var(gA, ddof=1) + np.cov(a1, gB)[0, 1] / np.var(gB, ddof=1)) / 2
                rel = den / np.sqrt(np.var(gA, ddof=1) * np.var(gB, ddof=1))    # reliability of the predictor
                return num / den, ols, rel
            N = okc.sum(); s0 = stat(np.arange(N))
            bs = np.array([stat(rng.integers(0, N, N)) for _ in range(1000)])
            J[f'IV:{kind}:n{n}:{p}'] = dict(b_iv=[float(s0[0]), float(bs[:, 0].std())], b_ols=[float(s0[1]), float(bs[:, 1].std())],
                                           reliability=[float(s0[2]), float(bs[:, 2].std())], N=int(N))
            print('IV', kind, n, p, J[f'IV:{kind}:n{n}:{p}'], flush=True)
json.dump(J, open(f'{OUT}/part4_split.json', 'w'), indent=1)

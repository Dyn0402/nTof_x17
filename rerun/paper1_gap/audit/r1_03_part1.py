"""r1_03_part1.py: gap fields of the five datasets (T golden, q_u90 golden and 6-25), bootstrap, comparisons. -> part1.json, fields.pkl"""
import sys
from r1_lib import *
K, LAM = 12, 0.1
NB = 150
BETA625 = 1.28
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
SETS = {}
for r in ('run1', 'run2'):
    m = D[r]['m']
    SETS[f'T_{r}'] = (m.U.values, m.W.values, np.log(m.T_amp.values), BETA_G)
for r in ('run1', 'run2', 'r625'):
    t = D[r]['tab']; ok = D[r]['qok']
    SETS[f'q_{r}'] = (t.U.values[ok], t.W.values[ok], np.log(D[r]['qamp'][ok]), BETA_G if r != 'r625' else BETA625)
# grid for the maps
GU = np.linspace(-1.65, 1.65, 23); GW = np.linspace(-1.45, 1.45, 21)
UU, WW = np.meshgrid(GU, GW, indexing='ij')
spl = Spl(K)
BG = spl.basis(UU.ravel(), WW.ravel())
rng = np.random.default_rng(7)
FLD = {}
for name, (U, W, y, beta) in SETS.items():
    B = spl.basis(U, W)
    y0 = y - y.mean()
    c0, keep = spl.fit(U, W, y0, LAM, B=B)
    cs = []
    for b in range(NB):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        cs.append(spl.fit(U, W, y0, LAM, wts=w, B=B)[0])
    FLD[name] = dict(c=c0, cb=np.array(cs), beta=beta, n=len(y), keep=float(keep.mean()), sd=float(y.std()))
    print(name, len(y), 'keep', keep.mean(), 'ln sd', y.std(), flush=True)
pickle.dump(FLD, open(f'{OUT}/fields.pkl', 'wb'))

def gmap(c, beta):
    return (G0 / (1 + beta)) * (BG @ c).reshape(UU.shape)

def lnmap(c):
    return (BG @ c).reshape(UU.shape)

def quad(Z):
    """quadratic surface fit on the grid: returns coefficients (c0, gu, gw, guu, gww, guw) and residual."""
    X = np.vstack([np.ones(Z.size), UU.ravel(), WW.ravel(), UU.ravel() ** 2 / 2, WW.ravel() ** 2 / 2, (UU * WW).ravel()]).T
    c = np.linalg.lstsq(X, Z.ravel(), rcond=None)[0]
    return c, (Z.ravel() - X @ c).reshape(Z.shape)

def stats(f):
    """dict of map statistics from a function name->map (callable on replicate index None = central)"""
    pass

J = {}
names = list(FLD)
# ------------------------------------------------ per dataset
CEN = {n: lnmap(FLD[n]['c']) for n in names}
BOOT = {n: np.array([lnmap(c) for c in FLD[n]['cb']]) for n in names}
for n in names:
    z = CEN[n] - CEN[n].mean(); b = BOOT[n] - BOOT[n].mean(axis=(1, 2), keepdims=True)
    beta = FLD[n]['beta']; fac = G0 / (1 + beta)
    qc, res = quad(z)
    qb = np.array([quad(bb)[0] for bb in b]); rb = np.array([quad(bb)[1].std() for bb in b])
    J[n] = dict(n=FLD[n]['n'], beta=beta, rms_lnamp=float(z.std()), rms_dg_mm=float(z.std() * fac),
                err_map_rms_mm=float(b.std(axis=0).mean() * fac),
                quad_mm={k: [float(qc[i] * fac), float(qb[:, i].std() * fac)] for i, k in enumerate(('c0', 'gu', 'gw', 'guu', 'gww', 'guw'))},
                fine_rms_mm=float(res.std() * fac), fine_rms_err=float(rb.std() * fac))
    print(n, json.dumps(J[n]), flush=True)

# ------------------------------------------------ comparisons: A vs B (slope of B on A, correlation), amplitude in ln units
def cmp(a, b, beta_a=None, beta_b=None, part='tot'):
    out = {}
    def f(za, zb):
        za = za - za.mean(); zb = zb - zb.mean()
        if part == 'fine':
            za = quad(za)[1]; zb = quad(zb)[1]
        else:
            pass
        r = np.corrcoef(za.ravel(), zb.ravel())[0, 1]
        s = np.sum(za * zb) / np.sum(za * za)
        return r, s
    r0, s0 = f(CEN[a], CEN[b])
    rr = np.array([f(BOOT[a][i], BOOT[b][(i * 7 + 3) % NB]) for i in range(NB)])
    # bootstrap: independent replicates of a and b
    return dict(r=float(r0), r_err=float(rr[:, 0].std()), slope=float(s0), slope_err=float(rr[:, 1].std()))

PAIRS = [('T_run1', 'T_run2'), ('q_run1', 'q_run2'), ('T_run1', 'q_run1'), ('T_run2', 'q_run2'),
         ('q_run1', 'q_r625'), ('q_run2', 'q_r625')]
J['cmp'] = {}
for a, b in PAIRS:
    for part in ('tot', 'fine'):
        J['cmp'][f'{a}|{b}|{part}'] = cmp(a, b, part=part)
        print(a, b, part, J['cmp'][f'{a}|{b}|{part}'], flush=True)
# golden pooled q vs r625, in ln units and in mm (golden beta .68, r625 beta 1.28 nominal)
# pooled golden T field = mean of run1 and run2 fields
CEN['T_gold'] = 0.5 * (CEN['T_run1'] + CEN['T_run2']); CEN['q_gold'] = 0.5 * (CEN['q_run1'] + CEN['q_run2'])
BOOT['T_gold'] = 0.5 * (BOOT['T_run1'] + BOOT['T_run2']); BOOT['q_gold'] = 0.5 * (BOOT['q_run1'] + BOOT['q_run2'])
for part in ('tot', 'fine'):
    J['cmp'][f'q_gold|q_r625|{part}'] = cmp('q_gold', 'q_r625', part=part)
    print('q_gold|q_r625', part, J['cmp'][f'q_gold|q_r625|{part}'])
# difference map 6-25 minus golden: quadratic parameters (ln units), per 100 mm
d = (CEN['q_r625'] - CEN['q_r625'].mean()) - (CEN['q_gold'] - CEN['q_gold'].mean())
qc, res = quad(d)
db = np.array([quad((BOOT['q_r625'][i] - BOOT['q_r625'][i].mean()) - (BOOT['q_gold'][i % NB] - BOOT['q_gold'][i % NB].mean()))[0] for i in range(NB)])
J['diff_625_gold_lnq'] = {k: [float(qc[i]), float(db[:, i].std())] for i, k in enumerate(('c0', 'gu', 'gw', 'guu', 'gww', 'guw'))}
J['diff_625_gold_lnq']['fine_rms'] = float(res.std())
print(J['diff_625_gold_lnq'])
json.dump(J, open(f'{OUT}/part1.json', 'w'), indent=1)
pickle.dump(dict(CEN=CEN, GU=GU, GW=GW, BOOTSD={n: BOOT[n].std(axis=0) for n in BOOT}), open(f'{OUT}/part1_maps.pkl', 'wb'))

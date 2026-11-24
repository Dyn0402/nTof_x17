"""r2_08_bd.py -- chamber D on the B-D cosmic lines (bd_lines.parquet, 2,793 lines, truth = B position + D position): does the timing-predicted tilt lower D's residual?
Slope of D in the line frame: view z (D x strips) = d_z/d_x = SGN_D x_tan_theta_D, view y = d_y/d_x = SGN_D y_tan_theta_D, SGN_D = -1 both (checked on the tables, run_145/149:
-0.931, -0.937 against x/y_tan_theta).  Train/test: event parity (as bd_map.py).  Models as r2_05 (k = 0/1, b nominal / free) with D's bench bow block (det7).
Cells 100 mm for the in-situ B-D map (bd_map.py)."""
import sys, os, json, pickle
os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
from r2_lib import *
import r2_04_compare as C4
import g_common as GC
from rerun.ntof.improve import angle_map as AM
FIELD = sys.argv[1] if len(sys.argv) > 1 else 'late'
EDG = np.arange(-200.0, 201.0, 100.0)
AM.EDGES, AM.CENTRES, AM.N_MIN = EDG, 0.5 * (EDG[1:] + EDG[:-1]), 60
BD = pd.read_parquet('/media/ucla/work/ntof_pass1/ana_r2/track_quality/bd_lines.parquet')
KEY = ['run', 'subrun', 'tag', 'event_id']
T = pd.read_parquet(f'{OUT}/tr_cosmic.parquet', columns=KEY + ['arm', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta'])
T = T[T.arm == 'D'].drop(columns='arm').drop_duplicates(KEY, keep=False)
M = BD.merge(T, on=KEY, how='inner')
print('B-D lines', len(BD), 'with D strip position', len(M))
SG = -1.0
bd = json.load(open(f'{C4.BUND}/D/bundle.json'))['bow']
c = C4.FL[('D', FIELD, 'all')]['c']
SPL = C4.SP
VIEWS = {'z': 'x', 'y': 'y'}          # line view -> D view
rng = np.random.default_rng(4)
NB = 150


def feats(M, p):
    U = np.clip((M.x_p0.to_numpy(float) + MID * M.x_tan_theta.to_numpy(float) - 200) / 100, -2, 2)
    W = np.clip((M.y_p0.to_numpy(float) + MID * M.y_tan_theta.to_numpy(float) - 200) / 100, -2, 2)
    AP = SG * C4.R._bow_poly(bd[f'a_{p}'], U, W)
    h = bd[f'h_{p}']
    H = SG * ((h['gu'] + h['kuu'] * U + h['kuw'] * W) / 100 if p == 'x' else (h['gw'] + h['kww'] * W + h['kuw'] * U) / 100)
    G = SG * C4.NOM * ((SPL.basis(U, W, 1, 0) @ c) / 100 if p == 'x' else (SPL.basis(U, W, 0, 1) @ c) / 100)
    return H, AP, G


def fit_lin(cols, y, clip=3.0):
    X = np.vstack(cols).T
    keep = np.ones(len(y), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        res = y - X @ coef
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < clip * sig
    return coef


rows = []
for lv, p in VIEWS.items():
    q = 'y' if lv == 'z' else 'z'
    j, r = M[f'j{lv}'].to_numpy(), M[f'res_{lv}'].to_numpy()
    ok = (M[f'res_{q}'].abs() < 0.10).to_numpy() & (np.abs(j) < 0.45) & (np.abs(r) < 0.15)
    H, AP, G = feats(M, p)
    xl, yl = M.x_local_d.to_numpy(), M.y_local_d.to_numpy()
    par = M.event_id.astype(int).to_numpy() % 2
    for tr in (0, 1):
        trm = ok & (par == tr)
        tem = ok & (par != tr)
        jt, rt = j[trm], r[trm]
        y_tr = rt + AP[trm]
        slope = r[tem] + j[tem]
        jj = j[tem]

        def ev(tag, corr):
            v = corr - jj
            b = [GC.mad(v[rng.integers(0, len(v), len(v))]) for _ in range(NB)]
            rows.append(dict(view=lv, train=tr, model=tag, sig=GC.mad(v), err=float(np.std(b)), n=int(tem.sum())))
        ev('none', slope)
        a_g, s_g, _ = AM.robust_line(jt, rt)
        ev('global', (slope - a_g) / (1 + s_g))
        mp = AM.fit_map(pd.DataFrame(dict(x=xl[trm], y=yl[trm], j=jt, r=rt, run=''))) | dict(centres=AM.CENTRES)
        ev('insitu 100mm B-D map', AM.correct(slope, xl[tem], yl[tem], mp))
        ev('insitu (AM.fit_map 100mm) const only', AM.correct(slope, xl[tem], yl[tem], dict(mp, a=np.full((4, 4), mp['a0']), s=np.full((4, 4), mp['s0']))))
        for kf in (0, 1):
            co = fit_lin([np.ones_like(jt), jt], y_tr - kf * H[trm] - G[trm])
            ev(f'tim_nom_k{kf}', (slope - (co[0] + kf * H[tem] + G[tem] - AP[tem])) / (1 + co[1]))
            co = fit_lin([np.ones_like(jt), jt], y_tr - kf * H[trm])
            ev(f'dish_only_k{kf}', (slope - (co[0] + kf * H[tem] - AP[tem])) / (1 + co[1]))
            co = fit_lin([np.ones_like(jt), jt, G[trm]], y_tr - kf * H[trm])
            ev(f'tim_fit_k{kf}', (slope - (co[0] + kf * H[tem] + co[2] * G[tem] - AP[tem])) / (1 + co[1]))
            rows.append(dict(view=lv, train=tr, model=f'coef_tim_fit_k{kf}', sig=float(co[2]), err=np.nan, n=0))
        co = fit_lin([np.ones_like(jt), jt, H[trm], G[trm]], y_tr)
        ev('tim_fit_kb', (slope - (co[0] + co[2] * H[tem] + co[3] * G[tem] - AP[tem])) / (1 + co[1]))
        rows.append(dict(view=lv, train=tr, model='coef_k', sig=float(co[2]), err=np.nan, n=0)); rows.append(dict(view=lv, train=tr, model='coef_b', sig=float(co[3]), err=np.nan, n=0))
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/bd_direct_{FIELD}.csv', index=False)
print(R.groupby(['view', 'model'], sort=False)[['sig', 'err', 'n']].mean().round(4).to_string())
print('n clean per view', {lv: int(((M[f"res_{'y' if lv=='z' else 'z'}"].abs() < 0.1) & (M[f"j{lv}"].abs() < 0.45) & (M[f"res_{lv}"].abs() < 0.15)).sum()) for lv in 'zy'})

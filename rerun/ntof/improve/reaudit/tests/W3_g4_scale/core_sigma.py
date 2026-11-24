"""Core sigma of the pointing residual (pm.centre definition: 1.4826 x median |r - m| over |r - m| < 0.3, m = windowed
median) in the band 0.18 <= |e| <= 0.44, |y| <= 125: Geant4 truth (level a), reco (b), against the data (T5)."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17'); sys.path.insert(0, '/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
import w3lib as L
warnings.simplefilter('ignore')
from ntof_cosmics.g4_digi import analyse as AN
AN.OUT = L.DIGI
def core(r):
    m = L.wmedian(r); k = np.abs(r - m) < 0.3
    return m, 1.4826 * np.median(np.abs(r[k] - m)), int(len(r))
rows = []
for lib, src in (('nose', (0.46, 0.32)), ('ear2_L19p5', (0.35, 1.11))):
    R = L.add_global(L.load_reduced(lib))
    S = R[R.full & (R.v_mesh.abs() <= 125) & (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5) & ((R.A != 'D') | R.wall)]
    for arm in 'ACD':
        g = S[S.A == arm]; e = L.expected(g, arm, src); b = (np.abs(e) >= 0.18) & (np.abs(e) <= 0.44)
        m, s, n = core((g.sh.to_numpy() - e)[b]); rows.append(dict(sample=f'G4 {lib} truth', arm=arm, centre=m, core_sigma=s, n=n))
    for arm in 'AC':
        R2 = AN.load(f'g4_{arm}_{lib}_pass1'); R2['A'] = arm
        h, n_, s_ = L.to_global(arm, R2.xl, np.full(len(R2), 30.228), R2.x_tan_theta)
        R2['h'], R2['n'] = h, n_
        g = R2[R2.x_ok & R2.y_ok & (np.abs(s_) < 0.5) & (R2.tan_v_reco.abs() < 0.5) & (R2.vl.abs() <= 125)]
        e = L.expected(g, arm, src); b = (np.abs(e) >= 0.18) & (np.abs(e) <= 0.44)
        sr = (L.FR[arm]['s'] * g.x_tan_theta).to_numpy()
        m, s, n = core((sr - e)[b]); rows.append(dict(sample=f'G4 {lib} reco', arm=arm, centre=m, core_sigma=s, n=n))
T = pd.read_csv('/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map/hmap_compare.csv')
for r in T[T.cell == 50].itertuples():
    rows.append(dict(sample='data (T5 core, nomap / map)', arm=r.arm, core_sigma=r.core_before, centre=r.core_after, n=r.n_tracks_valid))
D = pd.DataFrame(rows); D.to_csv('core_sigma.csv', index=False); print(D.round(4).to_string())

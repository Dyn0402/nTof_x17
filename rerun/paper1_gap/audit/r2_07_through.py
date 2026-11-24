"""r2_07_through.py -- through-going (muon-like) sample inside beam runs: late (>= 20 ms) beam-run events with one gated track in A and one in C
that lie on a straight A-C line (sep < 60 mm, each slope within 0.10 of the joined slope, both views) and whose line misses the capsule axis by > 100 mm.
Adds the timing columns -> tr_through.parquet (A and C rows)."""
import sys
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17')
from r2_lib import *
import pyarrow.dataset as ds
from sept26_prelim_analysis import pass1_tracks as P1
KEY = ['run', 'subrun', 'tag', 'event_id']
cols = KEY + ['arm', 'gated', 'censored_window', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_q_u90', 'y_q_u90',
              'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'chi2dof_x', 'chi2dof_y', 'x_quality_ok', 'y_quality_ok', 'drift_railed', 'n_cand_x', 'n_cand_y',
              't_since_flash_ns', 'x_local', 'y_local', 'tanx', 'tany']
runs = sorted({f.name.split('_stat')[0].split('_cosbounce')[0].replace('tracks_', '') for f in P1.files()})
out = []
dca = lambda pa, da, pc, dc: np.abs(np.einsum('ij,ij->i', pc - pa, np.cross(da, dc))) / np.linalg.norm(np.cross(da, dc), axis=1)
for r in runs:
    d = P1.read(cols, runs=[r], arms=('A', 'C'), filter=(ds.field('t_since_flash_ns') >= 2e7) & ds.field('gated'))
    if len(d) == 0: continue
    d = d[~d.censored_window.fillna(False).astype(bool)]
    d = d[d.groupby(KEY + ['arm']).p0_x.transform('size') == 1]
    A, C = d[d.arm == 'A'].drop(columns='arm'), d[d.arm == 'C'].drop(columns='arm')
    M = A.merge(C, on=KEY, suffixes=('_a', '_c'))
    if len(M) == 0: continue
    P = lambda s: M[[f'p0_x_{s}', f'p0_y_{s}', f'p0_z_{s}']].to_numpy(float)
    D = lambda s: M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)
    M['sep'] = dca(P('a'), D('a'), P('c'), D('c'))
    jz = M.p0_z_c - M.p0_z_a
    jx, jy = (M.p0_x_c - M.p0_x_a) / jz, (M.p0_y_c - M.p0_y_a) / jz
    ok = M.sep < 60
    for s in 'ac':
        ok &= ((M[f'd_x_{s}'] / M[f'd_z_{s}'] - jx).abs() < 0.10) & ((M[f'd_y_{s}'] / M[f'd_z_{s}'] - jy).abs() < 0.10)
    # distance of the joined line from the capsule axis at z = -4.5 (x = -10.9, y free: use x and y = -30)
    z0 = -4.5
    t = (z0 - M.p0_z_a) / jz
    xl, yl = M.p0_x_a + t * (M.p0_x_c - M.p0_x_a), M.p0_y_a + t * (M.p0_y_c - M.p0_y_a)
    ok &= np.hypot(xl + 10.9, yl + 30.0) > 100
    M = M[ok]
    if len(M) == 0: continue
    M['jx'], M['jy'] = jx[ok], jy[ok]
    out.append(M); print(r, len(M), flush=True)
M = pd.concat(out, ignore_index=True)
rows = []
for s, arm in (('a', 'A'), ('c', 'C')):
    cc = {c[:-2]: M[c] for c in M.columns if c.endswith(f'_{s}')}
    q = pd.DataFrame(cc); q['arm'] = arm
    for k in KEY: q[k] = M[k]
    rows.append(q)
T = pd.concat(rows, ignore_index=True)
T.to_parquet(f'{OUT}/tr_through.parquet', index=False)
print(len(M), 'lines', T.groupby('arm').size().to_dict())

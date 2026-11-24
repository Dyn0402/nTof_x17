"""Shared preparation for T9: per (chamber, view) long tables of residual + predictors, cosmics and beam."""
from pathlib import Path
import numpy as np, pandas as pd

D = Path('/media/ucla/work/claude-jobs/eb8200b3/T9_track_timing')
EDGES = np.arange(-200.0, 201.0, 50.0)
SOURCE_XZ = (-10.9, -4.5)          # pass1_tracks.SOURCE_XZ_MM
FTST_NS = 10.0
PRED = ['dTend', 'dT90', 'dT50', 'du50', 'du90', 'duend', 'dt0xy', 'dtc', 'dt0pin', 'dt0pin_o', 'lnbal', 'nstr',
        'chi2', 'lnq']
DESC = {'dTend': 'own q_uend / local median - 1', 'dT90': 'own q_u90 / local median - 1',
        'dT50': 'own q_u50 / local median - 1', 'du50': 'q_u50 x - y [ns]', 'du90': 'q_u90 x - y [ns]',
        'duend': 'q_uend x - y [ns]', 'dt0xy': 't0 x - y (+10 ns ftst) [ns]', 'dtc': 'dtc_ns [ns]',
        'dt0pin': 'own t0 - pin [ns]', 'dt0pin_o': 'other view t0 - pin [ns]', 'lnbal': 'ln(q_x/q_y)',
        'nstr': 'own n_strips', 'chi2': 'own chi2/dof (log)', 'lnq': 'own ln q_sum'}


def cell(x, y):
    nc = len(EDGES) - 1
    ix = np.clip(np.digitize(x, EDGES) - 1, 0, nc - 1)
    iy = np.clip(np.digitize(y, EDGES) - 1, 0, nc - 1)
    return ix * nc + iy


def rel_to_local(u, run, c, nmin=30):
    """u / median(u | run, 50 mm cell) - 1; cells with < nmin tracks in that run use cell median x run factor."""
    df = pd.DataFrame({'u': u, 'run': run, 'c': c})
    med_rc = df.groupby(['run', 'c']).u.transform('median')
    n_rc = df.groupby(['run', 'c']).u.transform('size')
    med_c = df.groupby('c').u.transform('median')
    fac = (df.u / med_c).groupby(df.run).transform('median')
    exp = np.where(n_rc >= nmin, med_rc, med_c * fac)
    return df.u.to_numpy() / exp - 1.0


def features(t: pd.DataFrame, p: str, run, rel_ref=None) -> pd.DataFrame:
    """t: columns x_*/y_* for one chamber.  p: the view.  Returns predictor frame."""
    q = 'y' if p == 'x' else 'x'
    c = cell(t['x_local'].to_numpy(), t['y_local'].to_numpy())
    F = pd.DataFrame(index=t.index)
    F['dTend'] = rel_to_local(t[f'{p}_q_uend'].to_numpy(float), run, c)
    F['dT90'] = rel_to_local(t[f'{p}_q_u90'].to_numpy(float), run, c)
    F['dT50'] = rel_to_local(t[f'{p}_q_u50'].to_numpy(float), run, c)
    F['du50'] = t['x_q_u50'] - t['y_q_u50']
    F['du90'] = t['x_q_u90'] - t['y_q_u90']
    F['duend'] = t['x_q_uend'] - t['y_q_uend']
    F['dt0xy'] = (t['x_t0'] + FTST_NS * t['x_ftst']) - (t['y_t0'] + FTST_NS * t['y_ftst'])
    F['dtc'] = t['dtc_ns']
    F['dt0pin'] = t[f'{p}_dt0']
    F['dt0pin_o'] = t[f'{q}_dt0']
    F['lnbal'] = np.log(t['x_q_sum'] / t['y_q_sum'])
    F['nstr'] = t[f'{p}_n_strips'].astype(float)
    F['chi2'] = np.log(t[f'chi2dof_{p}'].clip(lower=1e-3))
    F['lnq'] = np.log(t[f'{p}_q_sum'].clip(lower=1e-6))
    F['edge'] = np.maximum(t['x_local'].abs(), t['y_local'].abs())
    F['x_local'] = t['x_local']
    F['y_local'] = t['y_local']
    F['cell'] = c
    return F


def sub(M, s):
    """The chamber-s columns of the joined table, without suffix."""
    cols = {c: c[:-2] for c in M.columns if c.endswith('_' + s)}
    return M[list(cols)].rename(columns=cols)


def cosmic_long():
    M = pd.read_parquet(D / 'joined.parquet')
    rn = M.run.str.replace('run_', '').astype(int).to_numpy()
    out = {}
    for s, o, arm in (('a', 'c', 'A'), ('c', 'a', 'C')):
        t, to = sub(M, s), sub(M, o)
        for p in 'xy':
            q = 'y' if p == 'x' else 'x'
            F = features(t, p, M.run.to_numpy())
            F['run'] = rn
            F['j'] = M[f'j{p}']
            F['jq'] = M[f'j{q}']
            F['s'] = t[f'dm_{p}'] / t['dm_z']
            F['s_nomap'] = t[f'd_{p}'] / t['d_z']
            F['r'] = F.s - F.j
            F['rq'] = t[f'dm_{q}'] / t['dm_z'] - M[f'j{q}']
            F['ro'] = to[f'dm_{p}'] / to['dm_z'] - F.j
            F['roq'] = to[f'dm_{q}'] / to['dm_z'] - M[f'j{q}']
            F['clean'] = ((F.ro.abs() < 0.10) & (F.rq.abs() < 0.10) & (F.j.abs() < 0.45) & (F.r.abs() < 0.15))
            F['loose'] = (F.j.abs() < 0.45)
            out[(arm, p)] = F
    return out


def beam_long(path=D / 'beam.parquet'):
    cols = ['run', 'arm', 'x_local', 'y_local', 'p0_x', 'p0_z', 'd_x', 'd_z', 'x_q_uend', 'y_q_uend', 'x_q_u90',
            'y_q_u90', 'x_q_u50', 'y_q_u50', 'x_t0', 'y_t0', 'x_ftst', 'y_ftst', 'dtc_ns', 'x_dt0', 'y_dt0',
            'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'chi2dof_x', 'chi2dof_y', 't0_railed',
            'low_charge', 'xy_time_ok', 'xy_balance_ok', 'x_split_child', 'y_split_child', 'n_trk_arm']
    B = pd.read_parquet(path, columns=cols)
    clean = (~B.t0_railed.fillna(False).astype(bool) & ~B.low_charge.fillna(False).astype(bool)
             & B.xy_time_ok.fillna(False).astype(bool) & B.xy_balance_ok.fillna(False).astype(bool)
             & ~B.x_split_child.fillna(False).astype(bool) & ~B.y_split_child.fillna(False).astype(bool)
             & (B.n_trk_arm == 1))
    B = B[clean].reset_index(drop=True)
    out = {}
    for arm, g in B.groupby('arm'):
        g = g.reset_index(drop=True)
        F = features(g, 'x', g.run.to_numpy())
        F['run'] = g.run.str.replace('run_', '').astype(int).to_numpy()
        F['s'] = g.d_x / g.d_z
        F['p0_x'], F['p0_z'] = g.p0_x, g.p0_z
        F['s_exp'] = (g.p0_x - SOURCE_XZ[0]) / (g.p0_z - SOURCE_XZ[1])
        F['r'] = F.s - F.s_exp
        out[arm] = F
    return out


def miss_mm(F, s):
    """signed miss at the source plane z = z_s for slope s."""
    return F.p0_x.to_numpy() + (SOURCE_XZ[1] - F.p0_z.to_numpy()) * s - SOURCE_XZ[0]

"""Does a census-noisy (unmasked) channel under the window widen ln(qx/qy)?  M3-matched, no censored strip, unsaturated."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f15_charge_balance as F
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CEN = {'det2': 'det2_o22_long_det2', 'det3': 'det3_sat_det3', 'det4': 'det4_g_det4', 'det6': 'det6_g_det6_long', 'det7': 'det7_g_det7_long'}
root = os.path.expanduser('~/x17/cosmic_bench/model_audit')
out = {}
for det in F.DETS:
    df, ref, b = F.load(root, det)
    cen = json.load(open(f'{REPO}/rerun/channel_layer/{CEN[det]}.json'))
    both = df[df.x_ok & df.y_ok & (df.x_q_sum > 0) & (df.y_q_sum > 0)].copy()
    both['lr'] = np.log(both.x_q_sum / both.y_q_sum)
    idx = both.index.intersection(ref['x'].index).intersection(ref['y'].index)
    m = both.loc[idx]
    m = m[(m.x_n_flagged_strips == 0) & (m.y_n_flagged_strips == 0)]
    sat = np.zeros(len(m), bool)
    for p in 'xy':
        sat |= m[f'{p}_dof'].to_numpy() < 0.98 * m[f'{p}_n_strips'].to_numpy() * F.NSAMP
    m = m[~sat]
    near = np.zeros(len(m), bool)
    for p in 'xy':
        ch = np.array(cen[p]['noisy'], int)
        pos = 49.92 * (ch // 64) + (63 - ch % 64) * 0.78          # bench strip map (every June row, checked)
        p0 = m[f'{p}_p0'].to_numpy()
        if len(pos):
            near |= (np.abs(p0[:, None] - pos[None, :]) < 5.0).any(1)
    out[det] = dict(n=len(m), frac_noisy_near=float(near.mean()), w_noisy=F.rsig(m.lr[near]), w_clean=F.rsig(m.lr[~near]))
    # what else the residual follows: track inclination, column length and window size difference between views
    t = np.hypot(ref['x'].loc[m.index], ref['y'].loc[m.index]).to_numpy()
    out[det]['w_by_tan'] = [F.rsig(m.lr[(t >= lo) & (t < hi)]) for lo, hi in ((0, .1), (.1, .25), (.25, .5), (.5, 9))]
    out[det]['corr_uend_diff'] = float(np.corrcoef(m.lr, (m.x_q_uend - m.y_q_uend).to_numpy())[0, 1])
    out[det]['corr_nstrip_diff'] = float(np.corrcoef(m.lr, (m.x_n_strips - m.y_n_strips).to_numpy())[0, 1])
    out[det]['corr_chi2dof_diff'] = float(np.corrcoef(m.lr, np.log(m.x_chi2 / m.x_dof) - np.log(m.y_chi2 / m.y_dof))[0, 1])
    print(det, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in out[det].items()})
json.dump(out, open(os.path.join(HERE, 'noisy_check.json'), 'w'), indent=1)

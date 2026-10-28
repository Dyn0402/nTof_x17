#!/usr/bin/env python3
"""Summarise joint_review rows: the shared-profile x-y fit as a pairing statistic, and its angle resolution."""
import sys, json, pickle
import numpy as np

R = pickle.load(open(sys.argv[1], 'rb'))
right = [r for r in R if r['right']]
wrong = [r for r in R if not r['right']]


def s68(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(np.abs(x), 68)) if len(x) else np.nan


def stat(r):
    # cost of tying t0 and sharing the profile, in units of the per-dof chi2 of the untied fits
    sc = r['chi2_sep_at_prod'] / max(r['dof'], 1)
    return (r['chi2_joint'] - r['chi2_sep_at_prod']) / sc


d_right = np.array([stat(r) for r in right])
d_wrong = np.array([stat(r) for r in wrong])
# relative version
rel_right = np.array([r['chi2_joint'] / r['chi2_sep_at_prod'] - 1 for r in right])
rel_wrong = np.array([r['chi2_joint'] / r['chi2_sep_at_prod'] - 1 for r in wrong])
cut95 = np.percentile(d_right, 95)
cut90 = np.percentile(d_right, 90)
# AUC by rank
allv = np.concatenate([d_right, d_wrong]); lab = np.r_[np.zeros(len(d_right)), np.ones(len(d_wrong))]
order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
auc = (ranks[lab == 1].sum() - len(d_wrong) * (len(d_wrong) + 1) / 2) / (len(d_wrong) * len(d_right))

out = dict(
    n_right=len(right), n_wrong=len(wrong),
    time_median_s=float(np.median([r['time'] for r in R])),
    dchi2_over_scale=dict(right_median=float(np.median(d_right)), right_p90=float(cut90), right_p95=float(cut95),
                          wrong_median=float(np.median(d_wrong)), wrong_p10=float(np.percentile(d_wrong, 10)),
                          wrong_above_right_p95_pct=float(100 * np.mean(d_wrong > cut95)),
                          wrong_above_right_p90_pct=float(100 * np.mean(d_wrong > cut90)),
                          right_negative_pct=float(100 * np.mean(d_right < 0)),
                          auc=float(auc)),
    rel_cost=dict(right_median_pct=float(100 * np.median(rel_right)), wrong_median_pct=float(100 * np.median(rel_wrong))),
    alpha=dict(right_median=float(np.median([r['alpha'] for r in right])),
               right_rsig=float(0.5 * np.diff(np.percentile([r['alpha'] for r in right], [16, 84]))[0]),
               wrong_rsig=float(0.5 * np.diff(np.percentile([r['alpha'] for r in wrong], [16, 84]))[0])),
)
# angle resolution of the joint fit vs production, right pairs, with a ladder scale fitted per view
for v, key_w, key_pw, key_tr in (('x', 'wx', 'prod_wx', 'tan_ref_x'), ('y', 'wy', 'prod_wy', 'tan_ref_y')):
    tr = np.array([r[key_tr] for r in right])
    res = {}
    for arm, kw in (('joint', key_w), ('prod', key_pw)):
        w = np.array([r[kw] for r in right]) * 1e3
        sel = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
        vl = np.median(w[sel] / tr[sel])
        d = np.degrees(np.arctan(w / vl)) - np.degrees(np.arctan(tr)); d -= np.median(d)
        keep = np.abs(tr) < 0.6
        res[arm] = dict(v_ladder=float(vl), s68_deg=s68(d[keep]), far5_pct=float(100 * np.mean(np.abs(d[keep]) > 5)))
    # near-normal subset (|tan_ref| < 0.08 in THIS view): where timing carries no slope information
    nn = np.abs(tr) < 0.08
    for arm, kw in (('joint', key_w), ('prod', key_pw)):
        w = np.array([r[kw] for r in right]) * 1e3
        d = np.degrees(np.arctan(w / res[arm]['v_ladder'])) - np.degrees(np.arctan(tr)); d -= np.median(d)
        res[arm]['s68_nearnormal_deg'] = s68(d[nn]); res[arm]['n_nearnormal'] = int(nn.sum())
    dw = np.array([abs(r[key_w] - r[key_pw]) * 1e3 for r in right])
    res['same_w_1umns_pct'] = float(100 * np.mean(dw < 1.0))
    out[f'angles_{v}'] = res

json.dump(out, open(sys.argv[1].replace('.pkl', '_summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

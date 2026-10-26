#!/usr/bin/env python3
"""fleet_tables.py [--root ~/x17/cosmic_bench/model_audit] -- one consistent table across the five golden rows.

Inputs per chamber (pulled from EOS Analysis_rr1/<row>/wft/): final_physmask/{events.parquet, events.meta.json,
angles/per_event.npz, angles/angular_resolution.json} and the promoted calib_bundle/bundle.json.  Per view: angle
resolution (s68, robust sigma, bias) overall / head-on / inclined, far fraction, the reco ladder (w vs M3 tan, robust
slope = implied v) against the bundle v and the timing v, implied-v flatness, pull widths of the per-event tan error,
chi2/dof, t0_err health, candidate / competing-cluster statistics, x-y t0 coincidence and charge balance.
Writes <root>/fleet_tables.json and <root>/fleet_tables.md.
"""
import argparse, json, os
import numpy as np, pandas as pd

TIMING_V = {'det2': 39.3, 'det3': 35.8, 'det4': 33.0, 'det6': 24.17, 'det7': 36.8}   # C17/C21 (det6 at 28 mm)
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def rsig(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def slope(tr, w, lo=0.10, hi=0.45):
    """robust ladder slope dw/dtan [um/ns] over 0.10 < |tan| < 0.45 (median of w/tan after removing the median offset)."""
    m = np.isfinite(tr) & np.isfinite(w) & (np.abs(tr) > lo) & (np.abs(tr) < hi)
    if m.sum() < 30:
        return np.nan, np.nan
    c = np.median(w[np.abs(tr) < 0.05]) if (np.abs(tr) < 0.05).sum() > 20 else 0.0
    r = (w[m] - c) / tr[m]
    return float(np.median(r)), float(1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(m.sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.path.expanduser('~/x17/cosmic_bench/model_audit'))
    a = ap.parse_args(); out = {}; rows = []
    for det in DETS:
        d = os.path.join(a.root, det)
        b = json.load(open(f'{d}/calib_bundle/bundle.json'))
        meta = json.load(open(f'{d}/final_physmask/events.meta.json'))
        z = np.load(f'{d}/final_physmask/angles/per_event.npz')
        ar = json.load(open(f'{d}/final_physmask/angles/angular_resolution.json'))
        df = pd.read_parquet(f'{d}/final_physmask/events.parquet').drop_duplicates('event_id', keep=False).set_index('event_id')
        v = float(b['v_drift']); dt_xy = {int(k): float(x) for k, x in b['dt_xy'].items()}
        rec = dict(v_bundle=v, v_timing=TIMING_V[det], K=b['n_depth_bins'], t0_prior_sigma=b['t0_prior_sigma'],
                   t0_prior_in_table=meta['t0_prior'], w0=b['w0'], kw=b['kw'], dead={p: len(c) for p, c in b['dead'].items()},
                   hyper={k: b['hyper'][k] for k in b['hyper'] if k.startswith(('sigma_p0', 'Dp', 'rc_'))},
                   conditions=b['conditions'], w0kw_in_table=meta['angle_constants'], n_events=int(len(df)),
                   dead_seed=meta['selection'].get('dead_seed'), views={})
        for p in 'xy':
            eid = z[f'{p}_eid']; tr = z[f'{p}_tan_ref']; w = z[f'{p}_w'] * 1e3; tf_tab = z[f'{p}_tan_fit']
            # tan on the PROMOTED constants (the table carries the pre-finalize w0/kw)
            tf = (w - b['w0'][p]) / (b['kw'][p] * v)
            dth = np.degrees(np.arctan(tf) - np.arctan(tr)); dth_tab = np.degrees(np.arctan(tf_tab) - np.arctan(tr))
            far = np.abs(dth) > 5; at = np.abs(tr)
            sub = df.loc[eid]
            terr = sub[f'{p}_tan_err'].to_numpy(); chi = sub[f'{p}_chi2'].to_numpy(); dof = sub[f'{p}_dof'].to_numpy().astype(float)
            pull = (tf - tr) / terr
            cls = {'all': at < 9, 'headon': at < 0.1, 'inclined': at >= 0.1, 'steep': at >= 0.3}
            V = dict(n=int(len(eid)), far_pct=float(100 * far.mean()))
            for c, m in cls.items():
                mm = m & ~far
                V[c] = dict(n=int(m.sum()), s68=float(np.percentile(np.abs(dth[mm] - np.median(dth[mm])), 68)) if mm.sum() > 10 else np.nan,
                            rsig=rsig(dth[mm]), bias=float(np.median(dth[mm])) if mm.sum() else np.nan, far_pct=float(100 * far[m].mean()),
                            pull_rsig=rsig(pull[mm]), pull_rsig_notfloor=rsig(pull[mm & (terr > 0.0185)]),
                            frac_err_at_floor=float(np.mean(terr[m] <= 0.0185)))
            s, se = slope(tr, w)
            V['ladder_v'] = s; V['ladder_v_err'] = se; V['ladder_over_timing'] = s / TIMING_V[det]
            V['ladder_over_bundle_kw'] = s / (v * b['kw'][p])
            V['implied_v_bins'] = ar['planes'][p]['implied_v']; V['implied_v_spread'] = ar['planes'][p]['implied_v_spread']
            V['bias_tab_vs_promoted_deg'] = float(np.median(dth_tab) - np.median(dth))
            cd = chi / np.maximum(dof, 1)
            V['chi2dof'] = dict(p10=float(np.percentile(cd, 10)), p50=float(np.percentile(cd, 50)), p90=float(np.percentile(cd, 90)))
            te = sub[f'{p}_t0_err'].to_numpy()
            V['t0_err'] = dict(nan_frac=float(np.mean(~np.isfinite(te))), p50=float(np.nanmedian(te)), p90=float(np.nanpercentile(te, 90)))
            pe = sub[f'{p}_p0_err'].to_numpy(); V['p0_err_p50'] = float(np.nanmedian(pe)); V['p0_err_at_floor_frac'] = float(np.mean(pe <= 0.331))
            ue = sub[f'{p}_q_uend'].to_numpy()
            V['q_uend'] = dict(p10=float(np.nanpercentile(ue, 10)), p50=float(np.nanpercentile(ue, 50)), p90=float(np.nanpercentile(ue, 90)),
                               implausible_frac=float(np.mean(~((ue >= 250) & (ue <= 1100)))))
            V['T_from_uend_p50_over_v_ns'] = float(np.nanpercentile(ue, 50))
            V['gap_from_uend_mm'] = float(np.nanpercentile(ue, 50) * TIMING_V[det] * 1e-3)
            V['n_candidates_gt1_frac'] = float(np.mean(sub[f'{p}_n_candidates'] > 1))
            V['n_dropped_gt2_frac'] = float(np.mean(sub[f'{p}_n_dropped'] > 2))
            V['sat_frac_proxy_dof_lt_strips_x_nsamp'] = float(np.mean(dof < 0.98 * sub[f'{p}_n_strips'].to_numpy() * 32))
            V['n_strips_p50'] = float(np.median(sub[f'{p}_n_strips']))
            rec['views'][p] = V
        # event-level: x-y coincidence and charge balance on events with both views
        both = df[df.x_ok & df.y_ok]
        dtx = both.x_t0 - both.y_t0
        fd = (both.x_ftst - both.y_ftst).astype(int)
        pred = np.array([dt_xy.get(int(f), -18.8) for f in fd])
        res = dtx.to_numpy() - pred
        rec['xy'] = dict(n_both=int(len(both)), dt_resid_median=float(np.median(res)), dt_resid_rsig=rsig(res),
                         frac_outside_120ns=float(np.mean(np.abs(res) > 120)),
                         ln_qx_qy_median=float(np.median(np.log(both.x_q_sum / both.y_q_sum))),
                         ln_qx_qy_rsig=rsig(np.log(both.x_q_sum / both.y_q_sum)),
                         n_tracks=dict(df.n_tracks.value_counts().sort_index().astype(int).to_dict()),
                         frac_sigma_profile_unknown='not in table')
        out[det] = rec
        for p in 'xy':
            V = rec['views'][p]
            rows.append(f"| {det} {p} | {V['n']} | {V['all']['s68']:.3f} | {V['headon']['s68']:.3f} | {V['inclined']['s68']:.3f} | {V['all']['bias']:+.3f} | "
                        f"{V['far_pct']:.2f} | {V['ladder_v']:.2f}±{V['ladder_v_err']:.2f} | {TIMING_V[det]} | {v * b['kw'][p]:.2f} | "
                        f"{b['w0'][p]:+.3f} | {b['kw'][p]:.3f} | {V['all']['pull_rsig']:.2f} | {V['all']['frac_err_at_floor']:.2f} | "
                        f"{V['chi2dof']['p50']:.0f} | {V['t0_err']['nan_frac']:.3f} | {V['q_uend']['p50']:.0f} | {V['n_dropped_gt2_frac']:.3f} |")
    hdr = ('| det view | n | s68 all | s68 head-on | s68 incl | bias | far % | ladder v | timing v | kw·v | w0 | kw | pull rσ | err@floor | χ²/dof p50 | t0_err nan | uend p50 | ndrop>2 |\n'
           '|' + '---|' * 18)
    md = hdr + '\n' + '\n'.join(rows) + '\n'
    xy = ['| det | n both | Δt0 resid med | rσ | >120 ns | ln qx/qy med | rσ | t0 prior σ (bundle / table) |', '|' + '---|' * 8]
    for det in DETS:
        r = out[det]; X = r['xy']
        xy.append(f"| {det} | {X['n_both']} | {X['dt_resid_median']:+.1f} | {X['dt_resid_rsig']:.1f} | {X['frac_outside_120ns']:.3f} | "
                  f"{X['ln_qx_qy_median']:+.3f} | {X['ln_qx_qy_rsig']:.3f} | {r['t0_prior_sigma']} / {r['t0_prior_in_table']['sigma']} |")
    md += '\n' + '\n'.join(xy) + '\n'
    json.dump(out, open(os.path.join(a.root, 'fleet_tables.json'), 'w'), indent=1, default=float)
    open(os.path.join(a.root, 'fleet_tables.md'), 'w').write(md)
    print(md)


if __name__ == '__main__':
    main()

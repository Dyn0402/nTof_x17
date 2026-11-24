#!/usr/bin/env python3
"""T7_stability analysis: stability table, outliers, steps, specific checks, figures.

Reads cache/ from build.py.  Writes stability.parquet, outliers.csv, steps.csv, checks_*.csv, fig_*.png.
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path[:0] = [str(REPO)]
C = HERE / 'cache'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'figure.dpi': 130})

# ------------------------------------------------------------------ table
S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(C / 'stats/*.parquet')))], ignore_index=True)
reg = pd.read_csv(REPO / 'rerun/ntof/ntof_registry.csv')
reg = reg[reg.excluded != True]
reg['run'] = 'run_' + reg.run.astype(str)
ri = reg.groupby(['run', 'subrun']).agg(t_local=('t_local', 'min'), lat_epoch=('lat_epoch', 'first'),
                                        access_epoch=('access_epoch', 'first'), mask_era=('mask_era', 'first'),
                                        a_conn8_dead=('a_x_conn8_dead', 'first'), cm_x3=('a_cm_x3', 'first'),
                                        n_rows=('tag', 'size')).reset_index()
S = S.merge(ri, on=['run', 'subrun'], how='left')
S['rn'] = S.run.str[4:].astype(int)
S['t'] = pd.to_datetime(S.t_local)
S = S.sort_values(['t', 'run', 'subrun', 'arm']).reset_index(drop=True)
prod = S.rclass == 'production'
# derived
S['gated_per_late_trig'] = S.n_late_gated / S.n_trig_late
S['frac_trig_late_gated'] = S.n_ev_late_gated / S.n_trig_late
S['frac_trig_late_any'] = S.n_ev_late_any / S.n_trig_late
S['frac_late_trig'] = S.n_trig_late / S.n_trig
S['tracks_per_event'] = S.n_tracks / S.n_events_meta
S['gated_per_event'] = S.n_gated / S.n_events_meta
for p in 'xy':
    S[f'{p}_dt0_w68'] = 0.5 * (S[f'{p}_dt0_p84'] - S[f'{p}_dt0_p16'])
S['dtc_w68'] = 0.5 * (S.dtc_ns_p84 - S.dtc_ns_p16)
S['yx_q_ratio'] = S.y_q_sum_med / S.x_q_sum_med
S['T_end_over_vt'] = S.gap_end_nrmean  # alias kept for clarity
S.to_parquet(HERE / 'stability.parquet', index=False)

# sub-run order index (production only, per arm identical)
P = S[prod].copy()
order = P.drop_duplicates(['run', 'subrun'])[['run', 'subrun', 't']].reset_index(drop=True)
order['i'] = np.arange(len(order))
P = P.merge(order[['run', 'subrun', 'i']], on=['run', 'subrun'])
Cz = S[~prod].copy()
corder = Cz.drop_duplicates(['run', 'subrun'])[['run', 'subrun']].reset_index(drop=True)
corder['i'] = np.arange(len(corder))
Cz = Cz.merge(corder, on=['run', 'subrun'])

# quantities: name -> (column, count column for the statistical scale, arms)
Q = {
    'gated_per_late_trig': 'n_late_gated', 'frac_trig_late_gated': 'n_trig_late', 'frac_trig_late_any': 'n_trig_late',
    'frac_gated_late': 'n_late', 'frac_untimed': 'n_tracks', 'frac_late_trig': 'n_trig',
    'x_q_sum_med': 'n_late_gated', 'y_q_sum_med': 'n_late_gated', 'x_q_sum_p16': 'n_late_gated',
    'y_q_sum_p16': 'n_late_gated', 'yx_q_ratio': 'n_late_gated',
    'chi2dof_x_med': 'n_late_gated', 'chi2dof_y_med': 'n_late_gated', 'chi2dof_x_p84': 'n_late_gated',
    'chi2dof_y_p84': 'n_late_gated', 'x_n_strips_mean': 'n_late_gated', 'y_n_strips_mean': 'n_late_gated',
    'x_dt0_med': 'n_late_gated', 'y_dt0_med': 'n_late_gated', 'x_dt0_w68': 'n_late_gated',
    'y_dt0_w68': 'n_late_gated', 'dtc_ns_med': 'n_late_gated', 'dtc_w68': 'n_late_gated',
    'fL_t0_railed': 'n_late', 'fLG_t0_railed': 'n_late_gated', 'fL_censored_window': 'n_late',
    'fL_d_faint': 'n_late', 'fLG_x_split_child': 'n_late_gated', 'fLG_drift_railed': 'n_late_gated',
    'fLG_wall': 'n_late_gated', 'fLG_dca15': 'n_late_gated',
    'x_q_uend_nrmean': 'n_late_gated', 'drift_t_end_ns_nrmean': 'n_late_gated', 'v_t_med': 'n_late_gated',
    'gap_end_nrmean': 'n_late_gated', 'gap_xu_nrmean': 'n_late_gated',
    'x_local_mean': 'n_late_gated', 'y_local_mean': 'n_late_gated', 'x_local_rms': 'n_late_gated',
    'y_local_rms': 'n_late_gated', 'tanx_mean': 'n_late_gated', 'tany_mean': 'n_late_gated',
    'tanx_rms': 'n_late_gated', 'tany_rms': 'n_late_gated', 'xcross': 'n_late_gated', 'ycross': 'n_late_gated',
    'frac_2trk': 'n_ev_1p',
}
CQ = {k: v for k, v in Q.items() if not k.startswith(('frac_trig', 'gated_per_late', 'frac_untimed', 'frac_late_trig',
                                                      'x_dt0', 'y_dt0', 'xcross', 'ycross', 'fLG_wall', 'fLG_dca'))}
CQ.update({'tracks_per_event': 'n_events_meta', 'gated_per_event': 'n_events_meta'})
HALF = 7          # running median half-window in sub-runs
MIN_N = 300       # sub-runs with fewer late gated tracks are not scored


def score(T, quantities, label):
    rows = []
    for arm, g in T.groupby('arm'):
        g = g.sort_values('i')
        for q, ncol in quantities.items():
            if q not in g or g[q].notna().sum() < 10:
                continue
            v = g[q].to_numpy(float)
            n = g[ncol].to_numpy(float)
            ok = np.isfinite(v) & (g.n_late_gated.to_numpy() >= MIN_N if label == 'production' else
                                   g.n_gated.to_numpy() >= MIN_N)
            if ok.sum() < 10:
                continue
            idx = np.where(ok)[0]
            vv = v[ok]
            rm = np.array([np.median(np.r_[vv[max(0, j - HALF):j], vv[j + 1:j + 1 + HALF]]) for j in range(len(vv))])
            res = vv - rm
            nref = np.median(n[ok])
            s_run = 1.4826 * np.median(np.abs(res - np.median(res)))
            cm = np.median(vv)
            s_cmp = 1.4826 * np.median(np.abs(vv - cm))
            infl = np.maximum(1, np.sqrt(nref / np.maximum(n[ok], 1)))
            if q.startswith('f'):          # fractions: never narrower than the binomial error
                pb = np.clip(np.r_[rm], 0, 1)
                pb = np.maximum(pb, 3.0 / np.maximum(n[ok], 1))
                stat = np.sqrt(pb * (1 - np.minimum(pb, 0.999)) / np.maximum(n[ok], 1))
            else:
                stat = 0.0
            with np.errstate(all='ignore'):
                zr = res / np.maximum(s_run * infl, stat)
                zc = (vv - cm) / np.maximum(s_cmp * infl, stat)
            for k, j in enumerate(idx):
                rows.append(dict(sample=label, run=g.run.iloc[j], subrun=g.subrun.iloc[j], arm=arm, quantity=q,
                                 value=vv[k], expected_running=rm[k], z_running=zr[k], campaign_median=cm,
                                 z_campaign=zc[k], n=n[j], i=g.i.iloc[j]))
    return pd.DataFrame(rows)


Z = pd.concat([score(P, Q, 'production'), score(Cz, CQ, 'cosmic')], ignore_index=True)
Z['absz'] = np.maximum(np.abs(Z.z_running), 0.0)
out = Z[(Z.z_running.abs() >= 5) | ((Z.z_campaign.abs() >= 5) & (Z.z_running.abs() >= 3))].copy()
out = out.sort_values('absz', ascending=False)
out.round(5).to_csv(HERE / 'outliers.csv', index=False)
Z.round(5).to_parquet(HERE / 'zscores.parquet', index=False)

# run-level aggregation (production, weighted by n_late_gated) and run-level z
W = P[P.n_late_gated > 0].copy()
num = [q for q in Q if q in W]
agg = {}
for (run, arm), g in W.groupby(['run', 'arm']):
    w = g.n_late_gated.to_numpy(float)
    d = dict(run=run, arm=arm, rn=g.rn.iloc[0], t=g.t.min(), n_sub=len(g), n_late_gated=w.sum(),
             lat_epoch=g.lat_epoch.iloc[0], access_epoch=g.access_epoch.iloc[0])
    for q in num:
        v = g[q].to_numpy(float)
        m = np.isfinite(v)
        d[q] = np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan
    agg[(run, arm)] = d
R = pd.DataFrame(agg.values()).sort_values(['t', 'arm'])
R.to_csv(HERE / 'runs.csv', index=False)


# ------------------------------------------------------------------ steps (binary segmentation)
def segment(v, n, minseg=6, tmin=8.0, depth=0, off=0, found=None):
    found = [] if found is None else found
    if len(v) < 2 * minseg or depth > 3:
        return found
    best = None
    s = 1.4826 * np.median(np.abs(np.diff(v))) / np.sqrt(2)  # point-to-point scatter
    if not s > 0:
        return found
    for k in range(minseg, len(v) - minseg + 1):
        a, b = v[:k], v[k:]
        t = (np.median(b) - np.median(a)) / (s * np.sqrt(1 / len(a) + 1 / len(b)) * 1.25)
        if best is None or abs(t) > abs(best[1]):
            best = (k, t, np.median(b) - np.median(a), s)
    if best and abs(best[1]) >= tmin and abs(best[2]) > 2 * best[3]:
        k = best[0]
        found.append((off + k, best[1], best[2], best[3]))
        segment(v[:k], n[:k], minseg, tmin, depth + 1, off, found)
        segment(v[k:], n[k:], minseg, tmin, depth + 1, off + k, found)
    return found


steps = []
for arm, g in P.groupby('arm'):
    g = g[g.n_late_gated >= MIN_N].sort_values('i').reset_index(drop=True)
    for q in Q:
        if q not in g:
            continue
        m = g[q].notna().to_numpy()
        if m.sum() < 20:
            continue
        gg = g[m].reset_index(drop=True)
        for k, t, dlt, s in segment(gg[q].to_numpy(float), gg.n_late_gated.to_numpy(float)):
            steps.append(dict(arm=arm, quantity=q, before=f'{gg.run[k - 1]} {gg.subrun[k - 1]}',
                              after=f'{gg.run[k]} {gg.subrun[k]}', rn_before=gg.rn[k - 1], rn_after=gg.rn[k],
                              step=dlt, scatter=s, t=t, median_before=np.median(gg[q][:k]),
                              median_after=np.median(gg[q][k:])))
ST = pd.DataFrame(steps)
KNOWN = {(81, 84): '27 July access (E1->E2, A x conn 8 restored)', (102, 104): 'D latency L1 -> unplaced (run_104)',
         (104, 106): 'D latency unplaced -> L2 (run_105 cosmic)', (102, 106): 'D latency L1 -> L2'}


def known(a, b):
    for (x, y), s in KNOWN.items():
        if a <= x and b >= y and b - a <= 25:
            return s
    return ''


if len(ST):
    ST['known_epoch'] = [known(a, b) for a, b in zip(ST.rn_before, ST.rn_after)]
    ST = ST.sort_values(['arm', 'quantity', 'rn_after'])
ST.round(5).to_csv(HERE / 'steps.csv', index=False)

# ------------------------------------------------------------------ specific checks
# (a) gap = v_t x T per epoch;  (b) t0 - pin per epoch
P['epoch'] = np.where(P.access_epoch.str.startswith('pre'), 'E1', 'E2') + '_' + P.lat_epoch.astype(str)
chk = []
for (arm, ep), g in P[P.n_late_gated >= MIN_N].groupby(['arm', 'epoch']):
    d = dict(arm=arm, epoch=ep, n_sub=len(g), runs=f'{g.rn.min()}-{g.rn.max()}')
    for q in ('x_dt0_med', 'y_dt0_med', 'dtc_ns_med', 'v_t_med', 'drift_t_end_ns_nrmean', 'x_q_uend_nrmean',
              'gap_end_nrmean', 'gap_xu_nrmean', 'xcross', 'ycross', 'fLG_t0_railed', 'frac_gated_late'):
        v = g[q].dropna()
        if len(v):
            d[q] = v.median()
            d[q + '_sd'] = v.std()
    chk.append(d)
pd.DataFrame(chk).round(4).to_csv(HERE / 'checks_epochs.csv', index=False)

# gap correlation: does T scale as 1/v_t within an arm? fit T = G / v_t (+c)
gapfit = []
for arm, g in P[(P.n_late_gated >= MIN_N) & P.v_t_med.notna()].groupby('arm'):
    x = 1000.0 / g.v_t_med.to_numpy()
    for tcol in ('drift_t_end_ns_nrmean', 'x_q_uend_nrmean'):
        y = g[tcol].to_numpy()
        A = np.c_[x, np.ones_like(x)]
        (G, c0), *_ = np.linalg.lstsq(A, y, rcond=None)
        r = np.corrcoef(x, y)[0, 1]
        gapfit.append(dict(arm=arm, T=tcol, slope_mm=G, intercept_ns=c0, corr=r, vt_range=f'{g.v_t_med.min():.2f}-{g.v_t_med.max():.2f}',
                           T_range=f'{y.min():.0f}-{y.max():.0f}', gap_sd=(g.v_t_med * y / 1000).std(),
                           gap_med=(g.v_t_med * y / 1000).median()))
pd.DataFrame(gapfit).round(4).to_csv(HERE / 'checks_gap.csv', index=False)

# (c) per-run pooled crossings from the track cache (late gated, t0 <= 300)
from sept26_prelim_analysis import capsule_tof as CT
trk = pd.concat([pd.read_parquet(f, columns=['run', 'subrun', 'arm', 'pos', 'disp', 'ms', 'x_local', 'y_local'])
                 for f in sorted(glob.glob(str(C / 'trk/*.parquet')))], ignore_index=True)
trk = trk[trk.ms.notna()]
xc = []
for (run, arm), g in trk[trk.arm.isin(['A', 'C', 'D'])].groupby(['run', 'arm']):
    g = g[np.isfinite(g.pos) & np.isfinite(g.disp)]
    if len(g) < 300:
        continue
    x0, e0, n0 = CT.crossing(g.pos.to_numpy(), g.disp.to_numpy())
    xc.append(dict(run=run, arm=arm, xcross=x0, err=e0, n=n0))
XC = pd.DataFrame(xc)
XC['rn'] = XC.run.str[4:].astype(int)
XCp = XC.pivot(index='rn', columns='arm', values='xcross').join(XC.pivot(index='rn', columns='arm', values='err'),
                                                               rsuffix='_err').join(
    XC.pivot(index='rn', columns='arm', values='n'), rsuffix='_n')
XCp['AC_mean'] = 0.5 * (XCp.A + XCp.C)
XCp['AC_halfdiff'] = 0.5 * (XCp.A - XCp.C)
XCp.round(3).to_csv(HERE / 'checks_crossing_runs.csv')

# (d) occupancy holes: per sub-run late+all profiles vs the E2 campaign reference
occ = {}
for f in sorted(glob.glob(str(C / 'occ/*.npz'))):
    occ[Path(f).stem] = dict(np.load(f))
holes = []
pkeys = [f'{r[4:]}_{s}' for r, s in zip(order.run, order.subrun)]
E1 = {k for k in pkeys if k.startswith(('79_', '81_'))}
edges = np.linspace(-200, 200, 101)
for arm in 'ABCD':
    for sel in ('all', 'lg'):
        for p in ('x_local', 'y_local'):
            key = f'{arm}_{sel}_{p}'
            ref = sum(occ[k][key] for k in pkeys if k not in E1 and k in occ).astype(float)
            for k in pkeys:
                if k not in occ:
                    continue
                h = occ[k][key].astype(float)
                if h.sum() < 2000:
                    continue
                e = ref * h.sum() / ref.sum()
                with np.errstate(all='ignore'):
                    pull = (h - e) / np.sqrt(np.maximum(e, 1))
                    ratio = h / e
                bad = (ratio < 0.5) & (pull < -5) & (e > 20)
                # contiguous runs of >= 2 bins
                j = 0
                while j < len(bad):
                    if bad[j]:
                        k2 = j
                        while k2 + 1 < len(bad) and bad[k2 + 1]:
                            k2 += 1
                        if k2 - j + 1 >= 2:
                            holes.append(dict(key=k, arm=arm, sample=sel, view=p, lo=edges[j], hi=edges[k2 + 1],
                                              nbins=k2 - j + 1, ratio=float(np.nanmean(ratio[j:k2 + 1])),
                                              missing=float((e - h)[j:k2 + 1].sum()), frac_tracks_lost=float(
                                                  (e - h)[j:k2 + 1].sum() / h.sum())))
                        j = k2 + 1
                    else:
                        j += 1
H = pd.DataFrame(holes)
H.to_csv(HERE / 'checks_occupancy_holes.csv', index=False)

# ------------------------------------------------------------------ figures
runs_first = order.groupby('run').i.min()
EPOCH_LINES = [(order[order.run == 'run_84'].i.min() - 0.5, 'access 27 Jul'),
               (order[order.run == 'run_104'].i.min() - 0.5, 'run_104 (D unplaced)'),
               (order[order.run == 'run_106'].i.min() - 0.5, 'D L2')]
PAGES = {
    'rates': ['gated_per_late_trig', 'frac_trig_late_gated', 'frac_trig_late_any', 'frac_gated_late', 'frac_untimed',
              'frac_2trk'],
    'charge_fit': ['x_q_sum_med', 'y_q_sum_med', 'chi2dof_x_med', 'chi2dof_y_med', 'x_n_strips_mean',
                   'y_n_strips_mean'],
    'timing': ['x_dt0_med', 'y_dt0_med', 'dtc_ns_med', 'v_t_med', 'drift_t_end_ns_nrmean', 'gap_end_nrmean'],
    'flags': ['fLG_t0_railed', 'fL_censored_window', 'fL_d_faint', 'fLG_x_split_child', 'fLG_wall', 'fLG_dca15'],
    'occupancy': ['x_local_mean', 'y_local_mean', 'x_local_rms', 'y_local_rms', 'tanx_mean', 'tany_mean'],
    'pointing': ['xcross', 'ycross', 'tanx_rms', 'tany_rms', 'gap_xu_nrmean', 'fLG_drift_railed'],
}
outk = set(zip(out[out['sample'] == 'production'].run, out[out['sample'] == 'production'].subrun,
               out[out['sample'] == 'production'].arm, out[out['sample'] == 'production'].quantity))
for arm in 'ABCD':
    g = P[P.arm == arm].sort_values('i')
    for page, qs in PAGES.items():
        fig, axs = plt.subplots(3, 2, figsize=(10.5, 7.0), sharex=True)
        for ax, q in zip(axs.flat, qs):
            if q not in g or g[q].notna().sum() == 0:
                ax.set_title(f'{q} (n/a)')
                continue
            ok = g.n_late_gated >= MIN_N
            ax.plot(g.i[ok], g[q][ok], '.', ms=3, color='#3366aa')
            ax.plot(g.i[~ok], g[q][~ok], '.', ms=3, color='#bbbbbb')
            o = np.array([(r, s, arm, q) in outk for r, s in zip(g.run, g.subrun)])
            ax.plot(g.i[o], g[q][o], 'o', mfc='none', mec='#cc3311', ms=5)
            v = g[q][ok].to_numpy(float)
            if np.isfinite(v).sum() > 5:
                lo, hi = np.nanpercentile(v, [1, 99])
                pad = 0.25 * (hi - lo) + 1e-9
                ax.set_ylim(lo - pad, hi + pad)
            for x, lab in EPOCH_LINES:
                ax.axvline(x, color='#999999', lw=0.7, ls='--')
            ax.set_title(q)
            ax.grid(alpha=0.3)
        for ax in axs[-1]:
            ticks = runs_first[runs_first.index.str[4:].astype(int) % 1 == 0]
            sel = ticks.iloc[::3]
            ax.set_xticks(sel.values)
            ax.set_xticklabels([r[4:] for r in sel.index], rotation=90, fontsize=6.5)
            ax.set_xlabel('production sub-run in time order (labels: run number at its first sub-run)')
        fig.suptitle(f'Chamber {arm}: {page} (late >= 25 ms; grey = < {MIN_N} late gated tracks; red ring = outlier;'
                     f' dashed = access, run_104, run_106)', fontsize=9)
        fig.tight_layout()
        fig.savefig(HERE / f'fig_{arm}_{page}.png')
        plt.close(fig)

# crossing figure per run
fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for arm, c in (('A', '#3366aa'), ('C', '#cc3311')):
    axs[0].errorbar(XCp.index, XCp[arm], XCp[f'{arm}_err'], fmt='o', ms=3, color=c, label=arm)
axs[0].plot(XCp.index, XCp.AC_mean, 'k-', lw=0.8, label='(A+C)/2')
axs[0].set_ylabel('x crossing [mm]')
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].errorbar(XCp.index, XCp.D, XCp.D_err, fmt='o', ms=3, color='#228833', label='D z crossing')
axs[1].plot(XCp.index, XCp.AC_halfdiff, 'k.-', lw=0.8, label='(A-C)/2')
axs[1].set_ylabel('mm')
axs[1].set_xlabel('run')
axs[1].legend()
axs[1].grid(alpha=0.3)
fig.suptitle('Source crossing per run (late >= 25 ms gated tracks, t0 <= 300 ns, capsule_tof.crossing)')
fig.tight_layout()
fig.savefig(HERE / 'fig_crossing_runs.png')
plt.close(fig)

# gap figure
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, arm in zip(axs, 'ACD'):
    g = P[(P.arm == arm) & (P.n_late_gated >= MIN_N)]
    sc = ax.scatter(g.v_t_med, g.drift_t_end_ns_nrmean, c=g.rn, s=6, cmap='viridis')
    vv = np.linspace(g.v_t_med.min(), g.v_t_med.max(), 10)
    gm = (g.v_t_med * g.drift_t_end_ns_nrmean / 1000).median()
    ax.plot(vv, gm * 1000 / vv, 'k--', lw=0.8, label=f'T = {gm:.2f} mm / v_t')
    ax.set_xlabel('v_t [um/ns]')
    ax.set_ylabel('mean drift_t_end (non-railed) [ns]')
    ax.set_title(arm)
    ax.legend(fontsize=7)
fig.colorbar(sc, ax=axs, label='run')
fig.savefig(HERE / 'fig_gap_vt.png')
plt.close(fig)
print('outliers', len(out), 'steps', len(ST), 'holes', len(H))

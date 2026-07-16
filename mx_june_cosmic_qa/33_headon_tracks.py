#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
33_headon_tracks.py

Head-on (near-vertical) track identification from the cluster signature,
benchmarked against angle-based classifiers.

WHY a dedicated tagger: the micro-TPC time-fit angle needs a spatial lever
arm, which head-on tracks by definition lack (their unshared cluster is the
2-3 strip direct footprint) -- exactly the tracks where the angle estimator
is weakest are the ones we want to tag. But head-on tracks have their OWN
clean waveform signature on the RAW (shared) cluster:

  * the lead strip collects the WHOLE drift column -> maximal amplitude
    and long time-over-threshold;
  * the +-1 neighbours carry only resistive-shared charge:
      - amplitude-SYMMETRIC left/right    (a_asym ~ 0)
      - both DELAYED by the same +~69 ns  (t_asym ~ 0, t_delay > 0)
  * an inclined track instead has direct charge on the neighbours with a
    monotonic time ladder: one neighbour EARLY (mesh side), one late ->
    t_asym large, and the early one is not delayed at all;
  * after unsharing the footprint collapses to few strips (n_u small).

Classifiers benchmarked (truth = M3 v2 ray space angle):
  A  production time-fit |tan θ|      (hits cache; the "current algorithm")
  B  unshared+calibrated |tan θ|      (31's segment table; premium but only
                                       ~50 % coverage, biased against target)
  C  single signature features        (t_asym, a_asym, q_frac, tot_lead, n_u)
  D  Fisher LDA of the signature set  (numpy, train/test split by parity)
  E  production cluster size n_strips (the cheap baseline)

Outputs: ROC curves (per-coverage and coverage-corrected), efficiency vs
purity, feature distributions, tagged-sample |θ_ref| turn-on at working
points, features CSV cache.

Usage: ../.venv/bin/python 33_headon_tracks.py sat_det3 [--veto=50]
       [--rebuild] [--cut=5]   (head-on truth definition θ_ref < CUT deg)
Output: <alignment_tpc_vetoN>/headon/  headon_features.csv,
        headon_benchmark.png, headon_features.png + stdout table
"""
import os
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
REBUILD = '--rebuild' in sys.argv
CUT_DEG = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--cut=')), 5.0)
SAMPLE_NS = 60.0
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
A_LEAD_MIN = 300.0     # lead strip must be a real signal
N_PED_EVENTS = 300
CHUNK = 400
ALPHA = 0.5
V_SCALE = 34.0         # only for display; ROC invariant under monotone maps

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'headon')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
SEG_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'microtpc_metrics',
                       'microtpc_segments.csv')
FEAT_CSV = os.path.join(OUT, 'headon_features.csv')
CSHARE = {7: (0.449, 0.052),                      # det3 X (design)
          8: (0.432, 0.112),                      # det4 Y measured (det7 Y was 0.513/0.220)
          6: (0.370, 0.047),                      # det4 X measured 2026-07-12 (det7 X was 0.263/0.058)
          3: (0.250, 0.048), 4: (0.451, 0.204)}   # det6 X/Y measured 2026-07-12


def cfd_time(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a < THR_WF or ipk == 0:
        return np.nan
    lvl = 0.5 * a
    for i in range(1, ipk + 1):
        if w[i] >= lvl > w[i - 1]:
            return SAMPLE_NS * (i - 1 + (lvl - w[i - 1]) / (w[i] - w[i - 1]))
    return np.nan


def nsum(x, k):
    out = np.zeros_like(x)
    out[k:] += x[:-k]
    out[:-k] += x[k:]
    return out


def unshare(wb, c1, c2, alpha=ALPHA):
    n, ns = wb.shape
    if n < 3:
        return wb
    ab = np.zeros((5, n))
    ab[0, 2:] = alpha * c2
    ab[1, 1:] = alpha * c1
    ab[2, :] = 1.0
    ab[3, :-1] = alpha * c1
    ab[4, :-2] = alpha * c2
    X = np.zeros_like(wb)
    for s in range(ns):
        rhs = wb[:, s].copy()
        if s >= 1:
            rhs -= (1 - alpha) * c1 * nsum(X[:, s - 1], 1)
        if s >= 2:
            rhs -= (1 - alpha) * c2 * nsum(X[:, s - 2], 2)
        X[:, s] = solve_banded((2, 2), ab, rhs)
    return X


def build_features(ref, det):
    """One waveform pass -> per event & plane signature features."""
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    blocks, pos_of = {}, {}
    for feu in CFG.MX17_FEUS:
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                        [0 if plane_of_feu[feu] == 'x' else 1]
                        for ch in range(512)], dtype=float)
        pos_of[feu] = pos
        ok = np.where(np.isfinite(pos))[0]
        o = ok[np.argsort(pos[ok])]
        brk = np.where(np.diff(pos[o]) > 1.5 * PITCH_MM)[0]
        blocks[feu] = np.split(o, brk + 1)

    rows = []
    for feu in CFG.MX17_FEUS:
        c1, c2 = CSHARE[feu]
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(32, 512) for a in a0
                                      if a.size == 32 * 512]), axis=(0, 1))
            for lo in range(0, t.num_entries, CHUNK):
                hi = min(lo + CHUNK, t.num_entries)
                want = [i for i in range(lo, hi) if int(eids_all[i]) in ref]
                if not want:
                    continue
                arr = t.arrays(['eventId', 'amplitude'], entry_start=lo,
                               entry_stop=hi, library='np')
                for i in want:
                    j = i - lo
                    eid = int(arr['eventId'][j])
                    if arr['amplitude'][j].size != 32 * 512:
                        continue                         # malformed multi-frame event
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    wfm = wfm.T
                    # pick the block+cluster holding the largest strip signal
                    best = None
                    for blk in blocks[feu]:
                        wb = wfm[blk]
                        amax = wb.max(axis=1)
                        k = int(np.argmax(amax))
                        if best is None or amax[k] > best[0]:
                            best = (amax[k], k, blk, wb)
                    a_pk, k, blk, wb = best
                    if a_pk < A_LEAD_MIN or k < 2 or k > len(blk) - 3:
                        continue
                    amax = wb.max(axis=1)
                    # raw cluster around the lead (contiguous >= THR_HIT)
                    l0 = k
                    while l0 - 1 >= 0 and amax[l0 - 1] >= THR_HIT:
                        l0 -= 1
                    r0 = k
                    while r0 + 1 < len(blk) and amax[r0 + 1] >= THR_HIT:
                        r0 += 1
                    n_raw = r0 - l0 + 1
                    q_clu = float(amax[l0:r0 + 1].sum())
                    tL, t0, tR = (cfd_time(wb[k - 1]), cfd_time(wb[k]),
                                  cfd_time(wb[k + 1]))
                    aL, aR = float(amax[k - 1]), float(amax[k + 1])
                    tot = float(np.sum(wb[k] > THR_WF) * SAMPLE_NS)
                    # unshared footprint size
                    wu = unshare(wb[max(0, k - 6):k + 7], c1, c2)
                    n_u = int(np.sum(wu.max(axis=1) >= THR_HIT))
                    rows.append(dict(
                        eid=eid, plane=plane_of_feu[feu],
                        a_lead=float(a_pk), a_l=aL, a_r=aR,
                        t_lead=t0, t_l=tL, t_r=tR,
                        tot_lead=tot, n_raw=n_raw, n_u=n_u,
                        q_frac=float(a_pk / q_clu) if q_clu > 0 else np.nan))
            print(f'  {os.path.basename(fn)} done ({len(rows):,})')
    return pd.DataFrame(rows)


def roc(score, truth, n=200, higher_is_headon=True):
    """score: higher = more head-on-like (or flip). Returns FPR, TPR, thr."""
    s = np.asarray(score, float)
    m = np.isfinite(s)
    s, y = s[m], np.asarray(truth)[m].astype(bool)
    if not higher_is_headon:
        s = -s
    thr = np.quantile(s, np.linspace(0, 1, n))
    tpr = np.array([np.mean(s[y] >= t) for t in thr])
    fpr = np.array([np.mean(s[~y] >= t) for t in thr])
    auc = float(np.trapz(np.flip(tpr), np.flip(fpr)))
    return fpr, tpr, thr, auc, m.mean()


def eff_purity(score, truth, higher_is_headon=True, n=400):
    s = np.asarray(score, float)
    m = np.isfinite(s)
    s, y = s[m], np.asarray(truth)[m].astype(bool)
    if not higher_is_headon:
        s = -s
    thr = np.quantile(s, np.linspace(0.0, 1.0, n))
    eff, pur = [], []
    for t in thr:
        sel = s >= t
        if sel.sum() < 20:
            continue
        eff.append(np.mean(s[y] >= t))
        pur.append(np.mean(y[sel]))
    return np.array(eff), np.array(pur)


def main():
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    ref, prod = {}, {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
        prod[r.event_id] = (r.x_fit.slope_mm_per_ns * 1000.0,
                            r.y_fit.slope_mm_per_ns * 1000.0,
                            r.x_fit.n_strips, r.y_fit.n_strips)
    print(f'{len(ref):,} matched events; head-on truth: theta_ref < {CUT_DEG:g} deg')

    if os.path.exists(FEAT_CSV) and not REBUILD:
        ft = pd.read_csv(FEAT_CSV)
        print(f'Loaded cached features: {len(ft):,} rows')
    else:
        ft = build_features(ref, det)
        ft.to_csv(FEAT_CSV, index=False)
        print(f'Features cached: {len(ft):,} rows -> {FEAT_CSV}')

    # ---------------- event-level assembly ----------------
    eids = np.array(sorted(ref))
    th_sp = np.array([np.degrees(np.arctan(np.hypot(*ref[e]))) for e in eids])
    truth = th_sp < CUT_DEG
    print(f'prevalence: {100*truth.mean():.1f} % of matched events are head-on')

    # per-plane feature frames -> event-level (mean over available planes)
    ft = ft[np.isfinite(ft['t_lead'])].copy()
    ft['t_asym'] = np.abs(ft['t_r'] - ft['t_l'])                    # ns
    ft['t_delay'] = np.minimum(ft['t_l'], ft['t_r']) - ft['t_lead']  # ns
    ft['a_asym'] = np.abs(ft['a_r'] - ft['a_l']) / (ft['a_r'] + ft['a_l'])
    per_ev = ft.groupby('eid').agg(
        t_asym=('t_asym', 'mean'), t_delay=('t_delay', 'mean'),
        a_asym=('a_asym', 'mean'), q_frac=('q_frac', 'mean'),
        tot_lead=('tot_lead', 'mean'), n_u=('n_u', 'mean'),
        n_planes=('plane', 'nunique'))
    per_ev = per_ev.reindex(eids)

    # A: production combined |tan|
    S_prod = np.array([prod[e][:2] for e in eids])
    tan_prod = np.hypot(S_prod[:, 0], S_prod[:, 1]) / V_SCALE
    n_prod = np.array([prod[e][2] + prod[e][3] for e in eids], float)

    # B: unshared+calibrated combined |tan| from 31's table
    tan_unsh = np.full(len(eids), np.nan)
    if os.path.exists(SEG_CSV):
        seg = pd.read_csv(SEG_CSV)
        vv = {}
        for p in 'xy':
            d = seg[seg['plane'] == p]
            vv[p] = dict(zip(d['eid'], d['S_um_ns'] / 33.6))
        for i, e in enumerate(eids):
            ts = [abs(vv[p][e]) for p in 'xy' if e in vv[p]]
            if len(ts) == 2:
                tan_unsh[i] = np.hypot(*ts)

    # C/D: signature features (lower = more head-on for most)
    F = per_ev[['t_asym', 'a_asym', 'q_frac', 'tot_lead', 't_delay', 'n_u']].to_numpy()
    have_f = np.isfinite(F).all(axis=1)
    print(f'coverage: signature {100*have_f.mean():.1f} %, '
          f'production 100 %, unshared dual {100*np.isfinite(tan_unsh).mean():.1f} %')

    # Fisher LDA on standardized features, parity split (train even, test all;
    # report test-odd AUC to check overfit is nil for 6 features)
    Z = (F - np.nanmean(F[have_f], axis=0)) / np.nanstd(F[have_f], axis=0)
    tr = have_f & (eids % 2 == 0)
    te = have_f & (eids % 2 == 1)
    mu1 = Z[tr & truth].mean(axis=0)
    mu0 = Z[tr & ~truth].mean(axis=0)
    Sw = np.cov(Z[tr & truth].T) + np.cov(Z[tr & ~truth].T)
    w = np.linalg.solve(Sw, mu1 - mu0)
    lda = np.full(len(eids), np.nan)
    lda[have_f] = Z[have_f] @ w
    print('LDA weights:', {k: f'{v:+.2f}' for k, v in
                           zip(['t_asym', 'a_asym', 'q_frac', 'tot_lead', 't_delay', 'n_u'], w)})

    # ---------------- benchmark ----------------
    classifiers = [
        ('production |tanθ| (current algo)', -tan_prod, None, 'tab:blue'),
        ('unshared+cal |tanθ| (dual-seg only)', -tan_unsh, None, 'tab:cyan'),
        ('t_asym (neighbour time symmetry)', -per_ev['t_asym'].to_numpy(), None, 'tab:orange'),
        ('q_frac (lead-charge fraction)', per_ev['q_frac'].to_numpy(), None, 'tab:green'),
        ('tot_lead (lead time-over-thr)', per_ev['tot_lead'].to_numpy(), None, 'olive'),
        ('n_strips X+Y (cheap baseline)', -n_prod, None, 'gray'),
        ('signature LDA (6 features)', lda, None, 'crimson'),
    ]
    print(f'\n== ROC AUC (truth: theta_ref < {CUT_DEG:g} deg) ==')
    rocs = {}
    for name, s, _, c in classifiers:
        fpr, tpr, thr, auc, cov = roc(s, truth)
        rocs[name] = (fpr, tpr, auc, cov, c, s)
        print(f'  {name:38s} AUC = {auc:.3f}   (coverage {100*cov:.0f} %)')
    # LDA overfit check
    _, _, _, auc_te, _ = roc(lda[te], truth[te])
    print(f'  [LDA on odd-eid holdout: AUC = {auc_te:.3f}]')

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    ax = axes[0, 0]
    for name, (fpr, tpr, auc, cov, c, s) in rocs.items():
        ax.plot(fpr, tpr, '-', color=c, lw=2,
                label=f'{name}  AUC {auc:.3f}' + (f' ({100*cov:.0f}%)' if cov < 0.99 else ''))
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('false-positive rate (inclined tagged head-on)')
    ax.set_ylabel('efficiency for true head-on')
    ax.set_title(f'ROC — head-on = θ_ref < {CUT_DEG:g}°')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, (fpr, tpr, auc, cov, c, s) in rocs.items():
        eff, pur = eff_purity(s, truth)
        ax.plot(eff, 100 * pur, '-', color=c, lw=2)
    ax.axhline(100 * truth.mean(), color='k', ls=':', lw=1)
    ax.text(0.02, 100 * truth.mean() + 1, 'prevalence', fontsize=8)
    ax.set_xlabel('head-on efficiency'); ax.set_ylabel('purity of tagged sample [%]')
    ax.set_title('efficiency vs purity'); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

    # turn-on: theta_ref distribution of tagged events at eff=50% working point
    ax = axes[0, 2]
    bins = np.arange(0, 30, 1.0)
    ax.hist(th_sp, bins=bins, histtype='step', color='k', lw=1.5, label='all matched')
    for name in ['production |tanθ| (current algo)', 'signature LDA (6 features)']:
        fpr, tpr, auc, cov, c, s = rocs[name]
        sv = np.asarray(s, float)
        mfin_ = np.isfinite(sv)
        t50 = np.quantile(sv[mfin_ & truth], 0.5)   # 50% head-on efficiency
        sel = mfin_ & (sv >= t50)
        ax.hist(th_sp[sel], bins=bins, histtype='stepfilled', alpha=0.45, color=c,
                label=f'{name.split(" (")[0]} @50% eff (purity '
                      f'{100*np.mean(truth[sel]):.0f}%)')
    ax.axvline(CUT_DEG, color='k', ls='--', lw=1)
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('events')
    ax.set_title('what a tagged sample looks like'); ax.legend(fontsize=8)

    # feature distributions
    feats = [('t_asym', 't_asym = |t_R − t_L| [ns]', (0, 400)),
             ('a_asym', 'a_asym = |A_R−A_L|/(A_R+A_L)', (0, 1)),
             ('q_frac', 'lead-charge fraction', (0.2, 1.0))]
    for k, (col, lab, rng) in enumerate(feats):
        ax = axes[1, k]
        v = per_ev[col].to_numpy()
        for m_, lb, c in [(truth, f'θ<{CUT_DEG:g}° (head-on)', 'crimson'),
                          (~truth, f'θ>{CUT_DEG:g}° (inclined)', 'tab:blue')]:
            mm = m_ & np.isfinite(v)
            ax.hist(np.clip(v[mm], *rng), bins=50, range=rng, density=True,
                    histtype='step', lw=2, color=c, label=lb)
        ax.set_xlabel(lab); ax.set_ylabel('normalised')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'{CFG.RUN} — head-on track identification '
                 f'(signature vs angle estimators, M3 v2 truth)', fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'headon_benchmark.png'), dpi=155)

    # working-point table
    print('\n== working points (signature LDA) ==')
    fpr, tpr, auc, cov, c, s = rocs['signature LDA (6 features)']
    sv = np.asarray(s, float)
    mfin_ = np.isfinite(sv)
    for q, nm in [(0.8, 'eff 80%'), (0.5, 'eff 50%'), (0.2, 'eff 20%')]:
        t_ = np.quantile(sv[mfin_ & truth], 1 - q)
        sel = mfin_ & (sv >= t_)
        print(f'  {nm}: threshold {t_:+.2f}  tagged {sel.sum():5d}  '
              f'purity {100*np.mean(truth[sel]):.1f} %  '
              f'median θ_ref of tags {np.median(th_sp[sel]):.1f} deg')
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()

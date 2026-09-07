#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10b_hv_scan_efficiency.py -- efficiency vs amplification HV, on the CURRENT chain.

Why this file exists
--------------------
The deck's det3 HV curve came from CSVs written on **29 June 2026**
(``.../mx17_det3_saturday_scan_6-27-26/hv_scan{,2}/mx17_3/efficiency_vs_hv_scan{,2}.csv``).
They plateau at **81 %** while the SAME chamber, the SAME night, at the SAME
490 V reads **93.3 %** from ``long_run_resist_490V_drift_1000V``
(``wft/efficiency/efficiency_breakdown.txt``).  Three basis changes landed after
those CSVs were written and none of them was ever applied to the scan:

  1. **M3 recipe** (2026-07-13).  The old scan is on chi2<5 & NClus>=3 --
     2,206 rays on the 490 V sub-run against **938** on the golden
     chi2<1.0 & NClus=4.  A looser reference is a *worse* reference, and the
     efficiency is defined by a 5 mm match to it, so the extra rays land off
     track and read as detector inefficiency.  This is the dominant term.
  2. **Matched-filter reprocessing of the raw waveforms** (2026-07-24, ~+40 %
     hits) and the **relative significance floor** that fixes it
     (2026-07-25, ``cm.apply_significance_floor``, sigrel = 0.10).  Script 10
     never applied the floor; without it coherent noise inflates multiplicity
     and pushes ordinary muons over the >50-strip discharge veto.
  3. **Discharge accounting** (``mx_june_wft/02_efficiency.py``).  A muon that
     arrives while the chamber is discharging is not a detection failure; the
     golden accounting puts it in its own category rather than in the
     efficiency denominator.

Script ``10_hv_scan_efficiency.py`` is kept as the pre-2026-08 reference.  It
carries (1) but not (2) or (3), has no cache, and cannot see sub-runs whose
names are prefixed (``hv_scan_resist_490V_...``), which is why the saturday scan
was never re-run through it.

What this does, per sub-run
---------------------------
  1. combined_hits -> FEU cut -> ``apply_significance_floor(rel=0.10)`` -> strip
     map -> ``analyse_event`` in parallel.  Cached to the same
     ``cache/event_results.pkl`` + ``.meta.json`` contract as
     ``03_alignment_and_tpc.py --no-veto``, so the caches are interchangeable.
  2. M3 rays on the golden recipe (chi2<1.0 & NClus=4) from
     ``m3_tracking_root_v2`` when present.
  3. Geometry (z, theta, handedness) seeded from the run's long-run alignment --
     the detector does not move during a scan -- with translation re-fitted per
     sub-run.
  4. The ``02_efficiency.py`` category accounting, inside a box FIXED across the
     scan so the denominator region is identical at every voltage.  The box is
     the 0.5-99.5 percentiles of the SEED (long) run's reconstructed points --
     i.e. **the same box the published breakdown uses**, which is what makes the
     operating-point scan point directly comparable to it.  It must not be taken
     from the highest-HV sub-run: there the chamber discharges on half the
     triggers, the reco cloud is blown out past the real active area, and rays
     land in box corners the detector cannot see (has_any falls 99.9 -> 95 %).
     ``--box-from=best`` restores the old behaviour for comparison.

Two efficiencies are written, and they differ only in the discharge category:

  ``within_R``          reco within R mm over ALL crossings in the box,
                        discharges included in the denominator.  This is the
                        published convention -- ``02_efficiency.py``'s
                        ``within_R``, the 93.1 / 93.5 % on det3 -- and it is
                        the one to quote.  Verified by ``--closure``.
  ``within_R_nospark``  the same numerator over a denominator with the
                        discharge category removed.  A muon arriving while the
                        chamber is recovering is arguably not an efficiency
                        loss; this is that reading, and it runs ~2-3 points
                        higher at the operating point.  Secondary -- never mix
                        it with a published number.

Usage
-----
    ../.venv/bin/python 10b_hv_scan_efficiency.py <scan_key> [--r=5] [--refit]
    ../.venv/bin/python 10b_hv_scan_efficiency.py --list
    ../.venv/bin/python 10b_hv_scan_efficiency.py --closure <scan_key>

``--closure`` scores the scan's SEED sub-run (the long run) through this exact
code and prints it beside the published ``efficiency_breakdown_hits.txt``; the
two must agree to a few tenths of a point or the chain is wrong.

Output -> <cosmic_bench>/Analysis/<run>/<out>/<det>/efficiency_vs_hv.csv (+ .png)
Any ``--r`` other than the default 5 mm is written beside it as
``efficiency_vs_hv_r<R>mm.csv`` -- the plain name always means the 5 mm
published product, because five other scripts import it by that path.
"""
import argparse
import concurrent.futures
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import setup_paths, _Config, M3_CHI2_CUT, M3_MIN_NCLUS
setup_paths()
import uproot                                                       # noqa: E402
import cosmic_micro_tpc_analysis as cm                              # noqa: E402
from common.Mx17StripMap import RunConfig                           # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402

SIG_REL = cm.SIG_REL_FLOOR      # 0.10 -- the DET3_RECO_FIX floor, as 03 uses it
SPARK_HITS = 50                 # >50 strips after the floor = full-chamber discharge
BOX_PCT = (0.5, 99.5)           # same percentiles as 02_efficiency.py

_DET3_BASE = '/home/dylan/x17/cosmic_bench/det3/'
_D23_BASE = '/home/dylan/x17/cosmic_bench/det2_det3/'

# One entry per (run, detector, scan pass).  `prefix` selects the sub-runs:
# None -> `resist_<NNN>V...`, else `<prefix>_resist_<NNN>V...`.
SCANS = {
    # 27 June saturday det3 scan -- det3 in the TOP slot, FEU 7(X)/8(Y), z 702.
    # Two interleaved passes: hv_scan 425-525 V (10 V), hv_scan2 460-520 V.
    # This is the scan the deck uses; it is the only one that reaches below the
    # plateau, and it is the run mesh_ladder.csv comes from.
    'sat_det3_1': dict(run='mx17_det3_saturday_scan_6-27-26', base=_DET3_BASE,
                       det_name='mx17_3', feus=[7, 8], det_z=702.0,
                       prefix='hv_scan', out='hv_scan',
                       seed_subrun='long_run_resist_490V_drift_1000V'),
    'sat_det3_2': dict(run='mx17_det3_saturday_scan_6-27-26', base=_DET3_BASE,
                       det_name='mx17_3', feus=[7, 8], det_z=702.0,
                       prefix='hv_scan2', out='hv_scan2',
                       seed_subrun='long_run_resist_490V_drift_1000V'),
    # 22 June overnight -- det3 in the BOTTOM slot (FEU 3/4, z 232), det2 top.
    # Starts at 450 V, already on the plateau, so it cannot show a turn-on.
    'o22_det3': dict(run='mx17_det2_det3_overnight_6-22-26', base=_D23_BASE,
                     det_name='mx17_3', feus=[3, 4], det_z=232.0,
                     prefix=None, out='hv_scan', seed_subrun='long_run'),
    # det2 is FEU 6(X)/8(Y) in THIS run -- not 7/8, which is the weekend run's
    # layout.  qa_config says so in capitals and it still caught me: with 7/8
    # every sub-run analyses fine and reports 0 events with a valid X+Y point,
    # because the X view is simply absent.  Zero valid events on EVERY point of
    # a scan is an FEU mistake, not a detector result.
    'o22_det2': dict(run='mx17_det2_det3_overnight_6-22-26', base=_D23_BASE,
                     det_name='mx17_2', feus=[6, 8], det_z=702.0,
                     prefix=None, out='hv_scan', seed_subrun='long_run'),
}


# --------------------------------------------------------------------------- #
# per sub-run
# --------------------------------------------------------------------------- #

def cfg_for(sc, subrun):
    return _Config(f"{sc['det_name']}_{subrun}", sc['run'], subrun,
                   feus=sc['feus'], det_z=sc['det_z'],
                   det_name=sc['det_name'], base_path=sc['base'])


def m3_dir(cfg):
    """Prefer the tracking-v2 reprocessing, as qa_config._Config does."""
    v2 = f'{cfg.BASE_PATH}{cfg.RUN}/{cfg.SUB_RUN}/m3_tracking_root_v2/'
    if os.path.isdir(v2) and any(f.endswith('.root') for f in os.listdir(v2)):
        return v2
    return f'{cfg.BASE_PATH}{cfg.RUN}/{cfg.SUB_RUN}/m3_tracking_root/'


def load_hits(cfg, det):
    """combined_hits for this detector, with the significance floor applied.

    No spark veto: discharges are TAGGED by multiplicity in the accounting, not
    removed upstream -- the same choice 02_efficiency.py makes when it reads the
    un-vetoed cache.
    """
    d = cfg.combined_hits_dir
    fs = sorted(f for f in os.listdir(d) if f.endswith('.root') and '_datrun_' in f)
    if not fs:
        return None
    df = uproot.concatenate([f'{d}{f}:hits' for f in fs], library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)].copy()
    if not len(df):
        return None
    df = cm.apply_significance_floor(df, rel=SIG_REL)
    return cm._map_strip_positions(df, det)


def event_results(cfg, df, refit=False):
    """Per-event micro-TPC fits, cached on the 03_alignment_and_tpc contract."""
    cache = os.path.join(cfg.out_dir('cache'), 'event_results.pkl')
    meta_p = cache.replace('.pkl', '.meta.json')
    # The FEU pair belongs in the sidecar: 03_alignment_and_tpc.py's meta keys
    # only on sigrel+veto, so a cache built for the wrong X FEU reloads happily
    # and reports an empty detector.  Recording it turns that into a refit.
    meta = {'sigrel': SIG_REL, 'veto': None, 'feus': list(cfg.MX17_FEUS)}
    if os.path.exists(cache) and not refit:
        old = json.load(open(meta_p)) if os.path.exists(meta_p) else None
        # accept a sidecar written by 03 (no 'feus' key) if the rest matches
        if old is not None and all(old.get(k) == v for k, v in meta.items()
                                   if k in old):
            return pickle.load(open(cache, 'rb'))
        print(f'  cache built with {old}, want {meta} -> refitting')
    g = df.groupby('eventId')
    args = [(g.get_group(e).copy(), int(e)) for e in df['eventId'].unique()]
    nw = max(1, (os.cpu_count() or 1) - cm.N_FREE_THREADS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nw) as pool:
        res = list(cm._progress(pool.map(cm._analyse_event_worker, args),
                                total=len(args), desc='  events'))
    pickle.dump(res, open(cache, 'wb'))
    json.dump(meta, open(meta_p, 'w'))
    return res


def prepare(sc, subrun, seed, det, refit=False):
    """Everything for one sub-run except the box-dependent accounting."""
    cfg = cfg_for(sc, subrun)
    if not (os.path.isdir(cfg.combined_hits_dir) and os.path.isdir(m3_dir(cfg))):
        print(f'  [SKIP] {subrun}: missing hits or tracking'); return None
    df = load_hits(cfg, det)
    if df is None:
        print(f'  [SKIP] {subrun}: no detector hits'); return None

    results = event_results(cfg, df, refit=refit)
    n_valid = sum(r.has_both for r in results)
    if n_valid < 20:
        print(f'  [SKIP] {subrun}: only {n_valid} valid X+Y events'); return None

    rays = M3RefTracking(m3_dir(cfg), chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = seed.ref_x_sign * np.array(xa)

    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xa, an)

    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # Detection + discharge bookkeeping, from the hits, exactly as 02 does it:
    # the multiplicity that decides "discharge" is counted AFTER the floor, and
    # rays outside the detector's own event-id range are not scored at all.
    mult = df.groupby('eventId').size()
    fired = set(int(e) for e in mult.index)
    det_lo, det_hi = int(mult.index.min()), int(mult.index.max())
    mult_by_ev = {int(k): int(v) for k, v in mult.items()}
    n_firing = len(mult)
    n_spark_all = int((mult > SPARK_HITS).sum())

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    evn = [int(v) for v in evn]

    return dict(subrun=subrun, reco=reco, recpos=np.array(list(reco.values())),
                px=px, py=py, evn=evn, fired=fired, mult=mult_by_ev,
                det_lo=det_lo, det_hi=det_hi, n_valid=n_valid,
                n_firing=n_firing, n_spark_all=n_spark_all,
                params=params, n_rays_total=len(evn))


def score(a, box, R):
    """02_efficiency.py accounting for one sub-run inside a fixed box."""
    ax0, ax1, ay0, ay1 = box
    cat = dict(no_hit=0, hit_no_reco=0, spark=0, reco_far=0, reco_near=0)
    rl = []
    for e, x, y in zip(a['evn'], a['px'], a['py']):
        if e < a['det_lo'] or e > a['det_hi']:
            continue
        if not (np.isfinite(x) and np.isfinite(y)
                and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        if a['mult'].get(e, 0) > SPARK_HITS:
            cat['spark'] += 1
            continue
        if e in a['reco']:
            r = float(np.hypot(x - a['reco'][e][0], y - a['reco'][e][1]))
            rl.append(r)
            cat['reco_near' if r <= R else 'reco_far'] += 1
        elif e in a['fired']:
            cat['hit_no_reco'] += 1
        else:
            cat['no_hit'] += 1

    # Denominator = every M3 crossing of the box, discharges included: this is
    # what 02_efficiency.py divides by, and --closure checks it against the
    # published breakdown to the third decimal.
    n = sum(cat.values())
    n_ns = n - cat['spark']                  # secondary reading, see docstring
    rl = np.asarray(rl)
    eff = cat['reco_near'] / n if n else np.nan
    return dict(
        n_rays=n, n_rays_nospark=n_ns, n_near=cat['reco_near'],
        within_R=100 * eff,
        within_R_err=100 * np.sqrt(eff * (1 - eff) / n) if n else np.nan,
        within_R_nospark=100 * cat['reco_near'] / n_ns if n_ns else np.nan,
        reco_at_all=100 * (cat['reco_near'] + cat['reco_far']) / n if n else np.nan,
        reco_far=100 * cat['reco_far'] / n if n else np.nan,
        hit_no_reco=100 * cat['hit_no_reco'] / n if n else np.nan,
        no_hit=100 * cat['no_hit'] / n if n else np.nan,
        has_any=100 * (n - cat['no_hit']) / n if n else np.nan,
        spark_cat=100 * cat['spark'] / n if n else np.nan,
        spark_frac=100 * a['n_spark_all'] / a['n_firing'] if a['n_firing'] else np.nan,
        core_sigma_mm=_rstd(rl[rl < 15]) if len(rl) else np.nan,
        median_r_mm=float(np.median(rl)) if len(rl) else np.nan,
    )


def _rstd(v, ns=3, it=5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v)) if len(v) else np.nan


# --------------------------------------------------------------------------- #

def find_subruns(sc):
    import re
    pat = (rf'^{re.escape(sc["prefix"])}_resist_(\d+)V' if sc['prefix']
           else r'^resist_(\d+)V')
    run_dir = os.path.join(sc['base'], sc['run'])
    out = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(pat, name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            out.append((name, int(m.group(1))))
    return sorted(out, key=lambda t: t[1])


def seed_params(sc):
    p = os.path.join(os.path.dirname(sc['base'].rstrip('/')), 'Analysis',
                     sc['run'], sc['seed_subrun'], sc['det_name'],
                     'alignment_tpc_veto50', 'alignment.json')
    if not os.path.exists(p):
        sys.exit(f'No alignment seed at {p}\n'
                 f'Run 03_alignment_and_tpc.py on {sc["seed_subrun"]} first.')
    return cm.load_alignment(p), p


def out_dir(sc):
    d = os.path.join(os.path.dirname(sc['base'].rstrip('/')), 'Analysis',
                     sc['run'], sc['out'], sc['det_name'])
    os.makedirs(d, exist_ok=True)
    return d


def compare_superseded(od_pngs=()):
    """Before/after: the archived 29 June curve against this chain's.

    Both interleaved passes on one voltage axis.  The archived CSVs live in
    ``_superseded_20260629/`` with a README saying why they are wrong; this is
    the only place they are ever plotted, and only as the record of what the
    deck used to show.
    """
    sc = SCANS['sat_det3_1']
    root = os.path.join(os.path.dirname(sc['base'].rstrip('/')), 'Analysis',
                        sc['run'])
    new, old = [], []
    for out, tag in (('hv_scan', 'scan'), ('hv_scan2', 'scan2')):
        d = os.path.join(root, out, sc['det_name'])
        n = pd.read_csv(os.path.join(d, 'efficiency_vs_hv.csv'))
        o = pd.read_csv(os.path.join(d, '_superseded_20260629',
                                     f'efficiency_vs_hv_{tag}.csv'))
        new.append(n[['hv', 'within_R', 'within_R_err']])
        old.append(pd.DataFrame(dict(hv=o.x, within_R=o.eff_reco * 100,
                                     within_R_err=o.eff_reco_err * 100)))
    new = pd.concat(new).sort_values('hv')
    old = pd.concat(old).sort_values('hv')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(old.hv, old.within_R, yerr=old.within_R_err, fmt='s--', ms=6,
                lw=1.5, color='0.55', capsize=3,
                label='29 June 2026 — the curve the deck was using\n'
                      '(M3 chi2<5 & NClus$\\geq$3, no significance floor, '
                      'box from the top sub-run)')
    ax.errorbar(new.hv, new.within_R, yerr=new.within_R_err, fmt='o-', ms=7,
                lw=2, color='steelblue', capsize=4,
                label='28 August 2026 — this chain\n'
                      f'(golden M3 chi2<{M3_CHI2_CUT} & NClus={M3_MIN_NCLUS}, '
                      f'sigrel={SIG_REL}, fixed box)')
    ax.axhline(93.13, color='seagreen', lw=1.2, ls=':')
    ax.text(524, 93.9, 'published long run at 490 V: 93.13 %', fontsize=8,
            color='seagreen', ha='right')
    ax.set_xlabel('amplification (resistive-layer) HV [V]')
    ax.set_ylabel('efficiency: within 5 mm of the reference [%]')
    ax.set_title('det3, 27 June saturday scan — the same waveforms, twice',
                 fontsize=11)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    for p in od_pngs or (os.path.join(root, 'hv_scan', sc['det_name'],
                                      'efficiency_before_after.png'),):
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f'wrote {p}')
    plt.close(fig)


RADII = (0.5, 1.0, 2.0, 3.0, 5.0)


def scan_prepped(sc, refit=False):
    """(prepared sub-runs, fixed box) for one scan -- main()'s setup, reusable."""
    seed, _seed_path = seed_params(sc)
    rc = RunConfig(cfg_for(sc, sc['seed_subrun']).run_config_path,
                   cfg_for(sc, sc['seed_subrun']).MAP_CSV_PATH)
    det = rc.get_detector(sc['det_name'])
    prepped = []
    for name, hv in find_subruns(sc):
        a = prepare(sc, name, seed, det, refit=refit)
        if a is not None:
            a['hv'] = hv
            prepped.append(a)
    ref = prepare(sc, sc['seed_subrun'], seed, det, refit=refit)
    rp = ref['recpos']
    box = (*np.percentile(rp[:, 0], BOX_PCT), *np.percentile(rp[:, 1], BOX_PCT))
    return prepped, box


def radii_scan(keys=('sat_det3_1', 'sat_det3_2'), radii=RADII, refit=False):
    """The same events scored at several match radii.

    Everything except the radius is held fixed -- same caches, same box, same
    denominator -- so the only thing that moves between curves is how close the
    reconstructed point has to land.  Cheap: the per-event analysis is cached,
    only ``score`` re-runs.
    """
    rows = []
    for k in keys:
        sc = SCANS[k]
        prepped, box = scan_prepped(sc, refit=refit)
        for a in prepped:
            for R in radii:
                s = score(a, box, R)
                rows.append(dict(scan=k, hv=a['hv'], R_mm=R, subrun=a['subrun'],
                                 **{c: s[c] for c in
                                    ('within_R', 'within_R_err',
                                     'within_R_nospark', 'reco_at_all',
                                     'spark_frac', 'core_sigma_mm',
                                     'median_r_mm', 'n_rays')}))
    return pd.DataFrame(rows).sort_values(['R_mm', 'hv']).reset_index(drop=True)


def fig_radii(df, path, radii=RADII, hero=1.0):
    """Efficiency at several match radii, and the residual scale that sets them."""
    import matplotlib.cm as mcm
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.5, 1],
                                                  hspace=0.08))
    cols = mcm.viridis(np.linspace(.12, .82, len(radii)))
    for R, c in zip(radii, cols):
        d = df[df.R_mm == R].groupby('hv', as_index=False).first().sort_values('hv')
        big = R in (hero, 5.0)
        ax.errorbar(d.hv, d.within_R, yerr=d.within_R_err if big else None,
                    fmt='o-' if big else '.-', ms=7 if big else 4,
                    lw=2.2 if big else 1.1, color=c, capsize=3 if big else 0,
                    alpha=1.0 if big else .65, zorder=3 if big else 2,
                    label=f'within {R:g} mm' + (' (published cut)' if R == 5 else ''))
    d5 = df[df.R_mm == 5.0].groupby('hv', as_index=False).first().sort_values('hv')
    # Consistency curve, NOT a fit: scale the 5 mm efficiency by the Rayleigh
    # containment ratio implied by that sub-run's own median matched residual.
    # If the tight cut carried extra information about DETECTION, the measured
    # points would leave this line.
    sray = d5.median_r_mm / np.sqrt(2 * np.log(2))
    pred = d5.within_R * ((1 - np.exp(-hero ** 2 / (2 * sray ** 2)))
                          / (1 - np.exp(-25.0 / (2 * sray ** 2))))
    ax.plot(d5.hv, pred, '--', lw=1.3, color='crimson', alpha=.85, zorder=4,
            label=f'{hero:g} mm predicted from the 5 mm curve and the measured\n'
                  f'median residual (Rayleigh containment, not a fit)')
    ax.set_ylabel('efficiency: reconstructed within R of the reference [%]')
    # Headroom above 100 % so the legend never sits on a curve: five radii plus
    # the consistency line leave no clear band inside the data.
    ax.set_ylim(0, 134)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper center', ncol=3, framealpha=0,
              handlelength=2.4, columnspacing=1.4)
    ax.set_title('det3, 27 June saturday scan — the same events, five match radii\n'
                 'below ~2 mm the curve stops measuring detection and starts '
                 'measuring the residual width', fontsize=10)

    bx.plot(d5.hv, d5.median_r_mm, 'o-', ms=6, lw=1.8, color='#6a3d9a',
            label='median matched residual |r|')
    bx.plot(d5.hv, d5.core_sigma_mm, 's--', ms=5, lw=1.2, color='#b39ddb',
            label='robust width of the residual core')
    bx.axhline(0.224, color='0.45', lw=1.1, ls=':')
    bx.text(524, 0.245, 'M3 pointing at z = 702: 0.224 mm', fontsize=8,
            color='0.35', ha='right')
    bx.set_xlabel('amplification (resistive-layer) HV [V]')
    bx.set_ylabel('residual [mm]')
    bx.set_ylim(0, 1.38)
    bx.grid(alpha=.3)
    bx.legend(fontsize=8, loc='lower left')
    bx.annotate('', (425, 1.06), (462, 1.06),
                arrowprops=dict(arrowstyle='<->', color='#6a3d9a', lw=1.1))
    bx.text(443, 1.10, 'residual still improving here — 0.97 → 0.79 mm —\n'
                       'long after the 5 mm efficiency has flattened',
            fontsize=8, color='#6a3d9a', ha='center', va='bottom')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan_key', nargs='?')
    ap.add_argument('--r', type=float, default=5.0)
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--radii', action='store_true',
                    help='score the saturday scan at 0.5/1/2/3/5 mm from the '
                         'same caches and plot the family')
    ap.add_argument('--compare', action='store_true',
                    help='plot the archived 29 June saturday curve against this '
                         'chain\'s and stop (needs both already written)')
    ap.add_argument('--box-from', choices=('seed', 'best'), default='seed',
                    help='seed (default) = the long run\'s own box, the one the '
                         'published breakdown uses; best = the sub-run with the '
                         'most reco points, which at high HV is a discharge cloud')
    ap.add_argument('--closure', action='store_true',
                    help='score the seed sub-run through this chain instead')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.compare:
        compare_superseded()
        return

    if args.radii:
        sc = SCANS['sat_det3_1']
        od = os.path.join(os.path.dirname(sc['base'].rstrip('/')), 'Analysis',
                          sc['run'], 'hv_scan', sc['det_name'])
        df = radii_scan(refit=args.refit)
        df.to_csv(os.path.join(od, 'efficiency_vs_hv_radii.csv'), index=False)
        print()
        piv = df.pivot_table(index='hv', columns='R_mm', values='within_R')
        print(piv.round(2).to_string())
        fig_radii(df, os.path.join(od, 'efficiency_vs_hv_radii.png'))
        return

    if args.list or not args.scan_key:
        for k, v in SCANS.items():
            print(f'{k:12s}  {v["run"]}  {v["det_name"]}  '
                  f'prefix={v["prefix"]}  -> {v["out"]}/')
        return
    sc = SCANS[args.scan_key]
    seed, seed_path = seed_params(sc)
    rc = RunConfig(cfg_for(sc, sc['seed_subrun']).run_config_path,
                   cfg_for(sc, sc['seed_subrun']).MAP_CSV_PATH)
    det = rc.get_detector(sc['det_name'])

    if args.closure:
        a = prepare(sc, sc['seed_subrun'], seed, det, refit=args.refit)
        rp = a['recpos']
        box = (*np.percentile(rp[:, 0], BOX_PCT), *np.percentile(rp[:, 1], BOX_PCT))
        s = score(a, box, args.r)
        print(f'\nCLOSURE on {sc["seed_subrun"]} ({sc["det_name"]}):')
        for k in ('n_rays', 'within_R', 'reco_at_all', 'reco_far', 'hit_no_reco',
                  'no_hit', 'spark_cat', 'has_any', 'spark_frac',
                  'core_sigma_mm', 'within_R_nospark'):
            print(f'  {k:16s} {s[k]:10.3f}')
        pub = os.path.join(cfg_for(sc, sc['seed_subrun']).OUT_BASE, 'wft',
                           'efficiency', 'efficiency_breakdown_hits.txt')
        if os.path.exists(pub):
            print(f'\npublished ({pub}):\n' + open(pub).read())
        return

    subruns = find_subruns(sc)
    print(f'{len(subruns)} sub-runs: ' + ', '.join(f'{v}V' for _, v in subruns))
    print(f'seed: {seed_path}\n  {seed}')

    prepped = []
    for name, hv in subruns:
        print(f'\n{"="*66}\n{name}  ({hv} V)\n{"="*66}')
        a = prepare(sc, name, seed, det, refit=args.refit)
        if a is not None:
            a['hv'] = hv
            prepped.append(a)
    if not prepped:
        sys.exit('nothing analysed')

    if args.box_from == 'seed':
        ref = prepare(sc, sc['seed_subrun'], seed, det, refit=args.refit)
        if ref is None:
            sys.exit(f'seed sub-run {sc["seed_subrun"]} could not be prepared')
    else:
        ref = max(prepped, key=lambda a: len(a['recpos']))
    rp = ref['recpos']
    box = (*np.percentile(rp[:, 0], BOX_PCT), *np.percentile(rp[:, 1], BOX_PCT))
    print(f'\nFixed box from {ref["subrun"]} ({len(rp)} reco pts, '
          f'--box-from={args.box_from}): '
          f'x[{box[0]:.1f},{box[1]:.1f}] y[{box[2]:.1f},{box[3]:.1f}]  '
          f'({box[1]-box[0]:.1f} x {box[3]-box[2]:.1f} mm)')

    rows = []
    for a in prepped:
        s = score(a, box, args.r)
        rows.append(dict(hv=a['hv'], subrun=a['subrun'], n_valid=a['n_valid'],
                         n_firing=a['n_firing'], **s))
    df = pd.DataFrame(rows).sort_values('hv').reset_index(drop=True)

    # Legacy aliases, as FRACTIONS, under the exact names script 10 wrote, so the
    # five existing readers of efficiency_vs_hv.csv keep working and pick the
    # corrected numbers up automatically:
    #   mx_june_cosmic_qa/build_hv_scan_pdf.py, build_final_pdf.py
    #   mx_june_wft/report/make_grand_report.py
    #   mpgd26/make_flash_slides.py
    #   ntof_july_analysis/hv_tradeoff/hv_tradeoff.py
    # Everything named in the 02_efficiency vocabulary (within_R, has_any, ...)
    # is a PERCENT; everything in the legacy block is a fraction.  Mixed units in
    # one file is not nice, but a silent factor of 100 in a slide is worse, and
    # the two blocks are named differently on purpose.
    df['eff_reco'] = df['within_R'] / 100.0
    df['eff_reco_err'] = df['within_R_err'] / 100.0
    df['eff_anyhit'] = df['has_any'] / 100.0
    df['spark_frac_pct'] = df['spark_frac']
    df['spark_frac'] = df['spark_frac'] / 100.0

    od = out_dir(sc)
    # The default 5 mm product keeps the plain name -- five readers import it by
    # that path.  Any other radius is written beside it under its own stem, so a
    # tighter cut can never silently replace the published curve.
    stem = ('efficiency_vs_hv' if abs(args.r - 5.0) < 1e-9
            else f'efficiency_vs_hv_r{args.r:g}mm'.replace('.', 'p'))
    df.to_csv(os.path.join(od, f'{stem}.csv'), index=False)
    json.dump(dict(scan_key=args.scan_key, run=sc['run'], det=sc['det_name'],
                   prefix=sc['prefix'], R_mm=args.r, sigrel=SIG_REL,
                   spark_hits=SPARK_HITS, m3_chi2=M3_CHI2_CUT,
                   m3_min_nclus=M3_MIN_NCLUS, seed=seed_path,
                   box=[float(b) for b in box], box_pct=list(BOX_PCT),
                   box_from=args.box_from, ref_subrun=ref['subrun'],
                   basis='hits chain, 02_efficiency accounting',
                   units=('within_R/has_any/reco_*/spark_cat/spark_frac_pct are '
                          'PERCENT; the legacy eff_reco/eff_reco_err/eff_anyhit/'
                          'spark_frac columns are FRACTIONS')),
              open(os.path.join(od, f'{stem}.meta.json'), 'w'), indent=1)

    print(f'\n{"HV":>5} {"within_R":>9} {"+-":>6} {"noSpk":>8} {"has_any":>8} '
          f'{"recoAll":>8} {"spark%":>7} {"sigma":>6} {"rays":>6}')
    for _, r in df.iterrows():
        print(f'{r.hv:>5.0f} {r.within_R:>9.2f} {r.within_R_err:>6.2f} '
              f'{r.within_R_nospark:>8.2f} {r.has_any:>8.2f} {r.reco_at_all:>8.2f} '
              f'{r.spark_frac_pct:>7.2f} {r.core_sigma_mm:>6.3f} {r.n_rays:>6.0f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df.hv, df.within_R, yerr=df.within_R_err, fmt='o-', lw=2, ms=7,
                color='steelblue', capsize=4,
                label=f'within {args.r:g} mm of the reference (published convention)')
    ax.plot(df.hv, df.within_R_nospark, 'v--', ms=5, color='slategray', alpha=.8,
            label='same, discharge crossings removed from the denominator')
    ax.plot(df.hv, df.has_any, 's:', ms=5, color='darkorange', alpha=.8,
            label='fired at all')
    ax.set_xlabel('amplification (resistive-layer) HV [V]')
    ax.set_ylabel('efficiency [%]')
    ax.set_title(f'{sc["det_name"]} — {sc["run"]} / {sc["out"]}\n'
                 f'golden chain: M3 chi2<{M3_CHI2_CUT} & NClus={M3_MIN_NCLUS}, '
                 f'sigrel={SIG_REL}, fixed box')
    ax.set_ylim(0, 102); ax.grid(alpha=.3)
    axs = ax.twinx()
    axs.plot(df.hv, df.spark_frac, 'x:', color='crimson', ms=8, lw=1.5,
             label=f'discharge fraction (mult>{SPARK_HITS})')
    axs.set_ylabel('discharge fraction of firing events [%]', color='crimson')
    axs.tick_params(axis='y', labelcolor='crimson')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axs.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, f'{stem}.png'), dpi=200, bbox_inches='tight')
    print(f'\nwrote {od}/{stem}.{{csv,meta.json,png}}')


if __name__ == '__main__':
    main()

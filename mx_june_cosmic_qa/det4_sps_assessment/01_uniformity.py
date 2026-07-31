#!/usr/bin/env python3
"""
01_uniformity.py — is det4's inefficiency uniform, or are there good regions?

Repeats the `mx_june_wft/02_efficiency.py` per-ray categorisation, but keeps
the *position* of every ray and expresses it in detector-LOCAL coordinates, so
the five categories (no_hit / spark / hit_no_reco / reco_far / reco_near) can be
mapped across the chamber. The question this answers is the one that decides a
beam-test: if the loss is uniform, det4 is a low-gain chamber everywhere and a
test beam only measures that; if it is localised, there is a fiducial region
where det4 behaves like a working detector and a beam test is worth doing.

Control detectors are run through the identical code so "non-uniform" means
non-uniform *relative to a chamber we trust*.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/01_uniformity.py \
        g_det4 sat_det3 o22_long_det2

Outputs: uniformity_<key>.{npz,json,png}, written next to this script.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import compat                                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE            # noqa: E402

CATS = ('no_hit', 'spark', 'hit_no_reco', 'reco_far', 'reco_near')


def ref_to_det(x_ref, y_ref, params):
    """Inverse of cm._det_to_ref: aligned/reference frame -> detector-local mm."""
    th = np.deg2rad(params.theta_deg)
    c, s = np.cos(th), np.sin(th)
    cx, cy = params.centre_x, params.centre_y
    dx = np.asarray(x_ref, float) - cx - params.x_offset
    dy = np.asarray(y_ref, float) - cy - params.y_offset
    return c * dx + s * dy + cx, -s * dx + c * dy + cy


def categorise(key, source='wft', R=5.0):
    """Return a per-ray record array: local x/y, category code, residual."""
    cfg = get_config(key)
    if source == 'wft':
        align_path = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
        params = cm.load_alignment(align_path)
        df = compat.load_table(os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet'),
                               max_dropped=None)
        results = compat.as_event_results(df)
    else:
        align_path = os.path.join(cfg.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json')
        params = cm.load_alignment(align_path)
        results = pickle.load(open(os.path.join(cfg.OUT_BASE, 'cache',
                                                'event_results.pkl'), 'rb'))

    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # --- detection bookkeeping from hits (same convention as 02_efficiency) ---
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'significance'], library='pd')
    det_raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    fired = set(int(e) for e in det_raw['eventId'].unique())
    det_lo, det_hi = int(det_raw['eventId'].min()), int(det_raw['eventId'].max())
    mult_raw = det_raw.groupby('eventId').size()
    mult = (cm.apply_significance_floor(det_raw, rel=SIG_REL_FLOOR)
            .groupby('eventId').size().reindex(mult_raw.index).fillna(0).astype(int))
    mult_by_ev = mult.to_dict()

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    lx, ly = ref_to_det(px, py, params)

    out = dict(x=[], y=[], cat=[], r=[])
    for e, x, y, xl, yl in zip((int(v) for v in evn), px, py, lx, ly):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(xl) and np.isfinite(yl)):
            continue
        if mult_by_ev.get(e, 0) > SPARK_VETO_HITS:
            c, r = 'spark', np.nan
        elif e in reco:
            r = float(np.hypot(x - reco[e][0], y - reco[e][1]))
            c = 'reco_near' if r <= R else 'reco_far'
        elif e in fired:
            c, r = 'hit_no_reco', np.nan
        else:
            c, r = 'no_hit', np.nan
        out['x'].append(xl)
        out['y'].append(yl)
        out['cat'].append(CATS.index(c))
        out['r'].append(r)
    return {k: np.array(v) for k, v in out.items()}, cfg


def summarise(rec, key, cfg, out_dir, cell=25.0):
    """Bin into `cell`-mm squares, write maps + a uniformity summary."""
    ax0, ax1 = TRUE_ACTIVE['x']
    ay0, ay1 = TRUE_ACTIVE['y']
    inside = ((rec['x'] >= ax0) & (rec['x'] <= ax1)
              & (rec['y'] >= ay0) & (rec['y'] <= ay1))
    x, y, c = rec['x'][inside], rec['y'][inside], rec['cat'][inside]

    xe = np.arange(ax0, ax1 + cell, cell)
    ye = np.arange(ay0, ay1 + cell, cell)
    tot, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    maps = {}
    for i, name in enumerate(CATS):
        h, _, _ = np.histogram2d(x[c == i], y[c == i], bins=[xe, ye])
        maps[name] = h
    with np.errstate(invalid='ignore', divide='ignore'):
        eff = np.where(tot >= 20, maps['reco_near'] / tot, np.nan)
        anyh = np.where(tot >= 20, 1 - maps['no_hit'] / tot, np.nan)
        hnr = np.where(tot >= 20, maps['hit_no_reco'] / tot, np.nan)
        spk = np.where(tot >= 20, maps['spark'] / tot, np.nan)

    good = np.isfinite(eff)
    e = eff[good]
    summary = dict(
        run_key=key, detector=cfg.DET_NAME, cell_mm=cell,
        n_rays_active=int(inside.sum()),
        integrated=dict((k, float(maps[k].sum() / max(tot.sum(), 1))) for k in CATS),
        eff_mean=float(np.nansum(maps['reco_near']) / max(tot.sum(), 1)),
        eff_cells=dict(n=int(good.sum()),
                       median=float(np.median(e)), p10=float(np.percentile(e, 10)),
                       p90=float(np.percentile(e, 90)),
                       min=float(e.min()), max=float(e.max()),
                       spread_p90_p10=float(np.percentile(e, 90) - np.percentile(e, 10)),
                       # dispersion beyond binomial counting error
                       rel_rms=float(np.std(e) / max(np.mean(e), 1e-9))),
        area_frac_above=dict((f'{t:.2f}', float(np.mean(e >= t)))
                             for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)),
        has_any_cells=dict(median=float(np.nanmedian(anyh)),
                           p10=float(np.nanpercentile(anyh, 10))),
    )
    # binomial expectation for the observed cell spread: is the structure real?
    n_per = tot[good]
    p = summary['eff_mean']
    exp_rms = float(np.mean(np.sqrt(p * (1 - p) / n_per)))
    summary['eff_cells']['rms_observed'] = float(np.std(e))
    summary['eff_cells']['rms_binomial_expected'] = exp_rms
    summary['eff_cells']['excess_dispersion'] = float(
        np.sqrt(max(np.var(e) - exp_rms ** 2, 0.0)))

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'uniformity_{key}.npz'),
             xe=xe, ye=ye, tot=tot, eff=eff, anyh=anyh, hnr=hnr, spk=spk,
             **{f'n_{k}': v for k, v in maps.items()})
    with open(os.path.join(out_dir, f'uniformity_{key}.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ext = [xe[0], xe[-1], ye[0], ye[-1]]
    panels = [(eff, 'within 5 mm', 0, 1), (anyh, 'has any hit', 0, 1),
              (hnr, 'hit but no reco', 0, 1), (spk, 'spark', 0, 0.3),
              (tot.T * 0 + tot.T, 'rays / cell', None, None)]
    for ax, (m, t, vmn, vmx) in zip(axs.ravel(), panels):
        im = ax.imshow(m.T, origin='lower', extent=ext, aspect='equal',
                       vmin=vmn, vmax=vmx, cmap='viridis')
        ax.set_title(t)
        ax.set_xlabel('detector-local X [mm]')
        ax.set_ylabel('detector-local Y [mm]')
        fig.colorbar(im, ax=ax, fraction=0.046)
    axs[1, 2].hist(e, bins=20)
    axs[1, 2].set_title(f'cell efficiency spread\nmedian {np.median(e):.2f}, '
                        f'rms {np.std(e):.3f} (binom {exp_rms:.3f})')
    axs[1, 2].set_xlabel('within-5 mm efficiency per cell')
    fig.suptitle(f'{key} ({cfg.DET_NAME}) — uniformity, {cell:.0f} mm cells, '
                 f'{int(inside.sum()):,} active-area rays')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'uniformity_{key}.png'), dpi=110)
    print(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--source', default='wft', choices=('wft', 'hits'))
    ap.add_argument('--cell', type=float, default=25.0)
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    for key in args.keys:
        rec, cfg = categorise(key, source=args.source)
        summarise(rec, key, cfg, args.out, cell=args.cell)


if __name__ == '__main__':
    main()

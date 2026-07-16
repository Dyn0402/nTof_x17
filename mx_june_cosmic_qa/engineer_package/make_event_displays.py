#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_event_displays.py

Presentation-quality cosmic-muon event displays from the MX17 detector-3
micro-TPC data (Saturday 6-27 long run, resist 490 V / drift 1000 V),
for the engineer slide package.

Produces (in engineer_package/event_displays/, each as .png and .pdf):
  event_display_track_1..4     two-panel (X/Y plane) strip-time displays with
                               the fitted micro-TPC segment, at ~5/15/25/35 deg
  event_display_waveforms      "anatomy of an event": pedestal+CNS-subtracted
                               waveform heatmaps (both planes) + strip pulses
  event_display_spark          a quenched-discharge event for contrast
  event_display_gallery        2x3 grid of X-plane views (single-slide gallery)
  DISPLAYS.md                  file/event/caption index

Usage (from mx_june_cosmic_qa/):
  ../.venv/bin/python engineer_package/make_event_displays.py
      [--only=tracks,wf,spark,gallery] [--refresh]
      [--pick 1=<eid> 2=<eid> ...]     override auto-picked event per band
      [--wf-eid=<eid>] [--spark-eid=<eid>]
"""
import os
import sys
import glob
import json
import pickle
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
QA_DIR = os.path.dirname(HERE)
sys.path.insert(0, QA_DIR)
from qa_config import get_config, setup_paths          # noqa: E402
setup_paths()
import uproot                                          # noqa: E402
import cosmic_micro_tpc_analysis as cm                 # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles # noqa: E402

# ----------------------------------------------------------------- constants
CFG = get_config('sat_det3')            # det3, FEU 7 = X, FEU 8 = Y, z = 702
OUT = os.path.join(HERE, 'event_displays')
CACHE = os.path.join(OUT, '_cache')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')

SAMPLE_NS = 60.0
PITCH_MM = 0.78
GAP_MM = 30.0
V_UM_NS = 34.0                 # physical drift velocity at 1000 V [um/ns]
V_MM_NS = V_UM_NS / 1000.0
# velocity for converting the fitted strip-time slope to an angle. The
# core-strip OLS estimator used here recovers close to the geometric
# velocity (runbook: estimator shoot-out, script 22), so the physical
# value is the right conversion.
V_ANGLE = 34.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
RES_CUT_MM = 4.0               # tight ray-cluster match for display events
VETO = 50                      # spark veto (hit rows / event)
EID_MAX = 38900                # FEU 8 has no data beyond eid ~38,926 (file 003)
CLUSTER_GAP_MM = 3.0           # single tight cluster requirement for displays
CORE_FRAC = 0.30
N_PED_EVENTS = 300
DATE_STR = '27 June 2026'

# angle bands (deg, X-plane angle): 4 main displays + 2 gallery-only.
# The M3 telescope acceptance caps reference angles at ~23 deg; bands 3/4
# are therefore filled from the telescope-free tier (micro-TPC-only steep
# tracks, selected on internal consistency).
BANDS = {1: (3.5, 7.5), 2: (14.0, 18.0), 3: (24.0, 31.0), 4: (31.0, 45.0),
         5: (9.0, 12.5), 6: (19.0, 23.0)}

CHARGE_CMAP = 'viridis'
ACCENT = '#c2404d'             # fitted-segment / annotation accent
INK = '#1a1a2e'

plt.rcParams.update({
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.6,
    'axes.edgecolor': '#555555', 'axes.labelcolor': INK,
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'text.color': INK,
})


def save(fig, name):
    for ext, dpi in (('png', 220), ('pdf', None)):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=dpi,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.png/.pdf')


# ----------------------------------------------------------------- data load
def load_reference():
    """eid -> (tan_x_det, tan_y_det) for events matched to a single clean ray."""
    cache = os.path.join(CFG.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}',
                              'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    ray_counts = pd.Series(anum).value_counts()
    single_ray = set(ray_counts[ray_counts == 1].index.astype(int))
    th = np.deg2rad(best.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM \
                or np.isnan(r.ref_tan_theta_x) or r.event_id not in single_ray:
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    print(f'{len(ref):,} events matched to a single clean reference ray')
    return ref


def load_hits():
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate(
        [f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'time'],
        library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)]
    df['nrows'] = df.groupby('eventId')['eventId'].transform('size')
    df = cm._map_strip_positions(df, det)
    print(f'{len(df):,} hit rows over {df["eventId"].nunique():,} events')
    return df, det


def largest_cluster(pos, gap=CLUSTER_GAP_MM):
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > gap)[0]
    return max(np.split(o, breaks + 1), key=len)


# ----------------------------------------------------------- event selection
def fit_plane(gp, pcol):
    """OLS fit of core-strip times vs position. Returns dict or None."""
    pos = gp[pcol].to_numpy()
    t = gp['time'].to_numpy()
    amp = gp['amplitude'].to_numpy()
    core = amp >= CORE_FRAC * amp.max()
    if core.sum() < 3 or np.ptp(pos[core]) == 0:
        return None
    m, b = np.polyfit(pos[core], t[core], 1)
    if m == 0:
        return None
    tan_fit = 1000.0 / (V_ANGLE * m)          # dx/dt = v*tan  ->  tan = 1/(v m)
    resid_all = t - (m * pos + b)
    span_t = np.ptp(t[core])
    return dict(slope=m, intercept=b, tan_fit=tan_fit,
                theta_fit=np.degrees(np.arctan(tan_fit)),
                rms=float(np.sqrt(np.mean(resid_all ** 2))),
                span_t=float(span_t), n=len(gp), n_core=int(core.sum()),
                amp_max=float(amp.max()))


def build_candidates(hits, ref, refresh=False):
    csv = os.path.join(CACHE, 'candidates.csv')
    if os.path.exists(csv) and not refresh:
        cand = pd.read_csv(csv)
        print(f'loaded {len(cand):,} cached candidates')
        return cand
    hv = hits[(hits['nrows'] <= VETO) & (hits['eventId'] <= EID_MAX)]
    rows = []
    for eid, g in hv.groupby('eventId'):
        has_ref = eid in ref
        if has_ref:
            tx, ty = ref[eid]
            rec = {'eid': int(eid), 'theta_ref_x': np.degrees(np.arctan(tx)),
                   'theta_ref_y': np.degrees(np.arctan(ty))}
        else:
            rec = {'eid': int(eid), 'theta_ref_x': np.nan,
                   'theta_ref_y': np.nan}
        ok = True
        for p, pcol in (('x', 'x_position_mm'), ('y', 'y_position_mm')):
            gp = g[g[pcol].notna()]
            if not (6 <= len(gp) <= 40):
                ok = False
                break
            pos = gp[pcol].to_numpy()
            idx = largest_cluster(pos)
            if len(idx) != len(gp):          # stray hits -> not display-clean
                ok = False
                break
            f = fit_plane(gp, pcol)
            if f is None or f['span_t'] < 250.0:
                ok = False
                break
            rec.update({f'{p}_{k}': v for k, v in f.items()})
        if not ok:
            continue
        if has_ref:
            # fitted angle must agree with the telescope (both planes)
            if abs(rec['x_theta_fit'] - rec['theta_ref_x']) > 1.5:
                continue
            if abs(rec['y_theta_fit'] - rec['theta_ref_y']) > 3.0:
                continue
            rec['score'] = (
                rec['x_rms'] / rec['x_span_t']
                + 0.5 * rec['y_rms'] / rec['y_span_t']
                + 0.04 * abs(rec['x_theta_fit'] - rec['theta_ref_x'])
                + 0.015 * abs(rec['y_theta_fit'] - rec['theta_ref_y']))
        else:
            # telescope-free tier: only steep tracks (outside the M3
            # acceptance) with strong internal consistency in BOTH planes:
            # a full-gap drift-time span and long clean ladders.
            if not (24.0 <= abs(rec['x_theta_fit']) <= 50.0):
                continue
            if rec['x_n'] < 13 or abs(rec['y_theta_fit']) > 50.0:
                continue
            if not (450.0 <= rec['x_span_t'] <= 900.0):
                continue
            if not (450.0 <= rec['y_span_t'] <= 900.0):
                continue
            rec['score'] = (rec['x_rms'] / rec['x_span_t']
                            + 0.5 * rec['y_rms'] / rec['y_span_t'] + 0.015)
        rows.append(rec)
    cand = pd.DataFrame(rows).sort_values('score')
    # band-selection angle: telescope angle when available, else the fit
    cand['theta_sel'] = cand['theta_ref_x'].fillna(cand['x_theta_fit'])
    cand.to_csv(csv, index=False)
    amax = cand['theta_ref_x'].abs()
    print(f'{len(cand):,} display-clean candidates; |theta_ref_x| '
          f'p50/p90/max = {amax.median():.1f}/{amax.quantile(0.9):.1f}/'
          f'{amax.max():.1f} deg')
    return cand


def pick_events(cand, overrides):
    picks = {}
    for band, (lo, hi) in BANDS.items():
        if band in overrides:
            row = cand[cand['eid'] == overrides[band]]
            if len(row) == 0:
                raise SystemExit(f'override eid {overrides[band]} not a candidate')
            picks[band] = row.iloc[0]
            continue
        sel = cand[cand['theta_sel'].abs().between(lo, hi)]
        if len(sel) == 0:
            print(f'  WARNING: no candidate in band {band} ({lo}-{hi} deg)')
            continue
        picks[band] = sel.nsmallest(1, 'score').iloc[0]
    for band, row in sorted(picks.items()):
        r = (f"{row['theta_ref_x']:+.1f}" if np.isfinite(row['theta_ref_x'])
             else 'none')
        print(f"  band {band} ({BANDS[band][0]:.0f}-{BANDS[band][1]:.0f} deg): "
              f"eid {int(row['eid'])}  thx_ref={r} "
              f"fit={row['x_theta_fit']:+.1f}  ny={row['y_n']:.0f} "
              f"score={row['score']:.4f}")
    return picks


# --------------------------------------------------------- track displays
def draw_plane(ax, gp, pcol, fit, theta_ref, norm, label):
    pos = gp[pcol].to_numpy()
    t = gp['time'].to_numpy()
    amp = gp['amplitude'].to_numpy()
    # fitted segment (drawn first, under the markers)
    xx = np.linspace(pos.min() - 0.6, pos.max() + 0.6, 2)
    ax.plot(xx, fit['slope'] * xx + fit['intercept'], color=ACCENT, lw=2.2,
            zorder=2, solid_capstyle='round')
    sizes = 60 + 320 * (amp / amp.max())
    ax.scatter(pos, t, c=amp, cmap=CHARGE_CMAP, norm=norm, s=sizes,
               edgecolors='white', linewidths=1.2, zorder=3)
    pad_x = max(1.5, 0.15 * np.ptp(pos))
    ax.set_xlim(pos.min() - pad_x, pos.max() + pad_x)
    t_lo, t_hi = t.min(), t.max()
    pad_t = max(60, 0.12 * (t_hi - t_lo))
    ax.set_ylim(t_lo - pad_t, t_hi + pad_t)
    ax.set_xlabel(f'strip position, {label} [mm]')
    ax.set_ylabel('electron drift time [ns]')
    # secondary axis: drift time -> drift distance (relative to earliest strip)
    t0 = t_lo
    sec = ax.secondary_yaxis(
        'right', functions=(lambda tt: (tt - t0) * V_MM_NS,
                            lambda d: d / V_MM_NS + t0))
    sec.set_ylabel('drift distance [mm]', fontsize=13)
    sec.spines['right'].set_visible(True)
    sec.tick_params(labelsize=11, colors='#555555')
    sec.yaxis.label.set_color(INK)
    # angle annotation: put it in the corner away from the track
    corner = 0.04 if fit['slope'] > 0 else 0.60
    ax.text(corner, 0.975,
            f"track angle {abs(fit['theta_fit']):.0f}°",
            transform=ax.transAxes, ha='left', va='top', fontsize=14.5,
            color=ACCENT, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none',
                      pad=1.5))
    sub = (f"(telescope: {abs(theta_ref):.0f}°)"
           if np.isfinite(theta_ref) else '(outside telescope acceptance)')
    ax.text(corner, 0.905, sub,
            transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
            color='#888888',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none',
                      pad=1.0))


def make_track_figure(hits, row, n_label):
    eid = int(row['eid'])
    g = hits[hits['eventId'] == eid]
    fig = plt.figure(figsize=(13.2, 6.6))
    ax1 = fig.add_axes([0.065, 0.225, 0.295, 0.60])
    ax2 = fig.add_axes([0.50, 0.225, 0.295, 0.60])
    cax = fig.add_axes([0.925, 0.225, 0.014, 0.60])
    amps = g['amplitude'].to_numpy()
    norm = mcolors.Normalize(vmin=0, vmax=amps.max())
    for ax, (p, pcol, label) in zip((ax1, ax2),
                                    (('x', 'x_position_mm', 'X plane'),
                                     ('y', 'y_position_mm', 'Y plane'))):
        gp = g[g[pcol].notna()]
        fit = {k[2:]: row[k] for k in row.index if k.startswith(f'{p}_')}
        theta_ref = row[f'theta_ref_{p}']
        draw_plane(ax, gp, pcol, fit, theta_ref, norm, label)
        ax.set_title(label, fontsize=15, pad=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CHARGE_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('signal amplitude [ADC]', fontsize=12.5)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)
    fig.suptitle('Cosmic muon crossing detector 3 — micro-TPC readout',
                 fontsize=17.5, fontweight='bold', color=INK, y=0.965)
    fig.text(0.065, 0.022,
             'Each point is one strip pulse; the electron arrival time measures '
             'the depth of the ionization in the 30 mm drift gap.\n'
             'Point size and color scale with the collected charge; the red '
             'line is the fitted track segment.',
             fontsize=11.5, color='#666666', ha='left', va='bottom')
    fig.text(0.939, 0.022, f'event {eid} · {DATE_STR}',
             fontsize=9.5, color='#999999', ha='right', va='bottom')
    save(fig, f'event_display_track_{n_label}')


# --------------------------------------------------------- waveform helpers
def plane_positions(det, feu, axis):
    pos = np.full(512, np.nan)
    for ch in range(512):
        p = det.map_hit(feu, ch)
        if p is not None and p[axis] is not None:
            pos[ch] = p[axis]
    return pos


def file_pedestal(fn):
    """Per-channel pedestal (median over first N events x 32 samples), cached."""
    key = os.path.join(CACHE, os.path.basename(fn).replace('.root', '_ped.npy'))
    if os.path.exists(key):
        return np.load(key)
    t = uproot.open(fn)['nt']
    a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS,
                  library='np')['amplitude']
    stack = np.stack([a.reshape(32, 512) for a in a0]).astype(np.float32)
    ped = np.median(stack, axis=(0, 1))
    np.save(key, ped)
    return ped


def load_event_waveforms(feu, eid, cns=True):
    """(32, 512) pedestal(+CNS)-subtracted waveform image for one event."""
    for fn in sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root'))):
        key = os.path.join(CACHE,
                           os.path.basename(fn).replace('.root', '_eids.npy'))
        if os.path.exists(key):
            eids = np.load(key)
        else:
            eids = uproot.open(fn)['nt'].arrays(['eventId'],
                                                library='np')['eventId']
            np.save(key, eids)
        idx = np.where(eids == eid)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        t = uproot.open(fn)['nt']
        arr = t.arrays(['amplitude'], entry_start=i, entry_stop=i + 1,
                       library='np')['amplitude'][0]
        w = arr.reshape(32, 512).astype(np.float32) - file_pedestal(fn)
        if cns:
            cms = np.median(w.reshape(32, 8, 64), axis=2)
            w = w - np.repeat(cms, 64, axis=1)
        return w
    raise RuntimeError(f'event {eid} not found in decoded FEU {feu} files')


def plane_image(w, pos):
    """Sort channels by strip position -> (image, extent)."""
    ok = np.isfinite(pos)
    order = np.argsort(pos[ok])
    chans = np.where(ok)[0][order]
    p = pos[chans]
    img = w[:, chans]
    extent = [p[0] - PITCH_MM / 2, p[-1] + PITCH_MM / 2,
              -SAMPLE_NS / 2, 31.5 * SAMPLE_NS]
    return img, extent, p, chans


# --------------------------------------------------------- waveform figure
def make_waveform_figure(hits, row, det):
    import matplotlib.transforms as mtransforms
    eid = int(row['eid'])
    g = hits[hits['eventId'] == eid]
    t_all = g['time'].to_numpy()
    t_lo = max(-SAMPLE_NS / 2, t_all.min() - 350)
    t_hi = min(31.5 * SAMPLE_NS, t_all.max() + 500)
    t0 = t_all.min()

    fig = plt.figure(figsize=(13.2, 9.8))
    ax_x = fig.add_axes([0.065, 0.415, 0.46, 0.455])
    ax_y = fig.add_axes([0.615, 0.415, 0.24, 0.455])
    cax = fig.add_axes([0.935, 0.415, 0.013, 0.455])
    wf_axes = [fig.add_axes([x0, 0.065, 0.245, 0.21])
               for x0 in (0.065, 0.395, 0.725)]

    imgs = {}
    for p, feu, axis, ax, label in (('x', CFG.MX17_FEU_X, 0, ax_x, 'X plane'),
                                    ('y', CFG.MX17_FEU_Y, 1, ax_y, 'Y plane')):
        w = load_event_waveforms(feu, eid, cns=True)
        pos = plane_positions(det, feu, axis)
        img, extent, pvec, chans = plane_image(w, pos)
        imgs[p] = (img, extent, pvec, chans, w)
        gp = g[g[f'{p}_position_mm'].notna()]
        c_lo = gp[f'{p}_position_mm'].min()
        c_hi = gp[f'{p}_position_mm'].max()
        pad = 13.0 if p == 'x' else 7.0
        vmax = np.percentile(img, 99.98)
        im = ax.imshow(np.clip(img, 0, None), origin='lower', aspect='auto',
                       extent=extent, cmap=CHARGE_CMAP,
                       norm=mcolors.PowerNorm(0.6, vmin=0, vmax=vmax))
        ax.set_xlim(c_lo - pad, c_hi + pad)
        ax.set_ylim(t_lo, t_hi)
        ax.set_xlabel(f'strip position, {label[0]} [mm]')
        if p == 'x':
            ax.set_ylabel('drift time [ns]')
        ax.set_title(label, fontsize=15, pad=8,
                     loc='left' if p == 'x' else 'center')
        ax.grid(False)
        # secondary drift-distance axis on the Y-plane (right) panel
        if p == 'y':
            sec = ax.secondary_yaxis(
                'right', functions=(lambda tt: (tt - t0) * V_MM_NS,
                                    lambda d: d / V_MM_NS + t0))
            sec.set_ylabel('drift distance [mm]', fontsize=12.5)
            sec.tick_params(labelsize=11, colors='#555555')

    cb = fig.colorbar(im, cax=cax)
    cb.set_label('signal amplitude [ADC]', fontsize=12.5)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)

    # ---- three sample strip waveforms across the X-plane cluster ----
    img, extent, pvec, chans, w = imgs['x']
    gp = g[g['x_position_mm'].notna()].sort_values('x_position_mm')
    n = len(gp)
    sel = gp.iloc[[1, n // 2, n - 2]] if n >= 5 else gp.iloc[[0, n // 2, n - 1]]
    letters = ['a', 'b', 'c']
    ts = np.arange(32) * SAMPLE_NS
    trans = mtransforms.blended_transform_factory(ax_x.transData,
                                                  ax_x.transAxes)
    line_c = plt.get_cmap(CHARGE_CMAP)(0.38)
    fill_c = plt.get_cmap(CHARGE_CMAP)(0.60)
    for k, ((_, s), ax) in enumerate(zip(sel.iterrows(), wf_axes)):
        ch = int(s['channel'])
        wf = w[:, ch]
        ax.fill_between(ts, 0, wf, color=fill_c, alpha=0.35)
        ax.plot(ts, wf, color=line_c, lw=2.0)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_xlabel('drift time [ns]', fontsize=12.5)
        if k == 0:
            ax.set_ylabel('amplitude [ADC]', fontsize=12.5)
        ax.set_title(f'({letters[k]})  single strip at '
                     f'{s["x_position_mm"]:.1f} mm', fontsize=12.5)
        ax.set_xlim(0, 31 * SAMPLE_NS)
        ax.tick_params(labelsize=11)
        # marker on the heatmap (data x, axes-fraction y)
        ax_x.text(s['x_position_mm'], 1.015, f'({letters[k]})',
                  transform=trans, ha='center', va='bottom', fontsize=12,
                  color=INK, fontweight='bold')
        ax_x.plot([s['x_position_mm']], [0.955], marker='v', color='white',
                  ms=8, mec='#333333', mew=0.8, transform=trans,
                  clip_on=False, zorder=6)

    fig.suptitle('Anatomy of a cosmic-muon event — raw waveforms, '
                 'detector 3 micro-TPC', fontsize=17.5, fontweight='bold',
                 y=0.965)
    fig.text(0.065, 0.335,
             'The inclined muon leaves a diagonal charge stripe: strips hit '
             'deeper in the 30 mm drift gap arrive later. '
             'Below: pulses of three individual strips, marked (a)–(c).',
             fontsize=11.5, color='#666666', ha='left', va='top')
    ax_y.text(0.96, 0.025, f'event {eid} · {DATE_STR}',
              transform=ax_y.transAxes, fontsize=9.5, color='white',
              alpha=0.85, ha='right', va='bottom')
    save(fig, 'event_display_waveforms')


# ------------------------------------------------------------- spark figure
def find_spark_eid(hits):
    mult = hits.groupby('eventId').size()
    m7 = hits[hits['feu'] == CFG.MX17_FEU_X].groupby('eventId').size()
    m8 = hits[hits['feu'] == CFG.MX17_FEU_Y].groupby('eventId').size()
    both = pd.DataFrame({'m7': m7, 'm8': m8}).fillna(0)
    both = both[(both.index <= EID_MAX) & (both.min(axis=1) > 60)]
    both['tot'] = both.sum(axis=1)
    both = both.sort_values('tot', ascending=False)
    print('spark candidates (eid, nX, nY):')
    print(both.head(8).to_string())
    return [int(e) for e in both.index[:8]], mult


def make_spark_figure(hits, det, eid):
    g = hits[hits['eventId'] == eid]
    n_strips = len(g)
    fig = plt.figure(figsize=(13.2, 6.8))
    ax1 = fig.add_axes([0.06, 0.225, 0.40, 0.615])
    ax2 = fig.add_axes([0.51, 0.225, 0.40, 0.615])
    cax = fig.add_axes([0.925, 0.225, 0.014, 0.615])
    for p, feu, axis, ax, label in (('x', CFG.MX17_FEU_X, 0, ax1, 'X plane'),
                                    ('y', CFG.MX17_FEU_Y, 1, ax2, 'Y plane')):
        w = load_event_waveforms(feu, eid, cns=False)   # keep the common mode
        pos = plane_positions(det, feu, axis)
        img, extent, pvec, chans = plane_image(w, pos)
        vmax = np.percentile(img, 99.9)
        im = ax.imshow(np.clip(img, 0, None), origin='lower', aspect='auto',
                       extent=extent, cmap=CHARGE_CMAP,
                       norm=mcolors.PowerNorm(0.6, vmin=0, vmax=vmax))
        ax.set_xlabel(f'strip position, {label[0]} [mm]')
        if p == 'x':
            ax.set_ylabel('drift time [ns]')
        ax.set_title(label, fontsize=15, pad=8)
        ax.grid(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('signal amplitude [ADC]', fontsize=12.5)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)
    fig.suptitle('Quenched discharge (spark) — for contrast with muon tracks',
                 fontsize=17.5, fontweight='bold', y=0.965)
    fig.text(0.06, 0.03,
             f'A discharge fires {n_strips} strips at once across the full '
             '40 cm plane — unmistakably different from a muon track.\n'
             'The resistive strips quench it within a single readout window.',
             fontsize=11.5, color='#666666', ha='left', va='bottom')
    fig.text(0.939, 0.03, f'event {eid} · {DATE_STR}',
             fontsize=9.5, color='#999999', ha='right', va='bottom')
    save(fig, 'event_display_spark')


# ------------------------------------------------------------ gallery figure
def make_gallery(hits, picks):
    order = [b for b in sorted(picks, key=lambda b: abs(picks[b]['theta_sel']))]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
    fig.subplots_adjust(left=0.06, right=0.975, top=0.885, bottom=0.135,
                        wspace=0.24, hspace=0.30)
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    for ax, band in zip(axes.ravel(), order):
        row = picks[band]
        eid = int(row['eid'])
        g = hits[(hits['eventId'] == eid) & hits['x_position_mm'].notna()]
        pos = g['x_position_mm'].to_numpy()
        t = g['time'].to_numpy()
        amp = g['amplitude'].to_numpy()
        norm = mcolors.Normalize(vmin=0, vmax=amp.max())
        xx = np.linspace(pos.min() - 0.6, pos.max() + 0.6, 2)
        ax.plot(xx, row['x_slope'] * xx + row['x_intercept'], color=ACCENT,
                lw=2.0, zorder=2)
        ax.scatter(pos, t, c=amp, cmap=CHARGE_CMAP, norm=norm,
                   s=40 + 190 * amp / amp.max(), edgecolors='white',
                   linewidths=1.0, zorder=3)
        pad_x = max(1.2, 0.14 * np.ptp(pos))
        ax.set_xlim(pos.min() - pad_x, pos.max() + pad_x)
        pad_t = max(60, 0.12 * np.ptp(t))
        ax.set_ylim(t.min() - pad_t, t.max() + pad_t)
        ax.text(0.04, 0.96, f"{abs(row['x_theta_fit']):.0f}°",
                transform=ax.transAxes, ha='left', va='top', fontsize=16,
                color=ACCENT, fontweight='bold')
        ax.text(0.97, 0.03, f'event {eid}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, color='#999999')
        ax.tick_params(labelsize=10.5)
    for ax in axes[1]:
        ax.set_xlabel('strip position [mm]', fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel('drift time [ns]', fontsize=13)
    fig.suptitle('Cosmic muons in detector 3 — micro-TPC tracks at '
                 'increasing angle (X plane)', fontsize=17.5,
                 fontweight='bold', y=0.965)
    fig.text(0.06, 0.03,
             'Each point is one strip pulse; point size and color scale with '
             'the collected charge; the red line is the fitted track segment. '
             'Steeper tracks span more strips.',
             fontsize=11.5, color='#666666', ha='left', va='bottom')
    save(fig, 'event_display_gallery')


# -------------------------------------------------------------- DISPLAYS.md
def write_md(picks, wf_eid, spark_eid):
    angs = sorted(abs(picks[b]['x_theta_fit']) for b in picks)
    lines = [
        '# Detector 3 micro-TPC event displays (engineer slide package)',
        '',
        f'Run: `{CFG.RUN}/{CFG.SUB_RUN}` (resist 490 V, drift 1000 V, '
        f'{DATE_STR}). Drift velocity 34 um/ns; strip pitch 0.78 mm; '
        '30 mm drift gap; 32 samples x 60 ns readout.',
        '',
        '| file | event | angle (X plane) | caption |',
        '|---|---|---|---|',
    ]
    for k, band in enumerate(sorted(b for b in picks if b <= 4), start=1):
        r = picks[band]
        lines.append(
            f"| `event_display_track_{k}.png/.pdf` | {int(r['eid'])} | "
            f"{abs(r['x_theta_fit']):.0f} deg | Cosmic muon crossing the 30 mm "
            f"drift gap; each point is one strip pulse, the line is the fitted "
            f"track segment. |")
    lines += [
        f"| `event_display_waveforms.png/.pdf` | {wf_eid} | — | Raw "
        "waveforms of one inclined muon: the diagonal charge stripe shows "
        "deeper ionization arriving later; insets show single-strip pulses. |",
        f"| `event_display_spark.png/.pdf` | {spark_eid} | — | A quenched "
        "discharge for contrast: hundreds of strips fire simultaneously, "
        "trivially separated from muon tracks. |",
        f"| `event_display_gallery.png/.pdf` | {len(picks)} events | "
        f"{angs[0]:.0f}–{angs[-1]:.0f} deg | Gallery of micro-TPC muon "
        "tracks at increasing angle (X plane). |",
        '',
        'Selection: clean single cluster in both planes; events below ~23 deg '
        'are matched to a single reference-telescope ray with fitted angle '
        'consistent with the telescope; steeper events (tracks 3-4) lie '
        'outside the telescope acceptance and are selected on internal '
        'micro-TPC consistency alone (full-gap drift-time span and clean '
        'linear ladders in BOTH planes) — their quoted angles come from the '
        'micro-TPC fit only. Absolute drift time contains an arbitrary '
        'trigger offset; the secondary axis converts relative time to drift '
        'distance (34 um/ns).',
    ]
    with open(os.path.join(OUT, 'DISPLAYS.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  wrote DISPLAYS.md')


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='tracks,wf,spark,gallery')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--pick', nargs='*', default=[],
                    help='band=eid overrides, e.g. --pick 3=12345')
    ap.add_argument('--wf-eid', type=int, default=None)
    ap.add_argument('--spark-eid', type=int, default=None)
    args = ap.parse_args()
    only = set(args.only.split(','))
    overrides = {int(k): int(v) for k, v in
                 (s.split('=') for s in args.pick)}

    os.makedirs(CACHE, exist_ok=True)
    ref = load_reference()
    hits, det = load_hits()
    cand = build_candidates(hits, ref, refresh=args.refresh)
    picks = pick_events(cand, overrides)

    if 'tracks' in only:
        for k, band in enumerate(sorted(b for b in picks if b <= 4), start=1):
            make_track_figure(hits, picks[band], k)
    if args.wf_eid is None:
        # steepest of the picked events, capped at ~30 deg
        band = max((b for b in picks if abs(picks[b]['x_theta_fit']) < 31),
                   key=lambda b: abs(picks[b]['x_theta_fit']))
        wf_row = picks[band]
        wf_eid = int(wf_row['eid'])
    else:
        wf_eid = args.wf_eid
        wf_row = cand[cand['eid'] == wf_eid].iloc[0]
    if 'wf' in only:
        make_waveform_figure(hits, wf_row, det)
    spark_eid = args.spark_eid
    if spark_eid is None:
        spark_eid = find_spark_eid(hits)[0][0]
    if 'spark' in only:
        make_spark_figure(hits, det, spark_eid)
    if 'gallery' in only:
        make_gallery(hits, picks)
    write_md(picks, wf_eid, spark_eid)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()

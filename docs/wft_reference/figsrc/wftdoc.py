"""
Shared setup for the WFT reference-document figures.

Every figure in `docs/wft_reference/` is generated from real data by one of the
`f_*.py` scripts in this directory. Nothing is schematic unless it is an inline
SVG in the document itself.

Spine dataset: `sat_det3` — mx17_3, Saturday long run, resistive 490 V /
drift 1000 V, the run the whole waveform-first study was built on. Its live
`lp` calibration bundle, the 400-event ref-pinned calibration cache and the
reconstructed event table are all on disk beside the data.

Figures are rendered with a transparent background and mid-grey chrome so the
same PNG reads correctly on the document's light and dark themes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for _p in (REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
           os.path.join(REPO, 'cosmic_bench_analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

# --------------------------------------------------------------------- paths
RUN_KEY = 'sat_det3'
FIGDIR = os.environ.get(
    'WFT_DOC_FIGDIR',
    '/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17/'
    '6d4eafa1-3125-425d-94fe-b5fb7b7ea0b0/scratchpad/figs')

ANALYSIS = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
            'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3')
# Re-pointed 2026-08-13 to the frozen production products, then AGAIN on
# 2026-08-21 to `calib_bundle_r06`, when every product built on an inverted
# sharing kernel (c2 > c1) was retired and the gate in wft.calib made loading
# one an error. The promoted `events.parquet` beside it is the r06
# reconstruction, so the bundle and the table are the same calibration again.
# Overridable with WFT_DOC_BUNDLE, but only a physical bundle will load.
BUNDLE = os.environ.get(
    'WFT_DOC_BUNDLE', os.path.join(ANALYSIS, 'wft', 'calib_bundle_r06'))
CALIB_CACHE = os.path.join(ANALYSIS, 'wft', 'calib_work', 'calib_cache.pkl')
EVENTS = os.path.join(ANALYSIS, 'wft', 'events.parquet')
DET4_ANALYSIS = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
                 'mx17_det4_day_6-24-26/long_run/mx17_4')

# ------------------------------------------------------------------- palette
# Chosen to stay legible on both a near-white and a near-black card.
C = dict(
    blue='#3b82f6', orange='#f97316', green='#16a34a', red='#e11d48',
    purple='#8b5cf6', teal='#0d9488', pink='#db2777', olive='#a16207',
    grey='#6b7280', ink='#4b5563',
    x='#3b82f6', y='#e11d48',
    model='#f97316', data='#3b82f6', ref='#16a34a', prod='#db2777',
)

CHROME = '#6b7280'


def style():
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none',
        'savefig.transparent': True,
        'axes.edgecolor': CHROME,
        'axes.labelcolor': CHROME,
        'axes.titlecolor': CHROME,
        'xtick.color': CHROME,
        'ytick.color': CHROME,
        'text.color': CHROME,
        'axes.grid': True,
        'grid.color': CHROME,
        'grid.alpha': 0.18,
        'grid.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'legend.framealpha': 0.0,
        'legend.labelcolor': CHROME,
        'font.size': 9.5,
        'lines.linewidth': 1.6,
        'figure.dpi': 130,
    })


style()


def save(fig, name, tight=True, pad=0.25):
    """Write one figure PNG into FIGDIR and report its size."""
    os.makedirs(FIGDIR, exist_ok=True)
    if tight:
        fig.tight_layout(pad=pad)
    path = os.path.join(FIGDIR, name + '.png')
    fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    print(f'  {name}.png  {kb:6.0f} kB')
    return path


# ----------------------------------------------------------------- data load
def cfg():
    from qa_config import get_config, setup_paths
    setup_paths()
    return get_config(RUN_KEY)


def bundle(path=BUNDLE):
    from wft.calib import CalibrationBundle
    return CalibrationBundle.load(path)


def install(path=BUNDLE):
    """Load the det3 lp bundle and install it in wft.model. Returns the bundle."""
    from wft import model as wm
    cal = bundle(path)
    wm.use_calibration(cal)
    return cal


def calib_events(path=CALIB_CACHE):
    """The 400-event ref-pinned calibration cache: waveform windows plus the M3
    reference geometry (mesh anchor and rotated tangents) for each event."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def events_table(path=EVENTS):
    from wft import compat
    return compat.load_table(path)


def pick_events(evs, plane='x', tan_lo=0.18, tan_hi=0.40, min_amp=600,
                max_amp=3400, n=12):
    """Clean, clearly-inclined, unsaturated events — the ones worth displaying."""
    out = []
    for eid in sorted(evs):
        e = evs[eid]
        if plane not in e:
            continue
        t = abs(e[f'tan_{plane}'])
        if not (tan_lo <= t <= tan_hi):
            continue
        a = float(np.asarray(e[plane]['W']).max())
        if not (min_amp <= a <= max_amp):
            continue
        out.append(eid)
        if len(out) >= n:
            break
    return out


def label(ax, s, loc='upper left', **kw):
    ax.set_title(s, loc='left' if loc == 'upper left' else 'center', **kw)


def trim_window(P, keep_frac=0.02, pad=3):
    """Cut a calibration-cache window (which spans the whole ±5 mm reference
    corridor) down to the strips that actually carry charge, plus a pad — i.e.
    to what `wft.io.extract_window` would hand the fit in production.

    'Carries charge' uses the same significance idea as the production hit
    finder: above 5 sigma of that strip's own noise, and above a small fraction
    of the event's peak."""
    W = np.asarray(P['W'], float)
    noise = np.maximum(np.asarray(P['noise'], float), 3.0)
    amp = W.max(axis=1)
    live = np.where((amp > 5 * noise) & (amp > keep_frac * amp.max()))[0]
    if len(live) == 0:
        live = np.array([int(np.argmax(amp))])
    lo = max(0, live.min() - pad)
    hi = min(W.shape[0] - 1, live.max() + pad)
    sl = slice(lo, hi + 1)
    return dict(W=W[sl], pos=np.asarray(P['pos'], float)[sl],
                noise=np.asarray(P['noise'], float)[sl],
                ch=np.asarray(P['ch'])[sl])


def rank_events(evs, plane='x', tan_lo=0.15, tan_hi=0.40, min_amp=500,
                max_amp=3400, n_scan=250, n_keep=12, single_cluster=True):
    """Rank calibration-cache events by how cleanly the *reference-pinned*
    model describes them: low chi2/dof, unsaturated, clearly inclined, and
    (optionally) a single contiguous charge cluster.

    Used only to choose which events to *display*. The physics claims in the
    document are always population-level.
    """
    from wft import model as wm
    cal = bundle()
    if wm.CAL is None:
        wm.use_calibration(cal)
    out = []
    for eid in sorted(evs)[:n_scan]:
        e = evs[eid]
        if plane not in e:
            continue
        tan = e[f'tan_{plane}']
        if not (tan_lo <= abs(tan) <= tan_hi):
            continue
        P = trim_window(e[plane])
        W = P['W']
        a = float(W.max())
        if not (min_amp <= a <= max_amp):
            continue
        amp = W.max(axis=1)
        live = amp > 0.08 * amp.max()
        if single_cluster and (np.diff(np.where(live)[0]) > 1).any():
            continue
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        p0 = e[f'ref_mesh_{plane}']
        w = tan * cal.v_drift * 1e-3
        try:
            r = wm.fit_plane_raw(P, plane, p0, w, 400.0, fix_p0w=(p0, w))
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        out.append((r['chi2'] / max(r['dof'], 1), int(eid), a, float(tan)))
    out.sort()
    for c, eid, a, tan in out[:n_keep]:
        print(f'    event {eid:6d}  chi2/dof {c:7.1f}  peak {a:6.0f} ADC  '
              f'tan {tan:+.3f}')
    return [o[1] for o in out[:n_keep]]

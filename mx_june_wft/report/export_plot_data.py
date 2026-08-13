#!/usr/bin/env python3
"""
export_plot_data.py — the numbers behind every fleet figure, as plain CSV.

The report's figures are composites rendered at fixed scale: readable as a
summary, useless for looking closely at one thing, and impossible to re-cut a
different way. This writes the underlying data instead, so any plot can be
rebuilt — different binning, different cuts, different tool — without touching
the reconstruction.

Per key, into <OUT_BASE>/wft/plot_data/:

    rays.csv      one row per M3 reference ray inside the active box: where the
                  ray crossed, what (if anything) was reconstructed there, the
                  residual, both planes' fit quality, and the reference angle
                  the plane should have measured. Every per-detector figure in
                  the report is some projection of this table.
    summary.json  the scalars (efficiency, core sigma, angle bias/sigma68,
                  active box, alignment) plus provenance.
    COLUMNS.md    what each column means and its units.

Fleet level, into Analysis/fleet_report/plot_data/:

    fleet.csv     one row per detector — the fleet table, machine-readable.

    ../../.venv/bin/python mx_june_wft/report/export_plot_data.py [keys...]
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'mx_june_wft')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                    # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles    # noqa: E402
from make_june_figs import per_ray_table, JUNE_KEYS       # noqa: E402
from wft import compat                                    # noqa: E402

FLEET_REPORT = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report'
LETTER = {'g_det3_wknd': 'A', 'o22_long_det2': 'B', 'g_det6_long': 'C',
          'g_det7_long': 'D', 'g_det4': 'E'}

# Columns lifted from events.parquet onto the ray table. Everything a figure
# colours, cuts or bins by -- keep this list generous, the table is small.
FIT_COLS = ['x_ok', 'y_ok', 'x_p0', 'y_p0', 'x_w', 'y_w',
            'x_t0', 'y_t0', 'x_ftst', 'y_ftst',
            'x_tan_theta', 'y_tan_theta', 'x_theta_deg', 'y_theta_deg',
            'x_chi2', 'y_chi2', 'x_dof', 'y_dof',
            'x_p0_err', 'y_p0_err', 'x_tan_err', 'y_tan_err',
            'x_q_sum', 'y_q_sum', 'x_q_u50', 'y_q_u50',
            'x_n_strips', 'y_n_strips', 'x_n_dropped', 'y_n_dropped',
            'x_slope_reliable', 'y_slope_reliable',
            'x_quality_ok', 'y_quality_ok',
            'x_n_candidates', 'y_n_candidates', 'n_tracks', 'n_hits']

COLUMNS_MD = """# rays.csv — column reference

One row per M3 reference ray that crossed the detector's active box during this
run. This is the table every per-detector figure in the fleet report is a
projection of; rebuild any of them from here.

## Where the ray went (M3 reference, detector frame, mm)

| column | meaning |
|---|---|
| `event_id` | DAQ event number, joins to `events.parquet` |
| `x`, `y` | reference crossing point at the detector plane |
| `ref_tan_x`, `ref_tan_y` | reference track tangent, rotated into the detector frame — the angle each plane *should* measure |
| `ref_theta_x_deg`, `ref_theta_y_deg` | the same as angles |

## What was reconstructed there

| column | meaning |
|---|---|
| `det_x`, `det_y` | reconstructed position, aligned into the reference frame (NaN if no reco) |
| `dx`, `dy` | `det_ - ref`, the residual per axis |
| `r_mm` | radial residual |
| `has_any` | the detector produced hits for this event |
| `within` | reconstructed within 5 mm of the reference — the efficiency numerator |
| `spark` | event tagged as a discharge |
| `category` | `within` / `reco_far` / `hit_no_reco` / `no_hit` / `spark`, mutually exclusive |

## Per-plane fit quality (from `events.parquet`)

`x_*` and `y_*`: `ok` (plane fitted), `p0` (position), `w` (fitted width),
`tan_theta` / `theta_deg` (angle, **w0/kw applied in reco**), `chi2`, `dof`,
`p0_err` / `tan_err`, `q_sum` / `q_u50` (charge), `n_strips`, `n_dropped`
(strips in competing clusters), `slope_reliable` (the |tan| >= 0.08 gate —
retained for continuity, not recommended as a cut), `quality_ok`,
`n_candidates`. Plus `n_tracks`, `n_hits` per event.

## Reference track quality (M3)

`chi2_x`, `chi2_y`, `nclus_x`, `nclus_y` — the quantities the reference recipe
cuts on. The cut can only be **tightened** from this table: reconstruction ran
on the matched list, which is already `chi2 < 1.0` and `NClus >= 4`, so rays
outside the frozen recipe have no reconstruction to score.

## Angle residuals

`dtheta_x_deg`, `dtheta_y_deg` — reconstructed minus reference, per plane.
NaN where the plane did not fit or the reference angle is undefined.

## Caveats

- Rows exist for rays inside the active box only (0.5–99.5 percentile of the
  reconstructed footprint); the box is in `summary.json`.
- `ref_*` angles are the *rotated* tangents, i.e. already in the detector's
  own frame — do not rotate again.
- Positions are post-alignment. The alignment used is in `summary.json`.
"""


def ref_quality(cfg):
    """Per-event M3 track quality: the chi2 and NClus the recipe cuts on.

    Needed to ask how the answer moves with the reference cut. Note the cut can
    only be TIGHTENED from here: reconstruction ran on the matched list, which
    is already chi2 < M3_CHI2_CUT and NClus >= M3_MIN_NCLUS, so rays outside
    the frozen recipe have no reconstruction to score.
    """
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    rd = rays.ray_data
    import awkward as ak
    cols = {f: np.asarray(ak.to_numpy(rd[f]), float)
            for f in ('Chi2X', 'Chi2Y', 'NClusX', 'NClusY') if f in ak.fields(rd)}
    evn = np.asarray(ak.to_numpy(rd['evn']), int)
    return pd.DataFrame(dict(event_id=evn,
                             chi2_x=cols.get('Chi2X'), chi2_y=cols.get('Chi2Y'),
                             nclus_x=cols.get('NClusX'),
                             nclus_y=cols.get('NClusY'))
                        ).drop_duplicates('event_id')


def ref_angles(cfg, params):
    """Reference tangents rotated into the detector frame, per event id."""
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    table = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    results = compat.as_event_results(compat.load_table(table,
                                                        max_dropped=None))
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    out = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        out[int(r.event_id)] = (tx, ty)
    return out


def categorise(d):
    """One mutually exclusive label per ray, in the report's own order."""
    cat = np.full(len(d), 'no_hit', dtype=object)
    has = d['has_any'].to_numpy(bool)
    reco = np.isfinite(d['det_x'].to_numpy(float))
    cat[has & ~reco] = 'hit_no_reco'
    cat[has & reco] = 'reco_far'
    cat[d['within'].to_numpy(bool)] = 'within'
    cat[d['spark'].to_numpy(bool)] = 'spark'
    return cat


def export_key(key):
    cfg = get_config(key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    out = os.path.join(W, 'plot_data')
    os.makedirs(out, exist_ok=True)
    print(f'== {key} ({cfg.DET_NAME})')

    d, box, params = per_ray_table(cfg)
    d = d.copy()
    fits = pd.read_parquet(os.path.join(W, 'events.parquet'))
    keep = ['event_id'] + [c for c in FIT_COLS if c in fits.columns]
    d = d.merge(fits[keep].drop_duplicates('event_id'), on='event_id',
                how='left', suffixes=('', '_fit'))

    d = d.merge(ref_quality(cfg), on='event_id', how='left')

    ra = ref_angles(cfg, params)
    d['ref_tan_x'] = [ra.get(int(e), (np.nan, np.nan))[0] for e in d['event_id']]
    d['ref_tan_y'] = [ra.get(int(e), (np.nan, np.nan))[1] for e in d['event_id']]
    for p in ('x', 'y'):
        d[f'ref_theta_{p}_deg'] = np.degrees(np.arctan(d[f'ref_tan_{p}']))
        if f'{p}_theta_deg' in d:
            d[f'dtheta_{p}_deg'] = d[f'{p}_theta_deg'] - d[f'ref_theta_{p}_deg']
    d['dx'] = d['det_x'] - d['x']
    d['dy'] = d['det_y'] - d['y']
    d['category'] = categorise(d)

    d.to_csv(os.path.join(out, 'rays.csv'), index=False)
    with open(os.path.join(out, 'COLUMNS.md'), 'w') as f:
        f.write(COLUMNS_MD)

    def jload(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:                                     # noqa: BLE001
            return {}

    eff = jload(os.path.join(W, 'efficiency', 'efficiency_breakdown.json'))
    ang = jload(os.path.join(W, 'angles_w0corr', 'angular_resolution.json'))
    full = jload(os.path.join(W, 'angles_w0corr', 'angles_fullcoverage.json'))
    meta = jload(os.path.join(W, 'events.meta.json'))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    summary = dict(
        key=key, letter=LETTER.get(key), detector=cfg.DET_NAME,
        run=cfg.RUN, sub_run=cfg.SUB_RUN, n_rays=int(len(d)),
        active_box={k: float(v) for k, v in box.items()},
        alignment=dict(z_x=params.z_x, z_y=params.z_y,
                       theta_deg=params.theta_deg,
                       x_offset=params.x_offset, y_offset=params.y_offset,
                       ref_x_sign=params.ref_x_sign),
        efficiency=eff, angles=ang, angles_fullcoverage=full,
        bundle=os.path.basename(str(meta.get('calibration', ''))),
        angle_constants=meta.get('angle_constants'),
        categories={k: int(v) for k, v in
                    pd.Series(d['category']).value_counts().items()},
        provenance=dict(code_commit=commit,
                        source='mx_june_wft/report/export_plot_data.py'),
    )
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'   {len(d):,} rays -> {out}/rays.csv '
          f'({os.path.getsize(os.path.join(out, "rays.csv")) // 1024} kB)')
    return summary


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith('-')] or JUNE_KEYS
    rows = []
    for key in keys:
        s = export_key(key)
        e = s['efficiency'] or {}
        a = (s['angles'] or {}).get('planes', {})
        f = (s['angles_fullcoverage'] or {}).get('planes', {})
        rows.append(dict(
            letter=s['letter'], key=key, detector=s['detector'],
            run=s['run'], sub_run=s['sub_run'], n_rays=s['n_rays'],
            within_5mm_pct=e.get('within_R'),
            core_sigma_mm=e.get('core_sigma_mm'),
            bias_x_deg=(a.get('x') or {}).get('bias_deg'),
            bias_y_deg=(a.get('y') or {}).get('bias_deg'),
            sigma_x_deg=(a.get('x') or {}).get('sigma_deg'),
            sigma_y_deg=(a.get('y') or {}).get('sigma_deg'),
            s68_lt5_x_deg=(f.get('x') or {}).get('s68_lt5_deg'),
            s68_lt5_y_deg=(f.get('y') or {}).get('s68_lt5_deg'),
            bundle=s['bundle'],
        ))
    os.makedirs(os.path.join(FLEET_REPORT, 'plot_data'), exist_ok=True)
    fl = pd.DataFrame(rows).sort_values('letter')
    fl.to_csv(os.path.join(FLEET_REPORT, 'plot_data', 'fleet.csv'), index=False)
    print(f'\nfleet table -> {FLEET_REPORT}/plot_data/fleet.csv')
    print(fl[['letter', 'detector', 'n_rays', 'within_5mm_pct']].to_string(index=False))


if __name__ == '__main__':
    main()

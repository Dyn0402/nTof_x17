#!/usr/bin/env python3
"""
corrected_angles.py — the fleet's angles with the bundle's w->angle constants
APPLIED, through the standard 03_angles accounting.

Background (INVESTIGATION_2026-08-12.md): 9dd7d6e introduced per-plane angle
constants  tan = (w*1e3 - w0[plane]) / (kw[plane] * v)  and every production
bundle carries calibrated w0/kw — but f9e18d2's plane_fit rewrite silently
reverted the formula to  tan = w*1e3/v,  so the frozen campaign never applies
them. The fleet-wide Y-heavy angle bias in the campaign digest is, detector
by detector, arctan(w0_y/v) to within measurement error.

This script does NOT touch the frozen reco or the live accounting: per golden
key it writes a corrected copy of the campaign parquet (tan_theta/theta_deg
recomputed with the bundle constants; w, p0, everything else untouched) and
runs 03_angles on it with --table/--out into <OUT_BASE>/wft/angles_w0corr/.

    ../../.venv/bin/python mx_june_wft/quality_investigation/corrected_angles.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'mx_june_wft')]

from qa_config import get_config, setup_paths     # noqa: E402
setup_paths()
from fleet_state import FLEET                     # noqa: E402

PY = os.path.join(REPO, '.venv', 'bin', 'python')


def main():
    summary = {}
    for key in FLEET:
        cfg = get_config(key)
        W = os.path.join(cfg.OUT_BASE, 'wft')
        meta = json.load(open(os.path.join(W, 'events.meta.json')))
        # Since 2026-08-13 plane_fit applies the constants itself and stamps
        # it. Re-applying here would subtract w0 twice and divide by kw twice —
        # a second correction of the same size, in the same direction, on a
        # table that is already right. Tables reconstructed before the restore
        # carry no stamp and still need this pass, so the fleet can be mixed.
        if (meta.get('angle_constants') or {}).get('applied'):
            # Already corrected in reco: correcting again would subtract w0
            # twice and divide by kw twice. But the accounting still has to be
            # REFRESHED, not skipped — angles_w0corr/ is what the fleet report
            # and digest quote, so leaving the previous generation's files in
            # place would pair today's efficiencies with yesterday's angles and
            # nothing downstream would notice. Run the same accounting on the
            # live table, unmodified.
            print(f'{key}: angles corrected in reco — re-running accounting '
                  'on the live table (no second correction)')
            out_dir = os.path.join(W, 'angles_w0corr')
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run(
                [PY, os.path.join(REPO, 'mx_june_wft', '03_angles.py'), key,
                 '--table', os.path.join(W, 'events.parquet'),
                 '--out', out_dir], check=True, cwd=REPO)
            with open(os.path.join(out_dir, 'PROVENANCE.txt'), 'w') as f:
                f.write('Angles applied IN RECO (events.meta.json '
                        'angle_constants.applied=true); this directory is the '
                        'standard 03_angles accounting run on the live table '
                        'with no post-hoc correction.\n')
            a = json.load(open(os.path.join(out_dir,
                                            'angular_resolution.json')))
            summary[key] = {p: dict(bias=a['planes'][p]['bias_deg'],
                                    sigma=a['planes'][p]['sigma_deg'],
                                    vsp=a['planes'][p]['implied_v_spread'])
                            for p in ('x', 'y')}
            print(f'{key}: '
                  + '  '.join(f'{p}: bias {summary[key][p]["bias"]:+.2f} '
                              f'sigma {summary[key][p]["sigma"]:.2f} '
                              f'vsp {summary[key][p]["vsp"]:.2f}'
                              for p in ('x', 'y')))
            continue
        bdir = os.path.basename(str(meta['calibration']))
        b = json.load(open(os.path.join(W, bdir, 'bundle.json')))
        w0, kw, v = b.get('w0') or {}, b.get('kw') or {}, b['v_drift']
        if not w0 and not kw:
            print(f'{key}: WARNING bundle {bdir} carries no w0/kw — angles '
                  'cannot be corrected, and are NOT quotable')
            continue

        df = pd.read_parquet(os.path.join(W, 'events.parquet'))
        for p in ('x', 'y'):
            tan = ((df[f'{p}_w'] * 1e3 - w0.get(p, 0.0))
                   / (kw.get(p, 1.0) * v))
            df[f'{p}_tan_theta'] = tan
            df[f'{p}_theta_deg'] = np.degrees(np.arctan(tan))
        out_dir = os.path.join(W, 'angles_w0corr')
        os.makedirs(out_dir, exist_ok=True)
        table = os.path.join(out_dir, 'events_w0corr.parquet')
        df.to_parquet(table)
        # 03_angles reads sidecar meta/alignment relative to the key's live
        # tree, so only the table and the output dir are overridden.
        subprocess.run(
            [PY, os.path.join(REPO, 'mx_june_wft', '03_angles.py'), key,
             '--table', table, '--out', out_dir],
            check=True, cwd=REPO)
        a = json.load(open(os.path.join(out_dir, 'angular_resolution.json')))
        summary[key] = {p: dict(bias=a['planes'][p]['bias_deg'],
                                sigma=a['planes'][p]['sigma_deg'],
                                vsp=a['planes'][p]['implied_v_spread'])
                        for p in ('x', 'y')}
        print(f'{key}: '
              + '  '.join(f'{p}: bias {summary[key][p]["bias"]:+.2f} '
                          f'sigma {summary[key][p]["sigma"]:.2f} '
                          f'vsp {summary[key][p]["vsp"]:.2f}'
                          for p in ('x', 'y')))
    with open(os.path.join(HERE, 'corrected_angles_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()

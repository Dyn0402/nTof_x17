#!/usr/bin/env python3
"""compare_lines.py -- the angle truth of the phase-4 test: A-C cosmic lines of a configuration against the reference.

The beam sub-runs have no angle truth (the capsule pointing is blurred by scattering), so the test also reconstructs
ten beam-off sub-runs of run_149.  A cosmic muon that crosses chambers A and C gives a line from the two positions;
the residual of a view is its track slope minus the slope of that line (rerun/ntof/improve/line_residuals.py, built on
the unmapped slopes, the same lines the angle map is trained on).  This script builds the lines of both views and
compares them event by event:

    python rerun/ntof/improve/phase4/compare_lines.py --cfg <dir>/final --ref <ref dir>/final --out <dir> [--name cfg]

Per chamber and view it reports, for the lines found in BOTH sets (same run, subrun, tag, event_id): the clean width
of the residual in each set (1.4826 MAD, "clean" as rerun/ntof/improve/angle_map.py: the other chamber's same view and
this chamber's other view within 0.10 of the line, |line slope| < 0.45, |residual| < 0.15, evaluated in the REFERENCE
so both sets use the same lines), their difference with a paired bootstrap error, the same for near-normal lines
(|line slope| < 0.1, where the head-on rescue acts) and for steep ones (> 0.25); and the number of lines only one set
has.  Writes <out>/<name>_lines.txt and <name>_lines.csv.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
KEY = ['run', 'subrun', 'tag', 'event_id']
CLEAN, JMAX, RMAX = 0.10, 0.45, 0.15


def build(view: str, out: Path) -> pd.DataFrame:
    """line_residuals build on one analysis view (it reads $X17_PASS1_VIEW and writes $X17_TQ_OUT/lines.parquet)."""
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, X17_PASS1_VIEW=str(Path(view).resolve()), X17_TQ_OUT=str(out))
    subprocess.run([sys.executable, '-m', 'rerun.ntof.improve.line_residuals', 'build'], cwd=REPO, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    return pd.read_parquet(out / 'lines.parquet')


def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 20 else np.nan


def clean_mask(M, s, p):
    o, q = ('c' if s == 'a' else 'a'), ('y' if p == 'x' else 'x')
    return ((M[f'res_{p}_{o}'].abs() < CLEAN) & (M[f'res_{q}_{s}'].abs() < CLEAN)
            & (M[f'j{p}'].abs() < JMAX) & (M[f'res_{p}_{s}'].abs() < RMAX))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='cfg')
    ap.add_argument('--boot', type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    C = build(a.cfg, out / f'lines_{a.name}')
    R = build(a.ref, out / f'lines_{a.name}_ref')
    # the reference is cut to the acquisitions the configuration has
    acq = set(C[KEY[:3]].apply(tuple, axis=1))
    R = R[R[KEY[:3]].apply(tuple, axis=1).isin(acq)]
    M = R.merge(C, on=KEY, suffixes=('', '_cfg'))
    rng = np.random.default_rng(1)
    rows, txt = [], [f'{a.name} vs ref: A-C cosmic lines.  ref {len(R)}, {a.name} {len(C)}, in both {len(M)}, '
                     f'only ref {len(R) - len(M)}, only {a.name} {len(C) - len(M)}', '']
    txt.append(f'{"view":6s} {"sample":10s} {"n":>6s} {"sig ref":>9s} {"sig cfg":>9s} {"diff":>9s} {"+-":>7s}   (1e-3 tan)')
    for s in 'ac':
        for p in 'xy':
            ok = clean_mask(M, s, p)
            j = M[f'j{p}'].abs()
            for lab, m in (('all', ok), ('|j|<0.1', ok & (j < 0.1)), ('|j|>0.25', ok & (j > 0.25))):
                r0, r1 = M.loc[m, f'res_{p}_{s}'].to_numpy(float), M.loc[m, f'res_{p}_{s}_cfg'].to_numpy(float)
                if len(r0) < 30:
                    continue
                d = mad(r1) - mad(r0)
                bs = [mad(r1[i]) - mad(r0[i]) for i in (rng.integers(0, len(r0), len(r0)) for _ in range(a.boot))]
                rows.append(dict(chamber=s.upper(), view=p, sample=lab, n=len(r0), sig_ref=mad(r0), sig_cfg=mad(r1),
                                 diff=d, diff_err=float(np.nanstd(bs))))
                txt.append(f'{s.upper()} {p:4s} {lab:10s} {len(r0):6d} {1e3 * mad(r0):9.2f} {1e3 * mad(r1):9.2f} '
                           f'{1e3 * d:+9.2f} {1e3 * np.nanstd(bs):7.2f}')
    (out / f'{a.name}_lines.txt').write_text('\n'.join(txt) + '\n')
    pd.DataFrame(rows).to_csv(out / f'{a.name}_lines.csv', index=False)
    print('\n'.join(txt))


if __name__ == '__main__':
    main()

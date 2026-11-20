#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_chains.py -- the n_TOF downstream headline numbers on several analysis views side by side.

Each chain is (label, output root, focus label): the same PASS1_DOWNSTREAM.sh products read from each root.  Default:

    nomap     $WORK_DATA/ntof_pass1/ana_nomap   (v(t) view, as of 11-20)          focus label pass1nomap
    map       $WORK_DATA/ntof_pass1/ana_map     (+ in-situ angle map A, C, D)      focus label pass1map
    mapcens   $WORK_DATA/ntof_pass1/ana         (+ censored-window cut: the view)  focus label pass1

    python -m rerun.ntof.improve.compare_chains [--chains label:root:focuslabel,...] -> <out>/chains.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.improve.track_quality import out_dir  # noqa: E402

P = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1'
DEFAULT = f'nomap:{P}/ana_nomap:pass1nomap,map:{P}/ana_map:pass1map,mapcens:{P}/ana:pass1'
RES = REPO / 'rerun' / 'ntof' / 'stage3' / 'resolution'


def metrics(root: Path, flabel: str) -> list:
    """[(group, metric, value, better), ...]; better = 'up' / 'down' / '1' (closer to 1) / '0' / '' (no direction)."""
    M = []

    def add(g, m, v, b=''):
        M.append((g, m, float(v) if v is not None and np.isfinite(v) else np.nan, b))

    def safe(fn):
        try:
            fn()
        except Exception as e:  # a missing product is a missing row, not a failed comparison
            M.append(('missing', f'{fn.__name__}: {type(e).__name__}: {e}'[:120], np.nan, ''))

    def focus():
        S = pd.read_csv(REPO / f'ntof_cosmics/results/focus_split/focus_split_{flabel}.csv')
        S = S[(S.split == 'all') & (S.tan_lo == 0.1) & (S.tan_hi == 0.5)].set_index('arm').s_best
        add('capsule focus', 'A sharpest scale s (all, |tan| 0.1-0.5)', S['A'], '1')
        add('capsule focus', 'C sharpest scale s', S['C'], '1')
        add('capsule focus', 'A - C (antisymmetric: geometry or angle)', S['A'] - S['C'], '0')
        add('capsule focus', '(A + C)/2 (common: angle scale)', (S['A'] + S['C']) / 2, '1')
        F = pd.read_csv(REPO / f'ntof_cosmics/results/capsule_focus/focus_{flabel}.csv')
        F = F[~F.null & (F.tan_lo.round(2) == 0.2) & (F.tan_hi.round(2) == 0.3)]
        for arm, g in F.groupby('arm'):
            add('capsule focus', f'{arm} median miss at sharpest s, |tan| 0.2-0.3 [mm]', g.med_miss.min(), 'down')

    def tof():
        T = pd.read_csv(root / 'capsule_tof/tof.csv')
        T = T[T.chain == 'pass1']
        for arm, g in T.groupby('arm'):
            g = g.set_index('lo')
            for lo in (4.0, 12.0, 20.0, 40.0):
                add('capsule pointing', f'{arm} capsule-pointing excess, {lo:g} ms bin', g.loc[lo, 'excess'], 'up')
            add('capsule pointing', f'{arm} tracks per ms, 20-25 ms', g.loc[20.0, 'per_ms'], '')
            add('capsule pointing', f'{arm} source height (y crossing), 20-25 ms [mm]', g.loc[20.0, 'y_cross'], '')
        L = pd.read_csv(root / 'capsule_tof/tilt.csv')
        L = L[(L.chain == 'pass1') & (L.lo == -999)]
        for _, r in L.iterrows():
            add('capsule pointing', f'{r.arm} source {r.axis} crossing, late, all heights [mm]', r.cross, '')

    def pose():
        D = pd.read_csv(root / 'capsule_pose/data.csv')
        for _, r in D[D.window == 'late'].iterrows():
            add('capsule pose data', f'{r.arm} late x crossing [mm]', r.x_cross, '')
            add('capsule pose data', f'{r.arm} late y crossing [mm]', r.y_cross, '')
            add('capsule pose data', f'{r.arm} late in-focus fraction', r.f_focus, 'up')
        for fam in ('xb0_xc-16_-18_-20', 'xb-5_xc-10_-15', 'xb-10_xc-10_-15'):
            f = root / f'capsule_pose/pose2_{fam}.csv'
            if f.exists():
                add('capsule pose fit', f'best x-view chi2 (10 dof), {fam}', pd.read_csv(f).chi2.min(), 'down')
            f = root / f'capsule_pose/pose2z_{fam}.csv'
            if f.exists():
                add('capsule pose fit', f'best z-view chi2 (5 dof), {fam}', pd.read_csv(f).chi2_z.min(), 'down')

    def same_chamber():
        d = root / 'same_chamber/cmp_pass1'
        F = pd.read_csv(d / 'cmp_tof_fit.csv').set_index('arm')
        E = pd.read_csv(d / 'cmp_expectation.csv')
        E = E[E.eps_source.str.startswith('digitised')].set_index('arm')
        O = pd.read_csv(d / 'cmp_opening_chi2.csv')
        O = O[O.chain == 'pass1'].set_index(['arm', 'model'])
        W = pd.read_csv(d / 'cmp_wall_double_ratio.csv')
        W = W[W.chain == 'pass1'].set_index('arm')
        for arm in ('A', 'C'):
            add('same-chamber pairs', f'{arm} correlated fraction a [%]', F.loc[arm, 'a'] * 100, '')
            add('same-chamber pairs', f'{arm} rate fit chi2/ndf', F.loc[arm, 'chi2'] / F.loc[arm, 'ndf'], 'down')
            add('same-chamber pairs', f'{arm} data / Geant4', E.loc[arm, 'data_over_g4'], '1')
            add('same-chamber pairs', f'{arm} opening shape chi2/ndf vs G4 e+e-',
                O.loc[(arm, 'g4_ee'), 'chi2'] / O.loc[(arm, 'g4_ee'), 'ndf'], 'down')
            add('same-chamber pairs', f'{arm} opening shape chi2/ndf vs event-mixed',
                O.loc[(arm, 'mixed'), 'chi2'] / O.loc[(arm, 'mixed'), 'ndf'], '')
            add('same-chamber pairs', f'{arm} wall double ratio (G4 e+e- {W.loc[arm, "g4_double_ratio"]:.2f})',
                W.loc[arm, 'data_double_ratio'], '')
        for arm in ('A', 'C'):
            t = pd.read_csv(root / f'same_chamber_late/pass1_{arm}/compare.csv') if (
                root / f'same_chamber_late/pass1_{arm}/compare.csv').exists() else None
            if t is not None and 'n_pairs' in t:
                add('same-chamber pairs', f'{arm} late pairs', t.n_pairs.iloc[0], '')

    def wall():
        B = pd.read_csv(root / 'chamber_b/chamber_b_run_145.csv').set_index('arm')
        for arm, r in B.iterrows():
            add('scintillator wall (run_145)', f'{arm} track points at the fired segment', r.u_match, 'up')
            add('scintillator wall (run_145)', f'{arm} along-bar correlation', r.v_corr, 'up')
            add('scintillator wall (run_145)', f'{arm} wall-matched tracks', r.n, '')
        cal = json.load(open(root / 'scint_stack/ana/meta.json'))['cal']
        for arm, c in cal.items():
            add('scintillator stack', f'{arm} lambda (slope the wall edges want)', c['lam'], '')
            add('scintillator stack', f'{arm} wall offset from survey [mm]', c['off_w'], '0')

    def tight():
        C = pd.read_csv(root / 'tight_coincidence/cut_census_campaign.csv').set_index('topo')
        add('two-arm coincidences', 'tight pairs (all topologies)', C.loc['(all)', 'n_tight'], '')
        add('two-arm coincidences', 'tight back-to-back (through-goers)', C.loc['(all)', 'n_b2b_in_tight'], '')
        add('two-arm coincidences', 'tight non-back-to-back pairs', C.loc['(all)', 'n_tight_pair'], '')

    def vertex():
        V = pd.read_csv(root / 'pair_vertex/v3d_measure.csv')
        V = V[V['map'] == 'data'].set_index('cls')
        for cls, r in V.iterrows():
            add('pair vertex (3D)', f'{cls}: peak distance from the capsule [mm]', r.peak_dist_from_capsule, 'down')
            add('pair vertex (3D)', f'{cls}: excess significance [sigma]', r.box_sigma, 'up')
        I = pd.read_csv(root / 'pair_vertex/image_fit.csv')
        I = I[I.cut_mm == 150]
        for _, r in I.iterrows():
            add('pair vertex (image)', f'{r.selection} {r.coord} via {r.chamber}: centre [mm]', r.c, '')
            add('pair vertex (image)', f'{r.selection} {r.coord} via {r.chamber}: band excess', r.band_excess, 'up')

    def resolution():
        f = RES / f'resolution_{flabel}.csv'
        R = pd.read_csv(f)
        for (arm, ax), g in R.groupby(['arm', 'axis']):
            g = g.set_index('lo')
            for lo in (0.0, 0.12, 0.3):
                if lo in g.index:
                    add('resolution (run_149 A-C line)', f'{arm} {ax} sigma, |tan| bin from {lo:g}', g.loc[lo, 'sigma'], 'down')
            add('resolution (run_149 A-C line)', f'{arm} {ax} tail > 0.15, 0.30-0.45', g.loc[0.3, 'tail_frac'], 'down')

    for fn in (focus, tof, pose, same_chamber, wall, tight, vertex, resolution):
        safe(fn)
    return M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', default=DEFAULT)
    ap.add_argument('--out', default=str(out_dir() / 'chains.csv'))
    a = ap.parse_args()
    cols, frames = [], []
    for spec in a.chains.split(','):
        lab, root, fl = spec.split(':')
        m = pd.DataFrame(metrics(Path(root), fl), columns=['group', 'metric', lab, 'better'])
        frames.append(m.set_index(['group', 'metric', 'better']))
        cols.append(lab)
    T = pd.concat(frames, axis=1).reset_index()
    T.to_csv(a.out, index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 70)
    print(T.round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

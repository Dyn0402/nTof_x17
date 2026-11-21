#!/usr/bin/env python3
"""make_page.py -- the n_TOF pass-1 before/after page: every downstream metric against the chain it replaced.

Reads the stored products of both chains (no recomputation) and writes one self-contained HTML page (inline SVG charts
drawn by a small embedded JS library, hover tooltips, a drag-to-rotate 3D track viewer, light/dark), for the site's
/notes/.  Numbers and their meaning: rerun/ntof/PASS1_DOWNSTREAM.md.

    .venv/bin/python rerun/ntof/compare_page/make_page.py [--out FILE.html]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

OLD = Path('/media/dylan/data/x17/sept26_prelim')
WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
P1 = WD / 'ntof_pass1'
ANA = P1 / 'ana'
LOCAL = Path.home() / 'x17_local'
RNG = np.random.default_rng(20261120)


def r(x, n=4):
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def recs(df, cols, n=4):
    return [{c: r(v, n) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False)]


# ---------------------------------------------------------------------------------------------------------- data
def focus():
    out = {}
    for lab, f in (('before', REPO / 'ntof_cosmics/results/capsule_focus/focus_is2_v2.csv'),
                   ('after', REPO / 'ntof_cosmics/results/capsule_focus/focus_pass1.csv')):
        R = pd.read_csv(f)
        R = R[~R.null & ((R.tan_hi - R.tan_lo).round(2) == 0.1) & (R.tan_lo >= 0.1) & (R.tan_hi <= 0.5)]
        R = R.dropna(subset=['med_miss'])
        out[lab] = recs(R, ['arm', 'tan_lo', 'tan_hi', 'scale', 'med_miss', 'f10'], 3)
    S = {}
    for lab, f in (('before', REPO / 'ntof_cosmics/results/focus_split/focus_split_is2_v2.csv'),
                   ('after', REPO / 'ntof_cosmics/results/focus_split/focus_split_pass1.csv')):
        R = pd.read_csv(f)
        R = R[(R.tan_lo == 0.1) & (R.tan_hi == 0.5)]
        keep = ['all', 'ms 20-80', 'wall fired', 'chi2 < median', 'q/len > median', 'tanx > 0', 'tanx < 0']
        R = R[R.split.isin(keep)]
        S[lab] = recs(R, ['split', 'arm', 's_best', 's_err', 'med_miss', 'n'], 4)
    out['split'] = S
    return out


def capsule_tof():
    T = pd.read_csv(ANA / 'capsule_tof/tof.csv')
    T['chain'] = T.chain.map({'pass1': 'after', 'is2_v2_cf': 'before'})
    T['signal_per_ms'] = T.per_ms * T.excess
    return recs(T, ['chain', 'arm', 'lo', 'hi', 'per_ms', 'excess', 'signal_per_ms', 'x_cross', 'y_cross', 'y_err'], 4)


def pose_data():
    o = pd.read_csv(OLD / 'capsule_pose_ear2_pool/data.csv')
    n = pd.read_csv(ANA / 'capsule_pose/data.csv')
    rows = []
    for lab, d in (('before', o), ('after', n)):
        for x in d.itertuples():
            rows.append(dict(chain=lab, arm=x.arm, window=x.window, n=int(x.n), x_cross=r(x.x_cross, 2),
                             y_cross=r(x.y_cross, 2), f_focus=r(x.f_focus, 3)))
    return rows


def imaging():
    rows = []
    for lab, f in (('before', OLD / 'imaging_campaign/axis_per_run.csv'),
                   ('after', ANA / 'imaging_campaign/axis_per_run.csv')):
        d = pd.read_csv(f)
        d['num'] = d.run.str[4:].astype(int)
        for x in d.itertuples():
            rows.append(dict(chain=lab, run=int(x.num), x_source=r(x.x_source_mm, 2), x_err=r(x.x_err_mm, 2),
                             z_D=r(x.z_D_mm, 2)))
    return rows


def v3d():
    out = {}
    for lab, root in (('before', OLD / 'pair_vertex'), ('after', ANA / 'pair_vertex')):
        z = np.load(root / 'v3d_maps.npz')
        ex, ey = z['ex'], z['ey']
        maps = {}
        for cls in ('A–D, C–D', 'D–D', 'A–A, C–C', 'A–C', 'all'):
            E = z[f'{cls}|data'] - z[f'{cls}|bkg']
            xz = E.sum(1)
            xy = E.sum(2)
            maps[cls] = dict(xz=np.round(xz, 1).tolist(), xy=np.round(xy, 1).tolist(),
                             px=np.round(E.sum((1, 2)), 1).tolist(), pz=np.round(E.sum((0, 1)), 1).tolist(),
                             py=np.round(E.sum((0, 2)), 1).tolist(), total=r(float(E.sum()), 1))
        M = pd.read_csv(root / 'v3d_measure.csv')
        M = M[M['map'] == 'data']
        out[lab] = dict(ex=ex.tolist(), ey=ey.tolist(), maps=maps,
                        measure=recs(M, ['cls', 'peak_x', 'peak_z', 'peak_dist_from_capsule', 'box_excess',
                                         'box_sigma', 'xprof_peak', 'xprof_fwhm', 'zprof_peak', 'zprof_fwhm'], 2))
    out['capsule'] = dict(before=[-9.32, -3.45], after=[-9.09, -5.55])
    return out


def tracks3d(n_pairs=100, n_pts=2500):
    """A sample of vertexed opposing-arm pairs (both legs drawn chamber -> vertex) and a vertex cloud, per chain."""
    out = {}
    cols = ['arm1', 'arm2', 'sep_mm', 'vx_xz', 'vy_xz', 'vz_xz', 'vx', 'vy', 'vz', 'p0_x_1', 'p0_y_1', 'p0_z_1',
            'p0_x_2', 'p0_y_2', 'p0_z_2']
    for lab, root in (('before', OLD / 'pair_vertex'), ('after', ANA / 'pair_vertex')):
        P = pd.read_parquet(root / 'pairs_vertex.parquet', columns=cols)
        P = P[((P.arm1 + P.arm2).isin(['AD', 'CD', 'DA', 'DC'])) & (P.sep_mm < 20)
              & (P.vx.abs() < 100) & (P.vz.abs() < 100) & (P.vy.abs() < 180)]
        frac_near = float(((P.vx - (-9.1)) ** 2 + (P.vz - (-5)) ** 2 < 20 ** 2).mean())
        s = P.sample(min(n_pairs, len(P)), random_state=1)
        legs = [[r(v, 1) for v in (x.p0_x_1, x.p0_y_1, x.p0_z_1, x.p0_x_2, x.p0_y_2, x.p0_z_2, x.vx, x.vy, x.vz)]
                for x in s.itertuples()]
        c = P.sample(min(n_pts, len(P)), random_state=2)
        pts = np.round(c[['vx', 'vy', 'vz']].to_numpy(float), 1).tolist()
        out[lab] = dict(legs=legs, pts=pts, n=int(len(P)), frac_within_20mm=r(frac_near, 3))
    return out


def same_chamber():
    out = {}
    e = np.array([0, 10, 20, 30, 40, 60, 80, 120, 200, 400], float)
    for lab, base in (('before', LOCAL / 'same_chamber_late' / 'cf'), ('after', ANA / 'same_chamber_late' / 'pass1')):
        for arm in 'AC':
            P = pd.read_parquet(f'{base}_{arm}/pairs.parquet', columns=['mixed', 'sep_plane_mm', 'open_deg'])
            R, M = P[~P.mixed], P[P.mixed]
            hr = np.histogram(R.sep_plane_mm, e)[0]
            hm = np.histogram(M.sep_plane_mm, e)[0] * len(R) / max(len(M), 1)
            out[f'{lab}_{arm}'] = dict(edges=e.tolist(), real=hr.tolist(), mixed=np.round(hm, 1).tolist(),
                                       n_real=int(len(R)))
    # close-pair excess (real - mixed under 40 mm) per tracked event >= 12 ms; the track counts are those of
    # PASS1_DOWNSTREAM.md (1.78 M A, 1.41 M C); the censored-window cut removes only 87 late A tracks
    n_trk = dict(A=1.78e6, C=1.41e6)
    for arm in 'AC':
        x = out[f'after_{arm}']
        out[f'close_{arm}'] = r(100 * (sum(x['real'][:4]) - sum(x['mixed'][:4])) / n_trk[arm], 4)
    O = pd.read_csv(ANA / 'same_chamber/cmp_pass1/cmp_opening.csv')
    O['chain'] = O.chain.map({'pass1': 'after', 'cf': 'before'})
    out['opening'] = recs(O, ['arm', 'chain', 'mid', 'correlated', 'err', 'g4_ee'], 2)
    return out


def resolution():
    o = pd.read_csv(REPO / 'ntof_cosmics/results/tracking/pooled/resolution.csv').assign(chain='before')
    n = pd.read_csv(REPO / 'rerun/ntof/stage3/resolution/resolution_pass1.csv').assign(chain='after')
    d = pd.concat([o, n])
    d['mid'] = 0.5 * (d.lo + d.hi)
    return recs(d, ['chain', 'arm', 'axis', 'lo', 'hi', 'mid', 'n', 'sigma', 'tail_frac', 'bias'], 4)


def wall():
    o = pd.read_csv(OLD / 'chamber_b/chamber_b_run_145.csv').assign(chain='before')
    n = pd.read_csv(ANA / 'chamber_b/chamber_b_run_145.csv').assign(chain='after')
    d = pd.concat([o, n])
    return recs(d, ['chain', 'arm', 'n', 'u_match', 'u_match_shuffled', 'v_corr'], 3)


def scint_lambda():
    o = json.load(open('/media/dylan/data/x17/scint_stack/ana/meta.json'))['cal']
    n = json.load(open(ANA / 'scint_stack/ana/meta.json'))['cal']
    return [dict(arm=a, before=r(o[a]['lam'], 3) if a in o else None, after=r(n[a]['lam'], 3) if a in n else None)
            for a in 'ACD']


def pair_response():
    rows = []
    for arm in 'AC':
        for lib in ('nose', 'ear2_L19p5'):
            j = json.load(open(P1 / 'g4_digi/response' / f'g4_{arm}_{lib}_pass1_response.json'))
            rows.append(dict(arm=arm, lib=lib, p1=r(j['p_ge1_given_1'], 3), p2=r(j['p_ge2_given_2'], 3),
                             split=r(j['p_ge2_given_1_lt40'], 5)))
    return rows


# ---------------------------------------------------------------------------------------------------------- after values
def after_values():
    """The 'after' numbers the verdict table used to carry by hand, read from the analysis view's products."""
    a = {}
    for arm in 'AC':
        P = pd.read_parquet(ANA / f'same_chamber_late/pass1_{arm}/pairs.parquet', columns=['mixed'])
        a[f'pairs_{arm}'] = int((~P.mixed).sum())
    E = pd.read_csv(ANA / 'same_chamber/cmp_pass1/cmp_expectation.csv').groupby('arm').first()
    F = pd.read_csv(ANA / 'same_chamber/cmp_pass1/cmp_tof_fit.csv').set_index('arm')
    for arm in 'AC':
        a[f'dg4_{arm}'] = float(E.loc[arm, 'data_over_g4'])
        a[f'rate_chi2_{arm}'] = float(F.loc[arm, 'chi2'] / F.loc[arm, 'ndf'])
    X = pd.read_csv(ANA / 'capsule_pose/pose2_xb-10_xc-10_-15.csv')
    b = X.loc[X.chi2.idxmin()]
    a['pose_x'], a['pose_x_cross'], a['pose_x_shape'] = float(b.chi2), float(b.chi2_x), float(b.chi2_shape)
    a['pose_z'] = float(pd.read_csv(ANA / 'capsule_pose/pose2z_xb0_xc-16_-18_-20.csv').chi2_z.min())
    T = pd.read_csv(ANA / 'track_quality/chains.csv')
    a['tight'] = float(T[T.metric == 'tight pairs (all topologies)'].mapcens.iloc[0])
    a['tight_b2b'] = float(T[T.metric == 'tight back-to-back (through-goers)'].mapcens.iloc[0])
    a['tight_nb2b'] = float(T[T.metric == 'tight non-back-to-back pairs'].mapcens.iloc[0])
    a['calo_pairs'] = len(pd.read_parquet(ANA / 'calorimetry/mip_beam.parquet', columns=['pair'])) // 2
    import re
    m = re.search(r'A: corr\(tu, tan_raw_x\) = ([+-][\d.]+)', (ANA / 'logs/calo_beam.log').read_text())
    a['calo_corr_A'] = float(m.group(1))
    m = re.search(r'C: corr\(tu, tan_raw_x\) = ([+-][\d.]+)', (ANA / 'logs/calo_beam.log').read_text())
    a['calo_corr_C'] = float(m.group(1))
    return a


def views():
    """Section 7: no map / map / map + cut, the headline rows of compare_chains.py."""
    T = pd.read_csv(ANA / 'track_quality/chains.csv')
    keep = [
        ('resolution (run_149 A-C line)', 'A x sigma, |tan| bin from 0.12'),
        ('resolution (run_149 A-C line)', 'C x sigma, |tan| bin from 0.12'),
        ('resolution (run_149 A-C line)', 'A y sigma, |tan| bin from 0.12'),
        ('resolution (run_149 A-C line)', 'C y sigma, |tan| bin from 0.12'),
        ('capsule focus', 'A - C (antisymmetric: geometry or angle)'),
        ('capsule focus', '(A + C)/2 (common: angle scale)'),
        ('capsule pose data', 'A late x crossing [mm]'),
        ('capsule pose data', 'C late x crossing [mm]'),
        ('capsule pose data', 'A late y crossing [mm]'),
        ('capsule pose data', 'C late y crossing [mm]'),
        ('capsule pose data', 'A - C late x crossing [mm] (every pose: +2.3..2.7)'),
        ('capsule pointing', 'D source z crossing, late, all heights [mm]'),
        ('capsule pose fit', 'best x-view chi2 (10 dof), xb-10_xc-10_-15'),
        ('capsule pose fit', '  of which crossings (A, C x vs data), xb-10_xc-10_-15'),
        ('capsule pose fit', 'best shape chi2 over all poses, xb-10_xc-10_-15'),
        ('capsule pose fit', 'best z-view chi2 (5 dof), xb0_xc-16_-18_-20'),
        ('scintillator wall (run_145)', 'A along-bar correlation'),
        ('scintillator wall (run_145)', 'D track points at the fired segment'),
        ('scintillator wall (run_145)', 'D along-bar correlation'),
        ('scintillator wall (run_145)', 'D wall-matched tracks'),
        ('same-chamber pairs', 'A data / Geant4'),
        ('same-chamber pairs', 'C data / Geant4'),
        ('pair vertex (image)', 'A-D z via D: band excess'),
        ('pair vertex (3D)', 'all: excess significance [sigma]'),
        ('two-arm coincidences', 'tight pairs (all topologies)'),
    ]
    rows = []
    for g, m in keep:
        x = T[(T.group == g) & (T.metric == m)]
        if len(x):
            x = x.iloc[0]
            rows.append(dict(group=g, metric=m.strip(), nomap=r(x.nomap, 4), map=r(x['map'], 4),
                             mapcens=r(x.mapcens, 4), better=x.better if isinstance(x.better, str) else ''))
    return rows


def vd(b, a, up):
    """Verdict from the numbers: up = larger is better, False = smaller is better, 'abs' = closer to 0."""
    if up == 'abs':
        b, a = -abs(b), -abs(a)
    elif up is False:
        b, a = -b, -a
    if abs(a - b) <= 0.02 * max(abs(b), 1e-9):
        return 'same'
    return 'better' if a > b else 'worse'


# ---------------------------------------------------------------------------------------------------------- verdicts
def verdicts(D):
    """(group, metric, before, after, change text, verdict, why) -- the table at the top."""
    v = []
    AV = D['after_values']

    def add(g, m, b, a, unit, better, why, fmt='{:.3g}', rel=True):
        if b is None or a is None:
            ch = '—'
        elif rel and b not in (0,):
            ch = f'{(a - b) / abs(b) * 100:+.0f} %'
        else:
            ch = f'{a - b:+.1f} pts' if unit == '%' else f'{a - b:+.3g} {unit}'.strip()
        v.append(dict(group=g, metric=m, before=(fmt.format(b) + (' ' + unit if unit else '')) if b is not None else '—',
                      after=(fmt.format(a) + (' ' + unit if unit else '')) if a is not None else '—', change=ch,
                      verdict=better, why=why))
    sp = {(x['split'], x['arm'], lab): x['s_best'] for lab in ('before', 'after') for x in D['focus']['split'][lab]}
    cb = 0.5 * (sp[('all', 'A', 'before')] + sp[('all', 'C', 'before')])
    ca = 0.5 * (sp[('all', 'A', 'after')] + sp[('all', 'C', 'after')])
    add('Capsule focus', 'common focus scale (A, C mean): extra scale the angles need', (cb - 1) * 100, (ca - 1) * 100,
        '%', 'better', 'a common scale error is not geometry; it was the angle calibration', fmt='{:+.1f}', rel=False)
    db = (sp[('all', 'A', 'before')] - sp[('all', 'C', 'before')]) * 100
    da = (sp[('all', 'A', 'after')] - sp[('all', 'C', 'after')]) * 100
    add('Capsule focus', 'A − C focus difference', db, da, '%', vd(db, da, 'abs'),
        'antisymmetric between A and C; the angle map (A–C cosmic lines) removes most of it', fmt='{:+.1f}',
        rel=False)
    F = {lab: pd.DataFrame(D['focus'][lab]) for lab in ('before', 'after')}
    for arm in 'AC':
        mb = F['before'][(F['before'].arm == arm) & (F['before'].tan_lo == 0.2)].med_miss.min()
        ma = F['after'][(F['after'].arm == arm) & (F['after'].tan_lo == 0.2)].med_miss.min()
        add('Capsule focus', f'{arm}: sharpest median miss, |tan| 0.2–0.3', mb, ma, 'mm', vd(mb, ma, False),
            'the ~40 % more tracks pass 1 keeps point at the capsule less often', fmt='{:.1f}')
    T = pd.DataFrame(D['capsule_tof'])
    for arm in 'AC':
        for lo in (20.0,):
            g = T[(T.arm == arm) & (T.lo == lo)]
            b, a = g[g.chain == 'before'].iloc[0], g[g.chain == 'after'].iloc[0]
            add('Capsule pointing', f'{arm}: tracks per ms, 20–25 ms', b.per_ms, a.per_ms, '', 'better',
                'more tracks at the same purity', fmt='{:,.0f}')
            add('Capsule pointing', f'{arm}: capsule-pointing excess, 20–25 ms', b.excess, a.excess, '', 'same',
                'purity unchanged', fmt='{:.3f}')
        g = T[(T.arm == arm) & (T.lo == 4.0)]
        b, a = g[g.chain == 'before'].iloc[0], g[g.chain == 'after'].iloc[0]
        add('Capsule pointing', f'{arm}: capsule-pointing excess, 4–5 ms (early)', b.excess, a.excess, '', 'worse',
            '2.5–5× more early tracks, almost none from the capsule; the < 12 ms veto stays', fmt='{:.3f}')
    PD = pd.DataFrame(D['pose'])
    yb = {x.arm: x.y_cross for x in PD[(PD.chain == 'before') & (PD.window == 'late')].itertuples()}
    ya = {x.arm: x.y_cross for x in PD[(PD.chain == 'after') & (PD.window == 'late')].itertuples()}
    add('Capsule pointing', 'A − C source height difference (20–75 ms)', abs(yb['A'] - yb['C']), abs(ya['A'] - ya['C']),
        'mm', 'better', 'A and C now see the source at nearly the same height', fmt='{:.1f}')
    M = {lab: {x['cls']: x for x in D['v3d'][lab]['measure']} for lab in ('before', 'after')}
    for cls, why in (('A–D, C–D', 'the opposing-arm pairs that localise x'), ('A–A, C–C', 'same-chamber pairs'),
                     ('all', 'every class pooled')):
        add('3D pair vertex', f'{cls}: peak distance from the capsule', M['before'][cls]['peak_dist_from_capsule'],
            M['after'][cls]['peak_dist_from_capsule'], 'mm',
            vd(M['before'][cls]['peak_dist_from_capsule'], M['after'][cls]['peak_dist_from_capsule'], False), why,
            fmt='{:.1f}')
        add('3D pair vertex', f'{cls}: excess significance in the peak box', M['before'][cls]['box_sigma'],
            M['after'][cls]['box_sigma'], 'σ', vd(M['before'][cls]['box_sigma'], M['after'][cls]['box_sigma'], True),
            why, fmt='{:.1f}')
    add('3D pair vertex', 'D–D: peak distance from the capsule', M['before']['D–D']['peak_dist_from_capsule'],
        M['after']['D–D']['peak_dist_from_capsule'], 'mm',
        vd(M['before']['D–D']['peak_dist_from_capsule'], M['after']['D–D']['peak_dist_from_capsule'], False),
        'D–D pairs alone; z is now imaged through A–D/C–D', fmt='{:.1f}')
    R = pd.DataFrame(D['resolution'])
    for arm, ax in (('A', 'x'), ('C', 'x'), ('A', 'y'), ('C', 'y')):
        g = R[(R.arm == arm) & (R.axis == ax)]
        for lo in (0.0, 0.3):
            b = g[(g.chain == 'before') & (g.lo == lo)]
            a = g[(g.chain == 'after') & (g.lo == lo)]
            if len(b) and len(a):
                add('Angle resolution', f'{arm} {ax}: σ(tan), |tan| {lo:.2f}–{b.hi.iloc[0]:.2f}', b.sigma.iloc[0],
                    a.sigma.iloc[0], '', 'better', 'against the run_149 A–C cosmic line', fmt='{:.3f}')
    W = pd.DataFrame(D['wall'])
    for arm in 'ABCD':
        b, a = W[(W.chain == 'before') & (W.arm == arm)].iloc[0], W[(W.chain == 'after') & (W.arm == arm)].iloc[0]
        add('Scintillator wall', f'{arm}: track points at the fired wall segment', b.u_match, a.u_match, '', 'better',
            f'chance {a.u_match_shuffled:.2f}', fmt='{:.2f}')
    for x in D['lambda']:
        add('Scintillator wall', f'{x["arm"]}: slope factor λ the wall edges prefer', x['before'], x['after'], '',
            'better', 'the same 0.85 on A, C and D; the spread was calibration', fmt='{:.2f}')
    add('Same-chamber pairs', 'A: same-chamber pairs (≥ 12 ms)', 3452, AV['pairs_A'], '', 'better',
        'two-track reconstruction on; close pairs come back', fmt='{:,.0f}')
    add('Same-chamber pairs', 'C: same-chamber pairs (≥ 12 ms)', 2722, AV['pairs_C'], '', 'better', '', fmt='{:,.0f}')
    add('Same-chamber pairs', 'A: Geant4 P(2 tracks | 2 leptons) / P(≥1 | 1)', 0.163 / 0.537,
        0.476 / 0.875, '', 'better', 'the reconstruction loses fewer pairs', fmt='{:.2f}')
    add('Same-chamber pairs', 'A: data / Geant4, correlated pair fraction', 0.129, AV['dg4_A'], '',
        vd(0.129, AV['dg4_A'], True),
        'the rate fit now starts at 25 ms: the 12–20 ms population is not proportional to the rate', fmt='{:.3f}')
    add('Same-chamber pairs', 'C: data / Geant4, correlated pair fraction', 0.124, AV['dg4_C'], '',
        vd(0.124, AV['dg4_C'], True), '', fmt='{:.3f}')
    add('Same-chamber pairs', 'A: rate-fit χ²/ndf (pair fraction vs rate)', 6.4, AV['rate_chi2_A'], '',
        vd(6.4, AV['rate_chi2_A'], False), 'fitted from 25 ms (was 12 ms; χ²/ndf 67 there)', fmt='{:.1f}')
    add('Capsule pose', 'best x fit χ² (beam on capsule, 10 mm off frame)', 12.1, AV['pose_x'], '',
        vd(12.1, AV['pose_x'], False),
        f"shape part {AV['pose_x_shape']:.1f}; crossings part {AV['pose_x_cross']:.1f}: the toy puts A's x crossing "
        "2.3–2.7 mm right of C's for every pose, the data 0.1 mm", fmt='{:.1f}')
    add('Capsule pose', 'best z fit χ²_z (centred beam)', 1.0, AV['pose_z'], '', vd(1.0, AV['pose_z'], False), '',
        fmt='{:.1f}')
    add('Coincidences', 'two-arm tight coincident pairs (campaign)', 1452, AV['tight'], '', 'better', '',
        fmt='{:,.0f}')
    add('Calorimetry', 'A–C through-goer pairs', 20583, AV['calo_pairs'], '', 'better',
        'B–D pairs lost (B has no angle)', fmt='{:,.0f}')
    add('Calorimetry', 'A: corr(track tan x, joined-line slope)', 0.30, AV['calo_corr_A'], '', 'better', '',
        fmt='{:.2f}')
    return v


# ---------------------------------------------------------------------------------------------------------- html
def build(out):
    D = dict(focus=focus(), capsule_tof=capsule_tof(), pose=pose_data(), imaging=imaging(), v3d=v3d(),
             tracks3d=tracks3d(), same_chamber=same_chamber(), resolution=resolution(), wall=wall(),
             pair_response=pair_response())
    D['lambda'] = scint_lambda()
    D['after_values'] = after_values()
    D['views'] = views()
    D['verdicts'] = verdicts(D)
    html = (HERE / 'template.html').read_text()
    html = html.replace('/*__DATA__*/null', json.dumps(D, separators=(',', ':'), ensure_ascii=False))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    print(f'wrote {out} ({len(html) / 1e6:.2f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ANA / 'compare' / 'ntof-pass1-before-after.html'))
    a = ap.parse_args()
    build(a.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scint_ecal.py -- PLAN.md C1: is the plastic keVee scale right for the
energies a stopped X17 lepton leaves (3-5 MeV)?

C1.1  MIP CHECK.  The keVee scale is the 2026-07-28 two-source calibration
      (`srccal_energy_calib.json`): a line through the Cs-137 477 keVee and
      Y-88 699 keVee Compton edges, which the MIP (~3.4 MeV) sits FIVE times
      above.  A muon crossing 20 mm of PVT has a calculable most probable loss
      (`landau.mpv`: 3.36-3.42 MeV for any beta*gamma 5-100), so the MIP peak
      in mV is a third, independent point -- the one at the energy that
      matters.  Measured per bar on two samples (`mip_sample`):
        cosmic  run_149 beam-off muons;
        beam    in-beam through-goers at >= 10 ms after the flash.
      The direction (path length through the bar, cos) is the A-C / B-D line
      for the beam sample and the calibrated cosmic slope for cosmics.
      Selection: the predicted bar fired, both ends of the predicted wall group
      fired (the particle crossed the stack there), clear of the bar's outer
      edges and the L/R gap, not saturated.  The trigger needs a plastic bar
      above PLAS_THR (112-151 mV, ~0.8 MIP); ``unbiased`` keeps only events
      another arm triggered.
C1.3  SATURATION.  Where ``satuflag`` sets in, per bar, from every gated track
      of the campaign (scint-stack tables).
C1.4  TIME SINCE FLASH.  The beam MIP peak in bins of ms after the flash.

    python -m ntof_calorimetry.scint_ecal          # -> OUT/c1/*.csv, summary.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry.mip_sample import OUT, load  # noqa: E402
from ntof_scint_stack import ana as SA  # noqa: E402
from ntof_scint_stack.extract import PLAS_THR  # noqa: E402

C1 = OUT / 'c1'
SRCCAL = SA.SRCCAL
PLAS_MM = 20.0
BG = 30.0               # beta*gamma; Delta_p moves 1.8 % over 5-100
EDGE_MM = 20.0          # from the bar's outer edges
GAP_MM = 15.0           # from the L/R gap
BEAM_MS = 10.0
BEAM_SEP = 40.0


def calib_variants() -> pd.DataFrame:
    """Per plastic channel, keVee = (mV - b) / a under four readings of the
    source points.  ``line_477_699`` is what every product uses today."""
    J = json.loads(SRCCAL.read_text())['channels']
    rows = []
    for ch, c in J.items():
        if c.get('kind') != 'PSS':
            continue
        pts = {float(k): (v['mv'], v['err']) for k, v in c['points'].items()}
        E = np.array(sorted(pts))
        Y = np.array([pts[e][0] for e in E])
        S = np.array([pts[e][1] for e in E])
        r = dict(ch=ch, arm=ch[3], bar=int(ch[4]),
                 line_477_699=(c['mv_per_kevee'], c['offset_mv']),
                 origin_699=(pts[698.63][0] / 698.63, 0.0))
        if len(E) == 3:
            w = 1 / S ** 2
            A = np.vstack([E, np.ones_like(E)]).T
            a, b = np.linalg.lstsq(A * np.sqrt(w)[:, None], Y * np.sqrt(w), rcond=None)[0]
            r['line_all3'] = (a, b)
            r['edge1612_vs_line'] = pts[1612.06][0] / (c['mv_per_kevee'] * 1612.06 + c['offset_mv'])
        else:
            r['line_all3'] = (np.nan, np.nan)
            r['edge1612_vs_line'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def mip_selection(D: pd.DataFrame) -> pd.Series:
    m = ((D.n_trk == 1) & (D.bar > 0) & (D.d_edge_p > EDGE_MM) & (D.d_gap_p > GAP_MM)
         & D.pm_on & D.wboth_on & ~D.psat)
    if (D['sample'] == 'beam').any():
        m &= ((D['sample'] != 'beam') | ((D.ms >= BEAM_MS) & (D.sep < BEAM_SEP)))
    return m


def samples() -> pd.DataFrame:
    parts = []
    for s in ('cosmic', 'beam'):
        p = OUT / f'mip_{s}.parquet'
        if p.exists():
            parts.append(load(s))
    D = pd.concat(parts, ignore_index=True)
    D['e_mv'] = D.pa_on * D.cos              # path-corrected amplitude, mV per 20 mm
    D['path_mm'] = PLAS_MM / D.cos
    D['t_trig'] = trigger_floor(D)
    return D


def trigger_floor(D: pd.DataFrame) -> np.ndarray:
    """Per event, the lowest e_mv (= amp x cos) it could have been recorded
    with: PLAS_THR x cos where the hardware trigger NEEDED this bar (no other
    arm satisfied it, and the arm's other bar was below threshold), else 0."""
    thr = D.arm.map(PLAS_THR).to_numpy(float)
    if 'ntof_arm' in D:
        other = np.where(D['sample'] == 'cosmic', D.ntof_arm.astype(str) != D.arm, False)
    else:
        other = np.zeros(len(D), bool)
    oth_hw = np.zeros(len(D), bool)
    for a in 'ABCD':
        if f'hw_{a}' in D:
            oth_hw |= (D.arm != a).to_numpy() & D[f'hw_{a}'].fillna(False).to_numpy(bool)
    other = other | np.where(D['sample'] == 'beam', oth_hw, False)
    other_bar = np.nan_to_num(D.po_a_on.to_numpy()) >= thr
    need = ~other & ~other_bar
    return np.where(need, thr * D.cos.to_numpy(), 0.0)


def _fit_bar(x: pd.DataFrame, extra: float = 0.0, prior=None) -> dict:
    """Truncated fit on one bar's selection; ``extra`` raises every event's
    floor to extra x the rough peak (the closure test)."""
    rough = float(np.median(x.e_mv))
    t = np.maximum(x.t_trig.to_numpy(), FLOOR * rough)
    if extra:
        t = np.maximum(t, extra * rough * x.cos.to_numpy())
    return LD.fit_trunc(x.e_mv.to_numpy(), t, HI * rough,
                        res_prior=RES_PRIOR if prior is None else (prior or None))


#: Gaussian prior on sigma / MPV (see `landau.fit_trunc`).  Set from the
#: prior-free fits of the cosmic C and D bars, whose thresholds sit at
#: 0.55-0.75 of their peaks (`main` prints them as ``free``): 0.13-0.23.
RES_PRIOR = (0.18, 0.05)
#: every event is also floored at FLOOR x the bar's median (rejects the
#: non-MIP low deposits of partner-triggered beam events) and the fit stops
#: at HI x the median
FLOOR, HI = 0.55, 2.6


def per_bar(D: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    sel = mip_selection(D)
    rows = []
    groups = [('cosmic', D['sample'] == 'cosmic'), ('beam', D['sample'] == 'beam'),
              ('both', np.ones(len(D), bool))]
    for arm in 'ABCD':
        for bar in (1, 2):
            v = V[(V.arm == arm) & (V.bar == bar)].iloc[0]
            for samp, m in groups:
                x = D[sel & m & (D.arm == arm) & (D.bar == bar)]
                if len(x) < 40:
                    continue
                f = _fit_bar(x)
                exp = LD.mpv(float(np.median(x.path_mm)), BG) * float(np.median(x.cos))
                r = dict(arm=arm, bar=bar, ch=v.ch, sample=samp, n_sel=len(x),
                         frac_trig_needed=float((x.t_trig > 0).mean()),
                         cos_med=float(x.cos.median()), exp_mev=exp, **f)
                for k in (0.9, 1.0):
                    r[f'mpv_closure_{k:g}'] = _fit_bar(x, k)['mpv']
                fr = _fit_bar(x, prior=False)
                r['mpv_free'], r['sigma_free'] = fr['mpv'], fr['sigma']
                for pm in (0.13, 0.23):
                    r[f'mpv_prior_{pm:g}'] = _fit_bar(x, prior=(pm, 0.02))['mpv']
                for name in ('line_477_699', 'origin_699', 'line_all3'):
                    a, b = v[name]
                    r[f'mev_{name}'] = (f['mpv'] - b) / a / 1000
                    r[f'ratio_{name}'] = r[f'mev_{name}'] / exp
                r['mv_per_mev_mip'] = f['mpv'] / exp
                r['mv_per_mev_line'] = v['line_477_699'][0] * 1000
                r['thr_mev_line'] = (PLAS_THR[arm] - v['line_477_699'][1]) / v['line_477_699'][0] / 1000
                r['thr_mev_mip'] = PLAS_THR[arm] / r['mv_per_mev_mip']
                r['edge1612_vs_line'] = v['edge1612_vs_line']
                r['k_fill'] = float(x.k_fill.mean())
                rows.append(r)
    return pd.DataFrame(rows)


def path_check(D: pd.DataFrame) -> pd.DataFrame:
    """MPV in raw mV (no cos correction) against path: proportional => the
    path correction is right and the response is linear over 1-1.4 MIP."""
    sel = mip_selection(D) & (D['sample'] == 'cosmic')
    rows = []
    for arm in 'ABCD':
        x = D[sel & (D.arm == arm)].copy()
        if len(x) < 150:
            continue
        x['ab'] = pd.qcut(x.path_mm, 3, duplicates='drop')
        for b, g in x.groupby('ab', observed=True):
            # normalise each bar to its own overall MPV so the bars pool
            ref = x.bar.map({b: _fit_bar(x[x.bar == b])['mpv'] for b in x.bar.unique()})
            gr = g.assign(e_mv=g.pa_on / ref.loc[g.index], cos=1.0,
                          t_trig=g.t_trig / g.cos / ref.loc[g.index])
            f = _fit_bar(gr)
            rows.append(dict(arm=arm, path_mm=float(g.path_mm.median()), n=len(g),
                             mpv_rel=f['mpv'], err=f['mpv_err'],
                             exp_rel=LD.mpv(float(g.path_mm.median()), BG) / LD.mpv(PLAS_MM, BG)))
    return pd.DataFrame(rows)


def flash_time(D: pd.DataFrame) -> pd.DataFrame:
    sel = mip_selection(D.assign(ms=D.ms.fillna(-1))) & (D['sample'] == 'beam')
    rows = []
    for arm in 'AC':
        x = D[sel & (D.arm == arm)]
        ref = {bar: _fit_bar(x[x.bar == bar])['mpv'] for bar in (1, 2)}
        for lo, hi in ((10, 15), (15, 25), (25, 40), (40, 80)):
            g = x[(x.ms >= lo) & (x.ms < hi)]
            if len(g) < 60:
                continue
            rr = g.bar.map(ref)
            f = _fit_bar(g.assign(e_mv=g.e_mv / rr, t_trig=g.t_trig / rr))
            rows.append(dict(arm=arm, ms_lo=lo, ms_hi=hi, n=len(g), mpv_rel=f['mpv'],
                             err=f['mpv_err']))
    return pd.DataFrame(rows)


def saturation(R: pd.DataFrame) -> pd.DataFrame:
    """Per bar: ``satuflag`` is never set in this processing, so the ceiling
    is read off the amplitudes themselves -- the largest values, and whether
    they repeat (a clipped pulse fits to the same amplitude) -- from every
    gated track of the campaign.  In MeVee on the MIP scale (cosmic MPV /
    Delta_p).  The digitisers are 2 V full scale on a +950 mV baseline."""
    rows = []
    cols = ['arm', 'p1_amp_on', 'p1_sat_on', 'p2_amp_on', 'p2_sat_on']
    parts = [pd.read_parquet(f, columns=cols) for f in
             sorted(glob.glob(str(SA.out_dir() / 'tracks' / 'stack_run_*.parquet')))]
    T = pd.concat(parts, ignore_index=True)
    T['arm'] = T.arm.astype(str)
    for arm in 'ABCD':
        for bar in (1, 2):
            a = T[T.arm == arm][f'p{bar}_amp_on'].dropna().to_numpy()
            s = np.nan_to_num(T[T.arm == arm][f'p{bar}_sat_on'].dropna().to_numpy()) > 0
            top = np.sort(a)[-30:]
            v, c = np.unique(np.round(top, 1), return_counts=True)
            rep = v[c >= 2]
            r = R[(R.arm == arm) & (R.bar == bar) & (R['sample'] == 'cosmic')]
            k = float(r.mv_per_mev_mip.iloc[0]) if len(r) else np.nan
            rows.append(dict(arm=arm, bar=bar, n_hits=int(len(a)), n_satuflag=int(s.sum()),
                             amp_max=float(a.max()), amp_p9999=float(np.percentile(a, 99.99)),
                             repeated_top=float(rep.min()) if len(rep) else np.nan,
                             frac_gt_1v=float((a > 1000).mean()),
                             mv_per_mev_mip=k, max_mev_mip=float(a.max()) / k,
                             repeated_top_mev=(float(rep.min()) / k) if len(rep) else np.nan))
    return pd.DataFrame(rows)


#: systematics on the MIP-anchored scale, relative: Bichsel thin-layer MPV
#: (delta escape, Landau-Vavilov vs measured) ~3 %, Birks / keVee vs MIP ~2 %,
#: fit (closure at raised thresholds, prior range) per bar from the table
SYS_THEORY = 0.036


def product(R: pd.DataFrame, P: pd.DataFrame, S: pd.DataFrame) -> dict:
    """`calib_plastic_e.json`: the MIP-anchored plastic scale per bar, from
    the beam-off cosmics only (the beam through-goers do not all penetrate,
    `liquid_salvage.penetration`)."""
    out = dict(
        schema='ntof_calorimetry/calib_plastic_e/1',
        note=('mV per MeVee from the cosmic MIP peak in 20 mm PVT (Landau MPV, truncated at '
              'each event\'s trigger threshold) over the Bichsel Delta_p. Linear in path to '
              '1.45 MIP (~5 MeV). Beam-off, run_149 (post-23 July DREAM config, end of '
              'campaign). NOT measured: gain vs time since flash, beam/beam-off gain ratio, '
              'per-cell face map (use ntof_scint_stack plas_full maps).'),
        conditions=dict(run='run_149', ntof_runs='224678-224687', sample='cosmic muons'),
        path_check=P.to_dict(orient='records'), bars={})
    for _, r in R[R['sample'] == 'cosmic'].iterrows():
        fit_sys = np.nanmax(np.abs(np.array([r['mpv_closure_0.9'], r['mpv_closure_1'],
                                             r['mpv_prior_0.13'], r['mpv_prior_0.23']]) / r.mpv - 1))
        rel = float(np.sqrt((r.mpv_err / r.mpv) ** 2 + fit_sys ** 2 + SYS_THEORY ** 2))
        sat = S[(S.arm == r.arm) & (S.bar == r.bar)]
        out['bars'][r.ch] = dict(
            mv_per_mevee=round(r.mv_per_mev_mip, 3), rel_err=round(rel, 3),
            rel_err_stat=round(r.mpv_err / r.mpv, 3), rel_err_fit=round(float(fit_sys), 3),
            mip_mpv_mv=round(r.mpv, 2), resolution_sigma_over_mpv=round(r.sigma / r.mpv, 3),
            n=int(r.n), mv_per_mevee_srccal_line=round(r.mv_per_mev_line, 3),
            srccal_line_reads_mip_at=round(r.ratio_line_477_699, 3),
            trigger_threshold_mevee=round(r.thr_mev_mip, 3),
            adc_max_mevee=round(float(sat.max_mev_mip.iloc[0]), 1) if len(sat) else None)
    return out


def main() -> int:
    C1.mkdir(parents=True, exist_ok=True)
    V = calib_variants()
    D = samples()
    sel = mip_selection(D)
    print('MIP selection:', D[sel].groupby(['sample', 'arm']).size().to_dict())
    R = per_bar(D, V)
    P = path_check(D)
    F = flash_time(D)
    S = saturation(R)
    V2 = V.copy()
    for c in ('line_477_699', 'origin_699', 'line_all3'):
        V2[f'{c}_a'] = V2[c].map(lambda t: t[0])
        V2[f'{c}_b'] = V2[c].map(lambda t: t[1])
    V2.drop(columns=['line_477_699', 'origin_699', 'line_all3']).to_csv(C1 / 'calib_variants.csv', index=False)
    R.to_csv(C1 / 'mip_per_bar.csv', index=False)
    P.to_csv(C1 / 'path_check.csv', index=False)
    F.to_csv(C1 / 'flash_time.csv', index=False)
    S.to_csv(C1 / 'saturation.csv', index=False)
    (C1 / 'calib_plastic_e.json').write_text(json.dumps(product(R, P, S), indent=1))
    # the spectra, for the figures
    keep = ['sample', 'arm', 'bar', 'partner_hw', 'e_mv', 'pa_on', 'cos', 'path_mm', 'ms', 't_trig']
    D[sel][keep].to_parquet(C1 / 'mip_spectra.parquet', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(R[['ch', 'sample', 'n', 'frac_trig_needed', 'mpv', 'mpv_err', 'mpv_closure_0.9',
                 'mpv_closure_1', 'mpv_free', 'sigma_free', 'mpv_prior_0.13', 'mpv_prior_0.23',
                 'sigma', 'exp_mev',
                 'mev_line_477_699', 'ratio_line_477_699', 'ratio_origin_699', 'ratio_line_all3',
                 'edge1612_vs_line']].round(3).to_string(index=False))
        print(P.round(3).to_string(index=False))
        print(F.round(3).to_string(index=False))
        print(S.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

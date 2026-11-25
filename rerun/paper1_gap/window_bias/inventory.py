#!/usr/bin/env python3
"""inventory.py <data root> -- where does the drift column end against the end of the recorded waveform, per gap row?

For every |in30 arm of the bench timing stacks (rerun/paper1_gas/fit_v1, the T of Paper I Sec. 5 and of
rerun/paper1_gap/audit/gaps_all), rebuild the stack exactly as s4f_fit.py did (mask, interior cut, saturation cut,
rc_diffusion.stack, |tan_x| < 0.03), refit it once with x_width.fit_x (check: reproduces the stored T), and record

    t_end_stack   last stack time with data: rc_diffusion.stack keeps a time bin only when > 60 % of the events still
                  have a sample there (each event is aligned on its own leading edge, so its record ends at a different
                  time); this is where the fit stops seeing data
    t_end_ev      per-event record end after the leading edge (quantiles 10 / 50 / 90 %)
    col_end       dt + 60 + T of the stored fit: the stack time at which the last charge of the column reaches the
                  strips (predict_x's clock starts at GRID[0] = -300 ns, the template's at -360 ns from the pulse t0);
                  the falling edge is that end convolved with the response (peak +140 ns, 99 % of the area by ~+500 ns)
    col_end_exp   dt + g / v_ang, with g the geometric gap of the chamber and v_ang the angle-scale velocity (map1
                  bundle, gaps_all): where the column should end if the gap and the velocity are right
    margin        t_end_stack - col_end, and the same for col_end_exp

<data root>: a copy of cosmic_bench/ with x17rerun/paper1_gas/{fit_v1,Analysis/**/headon.pkl} and the arms' template
bundles (Analysis_pass1/<row>/wft/calib_bundle).  Writes results/inventory.csv and results/stacks.npz (the stacks, for
toy.py).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import s4_stacks  # noqa: E402
import s4f_fit  # noqa: E402
import x_width as xw  # noqa: E402
import gaps  # noqa: E402
from rc_diffusion import GRID, THR_ADC, OFFS  # noqa: E402
from width_vs_time import SAMPLE_NS  # noqa: E402

RES = os.path.join(HERE, 'results')
GAPS = os.path.join(REPO, 'rerun', 'paper1_gap', 'audit', 'gaps_all', 'gaps_all.csv')


def local(p, root):
    if '/cosmic_bench/' not in p:          # already a local copy (the fit_v2 replacements)
        return p
    return os.path.join(root, p.split('/cosmic_bench/')[1])


def arm_events(arm, root):
    """s4f_fit.main's selection of one arm: the events the fit used (unsaturated, any tan; stack() applies |tan_x|)."""
    ev = s4f_fit.load(local(arm['cache'], root))
    if arm.get('sel'):
        keep = set(json.load(open(local(arm['sel'][0], root) if '/cosmic_bench/' in arm['sel'][0] else arm['sel'][0]))[arm['sel'][1]])
        ev = {k: e for k, e in ev.items() if (k in keep) != bool(arm.get('sel_not'))}
    if arm.get('ch_min') is not None:
        ev = {k: e for k, e in ev.items() if e['x']['ch'][np.argmax(e['x']['W'].max(axis=1))] >= arm['ch_min']}
    if arm.get('mask_x'):
        mk = arm['mask_x'] if isinstance(arm['mask_x'], list) else json.loads(arm['mask_x'])
        ev = s4f_fit.mask(ev, set(mk), arm.get('guard'))
    if arm.get('edge'):
        (x0, x1), (y0, y1), d, side = arm['edge'][0]['x'], arm['edge'][0]['y'], arm['edge'][1], arm['edge'][2]
        ev = {k: e for k, e in ev.items()
              if (min(e['ref_mesh_x'] - x0, x1 - e['ref_mesh_x'], e['ref_mesh_y'] - y0, y1 - e['ref_mesh_y']) >= d) == (side == 'in')}
    sat, lo = arm.get('sat'), arm.get('amp_lo', 0.0)
    return {k: e for k, e in ev.items() if lo <= e['x']['W'].max() < (np.inf if sat is None else sat)}


def record_ends(ev):
    """Per event (as rc_diffusion.stack aligns it): record end after the leading edge [ns], for the stacked events."""
    out = []
    for e in ev.values():
        if abs(e['tan_x']) > xw.TAN:
            continue
        W = np.asarray(e['x']['W'], float)
        if 'bad' in e['x']:
            W = W.copy(); W[e['x']['bad']] = np.nan
        ic = int(np.nanargmax(np.nansum(W, 1)))
        if ic < 4 or ic > len(W) - 5 or not np.isfinite(W[ic]).all():
            continue
        s = np.nansum(W[ic - 4:ic + 5], 0)
        k = next((k for k in range(1, len(s)) if s[k - 1] < THR_ADC <= s[k]), None)
        if k is None or k < 3:
            continue
        lead = k - 1 + (THR_ADC - s[k - 1]) / (s[k] - s[k - 1])
        out.append(((len(s) - 1) - lead) * SAMPLE_NS)
    return np.array(out)


def main():
    root = sys.argv[1]
    os.makedirs(RES, exist_ok=True)
    F, T0 = {}, {}
    for f in glob.glob(os.path.join(root, 'x17rerun/paper1_gas', os.environ.get('FIT_DIR', 'fit_v2'), 'fit_T*.json')):
        d = json.load(open(f)); F.update(d)
        sp = json.load(open(f.replace('fit_T', 'spec_T')))
        for k in d:
            T0[k] = sp['T0']
            if 'replaces' in d[k]:          # fit_v2: a failed in30 fit replaced by another arm (det7, ch >= 256, all)
                d[k]['edge'] = None
    G = pd.read_csv(GAPS)
    rows, stacks = [], {}
    for name, arm in sorted(F.items()):
        if not name.endswith('|in30') or 'T' not in arm:
            continue
        key = name.split('|')[0]
        run, sub, det = arm['cache'].split('/paper1_gas/Analysis/')[1].split('/')[:3]   # keys like sat_det3 have no '__'
        ev = arm_events(arm, root)
        n, M = xw.stack(ev, 'x', tan_max=xw.TAN)
        if not np.all(np.isfinite(M), axis=0).any():
            print(f'{key}: empty stack (n {n}), skipped', flush=True)
            continue
        h = s4_stacks.pass_template(os.path.dirname(local(arm['template'], root)))
        r = xw.fit_x(h, M, T0[name])
        ok = np.all(np.isfinite(M), axis=0)
        t_end = float(GRID[ok].max())
        rec = record_ends(ev)
        g = G[(G.run == run) & (G.subrun == sub) & (G.det == 'det' + det.split('_')[1])]
        v_ang = float(g.v_ang.iloc[0]) if len(g) else np.nan
        dn = 'det' + det.split('_')[1]
        gg = gaps.GAP_GEOM.get(dn, 30.0)
        off = -300.0 - float(np.load(os.path.join(os.path.dirname(local(arm['template'], root)), 'arrays.npz'))['grid'][0])
        col_end = arm['dt'] + off + arm['T']          # stack time at which the last charge reaches the strips (pulse t0)
        col_exp = arm['dt'] + off + gg / v_ang * 1e3 if np.isfinite(v_ang) else np.nan
        rows.append(dict(key=key, det=dn, chamber={'det2': 'B', 'det3': 'A', 'det4': 'E', 'det6': 'C', 'det7': 'D'}[dn],
                         run=run, subrun=sub, n=n, T0=T0[name], T=arm['T'], T_err=arm['T_err'], dt=arm['dt'],
                         T_refit=r['T'], dt_refit=r['dt'], sig0=arm['sig0'], Dd=arm['Dd'], rms=r['rms'],
                         t_end_stack=t_end, rec_q10=float(np.quantile(rec, 0.1)), rec_q50=float(np.median(rec)),
                         rec_q90=float(np.quantile(rec, 0.9)), col_end=col_end, margin=t_end - col_end,
                         v_ang=v_ang, g_geom=gg, T_exp=gg / v_ang * 1e3 if np.isfinite(v_ang) else np.nan,
                         col_end_exp=col_exp, margin_exp=t_end - col_exp,
                         at_bound=bool(arm['T'] <= 0.6 * T0[name] + 1 or arm['T'] >= 1.5 * T0[name] - 1)))
        stacks[key] = M
        q = rows[-1]
        print(f"{q['chamber']} {sub[:40]:40s} n {n:4d} T {q['T']:6.0f} (refit {q['T_refit']:6.0f}) dt {q['dt']:5.0f} "
              f"end {t_end:5.0f} rec50 {q['rec_q50']:5.0f} margin {q['margin']:6.0f}  T_exp {q['T_exp']:6.0f} "
              f"margin_exp {q['margin_exp']:6.0f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'inventory.csv'), index=False)
    np.savez_compressed(os.path.join(RES, 'stacks.npz'), grid=GRID, **{k.replace('/', '_'): v for k, v in stacks.items()})


if __name__ == '__main__':
    main()

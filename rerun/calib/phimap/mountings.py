"""mountings.py -- assign every non-excluded registry row to a mounting and a role, write mountings.csv.

A mounting is one chamber, not moved: the grouping is by run directory, then checked against the row alignments (z, offsets,
theta; see check_alignment).  Rules (Dylan 2026-11-24 and the handoff, with the rows that had no gate table before):

  B_0617  det2 6-17                       (own mounting: offsets differ by 6.5 / 8 mm from 6-22)
  B_0622  det2 6-22 (long, longer, short + 16 mesh-HV rows)
  A_0622  det3 6-22 overnight + det3_test_6-22  (same slot and alignment within 0.4 mm)
  A_0623  det3 6-23 overnight             (same slot as 6-22 by alignment, 0.6 mm in y: kept separate, the foil changes with
                                           every mounting; the 6-22 -> 6-23 transfer is tested in RESULT.md)
  A_0625  det3 day 6-25                   (500 V drift)
  A_0627  det3 Saturday scan 6-27 + p2 overnight 6-28 (incl. the drift-field scan sub-runs)
  E       det4 6-23 + 6-24                (ONE joint map, Dylan: the cathode did not change; alignments differ)
  C_0626  det6 6-26 overnight + hv_scan   (one map PER ROW with >= 2000 gate events, the others borrow the nearest in time)
  D_0626  det7 6-26 overnight + hv_scan   (X view used only for reference x >= 200 mm)

Roles: train (row at the mounting's main drift field, enough events), tilt_only (another drift field: the tilt part of the
map, S = 0, unless the scale part is shown to help), borrowed (C rows below 2000 gate events: the map of the nearest C row in
time), nomap (mounting with < 2000 gate events in total, stays on the bow block).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rows as R  # noqa: E402

PUBLIC = {'mx17_2': 'B', 'mx17_3': 'A', 'mx17_4': 'E', 'mx17_6': 'C', 'mx17_7': 'D'}
MIN_TOTAL = 2000      # gate events of a mounting (or a C row) for a map
MIN_ROW = 60          # gate events of a row to fit its own constants


def mounting_of(r):
    det, run = r.det, r.run
    if det == 'mx17_2':
        return 'B_0617' if '6-17' in run else 'B_0622'
    if det == 'mx17_3':
        if '6-22' in run or '6-22' in r.subrun:
            return 'A_0622'
        if '6-23' in run:
            return 'A_0623'
        if '6-25' in run:
            return 'A_0625'
        return 'A_0627'
    if det == 'mx17_4':
        return 'E'
    if det == 'mx17_6':
        return 'C_0626'
    return 'D_0626'


def acq_minutes(tag):
    m = re.match(r'(\d\d)(\d\d)(\d\d)_(\d\d)H(\d\d)', tag)
    yy, mo, dd, hh, mi = (int(x) for x in m.groups())
    return ((dd * 24) + hh) * 60 + mi


def build():
    reg = R.reg()
    rec = []
    for k in reg.index:
        r = reg.loc[k]
        t = R.load_full(k)
        a = R.alignment(k)
        sel = t[t.sel]
        rec.append(dict(
            key=k, det=r.det, chamber=PUBLIC[r.det], run=r.run, subrun=r.subrun, mounting=mounting_of(r), drift_V=int(r.drift_V),
            resist_V=r.resist_V, cls=r.cls, K=int(r.K), calib_key=r.calib_key, align_key=r.align_key,
            n_table=len(t), n_gate=len(sel), cens_x=float(sel.cens_x.mean()) if len(sel) else np.nan,
            cens_y=float(sel.cens_y.mean()) if len(sel) else np.nan,
            cens_any=float((sel.cens_x | sel.cens_y).mean()) if len(sel) else np.nan,
            ntracks_unavailable=bool(t.attrs['ntracks_unavailable']), t_min=acq_minutes(t.acq_tag.min()), t_max=acq_minutes(t.acq_tag.max()),
            corr_x=float(np.corrcoef(sel.tan_ref_x, sel.x_tan_theta)[0, 1]) if len(sel) > 30 else np.nan,
            corr_y=float(np.corrcoef(sel.tan_ref_y, sel.y_tan_theta)[0, 1]) if len(sel) > 30 else np.nan,
            acq0=t.acq_tag.min(), z_x=a['z_x'], z_y=a['z_y'], theta=a['theta_deg'], x_off=a['x_offset'], y_off=a['y_offset']))
    df = pd.DataFrame(rec)
    out = []
    for m, g in df.groupby('mounting', sort=False):
        main = g.groupby('drift_V').n_gate.sum().idxmax()
        total = g.n_gate.sum()
        for i in g.index:
            row = df.loc[i]
            if row.n_gate < MIN_ROW:
                role = 'tiny'
            elif row.drift_V != main:
                role = 'tilt_only'
            else:
                role = 'train'
            out.append((i, main, role))
    for i, main, role in out:
        df.loc[i, 'main_V'] = main
        df.loc[i, 'role'] = role
    # total gate events at the main field, per mounting
    df['n_train_mount'] = df.groupby('mounting').apply(lambda g: g[g.role == 'train'].n_gate.sum()).reindex(df.mounting).values
    # unit (the thing a map is fitted for): the mounting, except C where each big row is its own unit
    df['unit'] = df.mounting
    c = df[df.mounting == 'C_0626']
    big = c[(c.n_gate >= MIN_TOTAL) & (c.role == 'train')]
    df.loc[big.index, 'unit'] = 'C_' + big.subrun.str.replace('_run', '', regex=False)
    # C rows below the threshold borrow the nearest big row in time (mid acq time)
    mid = lambda r_: 0.5 * (r_.t_min + r_.t_max)
    for i in c.index:
        if i in big.index:
            df.loc[i, 'donor'] = df.loc[i, 'unit']
            df.loc[i, 'role'] = 'train'
        else:
            d = {j: abs(mid(df.loc[i]) - mid(df.loc[j])) for j in big.index}
            j = min(d, key=d.get)
            df.loc[i, 'donor'] = df.loc[j, 'unit']
            df.loc[i, 'unit'] = df.loc[j, 'unit']
            df.loc[i, 'role'] = 'borrowed'
    # nomap mountings: fewer than MIN_TOTAL gate events, or pass-1 angles uncorrelated with the reference (A 6-23)
    df['note'] = ''
    for m, g in df.groupby('mounting'):
        tr_ = g[g.role == 'train']
        if m != 'C_0626' and tr_.n_gate.sum() < MIN_TOTAL:
            df.loc[g.index, 'role'] = 'nomap'
            df.loc[g.index, 'unit'] = ''
            df.loc[g.index, 'note'] = f'only {int(g.n_gate.sum())} gate events in total (< {MIN_TOTAL}): stays on the bow block'
        elif len(tr_) and np.average(np.nan_to_num(tr_.corr_x), weights=tr_.n_gate) < 0.3:
            df.loc[g.index, 'role'] = 'nomap'
            df.loc[g.index, 'unit'] = ''
            df.loc[g.index, 'note'] = 'pass-1 angles are uncorrelated with the M3 reference (corr %.2f): no angle map is possible' % np.average(np.nan_to_num(tr_.corr_x), weights=tr_.n_gate)
    df.loc[df.role == 'tiny', 'role'] = df.loc[df.role == 'tiny'].apply(lambda r_: 'nomap' if r_.unit == '' else 'tiny', axis=1)
    df['key_short'] = df.run.str.replace('mx17_', '', regex=False) + '/' + df.subrun
    return df


def check_alignment(df):
    """Spread of the alignment inside each mounting (z, offsets, theta)."""
    g = df.groupby('mounting')
    return pd.DataFrame({
        'rows': g.size(), 'z_x': g.z_x.agg(lambda s: f'{s.min():.0f}-{s.max():.0f}'), 'z_y': g.z_y.agg(lambda s: f'{s.min():.0f}-{s.max():.0f}'),
        'x_off': g.x_off.agg(lambda s: f'{s.min():.2f}..{s.max():.2f}'), 'y_off': g.y_off.agg(lambda s: f'{s.min():.2f}..{s.max():.2f}'),
        'theta': g.theta.agg(lambda s: f'{s.min():.2f}-{s.max():.2f}'), 'x_off_spread': g.x_off.agg(lambda s: s.max() - s.min()),
        'y_off_spread': g.y_off.agg(lambda s: s.max() - s.min()), 'theta_spread': g.theta.agg(lambda s: s.max() - s.min())})


if __name__ == '__main__':
    df = build()
    os.makedirs(R.OUTDIR, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    cols = ['key', 'chamber', 'mounting', 'unit', 'donor', 'role', 'drift_V', 'main_V', 'n_gate', 'n_table', 'cens_x', 'cens_y', 'cens_any',
            'ntracks_unavailable', 'corr_x', 'corr_y', 'note', 'z_x', 'z_y', 'x_off', 'y_off', 'theta', 'calib_key', 'align_key', 'run', 'subrun', 'K', 'acq0']
    df[cols].to_csv(os.path.join(here, 'mountings.csv'), index=False, float_format='%.4g')
    df[cols].to_csv(os.path.join(R.OUTDIR, 'mountings.csv'), index=False, float_format='%.4g')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    print(check_alignment(df).to_string())
    print(df.groupby(['mounting', 'unit', 'role']).agg(rows=('key', 'size'), N=('n_gate', 'sum'), cens=('cens_any', 'mean')).to_string())

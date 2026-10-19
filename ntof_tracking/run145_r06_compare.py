#!/usr/bin/env python3
"""
run145_r06_compare.py — the superseded run_145 imaging against the r06 one.

Reads two `imaging_summary.json` (the parked pre-correction one and the new
one) and prints the four checks HANDOFF_RUN145_R06 §4 asks for, per arm:

    1. does the spot still land inside the r <= 10 mm bore  (frac_in_capsule,
       r_core at k_opt)
    2. how far k_opt / k_track moved, and what that does to v_insitu
    3. do the arms agree with each other better or worse than before
    4. the external confirmation rate (SiPM segment AND plastic bar in time),
       which does not go through the fit's angle scale at all

    ../../.venv/bin/python ntof_tracking/run145_r06_compare.py \
        --old .../pre_r06_backup_20260819/imaging_fullcov/imaging_summary.json \
        --new .../imaging_r06/imaging_summary.json
"""
import argparse
import json
import os

import numpy as np

BASE = ('/media/dylan/data/x17/beam_july/analysis/wft/run_145/stat090_0000')
OLD = os.path.join(BASE, 'pre_r06_backup_20260819', 'imaging_fullcov',
                   'imaging_summary.json')
NEW = os.path.join(BASE, 'imaging_r06', 'imaging_summary.json')


def by_arm(p):
    d = json.load(open(p))
    return {r['arm']: r for r in d['results'] if 'error' not in r}


def g(r, *path, default=None):
    for k in path:
        if not isinstance(r, dict) or k not in r:
            return default
        r = r[k]
    return r


def fmt(a, b, unit='', pct=True, w=9):
    if a is None or b is None:
        return f'{"-":>{w}} {"-":>{w}} {"":>9}'
    s = f'{a:{w}.4g} {b:{w}.4g}'
    if pct and a:
        s += f'  {100 * (b - a) / abs(a):+7.1f} %'
    else:
        s += f'  {b - a:+9.4g}'
    return s + unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default=OLD)
    ap.add_argument('--new', default=NEW)
    a = ap.parse_args()
    o, n = by_arm(a.old), by_arm(a.new)
    arms = [x for x in 'ABCD' if x in o and x in n]
    print(f'old: {a.old}\nnew: {a.new}\narms: {",".join(arms)}\n')

    rows = [
        ('n_events',            ('n_events',),                    False),
        ('n 2-plane',           ('n_2plane',),                    False),
        ('n selected',          ('n_sel',),                       False),
        ('n wall-matched',      ('n_wall_matched',),              False),
        ('c2/c1 (kernel)',      None,                             False),
        ('k_track median',      ('k_track', 'median'),            True),
        ('k_track MAD',         ('k_track', 'mad'),               True),
        ('v_insitu (k_track)',  ('k_track', 'v_insitu'),          True),
        ('k_opt (image focus)', ('k_opt',),                       True),
        ('r_core @ k_opt [mm]', ('image_at_kopt', 'r_core'),      True),
        ('r_med  @ k_opt [mm]', ('image_at_kopt', 'r_med'),       True),
        ('frac in capsule',     ('image_at_kopt', 'frac_in_capsule'), True),
        ('k_phys (coincident)', ('k_phys',),                      True),
        ('r_core @ k_phys',     ('image_at_kphys_full', 'r_core'), True),
        ('confirmed / predictable', None,                         False),
    ]

    for arm in arms:
        ro, rn = o[arm], n[arm]
        print(f'=== arm {arm}')
        for lab, path, pct in rows:
            if lab.startswith('c2/c1'):
                print(f'  {lab:24s} (kernel ratio is a property of the bundle, '
                      'not the summary)')
                continue
            if lab.startswith('confirmed'):
                for r, tag in ((ro, 'old'), (rn, 'new')):
                    c = g(r, 'pointing_coincidence', 'n_coincident')
                    p = g(r, 'pointing_coincidence', 'n_predictable')
                    if c is not None and p:
                        print(f'  {lab if tag == "old" else "":24s} {tag}: '
                              f'{c:6d} / {p:6d} = {100 * c / p:5.1f} %')
                continue
            print(f'  {lab:24s} {fmt(g(ro, *path), g(rn, *path), pct=pct)}')
        print()

    # arm-to-arm agreement: the spread of v_insitu across arms, before/after
    for tag, d in (('old', o), ('new', n)):
        v = [g(d[x], 'k_track', 'v_insitu') for x in arms]
        v = [x for x in v if x is not None]
        if len(v) > 1:
            print(f'{tag}: v_insitu across arms  '
                  f'med={np.median(v):6.2f}  spread(p84-p16)='
                  f'{np.percentile(v, 84) - np.percentile(v, 16):6.2f}  '
                  + ' '.join(f'{x:.1f}' for x in v))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""mdreport.py -- the tables of RESULT.md, generated from result.json / the final parquets (report.py calls write_md)."""
import json
import os

import numpy as np

import rows as R

OUT = R.OUTDIR

# r3 (time-block CV, all events incl. railed, constants refit on the test block) targets: (s68, err) per view
R3 = {
    ('A_0627', 'long'): {'x': (0.688, 0.005), 'y': (0.599, 0.005)},
    ('A_0625', 'all'): {'x': (0.681, 0.010), 'y': (0.666, 0.008)},
    ('B_0622', 'long'): {'x': (0.675, 0.006), 'y': (0.735, 0.006)},
    ('D_0626', 'long'): {'x': (0.802, 0.019), 'y': (0.829, 0.013)},
    ('E', '6-23'): {'x': (2.354, 0.077), 'y': (1.377, 0.051)},
    ('E', '6-24'): {'x': (2.120, 0.048), 'y': (1.324, 0.042)},
    ('C_long', 'all'): {'x': (0.932, 0.010), 'y': (1.228, 0.014)},   # r3: the three C rows together, time5
}
R3_P1 = {('A_0627', 'long'): {'x': 0.923, 'y': 0.805}, ('A_0625', 'all'): {'x': 0.876, 'y': 1.030}, ('B_0622', 'long'): {'x': 0.961, 'y': 0.966},
         ('D_0626', 'long'): {'x': 0.968, 'y': 1.028}, ('E', '6-23'): {'x': 2.405, 'y': 1.627}, ('E', '6-24'): {'x': 2.141, 'y': 1.436},
         ('C_long', 'all'): {'x': 1.183, 'y': 1.480}}


def fmt(x, e=None, d=3):
    return f'{x:.{d}f}' + (f' +- {e:.{d}f}' if e is not None else '')


def write_md(df, fr, res):
    L_ = []
    A = L_.append
    A('## Mountings (`mountings.csv`: one line per registry row)\n')
    A('| mounting | chamber | map unit(s) | rows | gate events | main drift V | rows by role | railed (any view) | alignment spread (x off, y off, theta) | decision |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for m, g in df.groupby('mounting', sort=False):
        roles = ', '.join(f'{r} {n}' for r, n in g.role.value_counts().items())
        units = ', '.join(sorted(set(u for u in g.unit if u)) or ['none'])
        cens = np.average(g.cens_any.fillna(0), weights=np.maximum(g.n_gate, 1))
        sp = f'{g.x_off.max() - g.x_off.min():.2f}, {g.y_off.max() - g.y_off.min():.2f} mm, {g.theta.max() - g.theta.min():.2f} deg'
        dec = g.note[g.note != ''].iloc[0] if (g.note != '').any() else ''
        main = int(g.main_V.dropna().iloc[0]) if g.main_V.notna().any() else int(g.drift_V.mode().iloc[0])
        A(f'| {m} | {g.chamber.iloc[0]} | {units} | {len(g)} | {int(g.n_gate.sum())} | {main} | {roles} | {100 * cens:.1f} % | {sp} | {dec} |')
    A('')
    A('## Angle s68 [deg]: pass 1 against the map, 5-time-block cross-validation\n')
    A("Events: gate events of the unit's training rows (railed tracks of the view excluded; D X only for reference x >= 200 mm). "
      "`map CV` = each event from a map and row constants trained without its time block; `map (production)` = the shipped map on its own training events (in sample, for reference only).\n")
    A('| unit | view | N | pass 1 | map CV | gain (CV) | map (production) | far fraction (> 3 s68 of pass 1): pass 1 -> CV | K / K_S / lam (full fit) |')
    A('|---|---|---|---|---|---|---|---|---|')
    for u, r in res['units'].items():
        f0 = r['folds']['-1']
        for v in 'xy':
            t = r['train'].get(v)
            if not t:
                continue
            A(f"| {u} | {v.upper()} | {t['n']} | {fmt(t['p1'], t['p1_err'])} | {fmt(t['cv'], t['cv_err'])} | {100 * t['gain']:.1f} % ({t['diff']:+.3f} +- {t['diff_err']:.3f}) | {fmt(t['map'])} | {100 * t['far_p1']:.1f} % -> {100 * t['far_cv']:.1f} % | {f0['K']} / {f0['KS']} / {f0['lam']:g} |")
    A('')
    A('## Against the r3 targets (r3 rows: long, longer, short; r3 included railed tracks)\n')
    A('`excl.` = railed tracks of the view excluded from fit and statistics (production treatment); `incl.` = all selected events (C with the railed rescale; D and E railed tracks are only excluded in the production output, shown here for comparison with r3). '
      'r3 refit the two row constants on every test block; here the constants also come from the training blocks only.\n')
    A('| subset | view | r3 pass 1 | r3 map | ours pass 1 (excl.) | ours CV (excl.) | ours pass 1 (incl.) | ours CV (incl.) | (ours incl. - r3) / sigma |')
    A('|---|---|---|---|---|---|---|---|---|')
    sets = [('A_0627', 'long', 'A 6-27/28 (sat + p2 long rows)', 'long_rows', 'long_rows_incl_railed'),
            ('A_0625', 'all', 'A 6-25', 'train', 'train_incl_railed'),
            ('B_0622', 'long', 'B 6-22 (long, longer, short)', 'long_rows', 'long_rows_incl_railed'),
            ('D_0626', 'long', 'D 6-26 (long, longer, short; X: x >= 200)', 'long_rows', 'long_rows_incl_railed'),
            ('E', '6-23', 'E 6-23 (joint map)', 'E_6-23', 'E_6-23_incl_railed'),
            ('E', '6-24', 'E 6-24 (joint map)', 'E_6-24', 'E_6-24_incl_railed'),
            ('C_long', 'all', 'C long row (r3: all three C rows)', 'train', 'train_incl_railed')]
    for u, tag, name, ke, ki in sets:
        r = res['units'].get(u)
        if not r:
            continue
        for v in 'xy':
            ex, inc = (r.get(ke) or {}).get(v), (r.get(ki) or {}).get(v)
            tg = R3.get((u, tag), {}).get(v)
            if not ex or not inc:
                continue
            z = f"{(inc['cv'] - tg[0]) / np.hypot(inc['cv_err'], tg[1]):+.1f}" if tg else ''
            p1r3 = R3_P1.get((u, tag), {}).get(v)
            A(f"| {name} | {v.upper()} | {fmt(p1r3) if p1r3 else ''} | {fmt(tg[0]) if tg else ''} | {fmt(ex['p1'], ex['p1_err'])} | {fmt(ex['cv'], ex['cv_err'])} | {fmt(inc['p1'], inc['p1_err'])} | {fmt(inc['cv'], inc['cv_err'])} | {z} |")
    A('')
    c3 = res.get('C3')
    if c3:
        A('Chamber C, the three r3 rows together (own map for long and longer, the map of the nearest row in time for short), s68 [deg]; r3: pass 1 1.183 / 1.480, ONE map for the three rows 0.932 / 1.228, leave-one-row-out 1.037 / 1.285:\n')
        A('| railed tracks | view | N | pass 1 | map CV | map CV x railed rescale |')
        A('|---|---|---|---|---|---|')
        for tag in ('excl', 'incl'):
            for v in 'xy':
                t = c3[tag].get(v)
                if t:
                    A(f"| {'excluded' if tag == 'excl' else 'included'} | {v.upper()} | {t['n']} | {fmt(t['p1'], t['p1_err'])} | {fmt(t['cv'], t['cv_err'])} | {fmt(t['final_cv'], t['final_cv_err'])} |")
        A('')
    A('## Rows that were not in the training of their map\n')
    A('| unit | role | view | N | pass 1 | map CV | gain |')
    A('|---|---|---|---|---|---|---|')
    for u, r in res['units'].items():
        for role in ('tilt_only', 'borrowed', 'tiny'):
            for v in 'xy':
                t = (r.get(role) or {}).get(v)
                if t:
                    A(f"| {u} | {role} | {v.upper()} | {t['n']} | {fmt(t['p1'], t['p1_err'])} | {fmt(t['cv'], t['cv_err'])} | {100 * t['gain']:.1f} % ({t['diff']:+.3f} +- {t['diff_err']:.3f}) |")
    A('')
    tf = res['tilt_fields']
    if tf:
        A('Tilt part only (S = 0) against the full map on the rows at another drift field of A 6-27/28 (CV, unrailed):\n')
        A('| drift V | rows | N | X pass 1 | X tilt only | X full | Y pass 1 | Y tilt only | Y full |')
        A('|---|---|---|---|---|---|---|---|---|')
        for t in tf:
            g = lambda k: fmt(t[k]) if k in t else ''
            A(f"| {t['drift_V']} | {t['rows']} | {t['n']} | {g('x_p1')} | {g('x_tilt')} | {g('x_full')} | {g('y_p1')} | {g('y_tilt')} | {g('y_full')} |")
        A('')
        for u, d in res['tilt'].items():
            A(f"Pooled decision for {u}: " + '; '.join(f"{v.upper()}: pass 1 {fmt(d[v]['p1'][0])}, tilt only {fmt(d[v]['tilt'][0])}, full {fmt(d[v]['full'][0])}, full - tilt = {d[v]['full_minus_tilt']:+.3f} +- {d[v]['err']:.3f}" for v in 'xy' if v in d) + f" -> scale part used: {d['use_full']}.\n")
    A('## Railed tracks\n')
    A('| row (C) | view | railed events | factor 1/slope | slope |')
    A('|---|---|---|---|---|')
    pooled_done = set()
    for kv, c in res['censor'].items():
        k, v = kv.split('|')
        if c['n_railed'] >= 150:
            A(f"| {k[-48:]} | {v.upper()} | {c['n_railed']} | {c['factor']:.3f} | {c['slope']:.3f} |")
        elif v not in pooled_done:
            pooled_done.add(v)
            A(f"| all other C rows (< 150 railed events each), pooled | {v.upper()} | | {c['factor']:.3f} | {c['slope']:.3f} |")
    A('')
    A('Railed C tracks, residual x sign(tan_ref) [deg] (a negative mean = flatter than the reference):\n')
    A('| view | N | pass 1 mean / s68 | map CV mean / s68 | map CV x factor, mean / s68 |')
    A('|---|---|---|---|---|')
    for v, b in res['bias_C'].items():
        A(f"| {v.upper()} | {b['n']} | {b['p1_mean']:+.3f} / {b['p1_s68']:.3f} | {b['cv_mean']:+.3f} / {b['cv_s68']:.3f} | {b['resc_mean']:+.3f} / {b['resc_s68']:.3f} |")
    A('')
    A('## Rows with >= 300 gate events whose CV angle is worse than pass 1 by more than 2 sigma (paired bootstrap)\n')
    A('| row | unit | role | view | N | pass 1 | map CV | diff |')
    A('|---|---|---|---|---|---|---|---|')
    nworse = 0
    for k, r in res['rows'].items():
        for v, t in r['stats'].items():
            if t['n'] >= 300 and t['diff'] > 2 * t['diff_err']:
                nworse += 1
                A(f"| {k[-60:]} | {r['unit']} | {r['role']} | {v.upper()} | {t['n']} | {fmt(t['p1'])} | {fmt(t['cv'])} | {t['diff']:+.3f} +- {t['diff_err']:.3f} |")
    if not nworse:
        A('| none | | | | | | | |')
    A('')
    A('## Mid-gap position residual [mm] and edge zone\n')
    A('Position: s68 of (p0 + 14 tan) - (ref_mesh + 14 tan_ref), pass 1 -> map CV (unrailed events of the long rows). Edge zone: events inside the active area but within 30 mm of its border '
      '(never in a fit; production map, clamped to the domain; unrailed): angle s68 pass 1 -> map.\n')
    A('| unit | position X | position Y | edge X: N, pass 1 -> map (diff) | edge Y: N, pass 1 -> map (diff) |')
    A('|---|---|---|---|---|')
    for u, r in res['units'].items():
        p, e = r.get('position', {}), r.get('edge', {})
        fe = lambda t: f"{t['n']}, {t['p1']:.2f} -> {t['map']:.2f} ({t['diff']:+.2f} +- {t['diff_err']:.2f})" if t else ''
        A(f"| {u} | {p['x']['p1']:.3f} -> {p['x']['cv']:.3f} | {p['y']['p1']:.3f} -> {p['y']['cv']:.3f} | {fe(e.get('x'))} | {fe(e.get('y'))} |")
    A('')
    A('## Model selection: edge hits of the inner CV (6 fits per unit: full + 5 folds)\n')
    A(f"Grids: K in {{6, 9, 12, 16, 20, 24}}, K_S in {{0, 4, 6, 8, 10, 12, 14, 16}}, lam in 10^-6 ... 10^3 (10 values), K^2 + 2 K_S^2 <= N/15.\n")
    A('| unit | lam at grid edge | K at grid max | K_S at grid max | lam range chosen | K chosen | K_S chosen |')
    A('|---|---|---|---|---|---|---|')
    for u, r in res['units'].items():
        fo = list(r['folds'].values())
        A(f"| {u} | {sum(1 for f in fo if f['lam_edge'])} | {sum(1 for f in fo if f['k_edge'])} | {sum(1 for f in fo if f['ks_edge'])} | {min(f['lam'] for f in fo):g} .. {max(f['lam'] for f in fo):g} | {', '.join(str(f['K']) for f in fo)} | {', '.join(str(f['KS']) for f in fo)} |")
    A('')
    A('## Reproduction of the fitter by wft.reco.bow_pairs (block installed in the row bundle)\n')
    A('| unit | row | n events | max abs diff tan X | max abs diff tan Y |')
    A('|---|---|---|---|---|')
    for u in res['units']:
        uj = json.load(open(f'{OUT}/units/{u}.json'))
        rp = uj.get('reproduction')
        if rp:
            A(f"| {u} | {rp['row'][-50:]} | {rp['n']} | {rp['max_abs_diff_x']:.1e} | {rp['max_abs_diff_y']:.1e} |")
    A('')
    if res.get('diag_E'):
        A('## E with one map per day (diagnostic, not shipped)\n')
        A('| map | view | N | pass 1 | map CV | gain |')
        A('|---|---|---|---|---|---|')
        for u, d in res['diag_E'].items():
            for v in 'xy':
                t = d.get(v)
                if t:
                    A(f"| {u} | {v.upper()} | {t['n']} | {fmt(t['p1'], t['p1_err'])} | {fmt(t['cv'], t['cv_err'])} | {100 * t['gain']:.1f} % ({t['diff']:+.3f} +- {t['diff_err']:.3f}) |")
        A('')
    return '\n'.join(L_)

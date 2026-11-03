#!/usr/bin/env python3
"""final_tables.py build|run -- the audit-era bow estimators (xcheck.py, profiles.py, summary.py) on the FINAL tables.

build  (lxplus, EOS mounted; read-only) one per-event table per row in the layout fbcommon.load_path reads, from
       Analysis_pass1 (or Analysis_pass1_twotrack for chamber A's two golden rows) + the paper1_eff rays (the row's own
       waveform alignment): ref_mesh_{x,y}, tan_ref_{x,y} from rays_<det>.parquet joined to the pass table.
       -> OUT/<name>.parquet, OUT/<name>.bundle.json
run    (anywhere) xcheck.run + profiles.run on every table of OUT, exactly the audit estimator, results redirected to
       OUT/results/ (xcheck_<name>.json, profiles_<name>.json); then summary-style numbers -> OUT/results/summary_final.json
Row set = the audit's golden rows + the 19-run consistency set (fbcommon.DATASETS / FB_ROWS).
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
B = '/eos/experiment/ntof/data/x17/cosmic_bench'
P1, P1T = f'{B}/Analysis_pass1', f'{B}/Analysis_pass1_twotrack'
EFF, EFFT = f'{B}/x17rerun/paper1_eff', f'{B}/x17rerun/paper1_eff_twotrack'
C = 'c26__'
ROWS = {   # name -> (chamber, registry key, twotrack?)
    'det2': ('det2', 'o22_long_det2', 0), 'det3': ('det3', 'sat_det3', 1), 'det4': ('det4', 'g_det4', 0),
    'det6': ('det6', 'g_det6_long', 0), 'det7': ('det7', 'g_det7_long', 0),
    'det3_day625': ('det3', C + 'mx17_det3_day_6-25-26__long_run__mx17_3', 0),
    'det3_d700': ('det3', C + 'mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_700V__mx17_3', 0),
    'det3_d500': ('det3', C + 'mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_500V__mx17_3', 0),
    'det2_o22long': ('det2', C + 'mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 0),
    'det2_o17': ('det2', C + 'mx17_det1_det2_overnight_6-17-26__longer_run__mx17_2', 0),
    'det3_p2': ('det3', C + 'mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 1),
    'det3_o23': ('det3', C + 'mx17_det3_det4_overnight_6-23-26__long_run__mx17_3', 0),
    'det4_o23': ('det4', C + 'mx17_det3_det4_overnight_6-23-26__long_run__mx17_4', 0),
    'det6_longer': ('det6', C + 'mx17_det6_det7_overnight_6-26-26__longer_run__mx17_6', 0),
    'det7_longer': ('det7', C + 'mx17_det6_det7_overnight_6-26-26__longer_run__mx17_7', 0),
    'det6_hv505': ('det6', C + 'mx17_det6_det7_overnight_6-26-26__resist_det6_505V_det7_480V_drift_700V__mx17_6', 0),
    'det7_hv480': ('det7', C + 'mx17_det6_det7_overnight_6-26-26__resist_det6_505V_det7_480V_drift_700V__mx17_7', 0),
    'det6_hv530': ('det6', C + 'mx17_det6_det7_overnight_6-26-26__resist_det6_530V_det7_505V_drift_700V__mx17_6', 0),
    'det7_hv505': ('det7', C + 'mx17_det6_det7_overnight_6-26-26__resist_det6_530V_det7_505V_drift_700V__mx17_7', 0),
}
COLS = ('p0', 'w', 't0', 'q_u50', 'q_u90', 'q_uend', 'q_sum', 'n_seed', 'n_strips', 'chi2', 'dof', 'ftst', 'ok', 'n_flagged_strips')


def build(out, only=None, pass_root=None, eff_root=None, tt_pass=P1T, tt_eff=EFFT):
    import csv
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    os.makedirs(out, exist_ok=True)
    for n, (det, key, tt) in ROWS.items():
        if only and n not in only:
            continue
        r = reg[key]; rel = f"{r['run']}/{r['subrun']}/{r['det']}"
        p1, eff = (tt_pass, tt_eff) if tt else (pass_root or P1, eff_root or EFF)
        wdir = f'{p1}/{rel}/wft'
        m = json.load(open(f'{wdir}/events.meta.json'))
        rays = pd.read_parquet(f'{eff}/budget/{rel}/rays_{det}.parquet').drop_duplicates('event_id', keep=False)
        df = pd.read_parquet(f'{wdir}/events.parquet').drop_duplicates('event_id', keep=False)
        keep = ['event_id'] + [f'{p}_{c}' for p in 'xy' for c in COLS if f'{p}_{c}' in df.columns]
        T = rays[['event_id', 'mesh_x', 'mesh_y', 'tan_x', 'tan_y']].rename(
            columns=dict(mesh_x='ref_mesh_x', mesh_y='ref_mesh_y', tan_x='tan_ref_x', tan_y='tan_ref_y')).merge(df[keep], on='event_id')
        T.to_parquet(f'{out}/{n}.parquet')
        ac = m['angle_constants']
        json.dump(dict(key=key, rel=rel, pass_root=p1, v_drift=float(m['bundle']['v_drift']), kw=ac['kw'], w0=ac['w0']),
                  open(f'{out}/{n}.bundle.json', 'w'), indent=1)
        print(n, key, len(T), p1, flush=True)


def run1(args):
    d, n = args
    sys.path.insert(0, HERE)
    import fbcommon as F, xcheck as X, profiles as PR
    res = f'{d}/results'; os.makedirs(res, exist_ok=True)
    b = json.load(open(f'{d}/{n}.bundle.json'))
    det = ROWS[n][0]
    slot = {'det6': 'bottom', 'det3_day625': 'bottom', 'det3_o23': 'bottom', 'det6_longer': 'bottom', 'det6_hv505': 'bottom',
            'det6_hv530': 'bottom'}.get(n, 'top')
    T = F.load_path(f'{d}/{n}.parquet', b, det, slot=slot, note='final tables', name=n)
    F.load = lambda name, T=T: T
    X.load = F.load; PR.load = F.load
    X.RES = res; PR.RES = res
    X.run(n); PR.run(n)
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('build', 'run', 'gate', 'summary'))
    ap.add_argument('--out', required=True); ap.add_argument('--only', default=None); ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args(); only = a.only.split(',') if a.only else None
    if a.mode == 'build':
        build(a.out, only)
    elif a.mode == 'gate':
        sys.path.insert(0, HERE)
        import fbcommon as F, gate_bow as G
        slot = lambda n: 'bottom' if n in ('det6', 'det3_day625', 'det3_o23', 'det6_longer', 'det6_hv505', 'det6_hv530') else 'top'
        cache = {}

        def ld(n):
            if n not in cache:
                cache[n] = F.load_path(f'{a.out}/{n}.parquet', json.load(open(f'{a.out}/{n}.bundle.json')), ROWS[n][0],
                                       slot(n), 'final tables', n)
            return cache[n]
        G.load = ld
        G.DATASETS = {n: (ROWS[n][0],) for n in ROWS}
        os.makedirs(f'{a.out}/results', exist_ok=True)
        out = {n: G.gate(n) for n in F.DETS}
        json.dump(out, open(f'{a.out}/results/gate_bow.json', 'w'), indent=1)
        tr = {f'{ROWS[n][0]}->{n}': G.transfer(ROWS[n][0], n) for n in ROWS if n not in F.DETS}
        json.dump(tr, open(f'{a.out}/results/gate_transfer.json', 'w'), indent=1)
    else:
        import multiprocessing as mp
        sys.path.insert(0, HERE)
        names = [n for n in ROWS if os.path.exists(f'{a.out}/{n}.parquet') and (not only or n in only)]
        with mp.Pool(a.jobs) as p:
            for n in p.imap_unordered(run1, [(a.out, n) for n in names]):
                print('done', n, flush=True)

#!/usr/bin/env python3
"""
fleet_state.py — what is actually on disk, per chamber, right now.

Every number an analysis quotes comes from a file, and every file was produced
by one calibration bundle under one reconstruction configuration. This walks
the fleet and reports that mapping, so a reader (or an auditor) can tell at a
glance which generation each chamber's numbers belong to and whether they are
mutually comparable.

Reads, per run key:
  <det>/wft/events.meta.json                 which bundle produced the table
  <det>/wft/efficiency/efficiency_breakdown{,_hits}.json    position + detection
  <det>/wft/angles/angular_resolution.json                  angles
  <det>/wft/gap_study/gap_study.json                        drift column

    ../../.venv/bin/python mx_june_wft/fleet_state.py [--markdown] [keys ...]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

FLEET = ['sat_det3', 'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long']
DET = {'sat_det3': 'det3', 'o22_long_det2': 'det2', 'g_det4': 'det4',
       'g_det6_long': 'det6', 'g_det7_long': 'det7'}
# the erfc endpoint fit bounds in gap_merge/gap_study; a fit sitting on one of
# them is not a measurement
SIG_E_BOUND = 400.0


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def state(key):
    from qa_config import get_config
    W = os.path.join(get_config(key).OUT_BASE, 'wft')
    s = {'key': key, 'det': DET.get(key, key), 'wft_dir': W}

    m = jload(os.path.join(W, 'events.meta.json'))
    if m:
        b = m.get('bundle', {})
        h = b.get('hyper', {})
        s['kernel'] = 'RC-ladder (share_lp)' if h.get('share_lp') else 'legacy'
        s['v_drift'] = b.get('v_drift')
        s['n_events'] = m.get('n_events')
        rc = m.get('reco_config', {}) or {}
        s['reco_config'] = rc
        # the meta's bundle snapshot predates w0/kw; the authority is the
        # bundle directory the reco recorded
        cb = jload(os.path.join(str(m.get('calibration', '')), 'bundle.json')) or {}
        s['bundle_dir'] = os.path.basename(str(m.get('calibration', '')))
        s['w0'] = cb.get('w0') or rc.get('w0') or {}
        s['kw'] = cb.get('kw') or rc.get('kw') or {}
        s['bundle_note'] = (b.get('provenance', {}) or {}).get('note', '')
        s['reco_mtime'] = time.strftime(
            '%Y-%m-%d %H:%M', time.localtime(
                os.path.getmtime(os.path.join(W, 'events.parquet'))))

    # an analysis output older than the table it is supposed to describe is a
    # stale number, and mixing the two is exactly how a wrong figure survives
    t_reco = (os.path.getmtime(os.path.join(W, 'events.parquet'))
              if os.path.exists(os.path.join(W, 'events.parquet')) else 0)
    s['stale'] = sorted(
        os.path.basename(p) for p in
        (os.path.join(W, 'efficiency', 'efficiency_breakdown.json'),
         os.path.join(W, 'angles', 'angular_resolution.json'),
         os.path.join(W, 'alignment', 'alignment.json'))
        if os.path.exists(p) and os.path.getmtime(p) < t_reco - 60)

    e = jload(os.path.join(W, 'efficiency', 'efficiency_breakdown.json'))
    if e:
        s.update(within5=e.get('within_R'), core=e.get('core_sigma_mm'),
                 median_r=e.get('median_r_mm'), reco_at_all=e.get('reco_at_all'),
                 has_any=e.get('has_any'), spark=e.get('spark_frac'),
                 n_rays=e.get('n_rays'))
    eh = jload(os.path.join(W, 'efficiency', 'efficiency_breakdown_hits.json'))
    if eh:
        s.update(hits_within5=eh.get('within_R'), hits_core=eh.get('core_sigma_mm'))

    a = jload(os.path.join(W, 'angles', 'angular_resolution.json'))
    if a:
        for p in ('x', 'y'):
            pl = a['planes'].get(p, {})
            s[f'sig_{p}'] = pl.get('sigma_deg')
            s[f'bias_{p}'] = pl.get('bias_deg')
            s[f'vsp_{p}'] = pl.get('implied_v_spread')

    g = jload(os.path.join(W, 'gap_study', 'gap_study.json'))
    if g:
        pl = g.get('planes', {}).get('x', {})
        sh = (pl.get('fits') or {}).get('sharp', {})
        s['gap_mm'] = sh.get('gap_mm')
        s['gap_err'] = sh.get('gap_err')
        s['gap_v_geom'] = pl.get('v_geom')
        s['gap_railed'] = (abs(sh.get('sig_e', 0) - SIG_E_BOUND) < 1e-6
                           if 'sig_e' in sh else None)
        s['gap_bundle'] = os.path.basename(str(g.get('bundle', '')))
    return s


def f(v, spec='{:.2f}'):
    return spec.format(v) if isinstance(v, (int, float)) else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='*', default=None)
    ap.add_argument('--markdown', action='store_true')
    ap.add_argument('--export', default=None,
                    help='copy the source jsons of every quoted number into '
                         'this directory (audit snapshot; the analysis outputs '
                         'live on the data disk, which a reviewer may not have)')
    args = ap.parse_args()

    from qa_config import setup_paths
    setup_paths()
    rows = [state(k) for k in (args.keys or FLEET)]

    if args.export:
        import shutil
        os.makedirs(args.export, exist_ok=True)
        for s in rows:
            d = os.path.join(args.export, s['det'])
            os.makedirs(d, exist_ok=True)
            W = s['wft_dir']
            for rel in ('events.meta.json',
                        'efficiency/efficiency_breakdown.json',
                        'efficiency/efficiency_breakdown_cut.json',
                        'efficiency/efficiency_breakdown_hits.json',
                        'angles/angular_resolution.json',
                        'alignment/alignment.json',
                        'gap_study/gap_study.json'):
                src = os.path.join(W, rel)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(d, rel.replace('/', '__')))
        with open(os.path.join(args.export, 'fleet_state.json'), 'w') as fh:
            json.dump(rows, fh, indent=1, default=str)
        print(f'exported {len(rows)} chambers into {args.export}')

    if args.markdown:
        print('| chamber | reco generation | kernel | v [µm/ns] | within 5 mm '
              '(hits) | core σ [mm] | σ_θ X / Y [°] | bias X / Y [°] | column X [mm] |')
        print('|---|---|---|---|---|---|---|---|---|')
        for s in rows:
            gap = f(s.get('gap_mm')) + (' ⚠' if s.get('gap_railed') else '')
            gen = s.get('reco_mtime', '—')
            if s.get('stale'):
                gen += ' ⚠ **stale downstream**'
            print(f"| {s['det']} (`{s['key']}`) | {gen} | "
                  f"{s.get('kernel', '—')} | {f(s.get('v_drift'))} | "
                  f"{f(s.get('within5'))} ({f(s.get('hits_within5'))}) | "
                  f"{f(s.get('core'), '{:.3f}')} | "
                  f"{f(s.get('sig_x'))} / {f(s.get('sig_y'))} | "
                  f"{f(s.get('bias_x'), '{:+.2f}')} / {f(s.get('bias_y'), '{:+.2f}')} | "
                  f"{gap} |")
        return

    for s in rows:
        print(f"\n=== {s['det']}  ({s['key']})")
        if s.get('stale'):
            print(f"  ** STALE: {', '.join(s['stale'])} predate events.parquet **")
        print(f"  reco      {s.get('reco_mtime', '—')}  "
              f"{s.get('kernel', 'no table')}  v={f(s.get('v_drift'))}  "
              f"n={s.get('n_events', '—')}")
        rc = s.get('reco_config', {})
        print(f"  bundle    {s.get('bundle_dir', '—')}  "
              f"w0={ {k: round(v, 3) for k, v in s.get('w0', {}).items()} }  "
              f"kw={ {k: round(v, 3) for k, v in s.get('kw', {}).items()} }")
        print(f"  config    model_frac={rc.get('model_frac')} "
              f"prescan={rc.get('prescan_coarse')} chi2dof_bad={rc.get('chi2dof_bad')}"
              f"   [{s.get('bundle_note', '')[:50]}]")
        print(f"  position  within5 {f(s.get('within5'))} % "
              f"(hits {f(s.get('hits_within5'))})  core {f(s.get('core'), '{:.3f}')} "
              f"(hits {f(s.get('hits_core'), '{:.3f}')})  "
              f"reco_at_all {f(s.get('reco_at_all'))} %")
        print(f"  detection has_any {f(s.get('has_any'))} %  "
              f"spark {f(s.get('spark'))} %  rays {s.get('n_rays', '—')}")
        print(f"  angles    sigma {f(s.get('sig_x'))} / {f(s.get('sig_y'))} deg  "
              f"bias {f(s.get('bias_x'), '{:+.2f}')} / {f(s.get('bias_y'), '{:+.2f}')}  "
              f"v-spread {f(s.get('vsp_x'), '{:.1f}')} / {f(s.get('vsp_y'), '{:.1f}')}")
        print(f"  column X  {f(s.get('gap_mm'))} +- {f(s.get('gap_err'))} mm "
              f"(v_geom {f(s.get('gap_v_geom'))}, bundle {s.get('gap_bundle', '—')})"
              + ('  ** erfc width RAILED: not a measurement **'
                 if s.get('gap_railed') else ''))


if __name__ == '__main__':
    main()

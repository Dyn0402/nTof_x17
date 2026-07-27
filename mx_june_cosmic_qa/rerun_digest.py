#!/usr/bin/env python3
"""Digest for the full June rerun (RERUN_PLAN_2026-07-24.md §6).

Two modes:

  # BEFORE the rerun -- freeze the current on-disk numbers as the baseline
  rerun_digest.py --snapshot rerun_baseline.json sat_det3 g_det4 ...

  # AFTER  the rerun -- the driver calls this itself
  rerun_digest.py <main_log> <digest.md> sat_det3 g_det4 ...

Everything is read from the structured outputs the analysis scripts already
write (efficiency_breakdown.txt, efficiency_map_sliding.json,
angular_resolution.json, time_resolution.json, hybrid_summary.csv,
cache/cshare.json), plus the main log for the per-step OK/WARN tally and 03's
printed X/Y resolution.  Missing pieces are reported as n/a, never guessed.
"""
import os
import re
import sys
import csv
import json

import qa_config

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, 'rerun_baseline.json')
VETO_DIR = 'alignment_tpc_veto50'


# ---------------------------------------------------------------- harvesting
def _txt(path):
    try:
        with open(path, errors='ignore') as f:
            return f.read()
    except OSError:
        return ''


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _f(pat, txt, grp=1):
    m = re.search(pat, txt)
    return float(m.group(grp)) if m else None


def harvest(key):
    """All headline numbers for one key, from files on disk right now."""
    try:
        base = qa_config.get_config(key).OUT_BASE
    except Exception as e:                                    # unknown key
        return {'error': f'{type(e).__name__}: {e}'}

    d = {'out_base': base}

    # --- 09 efficiency breakdown (text) -----------------------------------
    bd = _txt(os.path.join(base, 'efficiency', 'efficiency_breakdown.txt'))
    if bd:
        d['n_rays'] = _f(r'active-area clean M3 rays:\s*(\d+)', bd)
        d['has_any'] = _f(r'has_any=([\d.]+)%', bd)
        d['within5'] = _f(r'within5mm=([\d.]+)%', bd)
        d['reco_at_all'] = _f(r'reco-at-all=([\d.]+)%', bd)
        d['core_sigma_mm'] = _f(r'core sigma\(\|r\|<15\)=([\d.]+) mm', bd)
        d['median_r_mm'] = _f(r'median \|r\|=([\d.]+) mm', bd)
        d['spark_frac'] = _f(r'spark_frac=([\d.]+)%', bd)
        d['reco_far'] = _f(r'reco_far\s*:\s*\d+\s*\(\s*([\d.]+)%\)', bd)

    # --- 12 sliding efficiency map ----------------------------------------
    sl = _json(os.path.join(base, 'efficiency', 'efficiency_map_sliding.json'))
    if sl:
        for k_src, k_dst in [('integrated_within', 'sliding_within'),
                             ('integrated_has_any', 'sliding_has_any')]:
            if sl.get(k_src) is not None:
                d[k_dst] = 100.0 * sl[k_src]
        d['sliding_n_rays_active'] = sl.get('n_rays_active')

    # --- 03 angular resolution / v_drift ----------------------------------
    ar = _json(os.path.join(base, VETO_DIR, 'angular_resolution.json'))
    for k in ('sigma_theta_x_deg', 'sigma_theta_y_deg',
              'v_drift_x_um_ns', 'v_drift_y_um_ns', 'n_events', 'reliable'):
        if k in ar:
            d[k] = ar[k]

    # --- 42 time resolution ------------------------------------------------
    tr = _json(os.path.join(base, VETO_DIR, 'time_resolution',
                            'time_resolution.json'))
    for k in ('single_strip_sigma_ns', 'lead_singleplane_ns',
              'abs_t0_sigma68_ns', 'v_drift_um_ns', 'n_events_dualplane'):
        if k in tr:
            d[k] = tr[k]

    # --- 26 measured charge sharing ---------------------------------------
    cs = _json(os.path.join(base, 'cache', 'cshare.json'))
    if cs:
        d['cshare'] = {str(k): list(v) for k, v in cs.items()}

    # --- 34 hybrid tracking ------------------------------------------------
    hp = os.path.join(base, VETO_DIR, 'hybrid', 'hybrid_summary.csv')
    if os.path.exists(hp):
        rows = []
        try:
            with open(hp) as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append((r['estimator'], r['band'],
                                     float(r['coverage']), float(r['bias_deg']),
                                     float(r['s68_deg'])))
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            rows = []
        hyb = {}
        for band in sorted({r[1] for r in rows}):
            inband = [r for r in rows if r[1] == band]
            best = min(inband, key=lambda r: r[4])
            prod = next((r for r in inband if 'production' in r[0].lower()), None)
            hyb[band] = {'best': {'estimator': best[0], 'coverage': best[2],
                                  'bias_deg': best[3], 's68_deg': best[4]}}
            if prod:
                hyb[band]['production'] = {'estimator': prod[0],
                                           'coverage': prod[2],
                                           'bias_deg': prod[3],
                                           's68_deg': prod[4]}
        if hyb:
            d['hybrid'] = hyb
        d['hybrid_model_saved'] = os.path.exists(
            os.path.join(base, VETO_DIR, 'hybrid', 'hybrid_model.json'))

    return d


# ---------------------------------------------------------------- log parsing
def parse_log(path, keys):
    """-> (per-key {ok, warn, warn_steps, res_x, res_y}, global tallies)."""
    txt = _txt(path)
    if not txt:
        return {}, {'ok': 0, 'warn': 0, 'warn_steps': [], 'missing_log': True}

    # split on the driver's per-key banner
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r'#{6,} PRIMARY (\S+) #{6,}', txt)]
    fleet = re.search(r'#{6,} FLEET / SCANS #{6,}', txt)
    bounds = marks + [(fleet.start() if fleet else len(txt), '_FLEET')]

    def tally(chunk):
        ok = re.findall(r'^\s*OK   : (.+)$', chunk, re.M)
        warn = re.findall(r'^\s*WARN : (.+)$', chunk, re.M)
        return {'ok': len(ok), 'warn': len(warn), 'warn_steps': warn}

    per_key = {}
    for i, (start, key) in enumerate(marks):
        end = bounds[i + 1][0]
        chunk = txt[start:end]
        info = tally(chunk)
        # 03 prints these once per invocation (veto50 --full, then --no-veto)
        info['res_x'] = re.findall(r'X resolution: ([\d.]+) \+/- ([\d.]+) mm', chunk)
        info['res_y'] = re.findall(r'Y resolution: ([\d.]+) \+/- ([\d.]+) mm', chunk)
        info['cshare_log'] = re.findall(
            r'FEU\s*(\d+):\s*c1 = ([\d.-]+)\s+c2 = ([\d.-]+)', chunk)
        info['cshare_written'] = 'cshare.json <-' in chunk
        per_key[key] = info

    glob = tally(txt)
    glob['fleet'] = tally(txt[fleet.start():]) if fleet else None
    glob['finished'] = 'Full June rerun finished' in txt
    m = re.search(r'started (.+)$', txt, re.M)
    glob['started'] = m.group(1).strip().rstrip('= ') if m else '?'
    m = re.search(r'Full June rerun finished (.+)$', txt, re.M)
    glob['ended'] = m.group(1).strip().rstrip('= ') if m else '(did not reach the end)'
    return per_key, glob


# ---------------------------------------------------------------- formatting
def fmt(v, spec='{:.2f}', na='n/a'):
    if v is None:
        return na
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def cmp_cell(new, old, spec='{:.2f}', pct_delta=True):
    """'new (old, +x%)' — the old-vs-new side-by-side the plan asks for."""
    s = fmt(new, spec)
    if old is None or new is None:
        return s + ('  (old n/a)' if old is None else '')
    try:
        if pct_delta and old:
            return f'{s}  (was {spec.format(old)}, {100*(new-old)/abs(old):+.0f}%)'
        return f'{s}  (was {spec.format(old)})'
    except (ValueError, TypeError, ZeroDivisionError):
        return s


ROWS = [
    ('rays (09, active area)',      'n_rays',              '{:.0f}'),
    ('has_any %',                   'has_any',             '{:.1f}'),
    ('within 5 mm %',               'within5',             '{:.1f}'),
    ('reco-at-all %',               'reco_at_all',         '{:.1f}'),
    ('reco_far %',                  'reco_far',            '{:.1f}'),
    ('core sigma r (mm)',           'core_sigma_mm',       '{:.2f}'),
    ('median r (mm)',               'median_r_mm',         '{:.2f}'),
    ('spark_frac %',                'spark_frac',          '{:.1f}'),
    ('sliding-map within %  (12)',  'sliding_within',      '{:.1f}'),
    ('sigma_theta X (deg)  (03)',   'sigma_theta_x_deg',   '{:.2f}'),
    ('sigma_theta Y (deg)  (03)',   'sigma_theta_y_deg',   '{:.2f}'),
    ('v_drift X (um/ns)   (03)',    'v_drift_x_um_ns',     '{:.1f}'),
    ('v_drift Y (um/ns)   (03)',    'v_drift_y_um_ns',     '{:.1f}'),
    ('single-strip sigma_t (ns) (42)', 'single_strip_sigma_ns', '{:.1f}'),
    ('lead single-plane (ns)   (42)', 'lead_singleplane_ns', '{:.1f}'),
    ('abs t0 sigma68 (ns)      (42)', 'abs_t0_sigma68_ns', '{:.1f}'),
]


def write_digest(log_path, out_path, keys):
    new = {k: harvest(k) for k in keys}
    old = _json(BASELINE)
    old_vals = old.get('keys', {}) if isinstance(old, dict) else {}
    per_key, glob = parse_log(log_path, keys)

    L = []
    L.append('# June cosmic rerun — results digest')
    L.append('')
    L.append(f'- log: `{log_path}`')
    L.append(f'- started: {glob.get("started", "?")}')
    L.append(f'- ended:   {glob.get("ended", "?")}')
    L.append(f'- steps:   {glob.get("ok", 0)} OK / {glob.get("warn", 0)} WARN')
    if old_vals:
        L.append(f'- baseline for the "was" column: `{BASELINE}`'
                 f' (snapshot {old.get("stamp") or "?"})')
    else:
        L.append('- **no baseline file** — "was" columns are blank '
                 f'(expected `{BASELINE}`)')
    L.append('')
    L.append('Numbers come from the analysis scripts\' own outputs '
             '(efficiency_breakdown.txt, efficiency_map_sliding.json, '
             'angular_resolution.json, time_resolution.json, '
             'hybrid_summary.csv, cache/cshare.json).')
    L.append('')

    # ---- headline table ---------------------------------------------------
    L.append('## Headline numbers')
    L.append('')
    hdr = '| quantity | ' + ' | '.join(keys) + ' |'
    L.append(hdr)
    L.append('|' + '---|' * (len(keys) + 1))
    for label, field, spec in ROWS:
        cells = []
        for k in keys:
            cells.append(cmp_cell(new[k].get(field),
                                  old_vals.get(k, {}).get(field), spec))
        L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    L.append('')

    # ---- per-detector detail ---------------------------------------------
    for k in keys:
        d = new[k]
        info = per_key.get(k, {})
        L.append(f'## {k}')
        L.append('')
        if d.get('error'):
            L.append(f'**config error: {d["error"]}**')
            L.append('')
            continue
        L.append(f'`{d["out_base"]}`')
        L.append('')
        L.append(f'- steps: {info.get("ok", 0)} OK / {info.get("warn", 0)} WARN')
        for w in info.get('warn_steps', []):
            L.append(f'  - WARN: {w}')

        # charge sharing (the per-detector constant the plan cares about)
        cs = d.get('cshare')
        if cs:
            txt = ', '.join(f'FEU {f}: c1={v[0]:.3f} c2={v[1]:.3f}'
                            for f, v in sorted(cs.items()))
            L.append(f'- **CSHARE measured (26 -> cache/cshare.json): {txt}**')
            oc = old_vals.get(k, {}).get('cshare')
            if oc:
                L.append('  - previously: ' + ', '.join(
                    f'FEU {f}: c1={v[0]:.3f} c2={v[1]:.3f}'
                    for f, v in sorted(oc.items())))
        else:
            L.append('- **CSHARE: not measured** — 26 found no usable leads; '
                     '27/28 fell back to their hardcoded (det7) dict. '
                     'Treat this detector\'s 27/28 output as unvalidated.')
            if info.get('cshare_log'):
                L.append(f'  - (log did show FEU rows: {info["cshare_log"]})')

        # 03's printed residual resolutions
        for axis in ('x', 'y'):
            vals = info.get(f'res_{axis}') or []
            if vals:
                L.append(f'- 03 {axis.upper()} resolution (per invocation): '
                         + ', '.join(f'{a} +/- {b} mm' for a, b in vals))

        # hybrid
        hyb = d.get('hybrid')
        if hyb:
            L.append('- hybrid tracking (34):')
            for band, v in sorted(hyb.items()):
                b = v['best']
                line = (f'  - band `{band}`: best = {b["estimator"]} — '
                        f'sigma68 {b["s68_deg"]:.2f} deg, '
                        f'coverage {100*b["coverage"]:.1f}%, '
                        f'bias {b["bias_deg"]:+.2f} deg')
                ob = (old_vals.get(k, {}).get('hybrid', {})
                      .get(band, {}).get('best'))
                if ob:
                    line += f'  (was {ob["s68_deg"]:.2f} deg / {ob["estimator"]})'
                L.append(line)
                p = v.get('production')
                if p:
                    L.append(f'    production time-fit: sigma68 '
                             f'{p["s68_deg"]:.2f} deg, coverage '
                             f'{100*p["coverage"]:.1f}%')
            L.append(f'  - model saved: {d.get("hybrid_model_saved")}')
        else:
            L.append('- hybrid tracking (34): no hybrid_summary.csv')
        L.append('')

    # ---- fleet / scans ----------------------------------------------------
    fl = glob.get('fleet')
    if fl:
        L.append('## Fleet / scan stage')
        L.append('')
        L.append(f'- steps: {fl["ok"]} OK / {fl["warn"]} WARN')
        for w in fl['warn_steps']:
            L.append(f'  - WARN: {w}')
        L.append('')

    L.append('## Review checklist (RERUN_PLAN §5)')
    L.append('')
    L.append('- [ ] det4: did 26 measure c1/c2 this time? '
             '(old verdict "gain-limited, hybrid not measurable" was suspected '
             'to be a trigger artifact — matched filter recovered ~1.5x its hits)')
    L.append('- [ ] any efficiency that moved by more than a few % vs the "was" column')
    L.append('- [ ] sigma_theta / v_drift consistent with the pre-rerun values')
    L.append('- [ ] WARN steps above: real failures or the expected Magboltz/env ones?')
    L.append('- [ ] only after review: propagate into JUNE_RESULTS_SUMMARY / paper docs')
    L.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'digest -> {out_path}')


def snapshot(out_path, keys):
    data = {'stamp': os.environ.get('SNAP_STAMP', ''),
            'note': 'pre-rerun baseline (previous-generation outputs on disk)',
            'keys': {k: harvest(k) for k in keys}}
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=1)
    n = sum(1 for k in keys if data['keys'][k].get('within5') is not None)
    print(f'baseline -> {out_path}  ({n}/{len(keys)} keys had efficiency numbers)')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(2)
    if a[0] == '--snapshot':
        snapshot(a[1], a[2:])
    else:
        write_digest(a[0], a[1], a[2:])

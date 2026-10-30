import csv, json, os, sys, glob
import numpy as np, pyarrow.parquet as pq
ROOTS = {'p1': '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1',
         'rr1': '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1',
         'fe': '/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe'}
reg = list(csv.DictReader(open(sys.argv[1])))
def lj(p):
    try: return json.load(open(p))
    except Exception: return None
def summ(root, r):
    d = f"{root}/{r['run']}/{r['subrun']}/{r['det']}/wft"
    o = {}
    if not os.path.exists(d + '/events.parquet'): return None
    m = lj(d + '/events.meta.json') or {}
    o['n_events'] = m.get('n_events')
    b = m.get('bundle', {}) if isinstance(m.get('bundle'), dict) else {}
    o['v'] = b.get('v_drift'); o['rc_D_y'] = (b.get('hyper') or {}).get('rc_D_y')
    o['chi2med_x'] = (m.get('quality_rule') or {}).get('chi2dof_median', {}).get('x')
    o['chi2med_y'] = (m.get('quality_rule') or {}).get('chi2dof_median', {}).get('y')
    o['ovr'] = len(m.get('diagnostic_overrides') or {})
    a = lj(d + '/angles/angular_resolution.json')
    if a:
        for v in 'xy':
            p = a['planes'].get(v, {})
            o[f's68_{v}'] = p.get('s68_deg'); o[f'bias_{v}'] = p.get('bias_deg'); o[f'nang_{v}'] = p.get('n')
            iv = p.get('implied_v'); o[f'ivspread_{v}'] = p.get('implied_v_spread')
            o[f'iv_{v}'] = float(np.nanmean(iv)) if iv else None
    e = lj(d + '/efficiency/efficiency_breakdown_cut.json')
    if e:
        for k in ['n_rays', 'within_R', 'reco_at_all', 'reco_far', 'hit_no_reco', 'no_hit', 'spark_frac', 'core_sigma_mm', 'median_r_mm']:
            o[k] = e.get(k)
    al = lj(d + '/alignment/alignment.json')
    if al: o['align'] = json.dumps(al)[:300]
    w = lj(d + '/calib_bundle/w0kw.json')
    if w:
        for v in 'xy':
            o[f'w0_{v}'] = w[v].get('w0'); o[f'kw_{v}'] = w[v].get('kw'); o[f'tfloor_{v}'] = w[v].get('tan_floor')
    bj = lj(d + '/calib_bundle/bundle.json')
    if bj:
        o['bundle_v'] = bj.get('v_drift'); o['has_bow'] = bool(bj.get('bow')); o['ndead'] = len(bj.get('dead') or [])
        o['bundle_runkey'] = bj.get('run_key')
    if root == ROOTS['p1']:
        st = sorted(glob.glob(f"{root}/{r['run']}/{r['subrun']}/{r['det']}/STAGES/*.json"))
        codes = set()
        for s in st:
            j = lj(s) or {}
            c = j.get('code') or j.get('code_hash') or (j.get('env') or {}).get('WFT_REPO', '')
            codes.add(str(c)[-7:])
        o['codes'] = ','.join(sorted(codes))
        try:
            t = pq.read_table(d + '/events.parquet', columns=['x_ok','y_ok','x_quality_ok','y_quality_ok','x_tan_theta','y_tan_theta','spark','n_tracks','x_p0','y_p0']).to_pandas()
            o['fx_ok'] = t.x_ok.mean(); o['fy_ok'] = t.y_ok.mean()
            o['fxq'] = t.x_quality_ok.mean(); o['fyq'] = t.y_quality_ok.mean()
            o['mtan_x'] = t.x_tan_theta[t.x_ok].median(); o['mtan_y'] = t.y_tan_theta[t.y_ok].median()
            o['nan_p0x'] = float(t.x_p0[t.x_ok].isna().mean()); o['multi'] = float((t.n_tracks > 1).mean())
            o['fspark'] = float(t.spark.mean())
        except Exception as ex:
            o['pq_err'] = str(ex)[:100]
    return o
out = []
for r in reg:
    row = {k: r[k] for k in ['key', 'det', 'run', 'subrun', 'resist_V', 'drift_V', 'cls', 'refit', 'K', 'calib_key']}
    for tag, root in ROOTS.items():
        s = summ(root, r)
        if s:
            for k, v in s.items(): row[f'{tag}.{k}'] = v
    out.append(row)
    print(r['key'], file=sys.stderr)
json.dump(out, open(sys.argv[2], 'w'), default=float)

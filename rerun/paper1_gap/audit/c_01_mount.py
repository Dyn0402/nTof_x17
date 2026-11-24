import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
import os; os.makedirs(OUT, exist_ok=True)
res = {}
for k, rel in ROWS.items():
    d = load(rel)
    al = json.load(open(f'{WORK}/pass1_all/{rel}/mx17_3/alignment/alignment.json'))
    b = json.load(open(f'{WORK}/pass1_all/{rel}/mx17_3/calib_bundle/bundle.json'))
    r = dict(n=len(d), theta=al['theta_deg'], zx=al['z_x'], zy=al['z_y'], xoff=al['x_offset'], yoff=al['y_offset'], refsign=al['ref_x_sign'],
             bundle_v=b['v_drift'], w0=b['w0'], kw=b['kw'], cond=b['conditions'], runkey=b['run_key'], bowsrc=b['bow'].get('source_row'),
             bowhash=hash(json.dumps(b['bow'], sort_keys=True)))
    for p in 'xy':
        v = view(d, p)
        c = CO.line(v['tr'], v['w'])
        r[f'ladder_{p}'] = dict(w0=c[0], v=c[1], w0e=c[2], ve=c[3], n=c[4])
        r[f'sign_tan_pass1_vs_ref_{p}'] = float(np.corrcoef(v['tp'], v['tr'])[0, 1])
    r['tan_ref_range'] = [float(d.tan_ref_x.abs().quantile(.95)), float(d.tan_ref_y.abs().quantile(.95))]
    res[k] = r
    print(k, json.dumps(r, default=str)[:1800])
json.dump(res, open(f'{OUT}/mount.json', 'w'), indent=1, default=str)

"""peek_res.py -- print the final_rule.json resolution summary (scratch inspection)."""
import json

d = json.load(open('rerun/sps_junk/results/final_rule.json'))
for run, R in d.items():
    print(run, {k: R[k] for k in R if k != 'est'}, list(R['est']))
    for e, E in R['est'].items():
        for v in ('X', 'Y'):
            V = E.get(v)
            if not V:
                print(' ', e, v, 'none')
                continue
            f = V.get('fit') or {}
            zs = [(z['pitch'], z['n'], round(z['sigma_res'] * 1e3), round(z['sigma_det4'] * 1e3)) for z in V['zones']]
            fs = f"{f['sigma_det4']*1e3:.0f}+-{f['sigma_det4_err']*1e3:.0f} fback {f['f_back']:.3f}" if f else 'no fit'
            print(f"  {e:6s} {v} res {V['sigma_res']*1e3:.0f} zones {zs} fit {fs} mix {V['mix']['sigma_det4']*1e3:.0f}+-{V['mix']['err']*1e3:.0f}")

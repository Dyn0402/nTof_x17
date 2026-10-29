"""T15: can ONE extra model parameter per chamber and view fix the universal shape mismatch (T9) and help the angles?
Dylan 10-29: "fix the model ... with some single extra parameter (maybe per detector per view)"; also "are X and Y
templates the same? separate ones are physically justified".

Base = the pass as planned (T8 A): fe_nobow bundle, sigma0 0.30/0.30 fixed, the bundle's D_rc,y, no t0 offset (delta
dropped, Dylan 10-29).  Families, each one parameter per view (an arm sets X and Y alike; each view is read on its own,
since a view's fit sees only its own keys):
  st<s>   template time scale: both views driven by the X template stretched about its t50, h(t / s) (rc_tmpl 'own'
          with tmpl_y = the stretched X template).  The temporal half of T9 (data narrower in time; T1 segment).
  dp<k>   drift-diffusion width Dp x k (one gas parameter, both views).  The lateral half, T13's lever.
  cf<f>   narrow prompt core: a fraction f of each landing at core_sigma 0.07 mm instead of sigma0 (wft.model
          core_frac_<plane>, a two-Gaussian footprint that survives the Y RC growth).  "Narrower core, same tails".
  yown_D<D>  Y on its OWN measured template with D_rc,y = D (T14 kept the plan D, so it counted the RC spreading
          twice: once in the template's undershoot and once in the kernel).  X untouched.
Held-out: heldout_arm.py --t0-fix on cache events 180-1500 (all 1320; no training here, a grid instead of a fit), LM,
WFT_COL_GUARD 0.01, on lxplus condor (one node per chamber x arm).  Compare: t5_compare (event-paired vs base).

Usage:
  python -I t15_one_param.py build OUT        local: bundles + caches + DAG into OUT (then rsync OUT to EOS)
  python -I t15_one_param.py compare DIR      DIR holds <det>_<arm>.npz -> results/t15/t15_<det>.json
  python -I t15_one_param.py table            markdown tables from results/t15/
"""
import json, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/bundles/{}/calib_bundle_fe_nobow')
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
EOS = '/eos/user/d/dneff/x17rerun/t15'
ARMS = {'base': {},
        'st092': dict(stretch=0.92), 'st096': dict(stretch=0.96), 'st104': dict(stretch=1.04),
        'dp085': dict(scale={'Dp': 0.85}), 'dp115': dict(scale={'Dp': 1.15}),
        'cf20': dict(set={'core_frac_x': 0.2, 'core_frac_y': 0.2}), 'cf40': dict(set={'core_frac_x': 0.4, 'core_frac_y': 0.4}),
        'yown_D0': dict(yown=0.0), 'yown_D1': dict(yown=1e-4), 'yown_D2': dict(yown=2e-4), 'yown_Dplan': dict(yown=None)}
SH = '''#!/bin/bash
# run_t15.sh <code_dir> <heldout_arm.py args...>
CODE=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg WFT_COL_GUARD=0.01
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/heldout_arm.py "$@"
'''
SUB = '''executable     = run_t15.sh
arguments      = $(code) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/t15.log
request_cpus   = 8
request_memory = 16000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def stretched(grid, t, s):
    return np.interp(grid / s, grid, t, left=0.0, right=0.0)


def build(out):
    for det in DETS:
        os.makedirs(f'{out}/caches', exist_ok=True)
        shutil.copy(CACHE.format(det), f'{out}/caches/{det}.pkl')
        for arm, spec in ARMS.items():
            d = f'{out}/bundles/{det}/{arm}'; os.makedirs(d, exist_ok=True)
            b = json.load(open(f'{SRC.format(det)}/bundle.json')); h = b['hyper']
            a = dict(np.load(f'{SRC.format(det)}/arrays.npz'))
            h['sigma_p0_x'] = h['sigma_p0_y'] = 0.30
            for k in ('sigma_p0_prof_x', 'sigma_p0_prof_y'):
                h.pop(k, None)
            for k, f in spec.get('scale', {}).items():
                h[k] = h[k] * f
            h.update(spec.get('set', {}))
            if 'stretch' in spec:
                a['tmpl_x'] = a['tmpl_y'] = stretched(a['grid'], a['tmpl_x'], spec['stretch'])
                h['rc_tmpl'] = 'own'
            if 'yown' in spec:
                h['rc_tmpl'] = 'own'
                if spec['yown'] is not None:
                    h['rc_D_y'] = spec['yown']
            json.dump(b, open(f'{d}/bundle.json', 'w'), indent=1)
            np.savez(f'{d}/arrays.npz', **a)
    L = []
    for det in DETS:
        for arm in ARMS:
            n = f'T15_{det}_{arm}'
            args = (f'--cache {EOS}/caches/{det}.pkl --bundle {EOS}/bundles/{det}/{arm} --out {EOS}/out/{det}_{arm}.npz '
                    f'--start 180 --n 1320 --jobs 8 --k 24 --t0-fix')
            L += [f'JOB {n} t15.sub', f'VARS {n} nodename="{n}" code="$(CODE)" args="{args}"', f'RETRY {n} 1']
    open(f'{out}/t15.dag', 'w').write('\n'.join(L) + '\n')
    open(f'{out}/t15.sub', 'w').write(SUB); open(f'{out}/run_t15.sh', 'w').write(SH); os.chmod(f'{out}/run_t15.sh', 0o755)
    print(f'{out}: {len(DETS) * len(ARMS)} nodes')


def compare(d):
    sys.path.insert(0, HERE)
    from t5_compare import main as cmp
    os.makedirs(f'{HERE}/results/t15', exist_ok=True)
    for det in DETS:
        if os.path.exists(f'{d}/{det}_base.npz'):
            cmp(d, det, 'base', 't15/t15')


def table():
    fam = [('st', 'template stretch s', ['st092', 'st096', 'st104']), ('dp', 'Dp x k', ['dp085', 'dp115']),
           ('cf', 'core fraction f (0.07 mm)', ['cf20', 'cf40']),
           ('yown', 'Y own template, D_rc,y', ['yown_D0', 'yown_D1', 'yown_D2', 'yown_Dplan'])]
    R = {det: json.load(open(f'{HERE}/results/t15/t15_{det}.json')) for det in DETS
         if os.path.exists(f'{HERE}/results/t15/t15_{det}.json')}
    for key, title, arms in fam:
        for v in ('xy' if key != 'yown' else 'y'):
            print(f'\n**{title}, {v.upper()}**: Δs68 [°] ± paired bootstrap · χ²/dof (base) · q ratio\n')
            print('| chamber | ' + ' | '.join(arms) + ' |'); print('|---|' + '---|' * len(arms))
            for det, r in R.items():
                cells = []
                for a in arms:
                    q = r.get(f'{v}_{a}')
                    cells.append('—' if q is None else f"{q['ds68']:+.3f} ± {q['ds68_err']:.3f} · {q['chi2']:.1f} ({q['chi2_base']:.1f}) · {q['q_ratio']:.3f}")
                print(f'| {det} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    {'build': lambda: build(sys.argv[2]), 'compare': lambda: compare(sys.argv[2]), 'table': table}[sys.argv[1]]()

"""T13: where does the remaining lateral mismatch live -- drift-diffusion width (Dp) or footprint shape?
det3, base = fe_nobow bundle at D_rc,y 3.44e-4 on main's fitter (the T6/T9 baseline 'cur_D3.44', sigma0 0.30 fixed).
Arms: Dp x 0.8 / x 1.2; pseudo-Voigt prompt footprint (wft.model lor_frac/lor_gamma, cloud_basics §23 ranges)
eta 0.2 gamma 0.6 mm and eta 0.3 gamma 0.8 mm on both views.  Each arm: t9 peak residuals (centre/±1/±2 vs k) and
the held-out A/B (heldout_arm --t0-fix, events 180-579) against the baseline.
Usage: python -I t13_width_shape.py  (work dir $T13_WORK, default the job tmp)"""
import json, os, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '../../..'))
PY = os.path.join(REPO, '.venv/bin/python')
WORK = os.environ.get('T13_WORK', '/home/dylan/.claude/jobs/f754a42b/tmp/t13')
BASE = '/home/dylan/.claude/jobs/f754a42b/tmp/t6/cur_D3.44'
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/det3.pkl')
ENV = dict(os.environ, WFT_COL_GUARD='0.01')
ARMS = {'dp080': dict(scale={'Dp': 0.8}), 'dp120': dict(scale={'Dp': 1.2}),
        'lor20': dict(set={'lor_frac_x': 0.2, 'lor_gamma_x': 0.6, 'lor_frac_y': 0.2, 'lor_gamma_y': 0.6}),
        'lor30': dict(set={'lor_frac_x': 0.3, 'lor_gamma_x': 0.8, 'lor_frac_y': 0.3, 'lor_gamma_y': 0.8})}

os.makedirs(WORK, exist_ok=True)
shutil.copytree(BASE, f'{WORK}/base', dirs_exist_ok=True)
for name, a in ARMS.items():
    d = f'{WORK}/{name}'; shutil.copytree(BASE, d, dirs_exist_ok=True)
    b = json.load(open(f'{d}/bundle.json'))
    for k, f in a.get('scale', {}).items():
        b['hyper'][k] = b['hyper'][k] * f
    b['hyper'].update(a.get('set', {}))
    json.dump(b, open(f'{d}/bundle.json', 'w'), indent=1)
for name in ['base'] + list(ARMS):
    rj = f'{HERE}/results/t13/det3_{name}_resid.json'; os.makedirs(os.path.dirname(rj), exist_ok=True)
    if not os.path.exists(rj):
        print(f'== {name} residuals', flush=True)
        subprocess.run([PY, '-I', f'{HERE}/t9_peak_residual.py', f'{WORK}/{name}', rj, 'det3'], env=ENV, check=True)
    out = f'{WORK}/det3_{name}.npz'
    if not os.path.exists(out):
        subprocess.run([PY, '-I', f'{REPO}/rerun/model_audit/heldout_arm.py', '--cache', CACHE, '--bundle', f'{WORK}/{name}',
                        '--out', out, '--start', '180', '--n', '400', '--jobs', '7', '--t0-fix'],
                       env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([PY, '-I', f'{HERE}/t5_compare.py', WORK, 'det3', 'base', 't13'], check=True)

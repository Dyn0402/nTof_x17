"""T8: fleet test of the calibration recipe (all five chambers, rr1 caches, main's fitter).

Per chamber, from the fe_nobow bundle:
  A  'plan'   : production template, sigma0 0.30/0.30, the bundle's D_rc,y, no t0 offset    (the pass as planned)
  B  'recal'  : production template, sigma0 fixed at 0.30 (a physical input, Dylan 10-29: same build -> same sigma0);
                rc_D_y (not det4: 2.2e-4 from H4) and t0_shift_x/y fitted
                jointly by recal.py --t0-pin --unsat-only on cache events 580-1079 (<= 80 NM evaluations)
  C  'heff'   : as B with the segment-free template h_eff (t4_closure.json, X view; rc_tmpl='x')
(A first version also freed sigma_p0_x/y: on det2 it ran to sigma0_x 0.058 mm, below the static-induction floor,
not converged after 126 evaluations -- sigma0 absorbs a footprint SHAPE error; results/t8/DIAG_det2_free_sigma0.json.)
Held-out events 180-579 (disjoint from the fit) through heldout_arm.py --t0-fix; t0 offsets are folded into t0_abs.
Usage: python -I t8_fleet.py fit|bundles|heldout|compare [det ...]   (work dir $T8_WORK, default the job tmp)"""
import json, os, shutil, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '../../..'))
PY = os.path.join(REPO, '.venv/bin/python')
WORK = os.environ.get('T8_WORK', '/home/dylan/.claude/jobs/f754a42b/tmp/t8')
SRC = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/bundles/{}/calib_bundle_fe_nobow')
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
ENV = dict(os.environ, WFT_COL_GUARD='0.01')
sys.path.insert(0, HERE)


def base_copy(det, name, tmpl=None):
    d = f'{WORK}/{det}/{name}'; os.makedirs(d, exist_ok=True)
    shutil.copy(f'{SRC.format(det)}/bundle.json', d)
    a = dict(np.load(f'{SRC.format(det)}/arrays.npz'))
    if tmpl is not None:
        a['tmpl_x'] = tmpl
    np.savez(f'{d}/arrays.npz', **a)
    b = json.load(open(f'{d}/bundle.json')); b['hyper']['sigma_p0_x'] = b['hyper']['sigma_p0_y'] = 0.30
    for k in ('sigma_p0_prof_x', 'sigma_p0_prof_y'):
        b['hyper'].pop(k, None)
    json.dump(b, open(f'{d}/bundle.json', 'w'), indent=1)
    return d


def fit(det):
    from t5_make_arms import heff_on_grid
    g = np.load(f'{SRC.format(det)}/arrays.npz')['grid']
    seeds = {'recal': base_copy(det, 'seed_recal'), 'heff': base_copy(det, 'seed_heff', heff_on_grid(det, g))}
    free = 't0_shift_x,t0_shift_y' + ('' if det == 'det4' else ',rc_D_y')
    for name, b in seeds.items():
        out = f'{WORK}/{det}/fit_{name}.json'
        if os.path.exists(out):
            continue
        cmd = [PY, '-I', f'{REPO}/sps_beam_test_26/analysis/plane_ratio/recal.py', '--cache', CACHE.format(det), '--bundle', b,
               '--arm', f't8_{name}', '--free', free, '--objective', 'cold', '--t0-pin', '--unsat-only', '--depth-bins', '24',
               '--p0-profile', '--p0-window', '2.0', '--p0-step', '0.2', '--train-start', '580', '--n-train', '500',
               '--out', out, '--jobs', '7', '--maxiter', '80']
        subprocess.run(cmd, env=ENV, check=True, stdout=open(f'{WORK}/{det}/fit_{name}.log', 'w'), stderr=subprocess.STDOUT)
        r = json.load(open(out))
        print(det, name, {k: round(r['hyper'][k], 5) for k in free.split(',') if k in r['hyper']}, r.get('t0_shift'),
              f"chi2 {r['chi2_seed']:.4e} -> {r['chi2']:.4e}", flush=True)


def bundles(det):
    if det == 'det4':
        pass
    os.makedirs(f'{WORK}/{det}', exist_ok=True)
    base_copy(det, 'A_plan')
    for name in ('recal', 'heff'):
        r = json.load(open(f'{WORK}/{det}/fit_{name}.json'))
        d = f'{WORK}/{det}/{"B_recal" if name == "recal" else "C_heff"}'
        shutil.copytree(f'{WORK}/{det}/seed_{name}', d, dirs_exist_ok=True)
        b = json.load(open(f'{d}/bundle.json'))
        for k in ('sigma_p0_x', 'sigma_p0_y', 'rc_D_y'):
            if k in r['hyper']:
                b['hyper'][k] = r['hyper'][k]
        for p, s in (r.get('t0_shift') or {}).items():
            b['t0_abs'][p] = {f: float(t) + float(s) for f, t in b['t0_abs'][p].items()}
        b.setdefault('provenance', {})['t8'] = dict(fit=f'fit_{name}.json', t0_shift=r.get('t0_shift'))
        json.dump(b, open(f'{d}/bundle.json', 'w'), indent=1)


def heldout(det):
    for arm in ('A_plan', 'B_recal', 'C_heff'):
        out = f'{WORK}/{det}_{arm}.npz'
        if os.path.exists(out):
            continue
        subprocess.run([PY, '-I', f'{REPO}/rerun/model_audit/heldout_arm.py', '--cache', CACHE.format(det), '--bundle',
                        f'{WORK}/{det}/{arm}', '--out', out, '--start', '180', '--n', '400', '--jobs', '7', '--t0-fix'],
                       env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compare(det):
    subprocess.run([PY, '-I', f'{HERE}/t5_compare.py', WORK, det, 'A_plan', 't8'], check=True)


if __name__ == '__main__':
    step = sys.argv[1]; dets = sys.argv[2:] or DETS
    for d in dets:
        globals()[step](d)

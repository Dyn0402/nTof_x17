#!/usr/bin/env python3
"""sat_arms.py -- held-out arms of the saturation treatment on lxplus condor, and their event-paired comparison.

Base = the planned pass (T15 base bundles: fe_nobow, sigma0 0.30, LM, t0 pinned), held-out events 180-1500 of each
chamber's rr1 cache, rerun/model_audit/heldout_arm.py --t0-fix --rails (per-sample ADC rails, rails.py).  An arm is a
set of wft.model switches passed as WFT_DIAG (all opt-in; base = none).

    sat_arms.py build OUT CODE [ROUND]   DAG + submit files into OUT (copy OUT to ~/x17rerun/satur/ on lxplus);
                                         CODE = git archive on EOS (/eos/user/d/dneff/x17rerun/code/code_<hash>)
    sat_arms.py compare DIR              DIR holds <det>_<arm>.npz -> results/arms/arms_<det>.json (t5_compare)
    sat_arms.py table                    markdown tables (all planes; saturated planes only)
    sat_arms.py emu DIR                  the emulated-rail diagnostic (emu<R> vs rail)
    sat_arms.py summary [ARM ...]        one line per arm and chamber (X all | X sat | Y all | Y sat)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DETS = ['det3', 'det2', 'det4', 'det6', 'det7']
EOS = '/eos/user/d/dneff/x17rerun'
#: signal-variance K [ADC] per chamber: 0.19 x ADC per electron (ws_floor KAPPA_FRAC), i.e. 11.5 on det3, scaled by
#: the toy's per-chamber median charge q50 (floor_toy.CHAMBERS: the toy's ADC-per-electron is q50 / its electron count)
Q50 = {'det2': 16558, 'det3': 11470, 'det4': 4314, 'det6': 22847, 'det7': 25340}
KSIG = {d: round(11.5 * q / Q50['det3'], 1) for d, q in Q50.items()}
R = 'model.SAT_RAIL=true'
ARMS = {
    'base': '',
    'rail': R,                                                              # clip mask = raw 4095 (per-sample rail)
    'floorub0': 'model.SAT_FLOOR=1',                                        # floor = upper bound, 3550 clip rule
    'railub': f'{R},model.SAT_FLOOR=1',                                     # rail + floor upper bound
    'railfree': f'{R},model.SAT_FLOOR=2',                                   # rail + floor censored, no bound
    'tail': f'{R},model.SAT_TAIL=true',                                     # rail + post-clip tail free
    'tailub': f'{R},model.SAT_TAIL=true,model.SAT_FLOOR=1',
    'tailL4ub': f'{R},model.SAT_TAIL=true,model.SAT_TAIL_MINLEN=4,model.SAT_FLOOR=1',
    'strip': f'{R},model.SAT_STRIP=true',                                   # whole clipped strip censored
    'm150': f'{R},model.SAT_MARGIN=150',                                    # soft compression below the rail
    'sv': 'model.SIG_VAR_K={K}',                                            # signal variance in the weights
    'sv_half': 'model.SIG_VAR_K={Kh}',
    'railub_sv': f'{R},model.SAT_FLOOR=1,model.SIG_VAR_K={{K}}',
    # r2: the clipped samples carry no bound (a clipped delta blob no longer forces charge onto the track model)
    'clipfree': f'{R},model.SAT_CLIP_FREE=true',
    'clipfree_ub': f'{R},model.SAT_CLIP_FREE=true,model.SAT_FLOOR=1',
    # r3: Huber loss on the live samples, C in noise sigma (a blob / mis-modelled bright sample pulls linearly)
    'rob10': 'model.ROBUST_C=10',
    'rob25': 'model.ROBUST_C=25',
    'rob50': 'model.ROBUST_C=50',
    'railub_rob25': f'{R},model.SAT_FLOOR=1,model.ROBUST_C=25',
    # r4: the diagnostic -- clip planes that never saturated at an emulated rail (heldout_arm --emulate-rail), with the
    # rail treatment; compare against 'rail' on the planes with no real clip and an emulated one
    'emu1500': R, 'emu2500': R,
    # r5: the emulated rail as a TREATMENT -- every sample above C censored as a lower bound (cap scan), rails kept
    'cap800': R, 'cap1000': R, 'cap1200': R, 'cap1500': R, 'cap2000': R,
    'cap1200ub': f'{R},model.SAT_FLOOR=1',
}
EXTRA = {'emu1500': ' --emulate-rail 1500', 'emu2500': ' --emulate-rail 2500', 'cap800': ' --emulate-rail 800',
         'cap1000': ' --emulate-rail 1000', 'cap1200': ' --emulate-rail 1200', 'cap1500': ' --emulate-rail 1500',
         'cap2000': ' --emulate-rail 2000', 'cap1200ub': ' --emulate-rail 1200'}
ROUNDS = {'r1': [a for a in ARMS if not a.startswith(('clipfree', 'rob', 'railub_rob'))], 'r2': ['clipfree', 'clipfree_ub'],
          'r3': ['rob10', 'rob25', 'rob50', 'railub_rob25'], 'r4': ['emu1500', 'emu2500'],
          'r5': ['cap800', 'cap1000', 'cap1200', 'cap1500', 'cap2000', 'cap1200ub']}
SH = '''#!/bin/bash
# run_sat.sh <code_dir> <WFT_DIAG or none> <heldout_arm.py args...>
CODE=$1; shift; DIAG=$1; shift
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg WFT_COL_GUARD=0.01
if [ "$DIAG" != "none" ]; then export WFT_DIAG=$DIAG; fi
echo "WFT_DIAG=$WFT_DIAG"
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/model_audit/heldout_arm.py "$@"
'''
SUB = '''executable     = run_sat.sh
arguments      = $(code) $(diag) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/sat.log
request_cpus   = 8
request_memory = 16000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def diag_of(arm, det):
    s = ARMS[arm].format(K=KSIG[det], Kh=round(KSIG[det] / 2, 1))
    return s or 'none'


def build(out, code, rnd='r1', arms=None, dets=None):
    os.makedirs(f'{out}/log', exist_ok=True)
    L = []
    for det in dets or DETS:
        for arm in arms or ARMS:
            n = f'SAT_{det}_{arm}'
            args = (f'--cache {EOS}/t15/caches/{det}.pkl --bundle {EOS}/t15/bundles/{det}/base '
                    f'--rails {EOS}/satur/rails/rails_{det}.pkl --out {EOS}/satur/{rnd}/{det}_{arm}.npz '
                    f'--start 180 --n 1320 --jobs 8 --k 24 --t0-fix' + EXTRA.get(arm, ''))
            L += [f'JOB {n} sat.sub', f'VARS {n} nodename="{n}" code="{code}" diag="{diag_of(arm, det)}" args="{args}"',
                  f'RETRY {n} 1']
    open(f'{out}/sat_{rnd}.dag', 'w').write('\n'.join(L) + '\n')
    open(f'{out}/sat.sub', 'w').write(SUB); open(f'{out}/run_sat.sh', 'w').write(SH); os.chmod(f'{out}/run_sat.sh', 0o755)
    print(f'{out}/sat_{rnd}.dag: {len(L) // 3} nodes')


# ------------------------------------------------------------------------------------------------ comparison
def angerr(R):
    tr, w = R[:, 2], R[:, 3]
    m = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45)
    c = np.median(w[np.abs(tr) < 0.05])
    s = np.median((w[m] - c) / tr[m])
    return np.degrees(np.arctan((w - c) / s) - np.arctan(tr)), s


def s68(d):
    k = np.abs(d) <= 5
    return float(np.percentile(np.abs(d[k] - np.median(d[k])), 68)) if k.sum() > 10 else np.nan


def compare_one(base, per, vi, sel=None, rng=None, nboot=300):
    """Event-paired: the ladder of each arm from ALL its planes of the view; s68/far/chi2/q on the selected planes."""
    B = {int(r[0]): r for r in base[base[:, 1] == vi]}; A = {int(r[0]): r for r in per[per[:, 1] == vi]}
    ids = sorted(set(A) & set(B))
    RA = np.array([A[i] for i in ids]); RB = np.array([B[i] for i in ids])
    dA, sA = angerr(RA); dB, sB = angerr(RB)
    m = np.ones(len(ids), bool) if sel is None else sel(RB)
    dA, dB, RA, RB = dA[m], dB[m], RA[m], RB[m]
    rng = rng or np.random.default_rng(3); boots = []
    for _ in range(nboot):
        j = rng.integers(0, len(dA), len(dA)); boots.append(s68(dA[j]) - s68(dB[j]))
    qr = np.median(RA[:, 15] / np.where(RB[:, 15] > 0, RB[:, 15], np.nan))
    return dict(n=int(m.sum()), s68=s68(dA), s68_base=s68(dB), ds68=s68(dA) - s68(dB), ds68_err=float(np.nanstd(boots)),
                far=float(100 * np.mean(np.abs(dA) > 5)), far_base=float(100 * np.mean(np.abs(dB) > 5)),
                chi2=float(np.median(RA[:, 4] / np.maximum(RA[:, 5], 1))), chi2_base=float(np.median(RB[:, 4] / np.maximum(RB[:, 5], 1))),
                ladder=float(sA), ladder_base=float(sB), q_ratio=float(np.nanmedian(qr)) if np.isfinite(qr).any() else np.nan,
                bias=float(np.median(dA[np.abs(dA) <= 5])), bias_base=float(np.median(dB[np.abs(dB) <= 5])))


SELS = {'all': None,
        'sat': lambda R: R[:, 16] > 0,                 # planes with a raw-clipped sample (the base row's own count)
        'unsat': lambda R: R[:, 16] == 0,
        'sat_heavy': lambda R: R[:, 16] >= 8}
#: emulated-clip diagnostic: planes with no real clip that the emulated rail does clip (the arm's own column 18)
EMU_SEL = lambda RB, RA: (RB[:, 16] == 0) & (RA[:, 18] > 0)


def compare(d, dets=None):
    import glob
    os.makedirs(f'{HERE}/results/arms', exist_ok=True)
    for det in dets or DETS:
        fb = f'{d}/{det}_base.npz'
        if not os.path.exists(fb):
            continue
        base = np.load(fb)['per']; out = {}
        for f in sorted(glob.glob(f'{d}/{det}_*.npz')):
            arm = os.path.basename(f)[len(det) + 1:-4]
            per = np.load(f)['per']
            for vi, v in enumerate('xy'):
                for sn, sel in SELS.items():
                    out[f'{v}_{sn}_{arm}'] = compare_one(base, per, vi, sel)
        json.dump(out, open(f'{HERE}/results/arms/arms_{det}.json', 'w'), indent=1)
        print(det, len(out) // (2 * len(SELS)), 'arms')


def table(arms=None, sels=('all', 'sat')):
    for det in DETS:
        p = f'{HERE}/results/arms/arms_{det}.json'
        if not os.path.exists(p):
            continue
        R = json.load(open(p)); have = sorted({k.split('_', 2)[2] for k in R})
        arms_ = [a for a in (arms or ARMS) if a in have]
        print(f'\n**{det}** — Δs68 [°] vs base ± paired bootstrap; far % ; χ²/dof (base) ; q ratio\n')
        print('| arm | ' + ' | '.join(f'{v.upper()} {s}' for v in 'xy' for s in sels) + ' |')
        print('|---|' + '---|' * (2 * len(sels)))
        for a in arms_:
            cells = []
            for v in 'xy':
                for s in sels:
                    q = R.get(f'{v}_{s}_{a}')
                    if q is None:
                        cells.append('—'); continue
                    if a == 'base':
                        cells.append(f"s68 {q['s68']:.3f} (n {q['n']}) · far {q['far']:.1f} · χ² {q['chi2']:.0f}")
                    else:
                        cells.append(f"{q['ds68']:+.3f} ± {q['ds68_err']:.3f} · far {q['far']:.1f} ({q['far_base']:.1f}) · "
                                     f"χ² {q['chi2']:.0f} ({q['chi2_base']:.0f}) · q {q['q_ratio']:.3f}")
            print(f'| {a} | ' + ' | '.join(cells) + ' |')


def emu(d, dets=('det3', 'det2', 'det7')):
    """The clip diagnostic: emu<R> against 'rail' (the same censor rule) on planes that never clipped but are clipped
    by the emulated rail; also those planes' s68 against the planes that really clipped."""
    out = {}
    for det in dets:
        fr = f'{d}/{det}_rail.npz'
        if not os.path.exists(fr):
            continue
        rail = np.load(fr)['per']
        for arm in ('emu1500', 'emu2500'):
            f = f'{d}/{det}_{arm}.npz'
            if not os.path.exists(f):
                continue
            per = np.load(f)['per']
            for vi, v in enumerate('xy'):
                A = {int(r[0]): r for r in per[per[:, 1] == vi]}
                hit = {e for e, r in A.items() if r[16] == 0 and r[18] > 0}
                sel = lambda R, hit=hit: np.array([int(r[0]) in hit for r in R])
                q = compare_one(rail, per, vi, sel)
                qs = compare_one(rail, rail, vi, SELS['sat'])
                out[f'{det}_{v}_{arm}'] = dict(q, real_sat_s68=qs['s68'], real_sat_far=qs['far'],
                                               frac_planes=len(hit) / max(len(A), 1))
                print(f"{det} {v} {arm}: {len(hit)} planes ({100 * len(hit) / max(len(A), 1):.0f} %) clipped only by the "
                      f"emulated rail: s68 {q['s68_base']:.3f} -> {q['s68']:.3f} ({q['ds68']:+.3f} ± {q['ds68_err']:.3f}), "
                      f"far {q['far_base']:.1f} -> {q['far']:.1f} %, chi2/dof {q['chi2_base']:.0f} -> {q['chi2']:.0f} | "
                      f"really clipped planes: s68 {qs['s68']:.3f}, far {qs['far']:.1f} %")
    os.makedirs(f'{HERE}/results/arms', exist_ok=True)
    json.dump(out, open(f'{HERE}/results/arms/emu.json', 'w'), indent=1)


def summary(arms=None):
    for det in DETS:
        p = f'{HERE}/results/arms/arms_{det}.json'
        if not os.path.exists(p):
            continue
        R = json.load(open(p)); b = lambda v, s_: R[f'{v}_{s_}_base']
        print(f"\n{det} base: X all {b('x','all')['s68']:.3f} sat {b('x','sat')['s68']:.3f} (n {b('x','sat')['n']}, far {b('x','sat')['far']:.1f}) | "
              f"Y all {b('y','all')['s68']:.3f} sat {b('y','sat')['s68']:.3f} (n {b('y','sat')['n']}, far {b('y','sat')['far']:.1f})")
        have = sorted({k.split('_', 2)[2] for k in R})
        for a in [a for a in (arms or ARMS) if a in have and a != 'base']:
            c = []
            for v in 'xy':
                for s_ in ('all', 'sat'):
                    q = R[f'{v}_{s_}_{a}']; c.append(f"{q['ds68']:+.3f}±{q['ds68_err']:.3f} f{q['far'] - q['far_base']:+.1f}")
            q, qy = R[f'x_sat_{a}'], R[f'y_sat_{a}']
            print(f'  {a:12s} ' + ' | '.join(c) + f"  | chi2 sat {q['chi2']:.0f}/{q['chi2_base']:.0f} {qy['chi2']:.0f}/{qy['chi2_base']:.0f}"
                  f" q {q['q_ratio']:.3f}/{qy['q_ratio']:.3f}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        rnd = (sys.argv[4:5] or ['r1'])[0]
        build(sys.argv[2], sys.argv[3], rnd, arms=ROUNDS.get(rnd), dets=sys.argv[5].split(',') if len(sys.argv) > 5 else None)
    elif cmd == 'compare':
        compare(sys.argv[2])
    elif cmd == 'emu':
        emu(sys.argv[2])
    elif cmd == 'summary':
        summary(sys.argv[2:] or None)
    else:
        table()

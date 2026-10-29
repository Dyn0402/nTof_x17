"""T8 summary: per chamber, the recalibrated (sigma0 fixed 0.30, pinned + unsaturated objective) D_rc,y and t0
offsets, and the held-out A (plan) vs B (recal) metrics from t5_compare's JSON.  Reads files, never logs.
Usage: python -I t8_summary.py  -> prints a markdown table; writes results/t8/summary.json"""
import json, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('T8_WORK', '/home/dylan/.claude/jobs/f754a42b/tmp/t8')
os.makedirs(f'{HERE}/results/t8', exist_ok=True)
rows = {}
for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
    f = f'{WORK}/{det}/fit_recal.json'
    if not os.path.exists(f):
        continue
    r = json.load(open(f)); shutil.copy(f, f'{HERE}/results/t8/{det}_fit_recal.json')
    b0 = json.load(open(f'{WORK}/{det}/seed_recal/bundle.json'))['hyper']
    cmp = f'{HERE}/results/t8_{det}.json'
    c = json.load(open(cmp)) if os.path.exists(cmp) else {}
    rows[det] = dict(d_x=r['t0_shift'].get('x'), d_y=r['t0_shift'].get('y'), rc_D_y=r['hyper'].get('rc_D_y'),
                     rc_D_y_bundle=b0.get('rc_D_y'), chi2=r['chi2'], chi2_seed=r['chi2_seed'], converged=r['converged'],
                     heldout={k: v for k, v in c.items() if k.endswith('B_recal')})
print('| chamber | δ_x [ns] | δ_y [ns] | D_rc,y fe → recal | train χ² Δ | held-out X s68 Δ (χ², ladder A→B) | Y s68 Δ (χ², ladder A→B) |')
print('|---|---|---|---|---|---|---|')
for det, r in rows.items():
    h = r['heldout']; cells = []
    for v in 'xy':
        q = h.get(f'{v}_B_recal')
        cells.append('—' if not q else f"{q['ds68']:+.3f} ± {q['ds68_err']:.3f} ({q['chi2_base']:.1f}→{q['chi2']:.1f}, {q['ladder_base']:.2f}→{q['ladder']:.2f})")
    print(f"| {det} | {r['d_x']:+.1f} | {r['d_y']:+.1f} | {r['rc_D_y_bundle']:.2e} → {r['rc_D_y']:.2e} | "
          f"{100*(r['chi2']/r['chi2_seed']-1):+.1f} % | {cells[0]} | {cells[1]} |")
json.dump(rows, open(f'{HERE}/results/t8/summary.json', 'w'), indent=1)

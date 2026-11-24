import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np, r4_lib as L
S = json.load(open(f'{L.OUT}/summary.json'))
bf = lambda V: np.interp(V, [int(k) for k in S['beta_eff']], list(S['beta_eff'].values()))
def line(r, beta=None):
    c, e = r['c'], r['e']
    f68 = 28.2 / 1.68; fb = 28.2 / (1 + (beta if beta is not None else bf(r['drift'])))
    return (f"| {r['label']} | {r['mesh']} | {r['drift']} | {r['N']} | {r['mean_q']:.0f} | {c[1]:+.3f} +- {e[1]:.3f} | {c[2]:+.3f} +- {e[2]:.3f} | "
            f"{c[3]:+.3f} +- {e[3]:.3f} | {c[4]:+.3f} +- {e[4]:.3f} | {c[5]:+.3f} +- {e[5]:.3f} | {r['amp'][0]:+.2f} +- {r['amp'][1]:.2f} | "
            f"{c[3]*f68:+.2f} / {c[4]*f68:+.2f} / {c[5]*f68:+.2f} | {c[3]*fb:+.2f} / {c[4]*fb:+.2f} / {c[5]*fb:+.2f} |")
hdr = ("| sub-run | mesh V | drift V | N | mean q_u90 (ns) | gu | gw | guu | gww | guw | golden-ridge amp. | guu / gww / guw in mm, beta 0.68 | same in mm, beta(V) |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
rows = S['rows']
main = [r for r in rows if not r['label'].startswith('6-27 hv_scan')]
out = '### Main table (coefficients of ln q_u90 per (100 mm)^2, bootstrap errors)\n\n' + hdr
for r in main:
    out += line(r, 1.28 if '6-25' in r['label'] else (0.68 if '6-22' in r['label'] else None)) + '\n'
out += '\n### Mesh scan at 1000 V drift (all 6-27 sub-runs)\n\n' + hdr
for r in rows:
    if r['label'].startswith('6-27 hv_scan'): out += line(r) + '\n'
open(f'{L.OUT}/tables.md', 'w').write(out)
print(out[:3500])

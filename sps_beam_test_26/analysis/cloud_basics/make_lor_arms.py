#!/usr/bin/env python3
"""make_lor_arms.py -- bench arms for the X footprint tails (FINDINGS §23).

From each chamber's rcm arm (the physical kernel, no refit) and the head-on X profile fit
in results/footprint_test.json ('mixlor': depth mixture + pseudo-Voigt tail):
  <det>_rcmlor   rcm + lor_frac_x, lor_gamma_x (the tail only)
  <det>_rcmlor2  the same + sigma_p0_x = the fit's prompt width
No refit of anything else.

    make_lor_arms.py <dir with arm_<det>_rcm.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    d = sys.argv[1]
    F = json.load(open(os.path.join(HERE, 'results', 'footprint_test.json')))
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        base = json.load(open(os.path.join(d, f'arm_{det}_rcm.json')))
        s0, a, eta, gam = F[det]['x']['mixlor']['par']
        for tag, extra in (('rcmlor', {}), ('rcmlor2', {'sigma_p0_x': s0})):
            arm = dict(base)
            arm['arm'] = f'{det}_{tag}'
            arm['hyper'] = dict(base['hyper'], lor_frac_x=eta, lor_gamma_x=gam, **extra)
            arm['source'] = f'rcm + X pseudo-Voigt tail from footprint_test mixlor (eta {eta:.3f}, gamma {gam:.3f})'
            json.dump(arm, open(os.path.join(d, f'arm_{det}_{tag}.json'), 'w'), indent=1)
            print(det, tag, {k: round(v, 4) for k, v in arm['hyper'].items()
                             if k in ('sigma_p0', 'sigma_p0_x', 'lor_frac_x', 'lor_gamma_x')})


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""satfin_score.py HO_DIR [--fleet DIR] [--big DIR] -- score the make_satfin_dag.py held-out arms with clean_check
(each arm's angle scale/zero from all its events of the view; event-paired bootstrap; charge level, x-y log-charge
sigma, pairing power; clean / blob / all classes from the raw waveforms + M3).

  <name>: cand vs base (no rails, the production path), per dataset of make_satfin_dag.py
  rails_<det>: cand (no rails) vs the fleet's capt (same recipe + SAT_RAIL with the rails sidecar), same events:
               what SAT_RAIL adds to the recipe (needs --fleet, the satur/fleet dir; det3big also --big, satur/big/out)
  base_<det>: base (no rails) vs the fleet's rail arm (SAT_RAIL alone): what the rail does without the cap
Writes results/satfin_<name>.json via clean_check.main and prints its tables.
"""
import argparse, glob, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_check                                                       # noqa: E402
from make_satfin_dag import ROWS, E, OUT                                 # noqa: E402

CACHE = {**{d: f'{E}/t15/caches/{d}.pkl' for d in ('det2', 'det3', 'det4', 'det6', 'det7')},
         'det3big': f'{E}/satur/big/det3_8000.pkl', **{n: f'{OUT}/caches/{n}.pkl' for n in ROWS}}


def stage(pairs):
    """pairs: {arm name in clean_check: [npz files]} -> temp dir laid out as clean_check reads it (det3big_<arm>_c*)."""
    t = tempfile.mkdtemp(prefix='satfin_')
    for arm, fs in pairs.items():
        for i, f in enumerate(sorted(fs)):
            shutil.copy(f, f'{t}/det3big_{arm}_c{i}.npz')
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ho'); ap.add_argument('--fleet', default=None)
    ap.add_argument('--big', default=None); ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--cache-dir', default=None, help='local copies: <dir>/<name>.pkl in place of the EOS caches')
    a = ap.parse_args()
    for name, cache in CACHE.items():
        if a.cache_dir:
            cache = os.path.join(a.cache_dir, f'{name}.pkl')
        if a.only and name not in a.only:
            continue
        base = glob.glob(f'{a.ho}/{name}_base_c*.npz'); cand = glob.glob(f'{a.ho}/{name}_cand_c*.npz')
        if not (base and cand and os.path.exists(cache)):
            continue
        print(f'\n## {name}: cand vs base (no rails)')
        clean_check.main(cache, stage({'rail': base, 'cand': cand}), 'cand', tag=f'satfin_{name}')
        ref = (sorted(glob.glob(f'{a.fleet}/{name}_capt.npz')) if a.fleet and name != 'det3big' else
               sorted(glob.glob(f'{a.big}/det3big_cap1200t05q_c*.npz')) if a.big and name == 'det3big' else [])
        if ref:
            print(f'\n## {name}: cand (no rails) vs capt (rails), what SAT_RAIL adds to the recipe')
            clean_check.main(cache, stage({'rail': ref, 'cand': cand}), 'cand', tag=f'satfin_rails_{name}')
        rr = (sorted(glob.glob(f'{a.fleet}/{name}_rail.npz')) if a.fleet and name != 'det3big' else
              sorted(glob.glob(f'{a.big}/det3big_rail_c*.npz')) if a.big and name == 'det3big' else [])
        if rr:
            print(f'\n## {name}: base (no rails) vs rail arm (SAT_RAIL alone)')
            clean_check.main(cache, stage({'rail': rr, 'base': base}), 'base', tag=f'satfin_base_{name}')


if __name__ == '__main__':
    main()

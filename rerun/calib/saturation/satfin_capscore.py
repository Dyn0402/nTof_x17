#!/usr/bin/env python3
"""satfin_capscore.py HO_DIR CACHE_DIR [ARM ...] -- clean_check of extra make_satfin_dag arms against base on every
dataset that has them (default arms: caponly cand; for the det3 pieces: caponly striponly cand).
Writes results/satfin_cap_<name>.json (or satfin_pieces_<name>.json when striponly is among the arms)."""
import glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_check, satfin_score                                         # noqa: E402

NAMES = ('det2', 'det3', 'det3big', 'det4', 'det6', 'det7', 'det2_o22long', 'det4_o23', 'det6_longer', 'det7_longer',
         'det3_p2', 'det3_d700', 'det3_d500', 'det3_day625')


def main(ho, ca, arms):
    tag = 'pieces' if 'striponly' in arms else 'cap'
    for n in NAMES:
        if not all(glob.glob(f'{ho}/{n}_{a}_c*.npz') for a in arms) or not os.path.exists(f'{ca}/{n}.pkl'):
            continue
        print(f'\n## {n}')
        d = satfin_score.stage({'rail': glob.glob(f'{ho}/{n}_base_c*.npz'), **{a: glob.glob(f'{ho}/{n}_{a}_c*.npz') for a in arms}})
        clean_check.main(f'{ca}/{n}.pkl', d, *arms, tag=f'satfin_{tag}_{n}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ['caponly', 'cand'])

#!/usr/bin/env python3
"""profile_arms.py out_tag tag1 tag2 [...] [--prefix rmF_]

Per-event profile over arms that differ in one nuisance hyper (e.g. sigma_p0): for each event and view keep the arm
with the lowest chi2 (same samples, same dof), and write it as ~/x17/cosmic_bench/det4dig/<prefix><out_tag>.npz with
the per-event table only (plus 'choice', the index of the arm taken), so compare.py / resid_show.py read it.  No
reference information enters the choice.
"""
import argparse, os
import numpy as np

D = os.path.expanduser('~/x17/cosmic_bench/det4dig/')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('tags', nargs='+')
    ap.add_argument('--prefix', default='rmF_')
    a = ap.parse_args()
    T = [{(int(r[0]), int(r[1])): r for r in np.load(D + f'{a.prefix}{t}.npz')['per']} for t in a.tags]
    keys = sorted(set.intersection(*[set(t) for t in T]))
    rows, choice = [], []
    for k in keys:
        i = int(np.argmin([t[k][4] for t in T]))
        rows.append(T[i][k]); choice.append(i)
    choice = np.array(choice); per = np.array(rows)
    for vi, v in enumerate('xy'):
        c = choice[per[:, 1] == vi]
        print(v, 'arm shares', [round(float((c == i).mean()), 3) for i in range(len(T))])
    np.savez(D + f'{a.prefix}{a.out}.npz', per=per, choice=choice, arms=np.array(a.tags))


if __name__ == '__main__':
    main()

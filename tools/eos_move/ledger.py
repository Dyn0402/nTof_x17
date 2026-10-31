#!/usr/bin/env python3
"""ledger.py [--out F] -- the move ledger from eosmove.py's manifests (run on lxplus): one row per moved tree
(old path, new path, files, bytes, all verified by size + adler32), then every duplicate kept in the user area."""
import argparse
import glob
import os

STATE = os.path.expanduser('~/x17rerun/eos_move/state')
U, X = '/eos/user/d/dneff', '/eos/experiment/ntof/data/x17'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=f'{U}/MOVED_TO_NTOF_2026-10-31.md')
    a = p.parse_args()
    rows, dups, tot = [], [], [0, 0]
    for f in sorted(glob.glob(os.path.join(STATE, '*.tsv'))):
        src = '/' + os.path.basename(f)[:-4].replace('__', '/')
        R = [l.rstrip('\n').split('\t') for l in open(f)]
        ver = [r for r in R if r[4] == 'verified']
        bad = [r for r in R if r[4] not in ('verified', 'dup', 'link')]
        nb = sum(int(r[1]) for r in ver)
        new = os.readlink(src) if os.path.islink(src) else ''
        if not new:
            note = os.path.join(src, 'MOVED.txt')
            new = next((l.split(': ', 1)[1].strip() for l in open(note) if l.startswith('New:')), '') \
                if os.path.exists(note) else '(not moved)'
        rows.append((src, new, len(ver), nb, sum(1 for r in R if r[4] == 'link'), len(bad)))
        tot[0] += len(ver)
        tot[1] += nb
        dups += [(src, r[0], int(r[1]), r[2], r[5]) for r in R if r[4] == 'dup']
    L = ['# Moved from the user EOS area to the n_TOF experiment area, 2026-10-31', '',
         'Tool: nTof_x17 `tools/eos_move/eosmove.py` (xrdcp across instances, then size + adler32 of every file on both',
         'sides from server-side listings; only then the user copy is removed and a symlink left at the old path).',
         'The symlinks resolve through FUSE (`/eos/...`), not through `root://eosuser.cern.ch` (another instance):',
         'read moved data by its new path in any xrootd URL.', '',
         f'**Total: {tot[0]} files, {tot[1] / 1e9:.1f} GB, all verified.**', '',
         '| old path | new path | files | GB | links | not verified |', '|---|---|---|---|---|---|']
    L += [f'| `{s}` | `{n}` | {nf} | {nb / 1e9:.2f} | {nl} | {b} |' for s, n, nf, nb, nl, b in rows]
    L += ['', f'## Duplicates kept in the user area ({len(dups)} files, {sum(d[2] for d in dups) / 1e9:.1f} GB)', '',
          'Identical (same size and adler32) to a file already in the experiment area; not copied, not deleted.', '',
          '| user file | GB | adler32 | identical experiment-area file |', '|---|---|---|---|']
    L += [f'| `{s}/{r}` | {sz / 1e9:.2f} | {c} | `{m}` |' for s, r, sz, c, m in dups]
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(f'{a.out}: {len(rows)} trees, {tot[0]} files, {tot[1] / 1e9:.1f} GB; {len(dups)} duplicates')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""eosmove.py -- move a directory tree from the user EOS instance to the n_TOF experiment instance, verified.

    eosmove.py copy   SRC DST [--dedupe-root R] [-j N]   copy every file not yet verified at DST
    eosmove.py verify SRC DST                             size + adler32 of every file, both sides
    eosmove.py finish SRC DST                             only if verify is clean: remove SRC, leave a symlink
                                                          SRC -> DST and SRC.MOVED.txt beside it

SRC is a /eos/user/... path, DST a /eos/experiment/... path (both FUSE paths on lxplus).  Copies go through
xrootd (root://eosuser.cern.ch -> root://eosexperiment.cern.ch) with an end-to-end adler32 check against the
source; verify re-reads both server-side checksums and sizes independently.  Symlinks inside SRC are recreated
at DST.  The manifest (SRC relpath, size, adler32 src/dst, status) goes to <state>/<tag>.tsv.

--dedupe-root R: a file whose basename exists anywhere under R (an experiment-area tree) with the same size and
adler32 is a DUPLICATE: it is not copied, it is listed (status dup, with the matching path), and `finish` keeps
the user copy (the tree then stays a real directory: moved files become per-file symlinks).
"""
import argparse
import concurrent.futures as cf
import datetime
import os
import shutil
import subprocess
import sys

USER, EXP = 'root://eosuser.cern.ch', 'root://eosexperiment.cern.ch'
STATE = os.path.expanduser('~/x17rerun/eos_move/state')


def host(p):
    return USER if p.startswith('/eos/user/') else EXP


def cksum(p, tries=4):
    for _ in range(tries):
        r = subprocess.run(['xrdfs', host(p).split('//')[1], 'query', 'checksum', p], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.split():
            parts = r.stdout.split()
            return parts[1] if len(parts) > 1 else parts[0]
    return None


def xsize(p, tries=4):
    """size by an xrootd stat (the destination side is never read through FUSE: a dir made through FUSE and
    filled through xrootd shows stale '?' entries on the same node for minutes)"""
    for _ in range(tries):
        r = subprocess.run(['xrdfs', host(p).split('//')[1], 'stat', p], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith('Size:'):
                    return int(line.split()[1])
        elif 'No such file' in r.stderr or '[3011]' in r.stderr:
            return None
    return None


def xmkdir(p):
    subprocess.run(['xrdfs', host(p).split('//')[1], 'mkdir', '-p', p], capture_output=True, text=True)


def eos_ln(target, link):
    env = dict(os.environ, EOS_MGM_URL=host(link))
    subprocess.run(['eos', '-b', 'file', 'symlink', '-f', link, target], capture_output=True, text=True, env=env)


def listing(root):
    """{relpath: (size, adler32)} of every file under root, from one server-side `eos find` (fast, no FUSE)"""
    env = dict(os.environ, EOS_MGM_URL=host(root))
    for _ in range(4):
        r = subprocess.run(['eos', '-b', 'find', '-f', '--checksum', '--size', root], capture_output=True, text=True,
                           env=env)
        if r.returncode == 0:
            break
    out = {}
    for line in r.stdout.splitlines():
        if ' size=' not in line:
            continue
        path, rest = line.split(' size=', 1)
        if '/.sys.' in path or ' -> ' in path:        # EOS-internal old versions (.sys.v#.<name>/): not data;
            continue                                  # symlinks are listed as "name -> target": links_of() has them
        kv = dict(x.split('=', 1) for x in ('size=' + rest).split() if '=' in x)
        out[os.path.relpath(path, root)] = (int(kv.get('size', -1)), kv.get('checksum', ''))
    return out


def links_of(src):
    r = subprocess.run(['find', src, '-type', 'l'], capture_output=True, text=True)
    return [(os.path.relpath(p, src), os.readlink(p)) for p in r.stdout.split('\n') if p]


def tag(src):
    return src.strip('/').replace('/', '__')


def walk(src):
    files, links, dirs = [], [], []
    for d, ds, fs in os.walk(src):
        rd = os.path.relpath(d, src)
        dirs.append(rd)
        for n in ds + fs:
            p = os.path.join(d, n)
            if os.path.islink(p):
                links.append((os.path.relpath(p, src), os.readlink(p)))
        for n in fs:
            p = os.path.join(d, n)
            if not os.path.islink(p):
                files.append((os.path.relpath(p, src), os.path.getsize(p)))
        ds[:] = [x for x in ds if not os.path.islink(os.path.join(d, x))]
    return files, links, dirs


def load(src):
    m = {}
    f = os.path.join(STATE, tag(src) + '.tsv')
    if os.path.exists(f):
        for line in open(f):
            r = line.rstrip('\n').split('\t')
            m[r[0]] = r
    return m


def save(src, m):
    os.makedirs(STATE, exist_ok=True)
    f = os.path.join(STATE, tag(src) + '.tsv')
    with open(f + '.tmp', 'w') as o:
        for k in sorted(m):
            o.write('\t'.join(m[k]) + '\n')
    os.replace(f + '.tmp', f)


def dedupe_index(root):
    idx = {}
    for d, _, fs in os.walk(root):
        for n in fs:
            p = os.path.join(d, n)
            try:
                idx.setdefault(n, []).append((p, os.path.getsize(p)))
            except OSError:
                pass
    return idx


def copy_one(src, dst, rel, size):
    s, d = os.path.join(src, rel), os.path.join(dst, rel)
    if xsize(d) == size and cksum(s) == cksum(d):
        return rel, 'ok', ''
    for _ in range(4):
        r = subprocess.run(['xrdcp', '--nopbar', '--force', '--path', '--cksum', 'adler32:source',
                            USER + '/' + s, EXP + '/' + d], capture_output=True, text=True)
        if r.returncode == 0:
            return rel, 'ok', ''
    return rel, 'fail', r.stderr.strip().replace('\t', ' ').replace('\n', ' ')[-300:]


#: 2026-10-31 incident: moving x17rerun/m3_audit (12 symlinks into cosmic_bench/june_tests) deleted the link
#: TARGETS (the golden rows' M3 v2 rays) and left a loop; they were restored from the n_TOF recycle bin
#: (restore_v2_rays.sh).  Until the link path is understood and tested, a tree with any symlink is refused.
def refuse_links(src, links, a):
    if links and not getattr(a, 'allow_links', False):
        sys.exit(f'{src}: {len(links)} symlink(s) inside; refusing (see the 2026-10-31 incident in README.md). '
                 'Move the real files, recreate the links by hand, or pass --allow-links after testing on a copy.\n'
                 + ''.join(f'  {r} -> {t}\n' for r, t in links[:20]))


def cmd_copy(a):
    files, links, dirs = walk(a.src)
    refuse_links(a.src, links, a)
    m = load(a.src)
    idx = dedupe_index(a.dedupe_root) if a.dedupe_root else {}
    todo = []
    for rel, size in files:
        base = os.path.basename(rel)
        if idx and base in idx:
            cs = cksum(os.path.join(a.src, rel))
            hit = [p for p, sz in idx[base] if sz == size and cksum(p) == cs]
            if hit:
                m[rel] = [rel, str(size), cs or '', cs or '', 'dup', hit[0]]
                continue
        if rel in m and m[rel][4] == 'verified':
            continue
        todo.append((rel, size))
    full = set()
    for rel, _ in files:
        d = os.path.dirname(rel)
        while d and d not in full:
            full.add(d)
            d = os.path.dirname(d)
    for rd in dirs:                                   # xrdcp makes every parent; only empty dirs need a mkdir
        if rd != '.' and rd not in full:
            xmkdir(os.path.normpath(os.path.join(a.dst, rd)))
    if len(todo) > 300 and os.path.basename(a.src) == os.path.basename(a.dst):
        # many small files: one recursive xrdcp (per-file process spawning runs at ~2 files/s on lxplus), then
        # only what the listings say is missing or different goes through the per-file path
        subprocess.run(['xrdcp', '--nopbar', '--force', '-r', '--parallel', str(a.j), '--cksum', 'adler32:source',
                        USER + '/' + a.src, EXP + '/' + os.path.dirname(a.dst) + '/'], capture_output=True, text=True)
        S, D = listing(a.src), listing(a.dst)
        todo = [(rel, size) for rel, size in todo if D.get(rel) != S.get(rel)]
    with cf.ThreadPoolExecutor(a.j) as ex:
        for i, (rel, st, err) in enumerate(ex.map(lambda t: copy_one(a.src, a.dst, *t), todo)):
            size = dict(files)[rel]
            m[rel] = [rel, str(size), '', '', 'copied' if st == 'ok' else 'FAIL', err]
            if i % 200 == 0:
                save(a.src, m)
    for rel, target in links:
        eos_ln(target, os.path.join(a.dst, rel))
        m[rel] = [rel, '0', '', '', 'link', target]
    save(a.src, m)
    print(f'{a.src}: {len(files)} files, {len(todo)} copied, {sum(1 for r in m.values() if r[4] == "dup")} dup, '
          f'{sum(1 for r in m.values() if r[4] == "FAIL")} FAIL, {len(links)} links')


def cmd_verify(a):
    S, D = listing(a.src), listing(a.dst)
    links = links_of(a.src)
    lk = set(rel for rel, _ in links)
    m = load(a.src)
    bad = 0
    for rel, (size, cs) in S.items():
        if rel in lk:
            continue
        r = m.get(rel)
        if r and r[4] == 'dup':
            continue
        d = D.get(rel)
        if d is None:
            m[rel] = [rel, str(size), cs, '', 'MISSING', '']
        else:
            ok = d[0] == size and bool(cs) and d[1] == cs
            m[rel] = [rel, str(size), cs, d[1], 'verified' if ok else 'MISMATCH', f'dst size {d[0]}']
        bad += m[rel][4] != 'verified'
    for rel in [k for k in m if k not in S and k not in lk]:
        del m[rel]                                    # stale manifest rows (no longer a file in SRC)
    for rel, target in links:
        env = dict(os.environ, EOS_MGM_URL=EXP)
        r = subprocess.run(['eos', '-b', 'stat', os.path.join(a.dst, rel)], capture_output=True, text=True, env=env)
        ok = r.returncode == 0
        m[rel] = [rel, '0', '', '', 'link' if ok else 'LINKMISSING', target]
        bad += not ok
    save(a.src, m)
    nb = sum(int(r[1]) for r in m.values() if r[4] == 'verified')
    print(f'{a.src}: verified {sum(1 for r in m.values() if r[4] == "verified")} files {nb / 1e9:.2f} GB, '
          f'dup {sum(1 for r in m.values() if r[4] == "dup")}, links {len(links)}, BAD {bad}')
    return bad


def cmd_finish(a):
    if cmd_verify(a):
        sys.exit('verify not clean: nothing removed')
    m = load(a.src)
    S, links = listing(a.src), links_of(a.src)
    refuse_links(a.src, links, a)
    if len(set(S) | set(r for r, _ in links)) != len([r for r in m.values() if r[4] in ('verified', 'dup', 'link')]):
        sys.exit('manifest does not cover the tree: nothing removed')
    dups = [r for r in m.values() if r[4] == 'dup']
    now = datetime.datetime.now().isoformat(timespec='seconds')
    nb = sum(int(r[1]) for r in m.values() if r[4] == 'verified')
    note = (f'Moved {now} by eosmove.py (nTof_x17 tools/eos_move).\nOld: {a.src}\nNew: {a.dst}\n'
            f'Files verified (size + adler32): {sum(1 for r in m.values() if r[4] == "verified")}, {nb} bytes.\n'
            f'Manifest: {os.path.join(STATE, tag(a.src) + ".tsv")}\n')
    if not dups:
        env = dict(os.environ, EOS_MGM_URL=USER)
        subprocess.run(['eos', '-b', 'rm', '-r', a.src], capture_output=True, text=True, env=env)
        if os.path.lexists(a.src):
            shutil.rmtree(a.src)                      # server-side rm (to the recycle bin) failed: FUSE fallback
        os.symlink(a.dst, a.src)
        open(a.src.rstrip('/') + '.MOVED.txt', 'w').write(note)
    else:
        # keep the duplicates as real files; every moved file becomes a per-file symlink to its new copy
        for r in m.values():
            if r[4] == 'verified':
                s = os.path.join(a.src, r[0])
                os.remove(s)
                os.symlink(os.path.join(a.dst, r[0]), s)
        note += f'Duplicates kept here (already in the experiment area, same adler32): {len(dups)}\n' + \
            ''.join(f'  {r[0]}  ==  {r[5]}\n' for r in sorted(dups))
        open(os.path.join(a.src, 'MOVED.txt'), 'w').write(note)
    print(f'finished {a.src} -> {a.dst} ({nb / 1e9:.2f} GB, {len(dups)} duplicates kept)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('copy', 'verify', 'finish'))
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--dedupe-root')
    p.add_argument('--allow-links', action='store_true')
    p.add_argument('-j', type=int, default=16)
    a = p.parse_args()
    a.src, a.dst = a.src.rstrip('/'), a.dst.rstrip('/')
    assert a.src.startswith('/eos/user/d/dneff/') and not a.src.startswith('/eos/user/d/dneff/storage')
    assert a.dst.startswith('/eos/experiment/ntof/data/x17/')
    {'copy': cmd_copy, 'verify': cmd_verify, 'finish': cmd_finish}[a.cmd](a)

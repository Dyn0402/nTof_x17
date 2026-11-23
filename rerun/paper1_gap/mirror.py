#!/usr/bin/env python3
"""mirror.py <local_root> -- copy what the det3 gap map needs from EOS into <local_root>/<same absolute EOS path>, over
xrootd (root://eospublic.cern.ch, adler32 checked), so rerun/inputs.py can plan and stage from the mirror unchanged.

Rows: the two det3 long runs at the operating point (sat_det3 = Run 1, 6-27; the 6-28 p2 overnight = Run 2).
Copied: run_config.json, m3_tracking_root_v2, decoded_root_fa80c06 (FEU 7/8 only), the decode audit of FEU 7/8, the
pass-1 alignment and calib_bundle of each row, and S4's cached head-on / inclined pickles of sat_det3.
Needs the XRootD bindings (~/.venvs/xrd/bin/python).
"""
import os
import sys

from XRootD import client

URL = 'root://eospublic.cern.ch'
CB = '/eos/experiment/ntof/data/x17/cosmic_bench'
ROWS = [('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V'),
        ('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check')]
FEUS = ('_07.', '_08.')
fs = client.FileSystem(URL)


def ls(p):
    st, l = fs.dirlist(p, client.flags.DirListFlags.STAT)
    if not st.ok:
        return []
    return [(e.name, e.statinfo.size, bool(e.statinfo.flags & client.flags.StatInfoFlags.IS_DIR)) for e in l]


def tree(p):
    for n, s, d in ls(p):
        if d:
            yield from tree(f'{p}/{n}')
        else:
            yield f'{p}/{n}', s


def main():
    root = sys.argv[1]
    want = []
    for run, sub in ROWS:
        rd = f'{CB}/june_tests/{run}'
        want.append((f'{rd}/run_config.json', None))
        want += [(f'{rd}/{sub}/m3_tracking_root_v2/{n}', s) for n, s, d in ls(f'{rd}/{sub}/m3_tracking_root_v2') if not d]
        want += [(f'{rd}/{sub}/decoded_root_fa80c06/{n}', s) for n, s, d in ls(f'{rd}/{sub}/decoded_root_fa80c06')
                 if n.endswith('.root') and any(f in n for f in FEUS)]
        ad = f'{CB}/x17rerun/decode_audit/{run}/{sub}'
        want += [(f'{ad}/{n}', s) for n, s, d in ls(ad) if n.endswith('.npz') and any(f in n for f in FEUS)]
        p1 = f'{CB}/Analysis_pass1/{run}/{sub}/mx17_3/wft'
        want += list(tree(f'{p1}/alignment')) + list(tree(f'{p1}/calib_bundle'))
    s4 = f'{CB}/x17rerun/s4_inputs/Analysis/{ROWS[0][0]}/{ROWS[0][1]}/mx17_3/s4'
    want += list(tree(s4))
    cp = client.CopyProcess()
    n = 0; tot = 0
    for src, size in want:
        dst = root + src
        if size is not None and os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cp.add_job(URL + '/' + src, dst, force=True, checksummode='end', checksumtype='adler32')
        n += 1; tot += size or 0
    print(f'{n} files, {tot / 1e9:.1f} GB to copy', flush=True)
    cp.prepare()
    st, res = cp.run()
    bad = [r for r in res if not r['status'].ok]
    print('done', st, 'failed', len(bad))
    for r in bad:
        print(r['status'].message)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

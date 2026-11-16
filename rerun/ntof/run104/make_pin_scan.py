#!/usr/bin/env python3
"""make_pin_scan.py -- D beam t0-pin scan: bundles and condor job list (2026-11-16).

The run_104 study (README.md) found that the L1 pin fits D's beam data ~1.5 % better than the L2 pin, on L1-era and
L2-era acquisitions alike, so the best D beam pin was never measured.  This scans it along the L2 -> L1 line,

    pin(f) = L2 + f (L1 - L2),   L2 = (x 18.3, y 20.0) ns, L1 = (x -17.8, y -11.2) ns  (t0_abs at ftst 0)

f = 0 is the main pass's L2-era pin, f = 1 its L1-era pin.  The scan bundles are the frozen D_E2_beam bundle with only
`t0_abs` (C - 10 ns * ftst, ftst 0..7) and `provenance.ntof_bundle.t0` changed; arrays.npz is copied byte for byte.
They are test bundles, written outside the frozen `Analysis_pass1/bundles/`.

Acquisitions: the 24 L2-era control acquisitions (runs 106, 108; run104/README.md) and 12 L1-era ones (runs 100, 102,
evenly spaced).  Existing outputs supply f = 0 and f = 1 for runs 106/108 (main pass, control 4673558) and f = 1 for
runs 100/102 (main pass); the jobs fill the rest.

    make_pin_scan.py --frozen DIR_WITH_D_E2_beam --main pass_main.txt.gz --control s11_d104_control_L1.txt --out DIR

writes DIR/bundles/D_E2_beam_f<NNN>/ and DIR/jobs.txt (ntof.sub format).  Upload the bundles to EOS_BASE/bundles/.
"""
import argparse, gzip, json, os, shutil

L1 = {'x': -17.8, 'y': -11.2}
L2 = {'x': 18.3, 'y': 20.0}
FS = (0.5, 1.5, 2.0, 3.0)
EOS_BASE = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/d_pin_scan'


def pin(f):
    return {p: round(L2[p] + f * (L1[p] - L2[p]), 2) for p in 'xy'}


def tag(f):
    return f'f{int(round(f * 100)):03d}'


def make_bundle(src, out, f):
    os.makedirs(out, exist_ok=True)
    shutil.copy2(f'{src}/arrays.npz', f'{out}/arrays.npz')
    b = json.load(open(f'{src}/bundle.json'))
    C = pin(f)
    b['t0_abs'] = {p: {str(k): round(C[p] - 10.0 * k, 4) for k in range(8)} for p in 'xy'}
    b['provenance']['ntof_bundle']['t0'] = dict(source=f'make_pin_scan.py f={f} (L2 + f (L1 - L2))', C=C,
                                                sigma=b['provenance']['ntof_bundle']['t0']['sigma'])
    json.dump(b, open(f'{out}/bundle.json', 'w'), indent=1)
    return C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frozen', required=True, help='directory holding the frozen D_E2_beam/')
    ap.add_argument('--main', required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src = f'{a.frozen}/D_E2_beam'
    ref = json.load(open(f'{src}/bundle.json'))
    assert ref['t0_abs']['x']['0'] == L2['x'] and ref['t0_abs']['y']['0'] == L2['y'], 'not the L2 bundle'
    op = gzip.open if a.main.endswith('.gz') else open
    main_rows = [l.split() for l in op(a.main, 'rt') if l.strip()]
    l1era = [r for r in main_rows if r[3] == 'D' and r[0] in ('100', '102')]
    pick = []
    for run in ('100', '102'):
        rr = [r for r in l1era if r[0] == run]
        n = 6
        pick += [rr[int(i * (len(rr) - 1) / (n - 1))] for i in range(n)]
    ctl = [l.split() for l in open(a.control) if l.strip()]
    acqs = [r[:4] for r in ctl] + [r[:4] for r in pick]
    jobs, meta = [], {}
    for f in FS:
        t = tag(f)
        meta[t] = make_bundle(src, f'{a.out}/bundles/D_E2_beam_{t}', f)
        for run, sub, tg, arm in acqs:
            jobs.append(f'{run} {sub} {tg} {arm} {EOS_BASE}/bundles/D_E2_beam_{t} {EOS_BASE}/{t}/run_{run}/{sub}/D')
    t = tag(0.0)   # L2 pin on the L1-era acquisitions: the frozen L2 bundle itself
    meta[t] = dict(L2)
    for run, sub, tg, arm in [r[:4] for r in pick]:
        jobs.append(f'{run} {sub} {tg} {arm} /eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/bundles/D_E2_beam '
                    f'{EOS_BASE}/{t}/run_{run}/{sub}/D')
    open(f'{a.out}/jobs.txt', 'w').write('\n'.join(jobs) + '\n')
    json.dump(dict(L1=L1, L2=L2, pins=meta, acquisitions=[' '.join(r) for r in acqs]),
              open(f'{a.out}/scan.json', 'w'), indent=1)
    print(f'{len(jobs)} jobs, {len(acqs)} acquisitions; pins', meta)


if __name__ == '__main__':
    main()

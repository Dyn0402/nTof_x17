#!/usr/bin/env python3
"""Audit june_tests archive for raw-vs-processed gaps.

For every <run>/<subrun> with raw_daq_data, group datrun fdfs by acquisition
(basename up to _NNN_FF), and per FEU compare against decoded_root,
hits_root, combined_hits_root (per file-number), and m3_tracking_root(_v2)
rays files (FEU 01 = M3).  Print one line per gap.
"""
import os
import re
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    '/afs/cern.ch/user/d/dneff/x17/cosmic_bench/june_tests'

PAT = re.compile(r'^(.*_datrun_\d{6}_\d{2}H\d{2})_(\d{3})_(\d{2})\.fdf$')


def listdir(p):
    try:
        return os.listdir(p)
    except OSError:
        return []


def scan_subrun(run, subrun, sdir):
    raw = os.path.join(sdir, 'raw_daq_data')
    fdfs = {}
    for f in listdir(raw):
        m = PAT.match(f)
        if m:
            acq, fnum, feu = m.group(1), m.group(2), m.group(3)
            fdfs.setdefault(acq, set()).add((fnum, feu))
    if not fdfs:
        return

    dec = set()
    for f in listdir(os.path.join(sdir, 'decoded_root')):
        m = re.match(r'^(.*_datrun_\d{6}_\d{2}H\d{2})_(\d{3})_(\d{2})\.root$', f)
        if m:
            dec.add((m.group(1), m.group(2), m.group(3)))
    hits = set()
    for f in listdir(os.path.join(sdir, 'hits_root')):
        m = re.match(r'^(.*_datrun_\d{6}_\d{2}H\d{2})_(\d{3})_(\d{2})_hits\.root$', f)
        if m:
            hits.add((m.group(1), m.group(2), m.group(3)))
    comb = set()
    for f in listdir(os.path.join(sdir, 'combined_hits_root')):
        m = re.match(r'^(.*_datrun_\d{6}_\d{2}H\d{2})_(\d{3})_feu-combined_hits\.root$', f)
        if m:
            comb.add((m.group(1), m.group(2)))
    rays = {}
    for ver in ('m3_tracking_root', 'm3_tracking_root_v2'):
        rays[ver] = set()
        for f in listdir(os.path.join(sdir, ver)):
            m = re.match(r'^(.*_datrun_\d{6}_\d{2}H\d{2})_(\d{3})_rays\.root$', f)
            if m:
                rays[ver].add((m.group(1), m.group(2)))
    has_v1 = os.path.isdir(os.path.join(sdir, 'm3_tracking_root'))
    has_v2 = os.path.isdir(os.path.join(sdir, 'm3_tracking_root_v2'))

    for acq in sorted(fdfs):
        pairs = fdfs[acq]
        feus = sorted({p[1] for p in pairs})
        fnums = sorted({p[0] for p in pairs})
        det_feus = [f for f in feus if f != '01']
        # which detector FEUs were ever decoded (if none, whole subrun undecoded)
        for feu in det_feus:
            missing_dec = sorted(fn for fn, fe in pairs
                                 if fe == feu and (acq, fn, feu) not in dec)
            if missing_dec:
                tot = sorted(fn for fn, fe in pairs if fe == feu)
                print(f'GAP decode  {run}/{subrun} acq={acq.split("_datrun_")[1]} '
                      f'feu={feu} missing={",".join(missing_dec)} of {len(tot)} files')
        if hits:
            for feu in det_feus:
                missing_h = sorted(fn for fn, fe in pairs
                                   if fe == feu and (acq, fn, feu) in dec
                                   and (acq, fn, feu) not in hits)
                if missing_h:
                    print(f'GAP hits    {run}/{subrun} acq={acq.split("_datrun_")[1]} '
                          f'feu={feu} missing={",".join(missing_h)}')
        if comb:
            missing_c = sorted(fn for fn in fnums if (acq, fn) not in comb)
            if missing_c:
                print(f'GAP combine {run}/{subrun} acq={acq.split("_datrun_")[1]} '
                      f'missing={",".join(missing_c)} of {len(fnums)} files')
        # partial combine: file present but one FEU's hits were missing at combine time
        if '01' in feus:
            for ver, present in rays.items():
                if ver == 'm3_tracking_root' and not has_v1:
                    continue
                if ver == 'm3_tracking_root_v2' and not has_v2:
                    continue
                missing_r = sorted(fn for fn in fnums if (acq, fn) not in present)
                if missing_r:
                    print(f'GAP m3-{"v2" if ver.endswith("v2") else "v1"}   '
                          f'{run}/{subrun} acq={acq.split("_datrun_")[1]} '
                          f'missing={",".join(missing_r)} of {len(fnums)} files')
        if not comb and not hits and all((acq, fn, feu) not in dec
                                         for fn, feu in pairs if feu != '01'):
            print(f'RAWONLY     {run}/{subrun} acq={acq.split("_datrun_")[1]} '
                  f'feus={",".join(feus)} files={len(fnums)}')


def main():
    for run in sorted(listdir(BASE)):
        rdir = os.path.join(BASE, run)
        if not os.path.isdir(rdir) or run in ('analysis', 'Analysis', 'pedestals',
                                              'test', '_m3_v2_condor'):
            continue
        for subrun in sorted(listdir(rdir)):
            sdir = os.path.join(rdir, subrun)
            if os.path.isdir(sdir):
                scan_subrun(run, subrun, sdir)
    print('AUDIT DONE')


if __name__ == '__main__':
    main()

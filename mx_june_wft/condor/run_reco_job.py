#!/usr/bin/env python3
"""
run_reco_job.py — one condor job = one campaign_manifest.csv row: fetch that
(subrun, detector)'s inputs from EOS, run the frozen wft reconstruction, leave
events(.candidates).parquet + meta in ./out/ for the wrapper to tar.

    python3 run_reco_job.py <row_index> [--bundle-name NAME] [--out-tag TAG]
                            [--manifest campaign_manifest.csv]
                            [--bundles bundles/] [--jobs N] [--vrefit]

Runs entirely inside the scratch dir it is started in. Expects, next to it:
    code/                unpacked repo (git archive of the freeze commit)
    campaign_manifest.csv
    bundles/<det>/<bundle_name>/     the frozen calibration bundles

--bundle-name overrides the manifest's bundle column (t0-prior gate arms).
--out-tag names the output subdir out/<key>[__TAG]/ so arms don't collide.
--vrefit (tier B): before reco, refit v_drift on this subrun with every kernel
hyper pinned to the frozen bundle (wft.calibrate --fix-hyper), then reco with
the refitted bundle. The refit bundle ships in the output for the record.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, 'code')
DATA = os.path.join(HERE, 'data')

# Hypers that stay pinned during a tier-B v-refit: everything the bench
# campaign froze (KERNEL_ARMS_2026-08-12.md) — only v (and w0 via the refit's
# own accounting) may move with drift field.
PINNED_HYPERS = ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')


def sh(cmd, **kw):
    print('[job]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch(eos_path, dest):
    """Copy a file or directory from EOS: xrdcp, falling back to fuse cp."""
    os.makedirs(os.path.dirname(dest.rstrip('/')), exist_ok=True)
    url = os.environ.get('EOS_URL', 'root://eospublic.cern.ch/')
    if os.path.isfile(eos_path):                   # fuse mount / local path
        shutil.copy2(eos_path, dest)
        return
    if os.path.isdir(eos_path):
        shutil.copytree(eos_path, dest, dirs_exist_ok=True)
        return
    try:
        sh(['xrdcp', '-r', '-N', url + eos_path, dest])
    except Exception:
        sh(['cp', '-r', eos_path, dest])


def _ls(eos_dir):
    url = os.environ.get('EOS_URL', 'root://eospublic.cern.ch/')
    if os.path.isdir(eos_dir):
        return os.listdir(eos_dir)
    out = subprocess.run(['xrdfs', url, 'ls', eos_dir], check=True,
                         capture_output=True, text=True).stdout
    return [os.path.basename(l.strip()) for l in out.splitlines() if l.strip()]


def _datrun_stem(name):
    """'..._datrun_260628_01H34_003_07.root' -> '260628_01H34' (acquisition)."""
    import re
    m = re.search(r'_datrun_(\d+_\d+H\d+)_', name)
    return m.group(1) if m else None


def fetch_decoded(eos_sub, dest, feus, m3_dir):
    """Only the detector's FEU files from decoded_root (…_NN.root), and only
    from acquisitions the M3 tracking dir covers at top level. A false-start
    acquisition restarts event ids at 0; its decoded events collide with the
    main run and get cross-matched against the wrong segment's rays (this put
    620 duplicate-id rows in g_det3_wknd's campaign table, 2026-08-13)."""
    os.makedirs(dest, exist_ok=True)
    src = eos_sub + '/decoded_root'
    names = _ls(src)
    m3_stems = {s for s in (_datrun_stem(n) for n in _ls(m3_dir)
                            if n.endswith('.root')) if s}
    pats = tuple(f'_{f:02d}.root' for f in feus)
    picked = [n for n in names if n.endswith(pats)]
    if m3_stems:
        skipped = [n for n in picked if _datrun_stem(n) not in m3_stems]
        if skipped:
            print(f'[job] skipping {len(skipped)} decoded files from '
                  f'acquisitions absent in the M3 dir: '
                  f'{sorted({_datrun_stem(n) for n in skipped})}', flush=True)
        picked = [n for n in picked if _datrun_stem(n) in m3_stems]
    if not picked:
        sys.exit(f'FATAL: no decoded files matching FEUs {feus} in {src}')
    for n in picked:
        fetch(f'{src}/{n}', os.path.join(dest, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('row', type=int)
    ap.add_argument('--manifest', default=os.path.join(HERE,
                                                       'campaign_manifest.csv'))
    ap.add_argument('--bundles', default=os.path.join(HERE, 'bundles'))
    ap.add_argument('--bundle-name', default=None)
    ap.add_argument('--out-tag', default=None)
    ap.add_argument('--jobs', type=int,
                    default=int(os.environ.get('RECO_JOBS', '8')))
    ap.add_argument('--vrefit', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help='smoke tests only — never for production rows')
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    row = rows[args.row]
    if row['tier'] == 'C' and not row['reason'].startswith('resist'):
        sys.exit(f'FATAL: row {args.row} is tier C ({row["reason"]}) — '
                 'not runnable')

    run, subrun, det = row['run'], row['subrun'], row['det']
    feux, feuy = int(row['feu_x']), int(row['feu_y'])
    eos_run = row['eos_run_dir']
    if not eos_run:
        sys.exit(f'FATAL: row {args.row} has no eos_run_dir')
    eos_sub = f'{eos_run}/{subrun}'

    # ---- stage inputs into the local bench layout _Config expects
    tree = os.path.join(DATA, row['tree'])
    sub_local = os.path.join(tree, run, subrun)
    fetch(f'{eos_run}/run_config.json',
          os.path.join(tree, run, 'run_config.json'))
    m3 = 'm3_tracking_root_v2' if row['has_m3v2'] == '1' else 'm3_tracking_root'
    fetch(f'{eos_sub}/{m3}', os.path.join(sub_local, m3))
    fetch_decoded(eos_sub, os.path.join(sub_local, 'decoded_root'),
                  [feux, feuy], m3_dir=f'{eos_sub}/{m3}')
    fetch(f'{eos_sub}/combined_hits_root',
          os.path.join(sub_local, 'combined_hits_root'))

    # ---- frozen code on the path, config synthesized from the row
    sys.path[:0] = [CODE, os.path.join(CODE, 'mx_june_cosmic_qa'),
                    os.path.join(CODE, 'cosmic_bench_analysis')]
    import qa_config
    from qa_config import _Config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    cfg = _Config(row['key'], run, subrun, feus=[feux, feuy],
                  det_z=float(row['det_z']), det_name=det,
                  zero_suppressed=False, base_path=tree + '/')
    qa_config.RUNS[row['key']] = cfg      # calibrate/cli resolve keys here

    bundle_name = args.bundle_name or row['bundle']
    bundle = os.path.join(args.bundles, det, bundle_name)
    if not os.path.isdir(bundle):
        sys.exit(f'FATAL: bundle {bundle} not staged')

    tag = f'__{args.out_tag}' if args.out_tag else ''
    out_dir = os.path.join(HERE, 'out', f'{row["key"]}{tag}')
    os.makedirs(out_dir, exist_ok=True)

    if args.vrefit:
        import runpy
        refit_out = os.path.join(out_dir, f'{bundle_name}_vrefit')
        b = json.load(open(os.path.join(bundle, 'bundle.json')))
        pins = ','.join(f'{h}={b["hyper"][h]}' for h in PINNED_HYPERS
                        if h in b.get('hyper', {}))
        argv = sys.argv
        # No --share-mode: the refit bundle must keep the campaign's kernel
        # branch. Production bundles carry share_mode null -> the loader falls
        # back to 'delay' (FREEZE_MPGD26 §2), and calibrate's default writes
        # 'delay'. Passing 'lp' here would have run these 7 tier-B rows on a
        # different kernel than the other 207.
        sys.argv = ['wft.calibrate', row['key'], '--seed-bundle', bundle,
                    '--fix-hyper', pins,
                    '--jobs', str(args.jobs), '--out', refit_out]
        print('[job]', ' '.join(sys.argv), flush=True)
        runpy.run_module('wft.calibrate', run_name='__main__')
        sys.argv = argv
        bundle = refit_out

    # ---- reco (mirrors wft.cli cmd_reco on the synthesized config)
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    from M3RefTracking import M3RefTracking, get_xy_angles
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    _xa, _ya, evn = get_xy_angles(rays.ray_data)
    filt = set(int(e) for e in evn)
    m3_has_nclus = bool(getattr(rays, 'has_nclus', True))
    print(f'[job] {len(filt):,} M3-matched events '
          f'(NClus branches: {m3_has_nclus})', flush=True)
    if not m3_has_nclus:
        # chi2-only fallback = a DIFFERENT selection than every local
        # accounting (g_det3_wknd 2026-08-13: 36,745 vs 26,670 events, 5.3 %
        # of good rays silently absent). Refuse rather than mislabel.
        sys.exit('FATAL: M3 NClus branches unavailable in '
                 f'{cfg.m3_tracking_dir} on this stack — the recipe would '
                 'silently degrade to chi2-only. Reprocess the run with '
                 'm3_tracking_root_v2 or run on a stack that reads these '
                 'files (JUNE_CONTINUITY_2026-08-13.md §5b).')
    cal = CalibrationBundle.load(bundle)
    out = os.path.join(out_dir, 'events.parquet')
    reconstruct_run(cfg, cal, out, event_filter=filt, jobs=args.jobs,
                    limit=args.limit, bundle_path=bundle)

    with open(os.path.join(out_dir, 'job_row.json'), 'w') as f:
        json.dump(dict(row, bundle_used=bundle_name,
                       vrefit=bool(args.vrefit), out_tag=args.out_tag or '',
                       off_conditions=(row['tier'] == 'C'),
                       n_matched=len(filt), m3_has_nclus=m3_has_nclus),
                  f, indent=1)
    print('[job] done:', out_dir, flush=True)


if __name__ == '__main__':
    main()

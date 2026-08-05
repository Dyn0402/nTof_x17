#!/usr/bin/env python3
"""Decode det4 (FEU 3) raw fdf -> waveforms + hits, for a flat-mount dataset.

Every analyzer setting is derived from the dataset's own record in
`datasets.py`, never assumed:

  --tps           sample_ns
  --thr           zs_sigma      (the FEU's own ZS threshold, from _thr.prg)
  --mf            round(1.7 * peaking_ns / sample_ns), the shaped pulse width
  --zs-baseline 1 because the DAQ ran with on-FEU pedestal subtraction, so the
                  waveforms are re-centred at 256 and the pedestal file's
                  per-channel raw means are NOT the right baseline

  python decode_dataset.py run63_operating [--jobs 3] [--subrun operating_03]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import datasets

SOFT = "/home/dylan/CLionProjects/mm_strip_reconstruction/cmake-build-release/"
DECODE = SOFT + "decoder/decode"
ANALYZE = SOFT + "waveform_analysis/analyze_waveforms"


def run(cmd, fatal=True):
    print("  $", " ".join(os.path.basename(c) if "/" in c else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        if fatal:
            raise SystemExit(f"failed: {cmd[0]}")
        print(f"  !! failed (non-fatal): {' '.join(cmd[:2])}")
        return r
    # The decoder shouts about dropped FEU packets.  Capturing its output for
    # error handling must not silence that -- a RAW run can lose a quarter of
    # its sample-groups and still exit 0, and nothing downstream can tell that
    # from genuinely quiet channels.  Echo the banner and the summary lines.
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith(" !!") or "DATA LOSS" in line or \
                "sample completeness" in line or "events MISSING" in line:
            print("     " + line.strip())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--subrun", default="", help="only this sub-run")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--feus", default="03",
                    help="comma list of FEU codes to decode, e.g. 03,01. "
                         "FEU1 carries BOTH uRWELL planes (ch 0-255 front, "
                         "256-511 back)")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    stage = D["stage"]
    mf = max(1, round(1.7 * D["peaking_ns"] / D["sample_ns"]))
    print(f"{args.dataset}: {D['mount']}, {D['gas']}, ZS {D['zs_sigma']} sigma, "
          f"{D['n_samples']}x{D['sample_ns']} ns, mf {mf} samples")

    jobs = []
    for sub, stem, _t0, idxs in D["subruns"]:
        if args.subrun and sub != args.subrun:
            continue
        d = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) \
            else stage
        ped_fdf = os.path.join(d, D["ped"])
        if not os.path.exists(ped_fdf):
            ped_fdf = os.path.join(stage, D["ped"])
        ped_root = os.path.join(d, "ped_03.root")
        if not os.path.exists(ped_root):
            print(f"[{sub}] pedestals")
            run([DECODE, ped_fdf, ped_root])
        for i in idxs:
            for feu in args.feus.split(","):
                # the datrun_ timestamp in the stem is the SUB-RUN start, which
                # differs per sub-run -- resolve '*' stems by glob
                cands = []
                for dd in (d, os.path.join(d, "raw_daq_data")):
                    pat = os.path.join(dd, f"{stem}{i}_{feu}.fdf")
                    cands += glob.glob(pat) if "*" in pat else \
                        ([pat] if os.path.exists(pat) else [])
                if not cands:
                    print(f"  !! missing {stem}{i}_{feu}.fdf, skipped")
                    continue
                fdf = cands[0]
                if os.path.getsize(fdf) == 0:
                    print(f"  !! empty {os.path.basename(fdf)}, skipped")
                    continue
                pr = ped_root if feu == "03" else os.path.join(d, f"ped_{feu}.root")
                if feu != "03" and not os.path.exists(pr):
                    pf = os.path.join(d, D["ped"].replace("_03.fdf", f"_{feu}.fdf"))
                    print(f"[{sub}] pedestals FEU {feu}")
                    run([DECODE, pf, pr])
                jobs.append((sub, d, i, fdf, pr, feu))

    def do(j):
        sub, d, i, fdf, ped_root, feu = j
        dec = os.path.join(d, f"dec_{sub}_{i}_{feu}.root")
        hit = os.path.join(d, f"hits_{sub}_{i}_{feu}.root")
        if not os.path.exists(dec):
            # per-file failures (truncated/empty fdf) must not kill the batch
            r = run([DECODE, fdf, dec], fatal=False)
            if r.returncode != 0:
                if os.path.exists(dec):
                    os.remove(dec)
                return f"{sub}/{i}/FEU{feu} DECODE-FAILED"
        if not os.path.exists(hit):
            # RAW runs carry no zero suppression and had on-FEU pedestal AND
            # common-mode subtraction switched OFF, so the waveforms sit on the
            # raw per-channel baselines (--zs-baseline 0, i.e. use the pedestal
            # file's means) and the common mode must be removed in software
            # (--cns 1).  ZS runs are the opposite on both counts.
            raw = bool(D.get("raw"))
            run([ANALYZE, dec, hit, ped_root,
                 "--tps", str(D["sample_ns"]), "--thr", str(D["zs_sigma"]),
                 "--mf", str(mf),
                 "--cns", "1" if raw else "0",
                 "--zs-baseline", "0" if raw else "1"])
        return f"{sub}/{i}/FEU{feu}"

    print(f"{len(jobs)} file groups, {args.jobs} threads")
    with ThreadPoolExecutor(max_workers=args.jobs) as p:
        for r in p.map(do, jobs):
            print("  done", r)


if __name__ == "__main__":
    main()

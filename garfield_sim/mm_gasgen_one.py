#!/usr/bin/env python3
"""
mm_gasgen_one.py — generate ONE Magboltz gas table (one HTCondor job)
======================================================================
mm_generate_gas.py runs every (gas, pressure) combination in a local
multiprocessing pool. That is fine on a workstation but wrong for lxplus:
interactive nodes kill long CPU hogs, and one hung Magboltz blocks the pool.

This script does exactly one (gas, pressure) table per process, so each one is
an independent HTCondor job. The composition and the field grid come from
mm_config.py (single source of truth — transfer it with the job).

The .gas file is written into the CURRENT directory; HTCondor transfers it back
to gas_tables/ via transfer_output_remaps in mm_gasgen.sub.

Usage:
    python3 mm_gasgen_one.py --gas-label Ar_CO2_iC4H10_93_5_2 \
                             --pressure-label CERN_450m [--ncoll 10]
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Generate one Magboltz gas table")
    p.add_argument("--gas-label",      required=True,
                   help="Label from mm_config.GAS_MIXTURES")
    p.add_argument("--pressure-label", required=True,
                   help="Key from mm_config.PRESSURES, e.g. CERN_450m")
    p.add_argument("--ncoll",          type=int, default=cfg.MAGBOLTZ_NCOLL,
                   help=f"Magboltz collision sets x10^7 (default {cfg.MAGBOLTZ_NCOLL})")
    p.add_argument("--outdir",         default=".",
                   help="Where to write the .gas file (default: CWD)")
    p.add_argument("--emin", type=float, default=cfg.E_GRID_MIN_VCM,
                   help=f"Field grid minimum in V/cm (default {cfg.E_GRID_MIN_VCM})")
    p.add_argument("--emax", type=float, default=cfg.E_GRID_MAX_VCM,
                   help=f"Field grid maximum in V/cm (default {cfg.E_GRID_MAX_VCM})")
    p.add_argument("--npts", type=int,   default=cfg.E_GRID_NPTS,
                   help=f"Field grid points (default {cfg.E_GRID_NPTS}). Magboltz "
                        "cost is roughly npts x ncoll, so narrowing the grid to the "
                        "amplification range is the cheapest way to buy wall time. "
                        "Safe for gain work: the avalanche is simulated with "
                        "AvalancheMicroscopic, which tracks on the cross sections, "
                        "not on this table. Anything that needs drift parameters "
                        "at low field needs the full default grid.")
    p.add_argument("--out-label",      default=None,
                   help="Write as <out-label>_<pressure>.gas instead of "
                        "<gas-label>_<pressure>.gas. Use to run the same "
                        "composition at a second nColl without clobbering the "
                        "production table.")
    return p.parse_args()


def main():
    args = parse_args()

    gas_cfg = next((g for g in cfg.GAS_MIXTURES if g["label"] == args.gas_label), None)
    if gas_cfg is None:
        sys.exit(f"[gasgen] ERROR: gas label {args.gas_label!r} not in mm_config.GAS_MIXTURES")
    if args.pressure_label not in cfg.PRESSURES:
        sys.exit(f"[gasgen] ERROR: pressure {args.pressure_label!r} not in mm_config.PRESSURES")

    pressure_torr = cfg.PRESSURES[args.pressure_label]
    components    = gas_cfg["components"]
    out_label     = args.out_label or args.gas_label
    fname         = os.path.join(args.outdir,
                                 f"{out_label}_{args.pressure_label}.gas")

    print(f"[gasgen] host       : {os.uname().nodename}", flush=True)
    print(f"[gasgen] gas        : {args.gas_label}  {components}", flush=True)
    print(f"[gasgen] pressure   : {args.pressure_label}  {pressure_torr:.2f} Torr", flush=True)
    print(f"[gasgen] temperature: {cfg.TEMP_K} K", flush=True)
    print(f"[gasgen] nColl      : {args.ncoll} x 10^7", flush=True)
    print(f"[gasgen] E grid     : {args.emin:.0f}-{args.emax:.0f} V/cm, "
          f"{args.npts} log-spaced points", flush=True)
    print(f"[gasgen] output     : {fname}", flush=True)

    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield  # noqa: F401  (registers ROOT.Garfield)
    import ctypes

    gas = ROOT.Garfield.MediumMagboltz()
    flat = [x for pair in components for x in pair]
    gas.SetComposition(*flat)
    gas.SetTemperature(cfg.TEMP_K)
    gas.SetPressure(pressure_torr)
    gas.SetFieldGrid(args.emin, args.emax, args.npts, True)

    t0 = time.time()
    gas.GenerateGasTable(args.ncoll)
    elapsed = time.time() - t0
    print(f"[gasgen] Magboltz done in {elapsed/60:.1f} min", flush=True)

    gas.WriteGasFile(fname)
    print(f"[gasgen] wrote {fname} "
          f"({os.path.getsize(fname)/1e6:.1f} MB)", flush=True)

    # Sanity: Townsend and attachment coefficients across the amplification range.
    print(f"[gasgen] {'E (V/cm)':>10} {'alpha (1/cm)':>14} {'eta (1/cm)':>12}", flush=True)
    for v in (400., 450., 500., 550., 600.):
        e = v / cfg.GAP_CM
        alpha = ctypes.c_double(0.)
        eta   = ctypes.c_double(0.)
        gas.ElectronTownsend(0., 0., e, 0., 0., 0., alpha)
        gas.ElectronAttachment(0., 0., e, 0., 0., 0., eta)
        print(f"[gasgen] {e:10.0f} {alpha.value:14.1f} {eta.value:12.3f}"
              f"   (V_mesh = {v:.0f} V)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
T13 — run the wft waveform reconstruction over SIMULATED decoded_root.

The response simulation (MX17_Geant, `response/digitizer/run.py --decoded-out`)
writes files in the exact `decoded_root` schema, and `wft/io.py` already reads
them unmodified. What was missing for T13 is the other half: wft's
*reconstruction* needs **seeds**, and `wft.reco._load_hits` gets those from the
run's `combined_hits` — which the simulation does not produce.

So this script supplies the one missing input and changes nothing in wft:

  1. Build a hits table FROM THE SIMULATED WAVEFORMS, using wft's own
     `FeuReader` so the pedestal and CNS treatment is bit-for-bit what data
     gets, and write it as a `hits` tree with the schema `_load_hits` expects
     (`eventId, feu, channel, amplitude, significance`).
  2. Take the REAL det3 run config and override only the paths, so the strip
     map, FEU ids and detector name come from the actual detector rather than
     from a hand-built stub.
  3. Call `wft.reco.reconstruct_run` unmodified.

WHY THIS IS NOT CHEATING. Seeds answer "which events and which strips carry a
track" — a detection question, which is exactly what hits are allowed to answer
(`wft/seed.py`: "the one place hits are allowed in"). Nothing here uses the
simulation's truth: the seed channels are found in the simulated waveforms by an
amplitude threshold and no time crosses the boundary. Using `truth.parquet` to
seed would have been cheating and would have inflated efficiency; it is
deliberately not read.

⚠️ **THE COMPARISON IS AT THE WAVEFORM LEVEL — RUN THIS ON BOTH LEGS.**
(Dylan, 2026-08-09: raw waveform analysis only; the C++ waveform analyzer and
the hits chain stay out of the loop entirely.) The seed table this builds comes
from the raw 32 x 60 ns samples, NOT from the analyzer — but wft's normal data
path seeds from `combined_hits`, which IS analyzer output. Seeding the
simulation one way and the data the other would put a detection-efficiency
difference straight into the comparison and charge it to the physics.

So this script is deliberately **dual-use**: point `--decoded-dir` at the real
det3 `decoded_root` to reconstruct the DATA leg the same way. Both legs must be
seeded by this code, at the same `--sigma`, or the comparison is not like for
like. Nothing downstream of here ever reads `combined_hits`.

    python3 mx17_sim_wft/t13_reco.py \
        --decoded-dir ~/x17/response_sim/stageB_w2/w2_rho2M/default \
        --bundle <det3 calib_bundle> --out events.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Same path set-up as wft/cli.py: qa_config lives in mx_june_cosmic_qa/ and the
# M3/reference helpers in cosmic_bench_analysis/.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mx_june_cosmic_qa"))
sys.path.insert(0, os.path.join(REPO, "cosmic_bench_analysis"))

# The DAQ analyzer's per-channel detection threshold. wft applies its OWN
# per-plane relative floor (0.10 x the plane's strongest strip) on top, so this
# only has to be loose enough not to pre-empt it — 5 sigma is the production
# figure the chain already reproduces (~6 channels over 5 sigma per muon).
SIGMA_THRESHOLD = 5.0


def build_hits(decoded_dir, feus, out_root, sigma=SIGMA_THRESHOLD):
    """Hits from simulated waveforms, in the schema wft._load_hits expects."""
    import uproot
    from wft.io import FeuReader

    rows = []
    n_events_total = 0
    median_noise = {}
    for feu in feus:
        pat = os.path.join(decoded_dir, "decoded_root", f"*_{feu:02d}.root")
        files = sorted(glob.glob(pat))
        if not files:
            raise FileNotFoundError(f"no decoded file matching {pat}")
        for path in files:
            rdr = FeuReader(path)
            noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
            n_events_total = max(n_events_total, rdr.n_entries)
            median_noise[feu] = float(np.median(rdr.noise))
            for eid, _ftst, wfm in rdr.iter_events():
                # wfm is [512, n_sample], pedestal- and CNS-corrected.
                amp = wfm.max(axis=1)
                sig = amp / noise
                ch = np.nonzero(sig >= sigma)[0]
                if len(ch) == 0:
                    continue
                rows.append(pd.DataFrame({
                    "eventId": np.full(len(ch), eid, dtype=np.int64),
                    "feu": np.full(len(ch), feu, dtype=np.int32),
                    "channel": ch.astype(np.int32),
                    "amplitude": amp[ch].astype(np.float32),
                    "significance": sig[ch].astype(np.float32),
                }))
            print(f"  {os.path.basename(path)}: {rdr.n_entries} events, "
                  f"median noise {np.median(rdr.noise):.1f} ADC")

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["eventId", "feu", "channel", "amplitude", "significance"])
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    # mktree + extend, NOT `f["hits"] = {...}`: the dict form writes an
    # RNTuple in current uproot, and `wft.reco._load_hits` reads with
    # library='pd', which only works on a TTree (an RNTuple comes back as an
    # awkward array and dies in `global_index` with "no field named 'index'").
    with uproot.recreate(out_root) as f:
        cols = {c: df[c].to_numpy() for c in df.columns}
        f.mktree("hits", {c: v.dtype for c, v in cols.items()})
        f["hits"].extend(cols)

    # SEED RATE IS ITSELF THE FIRST WAVEFORM-LEVEL OBSERVABLE. Because both
    # legs are seeded by this identical code at the identical threshold,
    # seeding is common-mode and cancels — so any residual difference in these
    # numbers between sim and data is NOT a nuisance to absorb, it is a
    # physics/noise-model discrepancy, and the cheapest one available. Dumped
    # next to the table so the two legs can be diffed directly.
    stats = {"n_events_seeded": int(df["eventId"].nunique()) if len(df) else 0,
             "n_events_total": int(n_events_total),
             "n_hits": int(len(df)),
             "sigma_threshold": float(sigma),
             "median_noise_adc": {str(k): float(v)
                                  for k, v in median_noise.items()},
             "per_feu": {}}
    for feu in feus:
        g = df[df["feu"] == feu] if len(df) else df
        nev = int(g["eventId"].nunique()) if len(g) else 0
        stats["per_feu"][str(feu)] = {
            "n_hits": int(len(g)),
            "n_events_with_hits": nev,
            "hits_per_seeded_event": (len(g) / nev) if nev else 0.0,
            "median_amplitude_adc": float(g["amplitude"].median()) if len(g) else 0.0,
            "median_significance": float(g["significance"].median()) if len(g) else 0.0,
        }
    with open(os.path.splitext(out_root)[0] + "_seedstats.json", "w") as f:
        json.dump(stats, f, indent=1)

    n_ev = stats["n_events_seeded"]
    print(f"  -> {out_root}: {len(df):,} hits over {n_ev:,}/{n_events_total:,} "
          f"events ({len(df)/max(n_ev,1):.1f} per seeded event, "
          f"{100*n_ev/max(n_events_total,1):.1f} % of events seeded)")
    for feu in feus:
        s = stats["per_feu"][str(feu)]
        print(f"     FEU {feu}: {s['n_hits']:,} hits, "
              f"{s['hits_per_seeded_event']:.2f}/event, median amp "
              f"{s['median_amplitude_adc']:.1f} ADC, median sig "
              f"{s['median_significance']:.1f}")
    return df


class _Redirected:
    """The real run config, with a few paths pointed elsewhere.

    A delegating wrapper rather than a copy-and-assign, because the config
    exposes `combined_hits_dir` (and friends) as read-only properties derived
    from BASE_PATH/RUN/SUB_RUN. Delegation also means every attribute NOT
    overridden here — strip map, FEU ids, detector name, M3 recipe — is the
    genuine article, which is the point: the bundle is only valid within the
    conditions of the run it was built for.
    """

    def __init__(self, base, **overrides):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_over", overrides)

    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_base"), name)


def sim_cfg(decoded_dir, hits_dir, run_key):
    """The REAL det3 config with only the paths redirected at the simulation."""
    from qa_config import get_config, setup_paths
    setup_paths()
    return _Redirected(
        get_config(run_key),
        BASE_PATH=os.path.dirname(os.path.dirname(decoded_dir)) + os.sep,
        RUN=os.path.basename(os.path.dirname(decoded_dir)),
        SUB_RUN=os.path.basename(decoded_dir),
        combined_hits_dir=hits_dir.rstrip(os.sep) + os.sep,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded-dir", required=True,
                    help="<...>/<RUN>/<SUB_RUN> holding decoded_root/")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-key", default="sat_det3",
                    help="run whose detector config the sim models")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=SIGMA_THRESHOLD)
    a = ap.parse_args()

    decoded_dir = os.path.abspath(os.path.expanduser(a.decoded_dir))
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run

    cal = CalibrationBundle.load(os.path.expanduser(a.bundle))
    print("bundle:", os.path.expanduser(a.bundle))
    print(cal.summary() if hasattr(cal, "summary") else "")

    hits_dir = os.path.join(decoded_dir, "sim_hits")
    hits_root = os.path.join(hits_dir, "sim_datrun_seed_00.root")
    print("\n[1/2] hits from simulated waveforms "
          f"(detection only, >= {a.sigma:g} sigma)")
    cfg = sim_cfg(decoded_dir, hits_dir, a.run_key)
    build_hits(decoded_dir, list(cfg.MX17_FEUS), hits_root, sigma=a.sigma)

    print("\n[2/2] wft reconstruction")
    reconstruct_run(cfg, cal, os.path.expanduser(a.out),
                    jobs=a.jobs, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

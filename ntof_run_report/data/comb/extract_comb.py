#!/usr/bin/env python3
"""Flash-anchored time-since-flash histogram, EOS edition.

Algorithm copied verbatim from nTof_x17_DAQ/projections/ipc_yield.py
:func:`extract_subrun` so the epochs extracted here are directly comparable to
the run_79 / run_82 / run_86 caches that script already wrote.

Runs on lxplus, against EOS, under LCG_105 (it needs uproot + awkward):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 extract_comb.py run_61 \
        sngPS_dr700_r540_002,sngPS_dr700_r525_018,sngPS_dr700_r520_004 \
        run_61_tsf.npz
    python3 extract_comb.py run_67 \
        m090On_dr700_r540_029,m090On_dr700_r525_038,m090On_dr700_r520_041 \
        run_67_tsf.npz
    python3 extract_comb.py run_77 stat090_0000,stat090_0001 run_77_tsf.npz

The sub-runs are chosen at (or nearest to) the production HV and plastic
threshold; the comb is a DAQ dead-time effect and does not depend on either, so
pooling a few neighbouring points only buys statistics.

`run_79_tsf.npz` and `run_86_tsf.npz` are copies of
`nTof_x17_DAQ/projections/cache/{run79,run_86_stat090_0000}_tsf.npz`, written by
`ipc_yield.py` itself on the DAQ machine.
"""
import glob, json, os, sys
import numpy as np
import uproot

RUNS = "/eos/experiment/ntof/data/x17/july_beam/runs"
FEU = "01"
TICK_MS = 1e-5
SPILL_GAP_MS = 200.0
SAT = 3500.0
MIN_SAT_CELLS = 40
BIN_MS = 0.05
TMIN, TMAX = 0.0, 81.0


def edges():
    return np.arange(TMIN, TMAX + BIN_MS, BIN_MS)


def extract_subrun(subrun_dir, e):
    files = sorted(glob.glob(f"{subrun_dir}/decoded_root/*_{FEU}.root"))
    if not files:
        return None
    ts_all, sat_all = [], []
    for f in files:
        try:
            t = uproot.open(f)["nt"]
            ts_all.append(t["timestamp"].array(library="np").astype(np.int64))
            amp = t["amplitude"].array(library="ak")
        except Exception as ex:
            print("   skip", os.path.basename(f), ex)
            continue
        import awkward as ak
        sat_all.append(ak.to_numpy(ak.sum(abs(amp) >= SAT, axis=1)).astype(np.int64))
    if not ts_all:
        return None
    ts = np.concatenate(ts_all)
    sat = np.concatenate(sat_all)
    o = np.argsort(ts)
    ts, sat = ts[o], sat[o]
    brk = np.where(np.diff(ts) * TICK_MS > SPILL_GAP_MS)[0]
    st = np.concatenate([[0], brk + 1])
    en = np.concatenate([brk + 1, [ts.size]])
    counts = np.zeros(len(e) - 1, dtype=np.int64)
    got = 0
    for s, en_ in zip(st, en):
        seg_ts, seg_sat = ts[s:en_], sat[s:en_]
        fi = np.where(seg_sat >= MIN_SAT_CELLS)[0]
        if fi.size == 0:
            continue
        got += 1
        counts += np.histogram((seg_ts - seg_ts[fi[0]]) * TICK_MS, bins=e)[0]
    return dict(counts=counts, n_spill_total=len(st), n_spill_flash=got,
                n_events=int(ts.size))


def main():
    run = sys.argv[1]
    subruns = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    out = sys.argv[3]
    e = edges()
    root = os.path.join(RUNS, run)
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    if subruns:
        names = [n for n in names if n in set(subruns)]
    counts = np.zeros(len(e) - 1, dtype=np.int64)
    tot = fl = nev = 0
    used = []
    for n in names:
        r = extract_subrun(os.path.join(root, n), e)
        if r is None:
            continue
        counts += r["counts"]
        tot += r["n_spill_total"]
        fl += r["n_spill_flash"]
        nev += r["n_events"]
        used.append(n)
        print(f"  {n:26s} spills {r['n_spill_total']:5d}  flash "
              f"{100*r['n_spill_flash']/max(r['n_spill_total'],1):5.1f}%  "
              f"events {r['n_events']:7d}", flush=True)
    np.savez_compressed(out, edges=e, counts=counts, n_spill_total=tot,
                        n_spill_flash=fl, n_events=nev, run=run, feu=FEU,
                        subruns=np.array(used))
    print(f"[{run}] spills {tot} flash {fl} events {nev} -> {out}")


if __name__ == "__main__":
    main()

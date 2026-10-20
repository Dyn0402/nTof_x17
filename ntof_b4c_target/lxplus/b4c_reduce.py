#!/usr/bin/env python3
"""b4c_reduce.py -- per-event, per-arm trigger and Micromegas observables for the
B4C July-2026 target simulation (MX17_Full_Geant branch b4c-slab).

    python3 b4c_reduce.py <out.npz> <file.root> [...]

Per event with any scored hit, and per arm (0..3):
  S   max single SiPM-bar edep [MeV]   (bar from u, 25 mm pitch, as analyze_trigger_thermal.py)
  P   max plastic-bar edep [MeV]       (BackScintL / BackScintR)
  MM  charged-particle edep in DriftGas [keV]
  EP / EM  1 if an e+ / e- deposits in that arm's DriftGas
Only hits within PROMPT_NS of the earliest scored hit of the event are kept: in
neutron mode the clock starts at the gun, so absolute times carry the neutron's
flight; the window keeps the capture-prompt cascade and drops late decays.
EventTree weight (capture bias) and the generated count are stored too.
"""
import sys
import numpy as np
import uproot

PROMPT_NS = 200.0
CODES = {"PlasticScint": 0, "BackScintL": 1, "BackScintR": 2, "DriftGas": 3}


def reduce_file(fp):
    with uproot.open(fp) as f:
        et = f["EventTree"].arrays(["eventID", "weight"], library="np")
        nev = len(et["eventID"])
        wmap = dict(zip(et["eventID"].tolist(), et["weight"].tolist())) if nev else {}
        if not f["HitTree"].num_entries:
            return None, nev, float(et["weight"].sum()) if nev else 0.0
        t = f["HitTree"].arrays(["eventID", "armID", "detType", "particle", "edep", "time", "u"],
                                library="np")
    det = t["detType"].astype(str)
    code = np.full(len(det), -1, np.int8)
    for k, c in CODES.items():
        code[np.char.startswith(det, k)] = c
    keep = code >= 0
    ev = t["eventID"][keep].astype(np.int64); arm = t["armID"][keep].astype(np.int64)
    code = code[keep]; edep = t["edep"][keep] * 1e-6; tm = t["time"][keep]
    u = t["u"][keep]; part = t["particle"][keep].astype(str)
    uev, inv = np.unique(ev, return_inverse=True)
    tmin = np.full(len(uev), np.inf); np.minimum.at(tmin, inv, tm)
    ok = (tm - tmin[inv]) <= PROMPT_NS
    ev, arm, code, edep, u, part, inv = ev[ok], arm[ok], code[ok], edep[ok], u[ok], part[ok], inv[ok]
    n = len(uev)
    S = np.zeros((n, 4)); P = np.zeros((n, 4)); MM = np.zeros((n, 4))
    EP = np.zeros((n, 4), bool); EM = np.zeros((n, 4), bool)
    # SiPM bars: sum per (event, arm, bar), then max over bars
    s = code == 0
    bar = np.clip((u[s] + 250.0) // 25.0, 0, 19).astype(np.int64)
    key = (inv[s] * 4 + arm[s]) * 20 + bar
    uk, ki = np.unique(key, return_inverse=True); es = np.zeros(len(uk)); np.add.at(es, ki, edep[s])
    np.maximum.at(S, (uk // 80, (uk // 20) % 4), es)
    for c in (1, 2):
        m = code == c
        key = inv[m] * 4 + arm[m]
        uk, ki = np.unique(key, return_inverse=True); es = np.zeros(len(uk)); np.add.at(es, ki, edep[m])
        np.maximum.at(P, (uk // 4, uk % 4), es)
    g = code == 3
    charged = g & (part != "gamma") & (part != "neutron")
    np.add.at(MM, (inv[charged], arm[charged]), edep[charged] * 1e3)
    EP[inv[g & (part == "e+")], arm[g & (part == "e+")]] = True
    EM[inv[g & (part == "e-")], arm[g & (part == "e-")]] = True
    w = np.array([wmap.get(int(e), 1.0) for e in uev])
    return dict(S=S.astype(np.float32), P=P.astype(np.float32), MM=MM.astype(np.float32),
                EP=EP, EM=EM, w=w), nev, float(et["weight"].sum())


def main():
    out, files = sys.argv[1], sys.argv[2:]
    acc = {k: [] for k in ("S", "P", "MM", "EP", "EM", "w")}
    ngen = 0; wsum = 0.0; nbad = 0
    for fp in files:
        try:
            r, nev, ws = reduce_file(fp)
        except Exception as e:  # truncated / unfinished file
            print("skip", fp, e, file=sys.stderr); nbad += 1; continue
        ngen += nev; wsum += ws
        if r is not None:
            for k in acc: acc[k].append(r[k])
    arr = {k: (np.concatenate(v) if v else np.zeros((0, 4))) for k, v in acc.items()}
    np.savez_compressed(out, ngen=ngen, wsum=wsum, nfiles=len(files) - nbad, **arr)
    print(out, "files", len(files) - nbad, "generated", ngen, "events with hits", len(arr["w"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""deck_data.py -- the numbers behind make_b4c_deck.py, from the reduced sim
arrays and the data-side census.  Writes OUT/deck_data.json.

    python -m ntof_b4c_target.deck_data
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

from . import expect as X

HERE = Path(__file__).resolve().parent
CENSUS = HERE / "data" / "activity_census.log"
EVENTS = HERE.parent / "ntof_run_report" / "data" / "events_per_subrun.csv"
LINES = [(2.533, "E1"), (4.444, "M1"), (4.711, "E1"), (6.740, "M1"), (7.007, "E1"),
         (8.917, "M1"), (11.448, "E1")]
# ³He reference (same geometry, current EAR2 profile; neutrons_thermal_ear2_L19p5):
HE3_GAS_ABS = 0.4482                 # beam neutrons absorbed by ³He(n,p) in the gas
HE3_SIG_NG = 55e-6 / 5333.0          # ³He(n,γ)/(n,p), thermal
HE3_ALPHA_M1 = 3.661e-3              # ipc_born.alpha_pair('M1', 20.58)
AL_CAP_PER_N = 0.0028459             # He3Cap_Al nCapture per beam neutron (same run, analog)
AL_IPC_PER_CAPT = 2.557e-3           # Σ I·α over the ²⁸Al lines > 2 MeV, all taken as E1 (ipc_born)


def stats(d, ngen):
    """Per-pair efficiencies and lepton layer reach (unweighted MC fractions)."""
    S = d["S"] / X.MIP_SIPM; P = d["P"] / X.MIP_PLASTIC; mm = d["MM"] >= X.MM_KEV
    leg = (S >= X.A_SIPM) & (P >= X.B_PLASTIC)
    lep = d["EP"] | d["EM"]                      # arm entered by a lepton (drift gas)
    split = (d["EP"][:, :, None] & d["EM"][:, None, :] & ~np.eye(4, dtype=bool)[None]).any((1, 2))
    nl = lep.sum()
    return dict(
        ngen=int(ngen),
        p_split=float(split.sum() / ngen),                       # e+ and e- in two different chambers
        p_leg1=float((leg.sum(1) >= 1).sum() / ngen),
        p_leg2=float((leg.sum(1) >= 2).sum() / ngen),
        p_pair=float((split & ((leg & mm).sum(1) >= 2)).sum() / ngen),
        n_pair=int((split & ((leg & mm).sum(1) >= 2)).sum()),
        reach_wall=float((lep & (S >= X.A_SIPM)).sum() / max(nl, 1)),
        reach_plastic_any=float((lep & (d["P"] > 0.1)).sum() / max(nl, 1)),
        reach_plastic_thr=float((lep & (P >= X.B_PLASTIC)).sum() / max(nl, 1)),
        n_lepton_arms=int(nl),
        # plastic deposit (MeV) of lepton arms that reached it, for a spectrum
        p_dep=np.histogram(d["P"][lep & (d["P"] > 0.1)], bins=np.arange(0, 12.01, 0.5))[0].tolist())


def census():
    rows = []
    for ln in CENSUS.read_text().splitlines():
        m = re.match(r"(run_\d+) (\S+) (\{.*\})", ln)
        if m:
            j = json.loads(m.group(3)); rows.append(dict(run=m.group(1), sub=m.group(2), **j))
    return rows


def timeline():
    """Recorded events per run and trigger family for runs 25-41 (event census)."""
    out = {}
    for r in csv.DictReader(open(EVENTS)):
        m = re.match(r"run_(\d+)$", r["run"])
        if not m or not 20 <= int(m.group(1)) <= 42:
            continue
        fam = r["subrun"].split("_")[0]
        fam = ("flash" if fam.startswith("flash") else "random" if fam.startswith(("rand", "frand", "fulltime"))
               else "scint" if fam.startswith("scint") else "other")
        d = out.setdefault(r["run"], {"stamp": r["stamp"]})
        d[fam] = d.get(fam, 0) + int(r["events"] or 0)
        d["stamp"] = min(d["stamp"], r["stamp"])
    return out


def main():
    res = {"lines": {}, "he3": {}}
    for e, k in LINES:
        d, n = X.load(f"ipc_r50__ipc_r50_{e:.3f}_{k}__*.npz")
        res["lines"][f"{e:.3f}"] = dict(stats(d, n), multipole=k,
                                        intensity=X.IPC_LINES[f"{e:.3f}_{k}"][0],
                                        alpha=X.IPC_LINES[f"{e:.3f}_{k}"][1])
    for tag in ("ipcM1", "ipcE0", "x17"):
        files = sorted(X.SIM_PARTS.glob(f"he3_ref__he3_{tag}__*.npz"))
        if files:
            d, n = X.load(f"he3_ref__he3_{tag}__*.npz")
            res["he3"][tag] = stats(d, n)
    c = X.N_PULSE
    res["per_pulse"] = dict(
        b4c_ng=X.CAPT_PER_PULSE,
        b4c_ipc=X.CAPT_PER_PULSE * sum(i / 100 * a for i, a in X.IPC_LINES.values()),
        he3_ng=c * HE3_GAS_ABS * HE3_SIG_NG,
        he3_ipc=c * HE3_GAS_ABS * HE3_SIG_NG * HE3_ALPHA_M1,
        he3_al_ng=c * AL_CAP_PER_N,
        he3_al_ipc=c * AL_CAP_PER_N * AL_IPC_PER_CAPT,
        n_pulse=c)
    res["census"] = census()
    res["timeline"] = timeline()
    (X.OUT / "deck_data.json").write_text(json.dumps(res, indent=1))
    for k, v in res["lines"].items():
        print(k, {q: (f"{v[q]:.3g}" if isinstance(v[q], float) else v[q]) for q in
                  ("p_split", "p_leg1", "p_leg2", "p_pair", "reach_wall", "reach_plastic_any", "reach_plastic_thr")})
    for k, v in res["he3"].items():
        print("he3", k, {q: f"{v[q]:.3g}" for q in ("p_split", "p_leg2", "p_pair", "reach_plastic_thr")})
    print(res["per_pulse"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Does the measured v(E) allow a richer isobutane fraction than 95/5?

Follow-up to the HV-slope test. That test showed det3's gain rises 1.52x
faster with mesh voltage than the Garfield ladder, and the leading suspect is
that the flowmeter-set mixture is richer in isobutane than the 95/5 the
simulation assumes — never assayed, same epistemic status as the humidity.

Isobutane fraction is doubly testable: it also moves the drift-velocity curve,
which this bench measured independently in June. So this script confronts the
iso hypothesis with v(E).

Measured leg
------------
The June waveform-first forward-fit v(HV) ladder (WAVEFORM_FIRST_THREADING.md
§14/§17), 300–1100 V on the det3 saturday scan — the same detector, the same
week, one day either side of the gain ladder. Waveform-derived throughout
(RECONSTRUCTION_BASIS.md); no combined_hits time is read anywhere.

  v(1000 V) = 36.7 ± 0.3 (fit) ± 0.9 (model)

Crucially, this v is gap-free: the forward fit gets it from arrival time
against a depth scale set by the reference track angle, not from filling a
nominal gap. The DRIFT GAP therefore enters only through the E axis
(E = V_drift / gap), where it is a genuine unknown — nominal 30 mm, but det3's
cathode is dished to 25.7–29.2 mm (GAP_STUDY_2026-07-30) and the
charge-visible column is 24.7 mm. So the gap is profiled as a free nuisance
over 24–32 mm rather than fixed, which is the conservative choice: it lets a
wrong mixture slide along the E axis to try to fit.

Model leg
---------
Every Magboltz drift-velocity grid in garfield_sim/results/ carrying
v_um_per_ns, parsed to (iso %, H2O %, other contaminant %). Curves computed at
a different pressure are put on the bench's 745.83 torr by the E/N scaling
v_p(E) = v_p0(E · p0/p), exact for drift velocity.

    python3 mx17_sim_wft/hv_slope/iso_ve.py --out-dir ~/x17/response_sim/hv_slope/iso_ve
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

BENCH = os.path.expanduser(
    "~/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/"
    "long_run_resist_490V_drift_1000V/mx17_3/waveform_first")
GRIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "garfield_sim", "results")
P_BENCH = 745.83          # torr, the pressure the Saclay grids were run at

# Uncertainty model, declared before fitting.
SIG_REL = 0.03            # per-point, uncorrelated (fit scatter + point-to-point)
SIG_FLOOR = 0.4           # um/ns
SCALE_PRIOR = 0.026       # common v-scale: the calibration's model bias
                          # (toy closure: strong mismatch deflates v by 2.6 %)
GAP_MM = np.arange(24.0, 32.01, 0.05)

# The 300 V point sits on a truncated window (U(25 mm) = 2083 ns against a
# ~1500 ns visible window), so every fit is reported on three point sets.
POINT_SETS = {"all": 0, "ge500": 500, "ge700": 700}

# det3's own charge-column endpoint, X plane, SAME subrun as the v ladder
# (mx_june_wft/bench/gap_study.py -> wft/gap_study/gap_study.json): a
# TIME, 757.5 +- 2.2 ns, before any velocity is applied. The published
# 27.9 +- 0.1 (stat) +- 1.0 (calib) mm column is this time times a velocity,
# so the +-1.0 mm calibration systematic IS the velocity ambiguity and must
# not be counted twice — variant B carries the time and lets the mixture
# supply the velocity. The +-25 ns systematic covers the endpoint definition
# (sharp 757.5 vs attachment 760.6) and the +1.5 mm amplitude dependence.
T_COL_NS = 757.5
T_COL_SYS_NS = 25.0
GAP_MECH_MAX = 30.6       # the chambers are mechanically 30 mm deep; det2
                          # reads 30.5, so a fitted gap above this is unphysical


def measured():
    ff = json.load(open(f"{BENCH}/drift_scan_v.json"))
    ff.update(json.load(open(f"{BENCH}/drift_scan_v_lowhv.json")))
    hv = sorted(int(k) for k in ff)
    v = [ff[str(h)]["v"] for h in hv]
    hv.append(1000)
    v.append(json.load(open(f"{BENCH}/hyper_v2.json"))["v"])
    o = np.argsort(hv)
    return np.array(hv, float)[o], np.array(v, float)[o]


_NAME = [
    (re.compile(r"^iso([\d.]+)_h2o([\d.]+)_n2([\d.]+)_o2([\d.]+)$"),
     lambda m: (float(m[1]), float(m[2]), float(m[3]) + float(m[4]))),
    (re.compile(r"^Ar_iso(\d+)_H2O([\d.]+)_N2_([\d.]+)$"),
     lambda m: (float(m[1]), float(m[2]), float(m[3]))),
    (re.compile(r"^Ar_iso(\d+)_H2O([\d.]+)_air([\d.]+)$"),
     lambda m: (float(m[1]), float(m[2]), float(m[3]))),
    (re.compile(r"^Ar_iso([\d.]+)_H2O([\d.]+)$"),
     lambda m: (float(m[1]), float(m[2]), 0.0)),
    (re.compile(r"^Ar_iso([\d.]+)_(?:N2|O2|air)[_]?([\d.]+)$"),
     lambda m: (float(m[1]), 0.0, float(m[2]))),
    (re.compile(r"^Ar[\d.]+_iso([\d.]+)_H2O([\d.]+)$"),
     lambda m: (float(m[1]), float(m[2]), 0.0)),
    (re.compile(r"^Ar[\d.]+_iso([\d.]+)$"),
     lambda m: (float(m[1]), 0.0, 0.0)),
]


def parse_mix(name, comps):
    """(iso %, H2O %, other contaminant %) or None if not an Ar/iso mixture."""
    if comps:
        d = {c[0].lower(): float(c[1]) for c in comps}
        if "ar" not in d or "ic4h10" not in d:
            return None
        other = sum(v for k, v in d.items() if k not in ("ar", "ic4h10", "h2o"))
        return d["ic4h10"], d.get("h2o", 0.0), other
    for rx, f in _NAME:
        m = rx.match(name)
        if m:
            return f(m)
    return None


def load_curves():
    out = {}
    for path in sorted(glob.glob(os.path.join(GRIDS, "**", "*.json"),
                                 recursive=True)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        p = d.get("pressure_torr")
        mix = d.get("mixtures") or {}
        if not p:
            continue
        comps_all = d.get("comps") or {}
        for k, pts in mix.items():
            if not pts or not isinstance(pts, list) or "v_um_per_ns" not in pts[0]:
                continue
            c = parse_mix(k, comps_all.get(k))
            if c is None:
                continue
            E = np.array([q["E_Vcm"] for q in pts], float)
            V = np.array([q["v_um_per_ns"] for q in pts], float)
            o = np.argsort(E)
            # E/N scaling onto the bench pressure
            E = E[o] * P_BENCH / p
            key = (round(c[0], 2), round(c[1], 3), round(c[2], 3))
            src = os.path.relpath(path, GRIDS)
            # prefer the higher-ncoll (finer) run if a mixture repeats
            if key in out and out[key]["ncoll"] >= (d.get("ncoll") or 0):
                continue
            out[key] = dict(E=E, V=V[o], src=src, name=k,
                            ncoll=d.get("ncoll") or 0, p=p)
    return out


def fit_one(cur, HV, VM, sig, gap_free=True, gap_fix=30.0, scale_free=True):
    """Profile the drift gap (E-axis) and a common v-scale. Returns the best
    chi2, gap, scale and the RMS residual in um/ns."""
    gaps = GAP_MM if gap_free else np.array([gap_fix])
    best = None
    w = 1.0 / sig ** 2
    for g in gaps:
        pred = np.interp(HV / (g / 10.0), cur["E"], cur["V"],
                         left=np.nan, right=np.nan)
        if not np.all(np.isfinite(pred)):
            continue
        if scale_free:
            # analytic minimum of sum w (s p - v)^2 + ((s-1)/prior)^2
            a = (w * pred * pred).sum() + 1.0 / SCALE_PRIOR ** 2
            b = (w * pred * VM).sum() + 1.0 / SCALE_PRIOR ** 2
            s = b / a
        else:
            s = 1.0
        r = s * pred - VM
        chi2 = float((w * r * r).sum() + ((s - 1) / SCALE_PRIOR) ** 2)
        if best is None or chi2 < best[0]:
            best = (chi2, float(g), float(s), float(np.sqrt(np.mean(r ** 2))))
    return best


def self_consistent_gap(cur, T_ns=T_COL_NS):
    """The gap a mixture implies from det3's measured charge-column TIME.

    The column at 1000 V drift spans the full mechanical depth (det2 control
    reads its full 30.5 mm with the same estimator), so
        gap = v(E = 1000 V / gap) * T_col
    which is a fixed point because the field depends on the gap. Solved by
    bisection on g - v(10000/g)*T/1000, monotone over the physical range.
    """
    def f(g):
        v = np.interp(10000.0 / g, cur["E"], cur["V"], left=np.nan, right=np.nan)
        return g - v * T_ns / 1000.0
    lo, hi = 20.0, 45.0
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def fit_tied(cur, HV, VM, sig):
    """Variant B: no free gap — it is whatever the mixture's own velocity and
    the measured column time imply. Only the v-scale is free."""
    out = []
    for dT in (-T_COL_SYS_NS, 0.0, +T_COL_SYS_NS):
        g = self_consistent_gap(cur, T_COL_NS + dT)
        if g is None:
            continue
        pred = np.interp(HV / (g / 10.0), cur["E"], cur["V"],
                         left=np.nan, right=np.nan)
        if not np.all(np.isfinite(pred)):
            continue
        w = 1.0 / sig ** 2
        a = (w * pred * pred).sum() + 1.0 / SCALE_PRIOR ** 2
        b = (w * pred * VM).sum() + 1.0 / SCALE_PRIOR ** 2
        s = b / a
        r = s * pred - VM
        chi2 = float((w * r * r).sum() + ((s - 1) / SCALE_PRIOR) ** 2)
        # a gap deeper than the chamber is not available to the fit
        pen = 0.0 if g <= GAP_MECH_MAX else ((g - GAP_MECH_MAX) / 0.5) ** 2
        out.append((chi2 + pen, float(g), float(s),
                    float(np.sqrt(np.mean(r ** 2))), chi2))
    return min(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/x17/response_sim/hv_slope/iso_ve"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    HV, VM = measured()
    SIG = np.hypot(SIG_REL * VM, SIG_FLOOR)
    curves = load_curves()
    print(f"measured: {[f'{h:.0f}V={v:.2f}' for h, v in zip(HV, VM)]}")
    print(f"{len(curves)} Ar/iso mixtures with a drift-velocity grid\n")

    res = {}
    for setname, lo in POINT_SETS.items():
        m = HV >= lo
        rows = []
        for key, cur in curves.items():
            iso, h2o, other = key
            f = fit_one(cur, HV[m], VM[m], SIG[m])
            if f is None:
                continue
            fx = fit_one(cur, HV[m], VM[m], SIG[m], gap_free=False,
                         scale_free=False)
            tb = fit_tied(cur, HV[m], VM[m], SIG[m])
            rows.append(dict(iso=iso, h2o=h2o, other=other, name=cur["name"],
                             src=cur["src"], ndof=int(m.sum()) - 2,
                             chi2=f[0], gap_mm=f[1], scale=f[2], rms=f[3],
                             chi2_fixed=None if fx is None else fx[0],
                             rms_fixed=None if fx is None else fx[3],
                             chi2_tied=None if tb is None else tb[0],
                             gap_tied=None if tb is None else tb[1],
                             scale_tied=None if tb is None else tb[2],
                             rms_tied=None if tb is None else tb[3]))
        rows.sort(key=lambda r: r["chi2"])
        res[setname] = rows

        print(f"===== point set {setname} (n = {int(m.sum())}) — best 12")
        print(f"{'iso%':>5} {'H2O%':>5} {'oth%':>5} {'chi2':>8} {'rms':>6} "
              f"{'gap':>6} {'scale':>6}   source")
        for r in rows[:12]:
            print(f"{r['iso']:5.1f} {r['h2o']:5.2f} {r['other']:5.2f} "
                  f"{r['chi2']:8.1f} {r['rms']:6.2f} {r['gap_mm']:6.2f} "
                  f"{r['scale']:6.3f}   {r['name']}")

        # profile: best achievable chi2 at each iso fraction, water and every
        # other contaminant free
        prof = {}
        for r in rows:
            k = r["iso"]
            if k not in prof or r["chi2"] < prof[k]["chi2"]:
                prof[k] = r
        print(f"\n  profiled over H2O/contaminants — best per iso fraction:")
        print(f"  {'iso%':>5} {'chi2':>8} {'dchi2':>7} {'rms':>6} {'H2O%':>5} "
              f"{'gap':>6} {'scale':>6}  n_mix")
        cmin = min(v["chi2"] for v in prof.values())
        for k in sorted(prof):
            r = prof[k]
            n = sum(1 for x in rows if x["iso"] == k)
            print(f"  {k:5.1f} {r['chi2']:8.1f} {r['chi2'] - cmin:7.1f} "
                  f"{r['rms']:6.2f} {r['h2o']:5.2f} {r['gap_mm']:6.2f} "
                  f"{r['scale']:6.3f}  {n}")
        res[setname + "_profile"] = {str(k): prof[k] for k in sorted(prof)}

        # variant B — gap tied to the measured column time, and variant C —
        # the June convention (gap fixed 30 mm, no v-scale), for reference
        proB, proC = {}, {}
        for r in rows:
            if r["chi2_tied"] is not None and (
                    r["iso"] not in proB or r["chi2_tied"] < proB[r["iso"]]["chi2_tied"]):
                proB[r["iso"]] = r
            if r["chi2_fixed"] is not None and (
                    r["iso"] not in proC or r["chi2_fixed"] < proC[r["iso"]]["chi2_fixed"]):
                proC[r["iso"]] = r
        bmin = min(v["chi2_tied"] for v in proB.values()) if proB else 0
        cmin2 = min(v["chi2_fixed"] for v in proC.values()) if proC else 0
        print("\n  B: gap TIED to the measured column time "
              f"({T_COL_NS:.0f} ns) | C: June convention (gap 30 mm, scale 1)")
        print(f"  {'iso%':>5} | {'chi2_B':>8} {'d':>7} {'gapB':>6} {'sclB':>6} "
              f"{'H2O':>5} | {'chi2_C':>9} {'d':>8} {'H2O':>5}")
        for k in sorted(set(proB) | set(proC)):
            b, c = proB.get(k), proC.get(k)
            bs = (f"{b['chi2_tied']:8.1f} {b['chi2_tied'] - bmin:7.1f} "
                  f"{b['gap_tied']:6.2f} {b['scale_tied']:6.3f} {b['h2o']:5.2f}"
                  ) if b else " " * 34
            cs = (f"{c['chi2_fixed']:9.1f} {c['chi2_fixed'] - cmin2:8.1f} "
                  f"{c['h2o']:5.2f}") if c else ""
            print(f"  {k:5.1f} | {bs} | {cs}")
        res[setname + "_profileB"] = {str(k): proB[k] for k in sorted(proB)}
        res[setname + "_profileC"] = {str(k): proC[k] for k in sorted(proC)}
        print()

    with open(os.path.join(a.out_dir, "iso_ve.json"), "w") as f:
        json.dump(dict(hv=HV.tolist(), v=VM.tolist(), sigma=SIG.tolist(),
                       gap_range=[float(GAP_MM[0]), float(GAP_MM[-1])],
                       scale_prior=SCALE_PRIOR, results=res), f, indent=1)
    print("wrote", os.path.join(a.out_dir, "iso_ve.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gain vs resist voltage on a FLAT dataset, and the kernel's invariance to it.

The bundle-transfer argument the whole `wft` calibration rests on is that the
resistive sharing kernel is a property of the *layer*, not of the operating
point -- so a kernel measured at one gain may be used at another.  The evidence
for that was run_56's flat 590 -> 625 V, a 6 % swing (`M70V_FLAT_ANALYSIS.md`),
plus the beam/bench cross-gas agreement.  run_66 is a factor ~1.9 in resist
voltage at normal incidence, which is the widest lever the campaign has.

Two measurements per resist plateau, both from hits alone (no pairing, no
track fit, no waveform extraction -- so this runs in minutes):

  gain proxy   the truncated-mean and median hit amplitude of the LEADING
               strip of each event, plus strips/event.  Absolute gain is not
               recoverable (ZS sigma and pedestal set are common here, which is
               what makes the plateaus comparable at all -- see the run_60
               warning in GAS_FLUSH_TIMELINE that gain is NOT comparable across
               runs with different ZS).  All nine plateaus share one sub-run,
               one pedestal set and one threshold, so ratios ARE meaningful.

  sharing      the model-independent charge-budget proxy: for each event, the
               ratio of the summed amplitude of the +-1 neighbours to the
               leading strip's, in the SAME time sample window.  This is the
               hits-level shadow of the kernel's c1 -- if the sharing kernel is
               a layer property it must be flat in gain, and if it is really a
               gain/threshold artefact it must track the amplitude.

The second is the point.  A sharing ratio that stays put while the gain proxy
moves by a factor of several is direct evidence for the invariance premise.

  ../../.venv/bin/python gain_scan_flat.py run66_flat_resist
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "det4_sps_assessment"))
import datasets                                          # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM     # noqa: E402

SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)
TIME_LO, TIME_HI = 200.0, 3800.0
DT_MATCH_NS = 180.0     # neighbour must be within this of the leading strip
#: amplitude-matched window for the censoring control (the kernel work's own
#: q0 window, RAW_RUN71_REANALYSIS / robust_waveforms.py)
QLO, QHI = 400.0, 3000.0


def trunc_mean(a, lo=10, hi=90):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan
    p, q = np.percentile(a, (lo, hi))
    core = a[(a >= p) & (a <= q)]
    return float(core.mean()) if len(core) else np.nan


def per_plateau(name):
    D = datasets.get(name)
    acc = {}
    for sub, stem, t0, idxs in D["subruns"]:
        d = os.path.join(D["stage"], sub)
        if not os.path.isdir(d):
            d = D["stage"]
        for fn in sorted(glob.glob(os.path.join(d, f"hits_{sub}_*_03.root"))):
            try:
                with uproot.open(fn) as f:
                    if "hits" not in f:
                        continue
                    a = f["hits"].arrays(
                        ["eventId", "trigger_timestamp_ns", "channel", "time",
                         "amplitude"], library="np")
            except Exception:
                continue
            tw = t0 + a["trigger_timestamp_ns"] / 1e9
            lab = datasets.plateau_of(name, tw)
            ok = (a["time"] >= TIME_LO) & (a["time"] <= TIME_HI) & (lab != "")
            if not ok.any():
                continue
            ev = a["eventId"][ok]
            ch = a["channel"][ok].astype(int)
            tt = a["time"][ok]
            am = a["amplitude"][ok]
            lb = lab[ok]
            for l in np.unique(lb):
                m = lb == l
                acc.setdefault(str(l), []).append(
                    (ev[m], ch[m], tt[m], am[m]))
    return D, acc


def analyse(chunks):
    """Leading-strip amplitude and the +-1 sharing ratio, per view."""
    ev = np.concatenate([c[0] for c in chunks])
    ch = np.concatenate([c[1] for c in chunks])
    tt = np.concatenate([c[2] for c in chunks])
    am = np.concatenate([c[3] for c in chunks])
    out = {}
    for v in ("x", "y"):
        m = VIEW[ch] == v
        if m.sum() < 500:
            continue
        e, c, t, q = ev[m], ch[m], tt[m], am[m]
        # leading strip per event: max amplitude
        order = np.lexsort((-q, e))
        e_s, c_s, t_s, q_s = e[order], c[order], t[order], q[order]
        first = np.r_[True, e_s[1:] != e_s[:-1]]
        lead_e, lead_c, lead_t, lead_q = (e_s[first], c_s[first], t_s[first],
                                          q_s[first])
        # map event -> leading strip index / time / amplitude
        emax = int(e.max()) + 1
        l_sidx = np.full(emax, -9999, np.int32)
        l_t = np.full(emax, np.nan)
        l_q = np.full(emax, np.nan)
        l_sidx[lead_e] = SIDX[lead_c]
        l_t[lead_e] = lead_t
        l_q[lead_e] = lead_q
        # neighbours: |d| == 1, time-matched to the leading strip
        d = SIDX[c] - l_sidx[e]
        near = (np.abs(d) == 1) & (np.abs(t - l_t[e]) <= DT_MATCH_NS)
        nb_sum = np.zeros(emax)
        np.add.at(nb_sum, e[near], q[near])
        has = np.isfinite(l_q) & (l_q > 0)
        ratio = nb_sum[has] / l_q[has]
        nstr = np.zeros(emax)
        np.add.at(nstr, e, 1.0)
        # --- THE CONTROL -------------------------------------------------
        # The raw sharing ratio above is ZS-censored and the censoring is
        # gain-dependent: a +-1 neighbour carries ~0.4 (Y) / ~0.5 (X) of the
        # leading strip, so as gain falls the neighbours cross the 4 sigma
        # threshold BEFORE the leading strip does, and the ratio falls for a
        # reason that has nothing to do with the kernel.  Restricting to a
        # fixed leading-strip amplitude window makes the plateaus compare
        # like with like: at the same q_lead the neighbour sits at the same
        # absolute ADC, so it is censored the same way at every gain.
        # If the matched ratio is flat while the raw one is not, the raw
        # variation was threshold, not physics.
        m_all = has & (l_q >= QLO) & (l_q <= QHI)
        idx = np.flatnonzero(m_all)
        matched = nb_sum[idx] / l_q[idx] if len(idx) else np.array([])
        out[v] = dict(
            n_events=int(has.sum()),
            q_lead_trunc=trunc_mean(l_q[has]),
            q_lead_med=float(np.median(l_q[has])),
            strips_per_ev=float(np.mean(nstr[has])),
            share_ratio_med=float(np.median(ratio)),
            share_ratio_trunc=trunc_mean(ratio),
            frac_with_nb=float(np.mean(ratio > 0)),
            n_matched=int(len(idx)),
            q_lead_matched=trunc_mean(l_q[idx]) if len(idx) else np.nan,
            share_matched_med=float(np.median(matched)) if len(matched) else np.nan,
            share_matched_trunc=trunc_mean(matched) if len(matched) else np.nan,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    D, acc = per_plateau(args.dataset)
    hv = {p[0]: (p[3], p[4]) for p in D["plateaus"]}
    res = {}
    for l, chunks in acc.items():
        r = analyse(chunks)
        if r:
            dv, rv = hv.get(l, (np.nan, np.nan))
            res[l] = dict(drift_V=float(dv), resist_V=float(rv), views=r)

    print(f"\n=== {args.dataset} ({D['mount']}, {D['gas']})")
    print("gain proxy = truncated-mean leading-strip amplitude [ADC]; "
          "share = median (sum |d|=1) / lead\n")
    # Sort and label by whichever voltage is actually being scanned -- this
    # file serves both the resist scan (run_66) and the drift scan (run_70),
    # and sorting a drift scan by its constant resist gives an arbitrary row
    # order that looks like noise.
    nres = len({round(x["resist_V"], 1) for x in res.values()})
    ndri = len({round(x["drift_V"], 1) for x in res.values()})
    scan_key = "resist_V" if nres >= ndri else "drift_V"
    hdr = "resist" if scan_key == "resist_V" else "drift"
    print(f"  (scanned quantity: {hdr}; the other is held)")
    for v in ("y", "x"):
        rows = [(x[scan_key], l, x["views"][v]) for l, x in res.items()
                if v in x["views"]]
        if not rows:
            continue
        rows.sort(key=lambda r: -r[0])
        print(f"  --- {v.upper()} view")
        print(f"  {hdr:>7} {'events':>8} {'q_lead':>8} {'strips':>7} "
              f"{'share':>7} | {'n(match)':>9} {'q(match)':>9} "
              f"{'share(match)':>12}")
        q0, w0 = rows[0][2]["q_lead_trunc"], rows[0][2]
        for rv, l, w in rows:
            print(f"  {rv:7.1f} {w['n_events']:8d} {w['q_lead_trunc']:8.1f} "
                  f"{w['strips_per_ev']:7.2f} {w['share_ratio_med']:7.3f} | "
                  f"{w['n_matched']:9d} {w['q_lead_matched']:9.1f} "
                  f"{w['share_matched_med']:12.3f}")
        wN = rows[-1][2]
        # the raw ratio's median collapses to exactly 0 once more than half
        # the events have both neighbours under threshold -- report that as
        # censored rather than dividing by it
        raw = (f"x{w0['share_ratio_med'] / wN['share_ratio_med']:.2f}"
               if wN["share_ratio_med"] > 1e-6 else "CENSORED to 0")
        print(f"  over the scan:  strips/ev x{w0['strips_per_ev'] / wN['strips_per_ev']:.2f}"
              f"   q_lead x{q0 / wN['q_lead_trunc']:.2f}"
              f"   share(raw) {raw}"
              f"   **share(matched) x{w0['share_matched_med'] / max(wN['share_matched_med'], 1e-9):.2f}**")

    out = args.out or D["stage"] + f"gain_scan_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

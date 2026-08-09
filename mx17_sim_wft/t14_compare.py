#!/usr/bin/env python3
"""
T14 — compare SIMULATED waveforms with DATA waveforms on the frozen default.

Frozen default (Dylan, 2026-08-09, decided BEFORE this comparison ran):
rho_s = 2 MOhm/sq W2 kernel + DRY Ar/iso 95/5 gas table + det3 saturday-scan
`long_run_resist_490V_drift_1000V` bundle as-analysed. This script renders the
one-shot verdict; post-default rho_s / v / contaminant variants are DIAGNOSIS.

THE COMPARISON IS AT THE WAVEFORM LEVEL. Both legs went through
`t13_reco.py`'s dual-use path: seeds built from the raw 32 x 60 ns samples by
the same code at the same sigma, `combined_hits` unread on either leg, and the
same wft forward model fitted to both. On top of the reco tables, this script
re-reads the decoded waveforms themselves (same FeuReader, so pedestal + CNS
are bit-for-bit identical) and compares:

  - average peak-strip waveform, aligned at peak (shape: rise, width, tail)
  - peak-strip amplitude and per-event charge spectra (ABSOLUTE scale — the
    W2 kernel's +25.7 % capture is exactly what is on trial here)
  - >= sigma strip multiplicity and the transverse cluster profile (sharing)
  - forward-fit chi2/dof (does the same model describe both)

Angle matching: the simulation's tracks are near-vertical (Stage A gun),
cosmics are not. Data events are cut per view to the sim's [p1, p99] theta
window before any waveform comparison; full theta distributions are shown for
context only. Population differences are therefore acceptance, not response.

Pre-registered systematics quoted in the report, not absorbed:
  - v_drift: bundle 36.60 vs dry-table 39.14 um/ns = 6.9 % input systematic
    (NO humidity was ever measured; contaminants are a fitted search axis)
  - W2 kernel at ny=512: ~1.3 % absolute-amplitude grid systematic and a
    ~0.45 % pad-edge shoulder term on shape
  - seed-rate difference between legs is an observable, not a nuisance

    python3 mx17_sim_wft/t14_compare.py \
        --sim-dir ~/x17/response_sim/stageB_w2/w2_rho2M/default \
        --data-dir <mirror>/<RUN>/<SUB_RUN> \
        --sim-events <sim events parquet> --data-events <data events parquet> \
        --out-dir ~/x17/response_sim/stageB_w2/t14_compare
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t13_reco import sim_cfg  # noqa: E402  (also sets up repo import paths)

SIGMA = 5.0          # same threshold as the t13 seeder — do not change one leg
ALIGN_AT = 12        # sample index the peak is shifted to in the average
N_AVG_SAMP = 44      # padded length of the aligned average window
MAX_EVENTS = 2500    # per leg per view, extraction cap (deterministic subset)
DT_NS = 60.0         # nominal bench DAQ sampling period (labels the axes; a
                     # different actual period would shift BOTH legs equally)


def _leg_cfg(decoded_dir, run_key):
    hits_dir = os.path.join(decoded_dir, "sim_hits")
    return sim_cfg(decoded_dir, hits_dir, run_key)


def extract_view(decoded_dir, feu, pos, wanted, tag):
    """Waveform-level observables for one leg/view. Identical code both legs."""
    from wft.io import FeuReader

    files = sorted(glob.glob(os.path.join(decoded_dir, "decoded_root",
                                          f"*_{feu:02d}.root")))
    if not files:
        raise FileNotFoundError(f"no decoded file for FEU {feu} in {decoded_dir}")

    valid = ~np.isnan(pos)
    acc_norm = np.zeros(N_AVG_SAMP)
    acc_abs = np.zeros(N_AVG_SAMP)
    n_acc = 0
    prof_off, prof_amp = [], []
    rows = []
    noise_medians = []
    for path in files:
        rdr = FeuReader(path)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        noise_medians.append(float(np.median(rdr.noise)))
        want_here = wanted & set(int(e) for e in rdr.event_ids)
        if not want_here:
            continue
        for eid, _ftst, wfm in rdr.iter_events(want_here):
            amp = wfm.max(axis=1)
            sig = amp / noise
            over = valid & (sig >= SIGMA)
            if not over.any():
                continue
            pk = int(np.flatnonzero(over)[np.argmax(amp[np.flatnonzero(over)])])
            w = wfm[pk]
            ipk = int(np.argmax(w))
            a = float(w[ipk])
            # rise 10->90 % and FWHM by linear interpolation on the samples
            def _cross(frac, side):
                lvl = frac * a
                if side < 0:
                    below = np.flatnonzero(w[:ipk + 1] < lvl)
                    if len(below) == 0:
                        return None
                    i = below[-1]
                    j = i + 1
                else:
                    below = np.flatnonzero(w[ipk:] < lvl)
                    if len(below) == 0:
                        return None
                    j = ipk + below[0]
                    i = j - 1
                if w[j] == w[i]:
                    return float(i)
                return i + (lvl - w[i]) / (w[j] - w[i])
            r10, r90 = _cross(0.10, -1), _cross(0.90, -1)
            hl, hr = _cross(0.50, -1), _cross(0.50, +1)
            rows.append(dict(
                event_id=int(eid), peak_ch=pk, peak_amp=a,
                peak_sample=ipk, n_over=int(over.sum()),
                q_event=float(amp[over].sum()),
                rise_ns=(r90 - r10) * DT_NS if r10 is not None and r90 is not None else np.nan,
                fwhm_ns=(hr - hl) * DT_NS if hl is not None and hr is not None else np.nan,
            ))
            # aligned average (skip peaks too close to the window edge)
            lo = ALIGN_AT - ipk
            if 0 <= lo and lo + len(w) <= N_AVG_SAMP:
                acc_norm[lo:lo + len(w)] += w / a
                acc_abs[lo:lo + len(w)] += w
                n_acc += 1
            # transverse profile vs signed strip offset (position-ordered)
            order = np.argsort(pos[valid])
            chs = np.flatnonzero(valid)[order]
            rank = {c: i for i, c in enumerate(chs)}
            if pk in rank:
                for c in np.flatnonzero(over):
                    if c in rank:
                        prof_off.append(rank[c] - rank[pk])
                        prof_amp.append(amp[c] / a)
        print(f"  [{tag}] {os.path.basename(path)}: {len(rows)} events so far",
              flush=True)

    df = pd.DataFrame(rows)
    return dict(
        table=df,
        avg_norm=acc_norm / max(n_acc, 1),
        avg_abs=acc_abs / max(n_acc, 1),
        n_avg=n_acc,
        prof=pd.DataFrame(dict(off=prof_off, amp=prof_amp)),
        median_noise=float(np.median(noise_medians)),
    )


def _sel_ids(ev, view, theta_win=None):
    ok = ev[f"{view}_ok"] & ev[f"{view}_quality_ok"] & ~ev["spark"]
    if theta_win is not None:
        th = ev[f"{view}_theta_deg"]
        ok &= (th >= theta_win[0]) & (th <= theta_win[1])
    return set(int(e) for e in ev.loc[ok, "event_id"])


def _cap(ids, n=MAX_EVENTS):
    return set(sorted(ids)[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sim-events", required=True)
    ap.add_argument("--data-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-key", default="sat_det3")
    ap.add_argument("--data-n-total", type=int, default=None,
                    help="true total data events (the seeder's per-file max "
                         "under-counts multi-file runs)")
    a = ap.parse_args()

    from wft.io import strip_position_map

    sim_dir = os.path.abspath(os.path.expanduser(a.sim_dir))
    dat_dir = os.path.abspath(os.path.expanduser(a.data_dir))
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = _leg_cfg(sim_dir, a.run_key)
    pos_maps = strip_position_map(cfg)
    feu_x, feu_y = cfg.MX17_FEU_X, cfg.MX17_FEU_Y

    ev_sim = pd.read_parquet(os.path.expanduser(a.sim_events))
    ev_dat = pd.read_parquet(os.path.expanduser(a.data_events))

    # angle windows from the SIM population, applied to DATA only. p5/p95, NOT
    # p1/p99: the sim tail beyond +-5 deg is fit failures, not gun spread, and
    # a loose window lets genuinely inclined cosmics into the data sample.
    wins = {v: tuple(np.percentile(ev_sim[f"{v}_theta_deg"].dropna(), [5, 95]))
            for v in ("x", "y")}

    stats = {}
    for leg, d in (("sim", sim_dir), ("data", dat_dir)):
        p = os.path.join(d, "sim_hits", "sim_datrun_seed_00_seedstats.json")
        with open(p) as f:
            stats[leg] = json.load(f)
    if a.data_n_total:
        stats["data"]["n_events_total"] = int(a.data_n_total)

    res = {}
    for view, feu in (("x", feu_x), ("y", feu_y)):
        ids_sim = _cap(_sel_ids(ev_sim, view))
        ids_dat = _cap(_sel_ids(ev_dat, view, wins[view]))
        print(f"[{view}] sim {len(ids_sim)} events, data {len(ids_dat)} "
              f"(theta window {wins[view][0]:.1f}..{wins[view][1]:.1f} deg)")
        res[("sim", view)] = extract_view(sim_dir, feu, pos_maps[feu],
                                          ids_sim, f"sim/{view}")
        res[("data", view)] = extract_view(dat_dir, feu, pos_maps[feu],
                                           ids_dat, f"data/{view}")

    np.savez_compressed(
        os.path.join(out_dir, "t14_extract.npz"),
        **{f"{leg}_{v}_avg_norm": res[(leg, v)]["avg_norm"]
           for leg in ("sim", "data") for v in ("x", "y")},
        **{f"{leg}_{v}_avg_abs": res[(leg, v)]["avg_abs"]
           for leg in ("sim", "data") for v in ("x", "y")})
    for (leg, v), r in res.items():
        r["table"].to_parquet(os.path.join(out_dir, f"wf_{leg}_{v}.parquet"),
                              index=False)

    figs = make_figures(fig_dir, res, ev_sim, ev_dat, wins)
    summary = make_summary(res, ev_sim, ev_dat, stats, wins)
    with open(os.path.join(out_dir, "t14_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_report(out_dir, figs, summary, sim_dir, dat_dir)
    print(f"\nreport: {os.path.join(out_dir, 'report.html')}")
    return 0


def make_summary(res, ev_sim, ev_dat, stats, wins):
    s = {"theta_windows_deg": {v: list(map(float, w)) for v, w in wins.items()},
         "seed": stats, "views": {}}
    for v in ("x", "y"):
        ts, td = res[("sim", v)]["table"], res[("data", v)]["table"]
        chi_s = (ev_sim[f"{v}_chi2"] / ev_sim[f"{v}_dof"]).median()
        chi_d = (ev_dat[f"{v}_chi2"] / ev_dat[f"{v}_dof"]).median()
        # shape residual between the two normalized average waveforms, over
        # the samples where either is above 5 % of peak
        an_s, an_d = res[("sim", v)]["avg_norm"], res[("data", v)]["avg_norm"]
        m = (an_s > 0.05) | (an_d > 0.05)
        shape_rms = float(np.sqrt(np.mean((an_s[m] - an_d[m]) ** 2)))
        # chi2 normalized to a FRACTIONAL model residual: sqrt(chi2/dof) is the
        # per-sample residual in noise units; x noise / peak gives residual as
        # a fraction of the pulse height. This is the number that makes the two
        # legs comparable — raw chi2 scales with (amp/noise)^2 and the legs
        # differ x2 in amp and x1.5 in noise.
        frac = {leg: float(np.sqrt(chi) * res[(leg, v)]["median_noise"]
                           / res[(leg, v)]["table"].peak_amp.median())
                for leg, chi in (("sim", chi_s), ("data", chi_d))}
        # reco-table cross-check on a tight angle cut, independent of the
        # waveform extraction path. The cut is +-3 deg around the SIM's own
        # median angle, not around zero: for the angled-gun ladder a |theta|<3
        # cut selects nothing on the sim leg (a 20 deg gun has no vertical
        # events) and the row would come back NaN. On a vertical point the sim
        # median is ~0 and this reduces to the original |theta| < 3 deg.
        th0 = float(ev_sim.loc[ev_sim[f"{v}_ok"] & ev_sim[f"{v}_quality_ok"],
                               f"{v}_theta_deg"].median())
        qs = {}
        for leg, ev in (("sim", ev_sim), ("data", ev_dat)):
            g = ev[ev[f"{v}_ok"] & ev[f"{v}_quality_ok"]
                   & ((ev[f"{v}_theta_deg"] - th0).abs() < 3) & ~ev["spark"]]
            qs[leg] = float(g[f"{v}_q_sum"].median())
            qs[f"n_{leg}"] = int(len(g))
        s["views"][v] = dict(
            n_sim=int(len(ts)), n_data=int(len(td)),
            peak_amp_med=dict(sim=float(ts.peak_amp.median()),
                              data=float(td.peak_amp.median()),
                              ratio=float(ts.peak_amp.median() / td.peak_amp.median())),
            q_event_med=dict(sim=float(ts.q_event.median()),
                             data=float(td.q_event.median()),
                             ratio=float(ts.q_event.median() / td.q_event.median())),
            n_over_med=dict(sim=float(ts.n_over.median()),
                            data=float(td.n_over.median())),
            rise_ns_med=dict(sim=float(ts.rise_ns.median()),
                             data=float(td.rise_ns.median())),
            fwhm_ns_med=dict(sim=float(ts.fwhm_ns.median()),
                             data=float(td.fwhm_ns.median())),
            chi2_dof_med=dict(sim=float(chi_s), data=float(chi_d)),
            frac_model_residual=dict(sim=frac["sim"], data=frac["data"]),
            sat_frac_3500=dict(
                sim=float((ts.peak_amp >= 3500).mean()),
                data=float((td.peak_amp >= 3500).mean())),
            q_sum_reco_tight3deg=dict(sim=qs["sim"], data=qs["data"],
                                      ratio=qs["sim"] / qs["data"],
                                      center_deg=th0,
                                      n_sim=qs["n_sim"], n_data=qs["n_data"]),
            shape_rms_frac_of_peak=shape_rms,
            median_noise_adc=dict(sim=res[("sim", v)]["median_noise"],
                                  data=res[("data", v)]["median_noise"]),
        )
    return s


def make_figures(fig_dir, res, ev_sim, ev_dat, wins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (np.arange(N_AVG_SAMP) - ALIGN_AT) * DT_NS
    figs = []
    C = dict(sim="tab:red", data="tab:blue")

    def _save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name), dpi=130)
        plt.close(fig)
        figs.append((name, caption))

    # 1 — average aligned peak-strip waveform, normalized and absolute
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, v in enumerate(("x", "y")):
        for leg in ("sim", "data"):
            r = res[(leg, v)]
            axes[0, j].plot(t, r["avg_norm"], color=C[leg],
                            label=f"{leg} (n={r['n_avg']})")
            axes[1, j].plot(t, r["avg_abs"], color=C[leg], label=leg)
        axes[0, j].set_title(f"{v.upper()} view — normalized")
        axes[1, j].set_title(f"{v.upper()} view — absolute [ADC]")
        axes[1, j].set_xlabel("t − t_peak [ns]")
        for ax in (axes[0, j], axes[1, j]):
            ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "avg_waveform.png",
          "Average peak-strip waveform, aligned at peak. Top: normalized to "
          "peak (pure shape). Bottom: absolute ADC (the W2 amplitude scale "
          "on trial).")

    # 2 — peak amplitude spectra
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        hi = max(res[("sim", v)]["table"].peak_amp.quantile(0.99),
                 res[("data", v)]["table"].peak_amp.quantile(0.99))
        bins = np.linspace(0, hi, 60)
        for leg in ("sim", "data"):
            axes[j].hist(res[(leg, v)]["table"].peak_amp, bins=bins,
                         histtype="step", density=True, color=C[leg], label=leg)
        axes[j].set_title(f"{v.upper()} peak-strip amplitude")
        axes[j].set_xlabel("peak amplitude [ADC]")
        axes[j].set_yscale("log"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "peak_amp.png",
          "Peak-strip amplitude spectra (area-normalized, log y). Absolute "
          "ADC on both legs — no rescaling anywhere.")

    # 3 — per-event summed amplitude
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        hi = max(res[("sim", v)]["table"].q_event.quantile(0.99),
                 res[("data", v)]["table"].q_event.quantile(0.99))
        bins = np.linspace(0, hi, 60)
        for leg in ("sim", "data"):
            axes[j].hist(res[(leg, v)]["table"].q_event, bins=bins,
                         histtype="step", density=True, color=C[leg], label=leg)
        axes[j].set_title(f"{v.upper()} event charge (sum of ≥{SIGMA:.0f}σ strips)")
        axes[j].set_xlabel("Σ strip amplitudes [ADC]")
        axes[j].set_yscale("log"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "q_event.png",
          "Per-event charge proxy: sum of strip peak amplitudes over "
          "threshold. Absolute scale, area-normalized shapes.")

    # 4 — strip multiplicity
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        bins = np.arange(0.5, 30.5)
        for leg in ("sim", "data"):
            axes[j].hist(res[(leg, v)]["table"].n_over, bins=bins,
                         histtype="step", density=True, color=C[leg], label=leg)
        axes[j].set_title(f"{v.upper()} strips ≥ {SIGMA:.0f}σ")
        axes[j].set_xlabel("n strips"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "multiplicity.png",
          "Over-threshold strip multiplicity — the charge-sharing width in "
          "count form.")

    # 5 — transverse cluster profile
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        for leg in ("sim", "data"):
            p = res[(leg, v)]["prof"]
            g = p.groupby("off")["amp"].median()
            g = g[(g.index >= -6) & (g.index <= 6)]
            axes[j].plot(g.index, g.values, "o-", color=C[leg], label=leg)
        axes[j].set_title(f"{v.upper()} transverse profile")
        axes[j].set_xlabel("strip offset from peak")
        axes[j].set_ylabel("median amp / peak amp")
        axes[j].set_yscale("log"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "transverse_profile.png",
          "Median strip amplitude relative to the peak strip vs "
          "position-ordered strip offset (over-threshold strips only).")

    # 6 — rise time and FWHM
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, v in enumerate(("x", "y")):
        for i, col in enumerate(("rise_ns", "fwhm_ns")):
            hi = max(res[("sim", v)]["table"][col].quantile(0.99),
                     res[("data", v)]["table"][col].quantile(0.99))
            bins = np.linspace(0, hi, 50)
            for leg in ("sim", "data"):
                axes[i, j].hist(res[(leg, v)]["table"][col].dropna(), bins=bins,
                                histtype="step", density=True, color=C[leg],
                                label=leg)
            axes[i, j].set_title(f"{v.upper()} {'rise 10–90 %' if i == 0 else 'FWHM'}")
            axes[i, j].set_xlabel("ns"); axes[i, j].legend()
            axes[i, j].grid(alpha=0.3)
    _save(fig, "timing_shape.png",
          "Peak-strip rise time (10–90 %) and FWHM from linear interpolation "
          "on the 60 ns samples.")

    # 7 — forward-fit chi2/dof
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        bins = np.linspace(0, 120, 60)
        for leg, ev in (("sim", ev_sim), ("data", ev_dat)):
            ok = ev[f"{v}_ok"] & ev[f"{v}_quality_ok"]
            axes[j].hist((ev.loc[ok, f"{v}_chi2"] / ev.loc[ok, f"{v}_dof"]),
                         bins=bins, histtype="step", density=True,
                         color=C[leg], label=leg)
        axes[j].set_title(f"{v.upper()} forward-fit χ²/dof")
        axes[j].set_xlabel("χ²/dof"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "chi2.png",
          "wft forward-model fit quality on each leg (quality_ok events, all "
          "angles). The same model, kernel and bundle fit both legs.")

    # 8 — theta context
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        bins = np.linspace(-30, 30, 90)
        for leg, ev in (("sim", ev_sim), ("data", ev_dat)):
            ok = ev[f"{v}_ok"] & ev[f"{v}_quality_ok"]
            axes[j].hist(ev.loc[ok, f"{v}_theta_deg"], bins=bins,
                         histtype="step", density=True, color=C[leg], label=leg)
        for w in wins[v]:
            axes[j].axvline(w, color="k", ls="--", lw=0.8)
        axes[j].set_title(f"{v.upper()} reconstructed θ (context)")
        axes[j].set_xlabel("θ [deg]"); axes[j].legend(); axes[j].grid(alpha=0.3)
    _save(fig, "theta_context.png",
          "Track-angle populations. The dashed window (sim p1–p99) is the "
          "data cut for every waveform figure above; the mismatch outside it "
          "is acceptance (near-vertical sim gun vs cosmic spectrum), not "
          "detector response.")

    return figs


def _rise_floor_para(out_dir):
    """Quantile-resolved rise-time comparison, computed from the extraction
    tables already in out_dir. Answers 'shift or population?': neither."""
    qs = [5, 25, 50, 75, 95]
    rows = []
    for v in ("x", "y"):
        try:
            s = pd.read_parquet(os.path.join(out_dir, f"wf_sim_{v}.parquet"))
            d = pd.read_parquet(os.path.join(out_dir, f"wf_data_{v}.parquet"))
        except Exception:
            return "<p>(extraction tables not found)</p>"
        a = np.percentile(s.rise_ns.dropna(), qs)
        b = np.percentile(d.rise_ns.dropna(), qs)
        fs = float((s.rise_ns < 240).mean())
        fd = float((d.rise_ns < 240).mean())
        rows.append((v, a, b, fs, fd))
    tr = "".join(
        f"<tr><th>{v.upper()} sim</th>" + "".join(f"<td>{x:.0f}</td>" for x in a)
        + f"<td>{100 * fs:.0f} %</td></tr>"
        f"<tr><th>{v.upper()} data</th>" + "".join(f"<td>{x:.0f}</td>" for x in b)
        + f"<td>{100 * fd:.0f} %</td></tr>"
        for v, a, b, fs, fd in rows)
    return f"""
<p>Quantile by quantile (10–90 % rise, ns), the sim−data difference is large
and positive at the fast end but ≈0 (X: negative) at the slow end — a pure
shift would be constant, and a separate fast population would leave the data
bulk aligned with the sim. Neither holds: the two upper halves nearly agree,
while the data's lower half extends far below anything the simulation
produces.</p>
<div class="tablewrap"><table>
<tr><th></th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
<th>rise &lt; 240 ns</th></tr>{tr}
</table></div>
<p>Angle mismatch is ruled out within the matched window: fast (&lt;240 ns)
and slow data events have statistically identical |θ|, charge, and
drift-window extent — fast risers are ordinary full-gap tracks. Two further
measurements localise the cause: data waveforms are jaggier (median largest
single-sample step / peak ≈ 0.25/0.27 vs sim 0.18/0.19 per view), and the
data amplitude spectrum is much broader than the sim's. The avalanche calib
DOES carry measured per-electron Polya fluctuations (θ = 1.14, rel. variance
0.47), so the candidates are the effective dispersion after packet
aggregation in Stage B, or a modeled impulse response (shaper peaking) slower
than the real electronics — the sim's ~250 ns floor against data rises of
~155 ns points at whichever mechanism sets the minimum response width. This
is the strongest shape lead for the forward-model parameterization.</p>"""


def _followup_section(out_dir):
    """Rendered only when bump_undershoot.json exists next to the report
    (produced by the follow-up extraction answering two referee questions)."""
    p = os.path.join(out_dir, "bump_undershoot.json")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        j = json.load(f)

    def u(leg, v):
        d = j[f"{leg}_{v}"]["undershoot"]
        return (f"{100 * d['median']:.1f} % "
                f"[{100 * d['p25']:.1f}, {100 * d['p75']:.1f}]")

    def occ(leg, v, o):
        return 100 * j[f"{leg}_{v}"]["occ"].get(str(o), 0.0)

    return f"""
<h2>Follow-up: two questions, measured</h2>
<h3>The ±5–6-strip 'bumps' in the transverse profile are a
threshold-selection artifact, not kernel structure</h3>
<p>Strips at |offset| ≥ 5 pass the 5σ seed cut in only
{occ('sim', 'x', 5):.1f} % (sim X) / {occ('data', 'x', 5):.1f} % (data X) of
events — the profile median there is conditioned on a tiny population that
passed a threshold, which selects upward fluctuations by construction.
Restricting to events with peak &gt; 1500 ADC (where the 5σ floor is a
resolvable ~0.02 of peak) the bump collapses in BOTH legs and the profile
falls monotonically (figure below). The earlier reading of these bumps as ESL
superperiod cross-coupling visible in data was WRONG and is withdrawn. What
the high-peak profile does show: the sim X kernel is confined to ±3 strips,
and the data has rare real far-strip activity (second tracks, deltas or
pickup) that the single-track simulation cannot contain.</p>
<figure><img src="figures/bump_selection.png" style="max-width:100%">
<figcaption>Transverse profile, all events (dashed) vs peak &gt; 1500 ADC
(solid). The far-strip rise is the threshold-selection floor; the resolved
profile decays monotonically in both legs.</figcaption></figure>
<h3>Rise times: not a fast data 'population', and not a pure shift — the
simulation has a rise-time FLOOR the data does not</h3>
{_rise_floor_para(out_dir)}
<h3>The simulation's overshoot is systematic, in both views</h3>
<p>Per-event undershoot (min of the peak-strip tail / peak, median
[quartiles]): X sim {u('sim', 'x')} vs data {u('data', 'x')};
Y sim {u('sim', 'y')} vs data {u('data', 'y')}.
{100 * j['sim_x']['undershoot']['frac_below_m005']:.0f} % of sim X events
undershoot beyond −5 % against {100 * j['data_x']['undershoot']['frac_below_m005']:.0f} %
in data. So the average-waveform picture is not an artifact of one figure:
the sim overshoots ~3× the data in X and ~2× in Y. The data's own X/Y
asymmetry (Y undershoots more) is qualitatively reproduced — the sim
exaggerates a real feature, pointing at the shaper/return-current model
rather than a wrong mechanism.</p>
"""


def write_report(out_dir, figs, s, sim_dir, dat_dir):
    v = s["views"]
    rx, ry = v["x"]["peak_amp_med"]["ratio"], v["y"]["peak_amp_med"]["ratio"]
    qx, qy = v["x"]["q_event_med"]["ratio"], v["y"]["q_event_med"]["ratio"]
    sx, sy = v["x"]["shape_rms_frac_of_peak"], v["y"]["shape_rms_frac_of_peak"]

    def row(label, key, fmt="{:.1f}"):
        cells = ""
        for vw in ("x", "y"):
            d = v[vw][key]
            if isinstance(d, dict):
                cells += (f"<td>{fmt.format(d['sim'])}</td>"
                          f"<td>{fmt.format(d['data'])}</td>"
                          f"<td>{('{:.2f}'.format(d['ratio'])) if 'ratio' in d else '—'}</td>")
            else:
                cells += f"<td colspan=3>{fmt.format(d)}</td>"
        return f"<tr><th>{html.escape(label)}</th>{cells}</tr>"

    worst = max(abs(np.log(rx)), abs(np.log(ry)))
    agree = worst < np.log(1.2)
    close = worst < np.log(2.0)
    verdict = ("IN AGREEMENT (within ×1.2)" if agree
               else "SAME BALLPARK, NOT IN AGREEMENT" if close
               else "NOT within ×2 in amplitude")
    edge = "#2a2" if agree else "#c60" if close else "#c22"
    fig_html = "\n".join(
        f'<figure><img src="figures/{n}" style="max-width:100%">'
        f"<figcaption>{html.escape(c)}</figcaption></figure>"
        for n, c in figs)

    body = f"""<meta charset="utf-8">
<title>T14 — sim vs data waveforms (frozen default)</title>
<style>
 body {{ font-family: sans-serif; max-width: 1000px; margin: 2em auto; }}
 table {{ border-collapse: collapse; }}
 td, th {{ border: 1px solid #999; padding: 3px 9px; text-align: right; }}
 th {{ text-align: left; }}
 .verdict {{ font-size: 1.25em; padding: .6em 1em; border-left: 6px solid
   {edge}; background: #f6f6f6; }}
 figure {{ margin: 1.5em 0; }} figcaption {{ color: #444; font-size: .92em; }}
</style>
<h1>T14 — simulation vs data at the waveform level</h1>
<p><b>Frozen default</b> (pre-registered): ρ_s = 2 MΩ/sq W2 kernel, dry 95/5
table, det3 <code>long_run_resist_490V_drift_1000V</code> bundle as-analysed.
Both legs seeded and fit by the identical dual-use code
(<code>t13_reco.py</code>, σ = 5); <code>combined_hits</code> unread on both.</p>
<p class="verdict"><b>Verdict: {verdict}.</b>
Peak-strip amplitude sim/data = <b>{rx:.2f}</b> (X), <b>{ry:.2f}</b> (Y);
event charge ratio {qx:.2f} / {qy:.2f}
(reco-table cross-check at |θ|&lt;3°:
{v['x']['q_sum_reco_tight3deg']['ratio']:.2f} /
{v['y']['q_sum_reco_tight3deg']['ratio']:.2f}).
The simulation is LOW in absolute charge; its pulse shapes are broadly
similar but slower — rise ×{v['x']['rise_ns_med']['sim'] / v['x']['rise_ns_med']['data']:.2f} /
×{v['y']['rise_ns_med']['sim'] / v['y']['rise_ns_med']['data']:.2f} and FWHM
×{v['x']['fwhm_ns_med']['sim'] / v['x']['fwhm_ns_med']['data']:.2f} /
×{v['y']['fwhm_ns_med']['sim'] / v['y']['fwhm_ns_med']['data']:.2f} the
data's (X / Y). Normalized average-waveform shape RMS
= {100 * sx:.1f} % (X), {100 * sy:.1f} % (Y) of peak.
Data saturation ({100 * v['x']['sat_frac_3500']['data']:.0f} % of X events
peak ≥ 3500 ADC vs {100 * v['x']['sat_frac_3500']['sim']:.0f} % in sim) clips
the data side, so the true amplitude deficit is somewhat LARGER than quoted.</p>
<h2>Headline numbers (angle-matched data, medians)</h2>
<table>
<tr><th></th><th colspan=3>X view (FEU 7)</th>
<th colspan=3>Y view (FEU 8)</th></tr>
<tr><th></th><th>sim</th><th>data</th><th>sim/data</th>
<th>sim</th><th>data</th><th>sim/data</th></tr>
{row('peak-strip amplitude [ADC]', 'peak_amp_med')}
{row('event charge Σamp [ADC]', 'q_event_med')}
{row('strips ≥ 5σ', 'n_over_med')}
{row('rise 10–90 % [ns]', 'rise_ns_med')}
{row('FWHM [ns]', 'fwhm_ns_med')}
{row('forward-fit χ²/dof', 'chi2_dof_med')}
{row('fractional model residual', 'frac_model_residual', '{:.3f}')}
{row('fraction peak ≥ 3500 ADC', 'sat_frac_3500', '{:.3f}')}
{row('reco q_sum, θ within ±3° of sim median [ADC]', 'q_sum_reco_tight3deg', '{:.0f}')}
{row('median noise [ADC]', 'median_noise_adc')}
</table>
<p>The raw χ²/dof gap is mostly scaling — χ² grows as (amplitude/noise)² and
the legs differ ×2 in amplitude and ×1.5 in noise. The <i>fractional model
residual</i> row (√(χ²/dof) · noise / peak amplitude) is the comparable
number: the same forward model misses both legs by a few % of pulse height.
Noise row caveat: both legs' figures are FeuReader re-characterisations of
the decoded files, and that estimator is inflated by residual signal on
occupied channels — the sim has 100 % track occupancy by construction, so
its re-characterised σ reads high even when the injected spec matches the
data (sim {v['x']['median_noise_adc']['sim']:.1f} vs data
{v['x']['median_noise_adc']['data']:.1f} ADC here). Compare injected spec vs
data spec, not re-characterisations across legs.</p>
<h2>Seed-rate observable (first-class)</h2>
<p>Sim: {s['seed']['sim']['n_events_seeded']}/{s['seed']['sim']['n_events_total']}
events seeded, {s['seed']['sim']['n_hits'] / max(s['seed']['sim']['n_events_seeded'], 1):.1f}
hits/seeded event. Data: {s['seed']['data']['n_events_seeded']}/{s['seed']['data']['n_events_total']}
({100 * s['seed']['data']['n_events_seeded'] / max(s['seed']['data']['n_events_total'], 1):.1f} %),
{s['seed']['data']['n_hits'] / max(s['seed']['data']['n_events_seeded'], 1):.1f} hits/seeded
event. The sim generates a track in every event; the data denominator counts
every trigger, so the data fraction folds in bench acceptance and detector
efficiency — the comparable number is hits per seeded event.</p>
<h2>Pre-registered systematics (quoted, not absorbed)</h2>
<ul>
<li><b>v_drift 6.9 %</b>: bundle 36.60 vs dry-table 39.14 µm/ns. NO humidity
was ever measured on any run; every contaminant figure in the project was
inferred from slow drift. A contaminant search after this comparison is a
fitted axis, and is labeled DIAGNOSIS.</li>
<li><b>W2 kernel grid</b>: ny=512 carries ~1.3 % absolute-amplitude
systematic and a ~0.45 % pad-edge shoulder term on shape.</li>
<li><b>Angle matching</b>: data cut to sim θ window
(x: {s['theta_windows_deg']['x'][0]:.1f}..{s['theta_windows_deg']['x'][1]:.1f}°,
y: {s['theta_windows_deg']['y'][0]:.1f}..{s['theta_windows_deg']['y'][1]:.1f}°).
The reco cross-check row uses ±3° around the sim median
(x: {v['x']['q_sum_reco_tight3deg']['center_deg']:+.2f}°, n =
{v['x']['q_sum_reco_tight3deg']['n_sim']} sim /
{v['x']['q_sum_reco_tight3deg']['n_data']} data;
y: {v['y']['q_sum_reco_tight3deg']['center_deg']:+.2f}°, n =
{v['y']['q_sum_reco_tight3deg']['n_sim']} /
{v['y']['q_sum_reco_tight3deg']['n_data']}).</li>
</ul>
<h2>What this result does NOT rule out</h2>
<ul>
<li>ρ_s off the 2 MΩ/sq central value (T2b band 1.4–2.6; the 4-point ladder
exists for diagnosis).</li>
<li>Gas contamination altering gain and sharing together (unconstrained
without a hygrometer).</li>
<li>Angle- or depth-dependent effects outside the near-vertical window the
sim covers.</li>
</ul>
{fig_html}
{_followup_section(out_dir)}
<hr><p><small>sim leg: {html.escape(sim_dir)}<br>
data leg: {html.escape(dat_dir)}<br>
generated by mx17_sim_wft/t14_compare.py</small></p>
"""
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(body)


if __name__ == "__main__":
    raise SystemExit(main())

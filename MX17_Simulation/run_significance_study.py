"""
X17 discovery-significance study: Configuration A vs Configuration B
====================================================================
Quantifies the trade-off between the two stack orderings:

  Config A (standard):       MM → SW → LS1 → LS2 → BS
      high double-trigger efficiency (~94% of both-MM pairs),
      but full calorimetry (both particles in BS) only ~26%.
  Config B (back-scint first): MM → SW → BS → LS1 → LS2
      trigger = SW ∧ BS → only ~57% trigger efficiency,
      but full calorimetry ~42%.

Significance metric
-------------------
Binned Asimov profile-likelihood significance (Cowan, Cranmer, Gross,
Vitells, EPJ C71 (2011) 1554):

    Z = sqrt( 2 Σ_i [ (s_i + b_i) ln(1 + s_i/b_i) − s_i ] )

summed over bins of a discriminating observable; with an optional
fractional background systematic the per-bin term is replaced by the
profiled version (eq. 20 of the same paper).  s_i and b_i are the
EXPECTED X17 and IPC counts in the full exposure:

    N_produced(X17) = X17_PER_PULSE × TOTAL_PULSES
    N_produced(IPC) = IPC_PER_PULSE × TOTAL_PULSES
    s_i = N_produced × (MC pairs in bin i) / (MC pairs produced)

Channels
--------
  angle : reconstructed opening angle of double-triggered MM pairs
          (uses the Geant4-response-smeared directions, i.e. the realistic
          ~13–15° angular resolution).  Large sample, no calorimetry needed.
  mass  : reconstructed invariant mass of full-calorimetry pairs (both
          particles hit BS on their MM arm).  Small sample, sharper
          discriminant.
  combo : quadrature sum of `mass` and the angle channel restricted to the
          non-BS remainder (expected-count subtraction — approximate, the
          two samples overlap only at expectation level).

Run:  venv/bin/python run_significance_study.py
"""

import io
import os
import sys
import contextlib
import dataclasses

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import (MicromegasSimulation, IPC_PER_PULSE,
                            X17_PER_PULSE, TOTAL_PULSES, RUN_DAYS)
from detector_config import cfg_A, cfg_B, N_WORKERS

N_MC_EVENTS = 200_000      # MC readout windows per (config, source) run

ANGLE_BINS = np.arange(0.0, 184.0, 4.0)     # deg
MASS_BINS  = np.arange(0.0, 26.0, 1.0)      # MeV
BG_SYST    = 0.10                           # fractional background systematic


# ── Significance ─────────────────────────────────────────────────────────────

def z_asimov(s, b, sigma_rel=0.0):
    """Binned Asimov significance. s, b = expected counts per bin.

    sigma_rel > 0 adds a per-bin fractional background uncertainty via the
    profiled formula (Cowan et al. eq. 20). Bins with b == 0 are floored to
    the smallest nonzero bin to avoid spurious infinite significance from
    MC holes.
    """
    s = np.asarray(s, float).copy()
    b = np.asarray(b, float).copy()
    if (b > 0).any():
        b[b <= 0] = b[b > 0].min()
    use = s > 0
    if not use.any():
        return 0.0
    s, b = s[use], b[use]
    if sigma_rel <= 0:
        z2 = 2.0 * ((s + b) * np.log1p(s / b) - s)
    else:
        sig2 = (sigma_rel * b) ** 2
        n = s + b
        t1 = n * np.log(n * (b + sig2) / (b**2 + n * sig2))
        t2 = (b**2 / sig2) * np.log1p(sig2 * s / (b * (b + sig2)))
        z2 = 2.0 * (t1 - t2)
    return float(np.sqrt(np.clip(z2, 0, None).sum()))


def z_profiled(s, b, edges, n_shape=0):
    """Asimov discovery significance with the background NORMALISATION AND
    SHAPE profiled (fit to the data), instead of assumed perfectly known.

    Background model:  b_i(θ) = b_i^MC · exp( θ₀ + θ₁·t_i + θ₂·t_i² + … )
    with t the bin centre mapped to [−1, 1]; n_shape = number of θ parameters
    (0 = fixed background → identical to z_asimov stat-only).

    Discovery test statistic on the Asimov dataset n_i = s_i + b_i:
        q₀ = 2 Σ_i [ n_i ln(n_i / b̂_i) − (n_i − b̂_i) ],   Z = √q₀
    where b̂ is the best-fit background-only model.  Because the signal+bkg
    model reproduces the Asimov data exactly, this is the full profile
    likelihood ratio.  The IPC-dominated sidebands (low angle / low mass)
    constrain θ in the fit — i.e. this emulates a data-driven background fit.
    """
    from scipy.optimize import minimize

    s = np.asarray(s, float).copy()
    b = np.asarray(b, float).copy()
    if (b > 0).any():
        b[b <= 0] = b[b > 0].min()
    n = s + b
    cen = 0.5 * (edges[:-1] + edges[1:])
    t = (cen - cen.mean()) / (0.5 * (cen[-1] - cen[0]))
    T = np.vstack([t**k for k in range(1, n_shape + 1)]) if n_shape >= 1 \
        else np.zeros((0, len(cen)))

    def nll(theta):
        m = b * np.exp(theta @ T) if n_shape else b
        m = np.clip(m, 1e-12, None)
        return float((m - n * np.log(m)).sum())

    if n_shape == 0:
        m_hat = b
    else:
        res = minimize(nll, np.zeros(n_shape), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 20000})
        m_hat = np.clip(b * np.exp(res.x @ T), 1e-12, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > 0, n * np.log(n / m_hat), 0.0) - (n - m_hat)
    return float(np.sqrt(np.clip(2.0 * terms.sum(), 0, None)))


def best_window(s_hist, b_hist, edges):
    """(lo, hi, S, B, S/sqrt(B)) of the contiguous window maximising S/√B."""
    best = (edges[0], edges[-1], 0.0, 0.0, 0.0)
    n = len(s_hist)
    cs, cb = np.concatenate([[0], np.cumsum(s_hist)]), np.concatenate([[0], np.cumsum(b_hist)])
    for i in range(n):
        for j in range(i + 1, n + 1):
            S, B = cs[j] - cs[i], cb[j] - cb[i]
            if B <= 0 or S <= 0:
                continue
            zsb = S / np.sqrt(B)
            if zsb > best[4]:
                best = (edges[i], edges[j], S, B, zsb)
    return best


# ── MC running ───────────────────────────────────────────────────────────────

def run_source(cfg, source, seed):
    """Run one config with only `source` pairs; return per-pair-normalised
    reco samples and the produced-pair count."""
    kw = dict(n_signal=0.0, n_background_pairs=0.0, n_random=0.0,
              n_events=N_MC_EVENTS, seed=seed)
    if source == "signal":
        kw["n_signal"] = 1.0
    else:
        kw["n_background_pairs"] = 1.0
    c = dataclasses.replace(cfg, **kw)
    sim = MicromegasSimulation(c)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(n_workers=N_WORKERS)

    key = "signal" if source == "signal" else "background_pair"
    n_prod = (sim.pair_stats["signal_produced"] if source == "signal"
              else sim.pair_stats["bg_produced"])

    sel_d  = np.array([s == key for s in sim.pair_sources_double], bool)
    sel_bs = np.array([s == key for s in sim.pair_sources_both_bs], bool)
    return {
        "n_produced":   n_prod,
        "angles_trig":  np.asarray(sim.angular_separations_double)[sel_d]
                        if len(sel_d) else np.array([]),
        "angles_bs":    np.asarray(sim.angular_separations_both_bs)[sel_bs]
                        if len(sel_bs) else np.array([]),
        "masses_bs":    np.asarray(sim.inv_masses_both_bs)[sel_bs]
                        if len(sel_bs) else np.array([]),
        "pair_stats":   dict(sim.pair_stats),
    }


def expected_hist(samples, bins, n_produced_mc, n_produced_exp):
    """Histogram of MC samples scaled to expected counts in the exposure."""
    h, _ = np.histogram(samples, bins=bins)
    return h * (n_produced_exp / max(n_produced_mc, 1))


# ── Main study ───────────────────────────────────────────────────────────────

def main():
    n_x17_exp = X17_PER_PULSE * TOTAL_PULSES
    n_ipc_exp = IPC_PER_PULSE * TOTAL_PULSES
    print(f"Exposure: {RUN_DAYS} days = {TOTAL_PULSES:,} pulses")
    print(f"  produced X17 pairs : {n_x17_exp:8.0f}")
    print(f"  produced IPC pairs : {n_ipc_exp:8.0f}")
    print(f"  MC events per run  : {N_MC_EVENTS:,}\n")

    results = {}
    for name, cfg, seed0 in [("A", cfg_A, 101), ("B", cfg_B, 202)]:
        print(f"── Config {name}: running signal + IPC MC ...")
        sig = run_source(cfg, "signal", seed0)
        bg  = run_source(cfg, "ipc",    seed0 + 1)

        # Expected-count histograms over the exposure
        s_ang = expected_hist(sig["angles_trig"], ANGLE_BINS,
                              sig["n_produced"], n_x17_exp)
        b_ang = expected_hist(bg["angles_trig"],  ANGLE_BINS,
                              bg["n_produced"],  n_ipc_exp)
        s_angbs = expected_hist(sig["angles_bs"], ANGLE_BINS,
                                sig["n_produced"], n_x17_exp)
        b_angbs = expected_hist(bg["angles_bs"],  ANGLE_BINS,
                                bg["n_produced"],  n_ipc_exp)
        s_mas = expected_hist(sig["masses_bs"], MASS_BINS,
                              sig["n_produced"], n_x17_exp)
        b_mas = expected_hist(bg["masses_bs"],  MASS_BINS,
                              bg["n_produced"],  n_ipc_exp)

        # Channel significances (stat-only and with background systematic)
        z_angle      = z_asimov(s_ang, b_ang)
        z_angle_syst = z_asimov(s_ang, b_ang, BG_SYST)
        z_mass       = z_asimov(s_mas, b_mas)
        z_mass_syst  = z_asimov(s_mas, b_mas, BG_SYST)

        # Profiled ladder: background norm / shape fit from the (Asimov) data
        z_ang_prof = {k: z_profiled(s_ang, b_ang, ANGLE_BINS, k)
                      for k in (0, 1, 2, 3)}
        z_mas_prof = {k: z_profiled(s_mas, b_mas, MASS_BINS, k)
                      for k in (0, 1, 2, 3)}

        # Combined: mass channel on the BS sample + angle channel on the
        # non-BS remainder (expected-count subtraction; approximate overlap
        # removal — BS pairs are a subset of triggered pairs to good accuracy)
        s_ang_nbs = np.clip(s_ang - s_angbs, 0, None)
        b_ang_nbs = np.clip(b_ang - b_angbs, 0, None)
        z_comb      = float(np.hypot(z_mass,      z_asimov(s_ang_nbs, b_ang_nbs)))
        z_comb_syst = float(np.hypot(z_mass_syst,
                                     z_asimov(s_ang_nbs, b_ang_nbs, BG_SYST)))

        w_ang = best_window(s_ang, b_ang, ANGLE_BINS)
        w_mas = best_window(s_mas, b_mas, MASS_BINS)

        results[name] = dict(
            s_ang=s_ang, b_ang=b_ang, s_mas=s_mas, b_mas=b_mas,
            z_angle=z_angle, z_angle_syst=z_angle_syst,
            z_mass=z_mass, z_mass_syst=z_mass_syst,
            z_comb=z_comb, z_comb_syst=z_comb_syst,
            z_ang_prof=z_ang_prof, z_mas_prof=z_mas_prof,
            w_ang=w_ang, w_mas=w_mas,
            sig_stats=sig["pair_stats"], n_sig_trig=s_ang.sum(),
            n_bg_trig=b_ang.sum(), n_sig_bs=s_mas.sum(), n_bg_bs=b_mas.sum(),
        )

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"X17 DISCOVERY SIGNIFICANCE — {RUN_DAYS}-day exposure "
          f"(Asimov profile likelihood)")
    print(f"{'='*74}")
    print(f"{'Metric':<44}{'Config A':>14}{'Config B':>14}")
    print("─" * 74)
    ra, rb = results["A"], results["B"]
    rows = [
        ("X17 pairs in angle channel (dbl trig)", "n_sig_trig", ".0f"),
        ("IPC pairs in angle channel",            "n_bg_trig",  ".0f"),
        ("X17 pairs in mass channel (both BS)",   "n_sig_bs",   ".0f"),
        ("IPC pairs in mass channel",             "n_bg_bs",    ".0f"),
        ("Z — angle channel (stat only)",         "z_angle",    ".2f"),
        (f"Z — angle channel ({BG_SYST:.0%} bg syst)", "z_angle_syst", ".2f"),
        ("Z — mass channel (stat only)",          "z_mass",     ".2f"),
        (f"Z — mass channel ({BG_SYST:.0%} bg syst)",  "z_mass_syst",  ".2f"),
        ("Z — combined (stat only)",              "z_comb",     ".2f"),
        (f"Z — combined ({BG_SYST:.0%} bg syst)", "z_comb_syst", ".2f"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<44}{ra[key]:>14{fmt}}{rb[key]:>14{fmt}}")
    print("─" * 74)
    print("Background-fit (profiled) ladder — how much Z survives when the")
    print("IPC spectrum must be fit from the data instead of assumed known:")
    prof_rows = [
        ("  angle: bkg fixed (= stat only)",      "z_ang_prof", 0),
        ("  angle: + free normalisation",          "z_ang_prof", 1),
        ("  angle: + norm + tilt",                 "z_ang_prof", 2),
        ("  angle: + norm + tilt + curvature",     "z_ang_prof", 3),
        ("  mass : bkg fixed (= stat only)",       "z_mas_prof", 0),
        ("  mass : + norm + tilt + curvature",     "z_mas_prof", 3),
    ]
    for label, key, k in prof_rows:
        print(f"{label:<44}{ra[key][k]:>13.2f}σ{rb[key][k]:>13.2f}σ")
    print("─" * 74)
    for name, r in results.items():
        lo, hi, S, B, zsb = r["w_ang"]
        print(f"Config {name} best angle window: {lo:.0f}–{hi:.0f}°  "
              f"S={S:.0f}  B={B:.0f}  S/√B={zsb:.2f}")
        lo, hi, S, B, zsb = r["w_mas"]
        print(f"Config {name} best mass window : {lo:.0f}–{hi:.0f} MeV  "
              f"S={S:.0f}  B={B:.0f}  S/√B={zsb:.2f}")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    a_cen = 0.5 * (ANGLE_BINS[:-1] + ANGLE_BINS[1:])
    m_cen = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])

    for row, (name, r) in enumerate(results.items()):
        ax = axes[row, 0]
        ax.step(a_cen, r["b_ang"], where="mid", color="#4a90d9", lw=2,
                label=f"IPC  (N={r['b_ang'].sum():.0f})")
        ax.step(a_cen, r["s_ang"] * 10, where="mid", color="#e84040", lw=2,
                label=f"X17 ×10  (N={r['s_ang'].sum():.0f})")
        imax = int(np.argmax(r["s_ang"]))
        ax.annotate("X17 SCALED ×10\n(true size is 10× smaller)",
                    xy=(a_cen[imax], r["s_ang"][imax] * 10),
                    xytext=(20, r["s_ang"][imax] * 10 * 1.15),
                    fontsize=9, color="#e84040", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e84040", lw=1.2))
        ax.set_xlabel("Reco opening angle [deg]")
        ax.set_ylabel(f"Expected pairs / {RUN_DAYS} d / 4°")
        ax.set_title(f"Config {name} — angle channel (double trigger)\n"
                     f"Z = {r['z_angle']:.2f}σ stat, "
                     f"{r['z_angle_syst']:.2f}σ w/ {BG_SYST:.0%} syst")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.step(m_cen, r["b_mas"], where="mid", color="#4a90d9", lw=2,
                label=f"IPC  (N={r['b_mas'].sum():.0f})")
        ax.step(m_cen, r["s_mas"] * 10, where="mid", color="#e84040", lw=2,
                label=f"X17 ×10  (N={r['s_mas'].sum():.0f})")
        ax.axvline(16.8, color="red", ls=":", lw=1)
        ax.text(0.03, 0.9, "X17 SCALED ×10", transform=ax.transAxes,
                fontsize=9, color="#e84040", fontweight="bold")
        ax.set_xlabel("Reco invariant mass [MeV]")
        ax.set_ylabel(f"Expected pairs / {RUN_DAYS} d / MeV")
        ax.set_title(f"Config {name} — mass channel (both back-scint)\n"
                     f"Z = {r['z_mass']:.2f}σ stat, "
                     f"{r['z_mass_syst']:.2f}σ w/ {BG_SYST:.0%} syst")
        ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    labels = ["angle", "mass", "combined"]
    x = np.arange(3); w = 0.35
    for i, (name, r) in enumerate(results.items()):
        vals  = [r["z_angle"], r["z_mass"], r["z_comb"]]
        vals_s = [r["z_angle_syst"], r["z_mass_syst"], r["z_comb_syst"]]
        col = "#2ca02c" if name == "A" else "#9467bd"
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, alpha=0.85,
               edgecolor="k", lw=0.5, label=f"Config {name} (stat)")
        ax.bar(x + (i - 0.5) * w, vals_s, w * 0.55, color="k", alpha=0.35,
               label=f"Config {name} ({BG_SYST:.0%} syst)")
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Asimov Z  [σ]")
    ax.set_title(f"Discovery significance, {RUN_DAYS} days")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    txt = [f"Exposure: {RUN_DAYS} d, {TOTAL_PULSES:,} pulses",
           f"Produced: X17 {n_x17_exp:.0f}, IPC {n_ipc_exp:.0f}",
           f"MC: {N_MC_EVENTS:,} windows / source / config", "",
           "Geant4 response: direction MS smearing +",
           "calorimetric mass (old-geometry tables)", ""]
    for name, r in results.items():
        st = r["sig_stats"]
        txt.append(f"Config {name}: both-MM eff "
                   f"{st['signal_efficiency']*100:.1f}%, "
                   f"trig eff {st['trigger_efficiency']*100:.1f}%, "
                   f"calor. compl. "
                   f"{st['calorimetry_complete_fraction']*100:.1f}%")
    ax.text(0.02, 0.95, "\n".join(txt), va="top", fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    fig.suptitle("X17 vs IPC discovery significance — Configuration A vs B",
                 fontsize=14)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "significance")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "significance_study.png")
    fig.savefig(out, dpi=130)
    print(f"\n[Output] Saved {out}")

    # ── Stacked, UNscaled view: what the data will actually look like ────
    fig2 = plt.figure(figsize=(15, 8))
    gs = fig2.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.22)
    for col, (name, r) in enumerate(results.items()):
        ax  = fig2.add_subplot(gs[0, col])
        axr = fig2.add_subplot(gs[1, col], sharex=ax)
        b, s = r["b_ang"], r["s_ang"]
        ax.bar(a_cen, b, width=4.0, color="#9dc3e6", edgecolor="#4a90d9",
               lw=0.4, label=f"IPC  (N={b.sum():.0f})")
        ax.bar(a_cen, s, width=4.0, bottom=b, color="#e84040",
               edgecolor="#a02020", lw=0.4,
               label=f"X17 on top, UNSCALED  (N={s.sum():.0f})")
        tot = s + b
        ax.errorbar(a_cen, tot, yerr=np.sqrt(tot), fmt="none",
                    ecolor="k", elinewidth=0.7, capsize=0, alpha=0.6,
                    label="±√N (expected stat. error)")
        ax.set_ylabel(f"Expected pairs / {RUN_DAYS} d / 4°")
        ax.set_title(f"Config {name} — angle channel, stacked & unscaled\n"
                     "(the X17 sliver is the entire signal)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            zbin = np.where(b > 0, s / np.sqrt(b), 0.0)
        axr.bar(a_cen, zbin, width=4.0, color="#e84040", alpha=0.8)
        axr.axhline(0, color="k", lw=0.6)
        axr.set_xlabel("Reco opening angle [deg]")
        axr.set_ylabel("S/√B per bin")
        axr.grid(alpha=0.3)

    fig2.suptitle(
        "Unscaled stacked spectra — the X17 excess as it will appear in data\n"
        "Significance comes from the COHERENT excess across many bins "
        "(per-bin S/√B below), not from any single visible bump",
        fontsize=12)
    out2 = os.path.join(out_dir, "significance_stacked.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"[Output] Saved {out2}")


if __name__ == "__main__":
    main()

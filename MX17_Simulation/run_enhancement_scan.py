"""
Discovery-power enhancement scan
================================
Toy study on top of the Geant4 response file: what analysis-level cuts buy
significance, given the irreducible target-wall multiple scattering?

For each generated pair (X17 or IPC, full relativistic kinematics):
  1. smear both directions with the measured P(ψ | KE) tables,
  2. require both (smeared) tracks to hit an MM panel (geometric acceptance),
  3. sample reconstructed energies from the calorimeter response,
  4. compute the reconstructed invariant mass (exact formula).

Then scan a symmetric-energy selection — min(E1_reco, E2_reco) > cut — and
report the Asimov Z of the mass spectrum vs cut value.  Physics: a min-energy
cut removes the asymmetric (soft) IPC pairs AND selects pairs with better
angular resolution (θ₀ ∝ 1/p), sharpening the X17 mass peak; the cost is
signal statistics.  X17 kinematics bound min KE ≳ 3.9 MeV, so cuts below that
are nearly free of signal loss.

The absolute normalisation uses the same exposure constants as
run_significance_study.py, scaled by a flat trigger-efficiency factor
(channel definition: double-trigger + full calorimetry assumed for all
accepted pairs — optimistic on calorimetric coverage, so read the RELATIVE
trend vs cut, not the absolute Z).

Run:  venv/bin/python run_enhancement_scan.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import (X17PhysicsSpectrum, IPCPhysicsSpectrum, Detector,
                            IPC_PER_PULSE, X17_PER_PULSE, TOTAL_PULSES,
                            RUN_DAYS)
from geant4_response import Geant4Response
from run_significance_study import z_asimov, MASS_BINS
from detector_config import GEO, G4_RESPONSE

N_PAIRS    = 120_000          # generated pairs per species
TRIG_EFF   = 0.93             # flat double-trigger factor (from Config A study)
ECUT_SCAN  = np.arange(0.0, 9.1, 0.5)    # MeV, min reco-energy cut
SEED       = 314159


def simulate_species(spec, resp, panels, n_pairs):
    """Generate, smear, accept and reconstruct pairs of one species.
    Returns dict of arrays for accepted pairs + the produced count."""
    pairs = spec.sample_pairs(n_pairs)
    out = {k: [] for k in ("mass_reco", "angle_reco", "e1", "e2", "angle_true")}
    origin = np.zeros(3)

    def hits_mm(d):
        return any(p.intersect(origin, d)[1] is not None for p in panels)

    for d1, d2, m_true, ang_true, ke1, ke2 in pairs:
        s1 = resp.smear_direction(d1, ke1)
        s2 = resp.smear_direction(d2, ke2)
        if not (hits_mm(s1) and hits_mm(s2)):
            continue
        e1 = resp.reco_energy(ke1, "electron")
        e2 = resp.reco_energy(ke2, "positron")
        cos = float(np.clip(np.dot(s1, s2), -1, 1))
        p1 = np.sqrt(max(e1**2 - 0.511**2, 0))
        p2 = np.sqrt(max(e2**2 - 0.511**2, 0))
        m2 = 2 * 0.511**2 + 2 * (e1 * e2 - p1 * p2 * cos)
        out["mass_reco"].append(np.sqrt(max(m2, 0.0)))
        out["angle_reco"].append(np.degrees(np.arccos(cos)))
        out["angle_true"].append(ang_true)
        out["e1"].append(e1)
        out["e2"].append(e2)
    return {k: np.asarray(v) for k, v in out.items()}, n_pairs


def main():
    np.random.seed(SEED)
    resp = Geant4Response(G4_RESPONSE, direction_estimator="first",
                          energy_method="ls", energy_corrected=True)
    # MM back-face planes, matching the fast-MC convention
    d_plane = GEO["detector_distance"] + GEO["mm_drift_gap"]
    panels = [Detector(s, d_plane, *GEO["detector_size"], "mm")
              for s in range(4)]

    print(f"Generating {N_PAIRS:,} pairs per species ...")
    x17, n_x17_mc = simulate_species(X17PhysicsSpectrum(), resp, panels, N_PAIRS)
    ipc, n_ipc_mc = simulate_species(IPCPhysicsSpectrum(), resp, panels, N_PAIRS)
    print(f"  accepted: X17 {len(x17['mass_reco']):,}  "
          f"IPC {len(ipc['mass_reco']):,}")

    n_x17_exp = X17_PER_PULSE * TOTAL_PULSES * TRIG_EFF
    n_ipc_exp = IPC_PER_PULSE * TOTAL_PULSES * TRIG_EFF
    w_x17 = n_x17_exp / n_x17_mc
    w_ipc = n_ipc_exp / n_ipc_mc

    min_e_x17 = np.minimum(x17["e1"], x17["e2"])
    min_e_ipc = np.minimum(ipc["e1"], ipc["e2"])

    z_mass, z_angle, n_sig, n_bg, s68_list = [], [], [], [], []
    ang_bins = np.arange(0, 184, 4.0)
    for cut in ECUT_SCAN:
        sx = min_e_x17 > cut
        si = min_e_ipc > cut
        hs, _ = np.histogram(x17["mass_reco"][sx], MASS_BINS)
        hb, _ = np.histogram(ipc["mass_reco"][si], MASS_BINS)
        z_mass.append(z_asimov(hs * w_x17, hb * w_ipc))
        ha_s, _ = np.histogram(x17["angle_reco"][sx], ang_bins)
        ha_b, _ = np.histogram(ipc["angle_reco"][si], ang_bins)
        z_angle.append(z_asimov(ha_s * w_x17, ha_b * w_ipc))
        n_sig.append(sx.sum() * w_x17)
        n_bg.append(si.sum() * w_ipc)
        d = (x17["angle_reco"] - x17["angle_true"])[sx]
        s68_list.append(0.5 * (np.percentile(d, 84) - np.percentile(d, 16))
                        if sx.sum() > 50 else np.nan)

    z_mass, z_angle = np.array(z_mass), np.array(z_angle)
    n_sig, n_bg, s68_arr = np.array(n_sig), np.array(n_bg), np.array(s68_list)

    i_best = int(np.nanargmax(z_mass))
    print(f"\nBest mass-channel cut: min reco E > {ECUT_SCAN[i_best]:.1f} MeV  "
          f"→ Z = {z_mass[i_best]:.2f}σ "
          f"(no cut: {z_mass[0]:.2f}σ, gain ×{z_mass[i_best]/z_mass[0]:.2f})")
    print(f"  S = {n_sig[i_best]:.0f}, B = {n_bg[i_best]:.0f}, "
          f"angular σ68 {s68_arr[0]:.1f}° → {s68_arr[i_best]:.1f}°")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(ECUT_SCAN, z_mass, "o-", color="#e84040", lw=2,
            label="Z (mass channel)")
    ax.plot(ECUT_SCAN, z_angle, "s--", color="#1f77b4", lw=2,
            label="Z (angle channel)")
    ax.axvline(3.9, color="grey", ls=":", lw=1)
    ax.text(3.95, ax.get_ylim()[0] + 0.1, "X17 kinematic\nmin-KE bound",
            fontsize=8, color="grey")
    ax.set_xlabel("Cut: min(E1, E2) reco  [MeV]")
    ax.set_ylabel("Asimov Z  [σ]")
    ax.set_title(f"Significance vs symmetric-energy cut\n"
                 f"({RUN_DAYS}-day exposure, trigger eff ×{TRIG_EFF})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(ECUT_SCAN, n_sig, "o-", color="#e84040", lw=2, label="X17 (S)")
    ax2.plot(ECUT_SCAN, n_bg, "s--", color="#4a90d9", lw=2, label="IPC (B)")
    ax.set_xlabel("Cut: min(E1, E2) reco  [MeV]")
    ax.set_ylabel("Expected X17 pairs", color="#e84040")
    ax2.set_ylabel("Expected IPC pairs", color="#4a90d9")
    ax.set_title("Sample sizes vs cut\n(IPC soft-asymmetric pairs removed first)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    v = ~np.isnan(s68_arr)
    ax.plot(ECUT_SCAN[v], s68_arr[v], "o-", color="#2ca02c", lw=2)
    ax.set_xlabel("Cut: min(E1, E2) reco  [MeV]")
    ax.set_ylabel("X17 opening-angle σ68  [deg]")
    ax.set_title("Angular resolution of surviving X17 pairs\n"
                 "(θ₀ ∝ 1/p: the cut buys resolution too)")
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    fig.suptitle("Enhancement scan: symmetric-energy selection "
                 "(Geant4-response toy, MM-double acceptance)", fontsize=13)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "significance")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "enhancement_scan.png")
    fig.savefig(out, dpi=130)
    print(f"[Output] Saved {out}")


if __name__ == "__main__":
    main()

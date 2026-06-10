"""
IPC background decomposition: He-3 vs target-material captures + combinatorics
==============================================================================
Answers three questions for the MX17 experiment (E_n < 1 keV at n_TOF):

  1. CAPTURE BUDGET — where do beam neutrons get captured?  The beam enters
     along the He-3 cylinder axis through 1.2 mm CFRP + 0.5 mm Al end caps.
     All capture cross sections are 1/v below 1 keV; the n_TOF flux is taken
     iso-lethargic (dΦ/dlnE = const) from 25 meV to 1 keV.

  2. PAIR YIELD PER SOURCE — e⁺e⁻ pairs per incident neutron from
       (a) He-3 IPC:   ³He(n,γ)⁴He branch (σ_γ/σ_np ≈ 1.0e-8, Wolfs 1989)
                       × IPC coefficient of the 20.58 MeV transition (~3.6e-3)
       (b) Al-wall:    ²⁷Al(n,γ)²⁸Al (σ_th = 0.231 b, Q = 7.725 MeV) — every
                       capture radiates a γ cascade; each γ > 2mₑ can convert
                       internally (IPC) or externally (target walls, ~0.7% X₀)
       (c) CFRP:       ¹²C(n,γ)¹³C (σ_th = 3.5 mb, Q = 4.95 MeV) and
                       ¹H(n,γ)D in the epoxy (σ_th = 0.332 b, Eγ = 2.22 MeV)
     The punchline: wall captures make ~10⁷ more pairs *per capture* than
     He-3 (no 1e-8 radiative suppression), so even tiny wall capture
     fractions dominate raw pair production.

  3. WHAT TRIGGERS — wall pairs have ≤ 7.7 MeV total; the measured
     per-particle trigger curve (Geant4: logistic, midpoint 4.1 MeV) means
     BOTH particles can never pass the double-trigger threshold
     (E₁+E₂ ≤ 6.7 MeV KE < 2×4.1 MeV).  Wall IPC is therefore a SINGLES and
     COMBINATORIAL background, not a direct double-trigger one.  The
     combinatorial rate is estimated from the in-band singles density.

Physics inputs (all 1/v extrapolated below 1 keV):
  σ_np(³He)  = 5333 b   thermal   [ENDF/B-VIII]
  σ_nγ(³He)  = 54 µb    thermal   [Wolfs et al., PRL 63, 2721 (1989)]
  σ_nγ(²⁷Al) = 0.231 b  thermal   [ENDF; IAEA PGAA database]
  σ_nγ(¹²C)  = 3.53 mb  thermal   [ENDF]
  σ_nγ(¹H)   = 0.332 b  thermal   [ENDF]
  ²⁸Al cascade: representative IAEA-PGAA lines > 2 MeV (see _AL_LINES)
  IPC coefficient α(E): log-interpolated anchors from Rose's pair-conversion
     tables (E1, low Z); ~2.6e-4 at 2.2 MeV → ~3.6e-3 at 20.6 MeV.
  Trigger curve: P(PlScint∧LS1 | KE, aimed) = logistic(x0=4.1, w=0.63 MeV),
     measured from the old-geometry Geant4 dataset (this session).

Run:  venv/bin/python run_ipc_background_study.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import (_generate_pair_kinematics, _sample_ipc_mass,
                            Detector, IPC_PER_PULSE, X17_PER_PULSE,
                            TOTAL_PULSES, RUN_DAYS, _ME_MEV)
from geant4_response import Geant4Response
from run_significance_study import z_asimov, z_profiled, ANGLE_BINS
from detector_config import GEO, G4_RESPONSE

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "backgrounds")

# ── Nuclear data ─────────────────────────────────────────────────────────────
E_TH_EV   = 0.0253          # thermal reference energy
SIG_HE3_NP = 5333.0         # b
SIG_HE3_NG = 54e-6          # b  (Wolfs 1989: 54 ± 6 µb)
SIG_AL_NG  = 0.231          # b
SIG_C_NG   = 3.53e-3        # b
SIG_H_NG   = 0.332          # b

# Number densities [1/cm³]
N_HE3  = 62.7e-3 / 3.016 * 6.022e23          # 500 bar gas: 1.25e22
N_AL   = 2.70 / 26.98 * 6.022e23             # 6.03e22
# CFRP 1.55 g/cm³, mass fractions ~ C:0.85, H:0.04, O:0.11 (fiber + epoxy)
N_C_CFRP = 1.55 * 0.85 / 12.011 * 6.022e23   # 6.6e22
N_H_CFRP = 1.55 * 0.04 / 1.008  * 6.022e23   # 3.7e22

# Path lengths along the beam axis [cm]
L_HE3, L_AL, L_CFRP = 5.0, 0.05, 0.12

# ²⁸Al prompt capture γ lines > 2 MeV (E [MeV], intensity per capture)
# Representative subset of the IAEA PGAA evaluation; Σ I ≈ 1.1 γ/capture
# above 2 MeV (total cascade multiplicity ~2.7 including soft lines).
_AL_LINES = [
    (7.724, 0.275), (7.693, 0.044), (6.316, 0.019), (6.102, 0.020),
    (5.134, 0.020), (4.903, 0.019), (4.734, 0.055), (4.691, 0.026),
    (4.660, 0.029), (4.259, 0.070), (4.133, 0.068), (3.876, 0.035),
    (3.591, 0.046), (3.539, 0.072), (3.465, 0.085), (3.034, 0.082),
    (2.960, 0.036), (2.821, 0.045), (2.590, 0.035), (2.283, 0.047),
]
_C_LINES = [(4.945, 0.675), (3.684, 0.323)]      # ¹³C: GS + 3.68 cascade
_H_LINES = [(2.223, 1.0)]                        # deuteron formation

# IPC pair-conversion coefficient α(Eγ): log-E interpolation of Rose-table
# anchors (E1, Z≈0).  M1 is ~20-30% lower at high E — within our error budget.
_ALPHA_E  = np.array([1.1, 2.2, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 20.6])
_ALPHA_V  = np.array([2e-5, 2.6e-4, 4.5e-4, 6.5e-4, 8.0e-4, 9.5e-4,
                      1.2e-3, 1.9e-3, 2.8e-3, 3.6e-3])


def alpha_ipc(e_mev):
    return float(np.interp(np.log(e_mev), np.log(_ALPHA_E), _ALPHA_V))


# External conversion probability for a γ crossing the FAR target wall
# (0.5 mm Al + 1.2 mm CFRP ≈ 0.75% X₀): P ≈ (7/9)·x/X₀ at high E.
_XOVER_WALL = 0.05 / 8.897 + 0.12 / (42.70 / 1.55)
P_EXT_WALL  = (7.0 / 9.0) * _XOVER_WALL

# Per-particle trigger probability GIVEN the particle is aimed at an arm
# (energy part only; geometry handled separately by the MM panels).
# Fitted to the CONDITIONAL Geant4 curve (unconditional / 0.548 plateau) in
# the threshold region 2–5 MeV, where the soft wall pairs live:
#   KE [MeV]:   2.25   2.75   3.25   3.75   4.25   4.75   5.25
#   P(cond):    0.005  0.036  0.16   0.43   0.61   0.75   0.84
TRIG_X0, TRIG_W = 3.85, 0.35   # MeV


def p_trig(ke_mev):
    return 1.0 / (1.0 + np.exp(-(np.asarray(ke_mev) - TRIG_X0) / TRIG_W))


# ── 1. Capture budget ────────────────────────────────────────────────────────

def capture_budget(n_lethargy=400):
    """Iso-lethargic flux 25 meV–1 keV through CFRP|Al|He3|Al|CFRP along the
    axis.  Returns dict of capture probabilities per incident neutron."""
    lne = np.linspace(np.log(0.0253), np.log(1000.0), n_lethargy)
    e_ev = np.exp(lne)
    sqrt_fac = np.sqrt(E_TH_EV / e_ev)          # 1/v scaling

    def tau(n, sig_th, L):                      # optical depth per layer
        return n * (sig_th * 1e-24) * sqrt_fac * L

    t_cfrp = tau(N_C_CFRP, SIG_C_NG, L_CFRP) + tau(N_H_CFRP, SIG_H_NG, L_CFRP)
    t_al   = tau(N_AL, SIG_AL_NG, L_AL)
    t_he3  = tau(N_HE3, SIG_HE3_NP + SIG_HE3_NG, L_HE3)

    out = {k: np.zeros_like(e_ev) for k in
           ("cfrp_in", "al_in", "he3", "al_out", "cfrp_out")}
    surv = np.ones_like(e_ev)
    for key, t in [("cfrp_in", t_cfrp), ("al_in", t_al), ("he3", t_he3),
                   ("al_out", t_al), ("cfrp_out", t_cfrp)]:
        cap = surv * (1.0 - np.exp(-t))
        out[key] = cap
        surv = surv * np.exp(-t)

    # lethargy average (flat weights)
    res = {k: float(v.mean()) for k, v in out.items()}
    res["transmitted"] = float(surv.mean())
    res["_e_ev"], res["_curves"] = e_ev, out
    return res


# ── 2. Pair yields per incident neutron ─────────────────────────────────────

def pair_yields(budget):
    """pairs / incident neutron, per source and mechanism."""
    p_he3 = budget["he3"]
    p_al  = budget["al_in"] + budget["al_out"]
    p_cf  = budget["cfrp_in"] + budget["cfrp_out"]

    # He-3: radiative branch is σγ/(σnp+σγ) of captures, then IPC
    y_he3_ipc = p_he3 * (SIG_HE3_NG / SIG_HE3_NP) * alpha_ipc(20.58)

    def cascade_yield(lines, mech="ipc"):
        if mech == "ipc":
            return sum(I * alpha_ipc(E) for E, I in lines)
        return sum(I * P_EXT_WALL for E, I in lines)   # external conv.

    # CFRP captures split C/H by their optical-depth share
    tc = N_C_CFRP * SIG_C_NG
    th = N_H_CFRP * SIG_H_NG
    fc, fh = tc / (tc + th), th / (tc + th)

    y = {
        "He3 IPC (20.58 MeV)":        y_he3_ipc,
        "Al wall IPC":                p_al * cascade_yield(_AL_LINES, "ipc"),
        "Al wall ext. conv.":         p_al * cascade_yield(_AL_LINES, "ext"),
        "CFRP(C) IPC":                p_cf * fc * cascade_yield(_C_LINES, "ipc"),
        "CFRP(C) ext. conv.":         p_cf * fc * cascade_yield(_C_LINES, "ext"),
        "CFRP(H) IPC (2.22 MeV)":     p_cf * fh * cascade_yield(_H_LINES, "ipc"),
    }
    return y


# ── 3. Detector response toy ─────────────────────────────────────────────────

def simulate_source(lines_or_he3, n_pairs, resp, panels, rng_seed=0):
    """Generate IPC pairs for a source, smear with the Geant4 response,
    apply MM-panel geometry + the energy-dependent trigger curve.

    Returns dict with per-pair arrays: opening angle (reco), KEs, category
    counts (double-trig / single-trig / none).
    """
    np.random.seed(rng_seed)
    origin = np.zeros(3)

    def hits_arm(d):
        for ip, p in enumerate(panels):
            if p.intersect(origin, d)[1] is not None:
                return ip
        return -1

    if lines_or_he3 == "he3":
        line_e = np.full(n_pairs, 20.58)
    else:
        es = np.array([E for E, _ in lines_or_he3])
        ws = np.array([I for _, I in lines_or_he3]); ws = ws / ws.sum()
        line_e = es[np.random.choice(len(es), size=n_pairs, p=ws)]

    out = dict(ang=[], ke1=[], ke2=[], cls=[])
    for etr in line_e:
        mee = _sample_ipc_mass(etr)
        d1, d2, _m, ke1, ke2 = _generate_pair_kinematics(mee, etr)
        s1 = resp.smear_direction(d1, ke1) if ke1 > 0.05 else d1
        s2 = resp.smear_direction(d2, ke2) if ke2 > 0.05 else d2
        a1, a2 = hits_arm(s1), hits_arm(s2)
        t1 = (a1 >= 0) and (np.random.random() < p_trig(ke1))
        t2 = (a2 >= 0) and (np.random.random() < p_trig(ke2))
        if t1 and t2 and a1 != a2:
            cls = 2          # double trigger (different arms)
        elif t1 or t2:
            cls = 1          # single trigger
        else:
            cls = 0
        cos = float(np.clip(np.dot(s1, s2), -1, 1))
        out["ang"].append(np.degrees(np.arccos(cos)))
        out["ke1"].append(ke1); out["ke2"].append(ke2); out["cls"].append(cls)
    return {k: np.asarray(v) for k, v in out.items()}


# ── 4. Combinatorial background ──────────────────────────────────────────────

def combinatorial_rate(singles_per_band, band_ms=9.0, window_ns=200.0,
                       p_mm=0.36):
    """Expected fake MM pairs per pulse from uncorrelated singles.

    singles_per_band : triggering single tracks per pulse in the E<1 keV TOF
                       band (TOF ≈ 44 µs → 9 ms at ~19.5 m ⇒ band ≈ 9 ms)
    window_ns        : pairing coincidence window (|Δt| < window)
    p_mm             : per-track MM-panel acceptance (≈ 36 % from fast-MC)

    Poisson pairing: N_pair ≈ R²·(2τ)/(2T) with R singles/band, T band width.
    """
    R, T = singles_per_band, band_ms * 1e6   # ns
    n_raw  = R * R * (2 * window_ns) / (2 * T)
    return n_raw * p_mm * p_mm


def combinatorial_angle_shape(n=50000, seed=3):
    """Opening-angle distribution of two UNCORRELATED tracks that both hit
    MM panels (isotropic emission from the target)."""
    rng = np.random.default_rng(seed)
    d_plane = GEO["detector_distance"] + GEO["mm_drift_gap"]
    panels = [Detector(s, d_plane, *GEO["detector_size"], "mm")
              for s in range(4)]
    origin = np.zeros(3)

    def rand_dir(m):
        phi = rng.uniform(0, 2 * np.pi, m)
        ct  = rng.uniform(-1, 1, m)
        st  = np.sqrt(1 - ct**2)
        return np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)

    def hits(d):
        return any(p.intersect(origin, d)[1] is not None for p in panels)

    angs = []
    while len(angs) < n:
        d1, d2 = rand_dir(1)[0], rand_dir(1)[0]
        if hits(d1) and hits(d2):
            angs.append(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1))))
    return np.asarray(angs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. capture budget
    bud = capture_budget()
    print("═" * 72)
    print("1. CAPTURE BUDGET  (per incident neutron, iso-lethargic 25 meV–1 keV)")
    print("═" * 72)
    for k in ("cfrp_in", "al_in", "he3", "al_out", "cfrp_out", "transmitted"):
        print(f"   {k:<12s}: {bud[k]:.3e}")

    # 2. pair yields
    yields = pair_yields(bud)
    y_he3 = yields["He3 IPC (20.58 MeV)"]
    print("\n" + "═" * 72)
    print("2. PAIR YIELDS  (e+e- pairs per incident neutron)")
    print("═" * 72)
    for k, v in yields.items():
        print(f"   {k:<26s}: {v:.3e}   (×{v/y_he3:7.1f} vs He3 IPC)")
    y_wall = sum(v for k, v in yields.items() if "He3" not in k)
    print(f"   {'WALL TOTAL':<26s}: {y_wall:.3e}   (×{y_wall/y_he3:7.1f} vs He3 IPC)")

    # 3. detector response per source
    resp = Geant4Response(G4_RESPONSE)
    d_plane = GEO["detector_distance"] + GEO["mm_drift_gap"]
    panels = [Detector(s, d_plane, *GEO["detector_size"], "mm")
              for s in range(4)]

    print("\n" + "═" * 72)
    print("3. TRIGGER CLASSIFICATION  (MC, per source)")
    print("═" * 72)
    N_MC = 40000
    src_mc = {}
    for name, src, seed in [("He3 IPC", "he3", 11),
                            ("Al wall", _AL_LINES, 22),
                            ("CFRP(C)", _C_LINES, 33)]:
        mc = simulate_source(src, N_MC, resp, panels, seed)
        src_mc[name] = mc
        f_dbl = (mc["cls"] == 2).mean()
        f_sgl = (mc["cls"] == 1).mean()
        print(f"   {name:<10s}: double-trig {f_dbl*100:6.3f}%   "
              f"single-trig {f_sgl*100:5.1f}%   (of produced pairs)")

    # absolute rates: He3-IPC produced = IPC_PER_PULSE (Alberto), others scaled
    print("\n" + "═" * 72)
    print(f"4. ABSOLUTE RATES over {RUN_DAYS} days ({TOTAL_PULSES:,} pulses)")
    print("═" * 72)
    n_he3_prod = IPC_PER_PULSE * TOTAL_PULSES
    rates = {}
    for name, ykey in [("He3 IPC", "He3 IPC (20.58 MeV)")]:
        rates[name] = n_he3_prod
    n_al_prod  = n_he3_prod * (yields["Al wall IPC"] +
                               yields["Al wall ext. conv."]) / y_he3
    n_cf_prod  = n_he3_prod * (yields["CFRP(C) IPC"] +
                               yields["CFRP(C) ext. conv."]) / y_he3
    rates["Al wall"], rates["CFRP(C)"] = n_al_prod, n_cf_prod
    for name in ("He3 IPC", "Al wall", "CFRP(C)"):
        mc = src_mc[name]
        nd = rates[name] * (mc["cls"] == 2).mean()
        ns = rates[name] * (mc["cls"] == 1).mean()
        print(f"   {name:<10s}: produced {rates[name]:12,.0f}   "
              f"double-trig {nd:10,.1f}   singles {ns:12,.0f}")

    # singles per pulse in band (for combinatorics)
    s_wall_pp = (rates["Al wall"] * (src_mc["Al wall"]["cls"] == 1).mean() +
                 rates["CFRP(C)"] * (src_mc["CFRP(C)"]["cls"] == 1).mean() +
                 rates["He3 IPC"] * (src_mc["He3 IPC"]["cls"] == 1).mean()
                 ) / TOTAL_PULSES
    print(f"\n   IPC-induced triggering singles/pulse : {s_wall_pp:.3f}")

    # 5. combinatorial
    print("\n" + "═" * 72)
    print("5. COMBINATORIAL BACKGROUND  (uncorrelated in-band singles)")
    print("═" * 72)
    comb_shape = combinatorial_angle_shape()
    print(f"   {'singles/band':>14s} {'fake MM pairs/pulse':>22s} "
          f"{'pairs/30d':>12s}")
    for s_band in [0.5, 1, 2, 5, 10, 20, 90]:
        r = combinatorial_rate(s_band)
        print(f"   {s_band:>14.1f} {r:>22.3e} {r*TOTAL_PULSES:>12.1f}")
    # default working point: Dylan's 200/spill flat over 20 ms → ~45% in band,
    # times the fraction that actually trigger an arm (~0.5 assumed inside)
    S_BAND_DEF = 90.0
    n_comb_30d = combinatorial_rate(S_BAND_DEF) * TOTAL_PULSES
    print(f"   default (200/spill flat in TOF → ~{S_BAND_DEF:.0f}/band): "
          f"{n_comb_30d:,.0f} fake pairs / {RUN_DAYS} d")

    # 6. stacked angle spectrum + significance impact
    a_cen = 0.5 * (ANGLE_BINS[:-1] + ANGLE_BINS[1:])

    def hist_density(angles, n_total):
        h, _ = np.histogram(angles, ANGLE_BINS)
        return h / max(len(angles), 1) * n_total

    # double-trigger components
    he3_dbl  = src_mc["He3 IPC"]["ang"][src_mc["He3 IPC"]["cls"] == 2]
    al_dbl   = src_mc["Al wall"]["ang"][src_mc["Al wall"]["cls"] == 2]
    n_he3_d  = rates["He3 IPC"] * (src_mc["He3 IPC"]["cls"] == 2).mean()
    n_al_d   = rates["Al wall"] * (src_mc["Al wall"]["cls"] == 2).mean()

    b_he3  = hist_density(he3_dbl, n_he3_d)
    b_al   = hist_density(al_dbl, n_al_d) if len(al_dbl) else np.zeros(len(a_cen))
    b_comb = hist_density(comb_shape, n_comb_30d)

    # X17 signal from the same toy machinery (consistency)
    from MX17_Simulator import X17PhysicsSpectrum
    np.random.seed(77)
    x_ang = []
    for d1, d2, m, ang, ke1, ke2 in X17PhysicsSpectrum().sample_pairs(N_MC):
        s1 = resp.smear_direction(d1, ke1); s2 = resp.smear_direction(d2, ke2)
        a1 = next((i for i, p in enumerate(panels)
                   if p.intersect(np.zeros(3), s1)[1] is not None), -1)
        a2 = next((i for i, p in enumerate(panels)
                   if p.intersect(np.zeros(3), s2)[1] is not None), -1)
        if (a1 >= 0 and a2 >= 0 and a1 != a2
                and np.random.random() < p_trig(ke1)
                and np.random.random() < p_trig(ke2)):
            x_ang.append(np.degrees(np.arccos(np.clip(np.dot(s1, s2), -1, 1))))
    x_ang = np.asarray(x_ang)
    n_x17_d = X17_PER_PULSE * TOTAL_PULSES * len(x_ang) / N_MC
    s_x17 = hist_density(x_ang, n_x17_d)

    b_tot = b_he3 + b_al + b_comb
    z_fixed = z_asimov(s_x17, b_tot)
    z_prof3 = z_profiled(s_x17, b_tot, ANGLE_BINS, 3)
    z_he3only = z_asimov(s_x17, b_he3)
    print("\n" + "═" * 72)
    print("6. SIGNIFICANCE IMPACT  (toy double-trigger channel)")
    print("═" * 72)
    print(f"   X17 double-trig pairs/{RUN_DAYS}d : {s_x17.sum():8.1f}")
    print(f"   He3-IPC background            : {b_he3.sum():10,.1f}")
    print(f"   Al-wall double-trig bkg       : {b_al.sum():10,.1f}")
    print(f"   Combinatorial bkg (default)   : {b_comb.sum():10,.1f}")
    print(f"   Z (He3-IPC only, shape known) : {z_he3only:6.2f}σ")
    print(f"   Z (all bkg, shapes known)     : {z_fixed:6.2f}σ")
    print(f"   Z (all bkg, norm+tilt+curv fit): {z_prof3:5.2f}σ")

    # ── 7. Mitigation: per-leg energy threshold ──────────────────────────
    # An offline (or trigger-level pulse-height) requirement that BOTH legs
    # carry KE above E_cut.  Wall pairs share ≤ 6.7 MeV of KE, so a cut at
    # 4–5 MeV removes them kinematically; X17 legs are bounded below at
    # ~3.9 MeV, so the signal cost is modest.  Combinatorial singles from
    # wall captures are also < 7 MeV, so the same cut suppresses them.
    print("\n" + "═" * 72)
    print("7. MITIGATION — require both legs KE > E_cut (toy)")
    print("═" * 72)
    print(f"   {'E_cut':>6s} {'X17':>8s} {'He3-IPC':>10s} {'Al wall':>12s} "
          f"{'comb':>10s} {'Z(known)':>9s} {'Z(fit)':>8s}")

    def dbl_after_cut(mc, n_prod, cut):
        sel = (mc["cls"] == 2) & (mc["ke1"] > cut) & (mc["ke2"] > cut)
        return mc["ang"][sel], n_prod * sel.mean()

    # X17 KEs for the cut: regenerate selection arrays alongside x_ang
    np.random.seed(77)
    x_ang2, x_min_ke = [], []
    for d1, d2, m, ang, ke1, ke2 in X17PhysicsSpectrum().sample_pairs(N_MC):
        s1 = resp.smear_direction(d1, ke1); s2 = resp.smear_direction(d2, ke2)
        a1 = next((i for i, p in enumerate(panels)
                   if p.intersect(np.zeros(3), s1)[1] is not None), -1)
        a2 = next((i for i, p in enumerate(panels)
                   if p.intersect(np.zeros(3), s2)[1] is not None), -1)
        if (a1 >= 0 and a2 >= 0 and a1 != a2
                and np.random.random() < p_trig(ke1)
                and np.random.random() < p_trig(ke2)):
            x_ang2.append(np.degrees(np.arccos(np.clip(np.dot(s1, s2), -1, 1))))
            x_min_ke.append(min(ke1, ke2))
    x_ang2, x_min_ke = np.asarray(x_ang2), np.asarray(x_min_ke)

    # combinatorial: each leg is a wall single with KE from the Al spectrum;
    # approximate the cut survival as the squared fraction of triggering
    # wall singles above the cut
    al = src_mc["Al wall"]
    sgl_ke = np.concatenate([al["ke1"][al["cls"] >= 1], al["ke2"][al["cls"] >= 1]])
    sgl_ke = sgl_ke[p_trig(sgl_ke) > np.random.random(len(sgl_ke))]

    for cut in [0.0, 3.0, 4.0, 5.0, 6.0]:
        xs = x_ang2[x_min_ke > cut]
        n_x = X17_PER_PULSE * TOTAL_PULSES * len(xs) / N_MC
        s_c = hist_density(xs, n_x) if len(xs) else np.zeros(len(a_cen))

        h_he3, n_he3c = dbl_after_cut(src_mc["He3 IPC"], rates["He3 IPC"], cut)
        b_he3c = hist_density(h_he3, n_he3c) if len(h_he3) else np.zeros(len(a_cen))
        h_al, n_alc = dbl_after_cut(src_mc["Al wall"], rates["Al wall"], cut)
        b_alc = hist_density(h_al, n_alc) if len(h_al) else np.zeros(len(a_cen))

        f_comb = (sgl_ke > cut).mean() ** 2 if len(sgl_ke) else 0.0
        b_combc = b_comb * f_comb

        b_t = b_he3c + b_alc + b_combc
        zf = z_asimov(s_c, b_t)
        zp = z_profiled(s_c, b_t, ANGLE_BINS, 3)
        print(f"   {cut:>5.1f} {s_c.sum():>8.1f} {b_he3c.sum():>10.1f} "
              f"{b_alc.sum():>12.1f} {b_combc.sum():>10.1f} "
              f"{zf:>8.2f}σ {zp:>7.2f}σ")

    # ── plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]   # capture curves vs energy
    e_ev, curves = bud["_e_ev"], bud["_curves"]
    for k, c in [("he3", "#4a90d9"), ("al_in", "#d62728"),
                 ("cfrp_in", "#ff7f0e")]:
        ax.loglog(e_ev, np.clip(curves[k], 1e-12, None), lw=2,
                  label={"he3": "He-3 gas", "al_in": "Al end cap (entry)",
                         "cfrp_in": "CFRP end cap (entry)"}[k], color=c)
    ax.set_xlabel("Neutron energy [eV]")
    ax.set_ylabel("Capture probability / neutron")
    ax.set_title("Where neutrons capture vs energy\n(beam enters through CFRP+Al end cap)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]   # pair yields bar
    names = list(yields)
    vals  = [yields[k] for k in names]
    ax.barh(np.arange(len(names)), vals, color=["#4a90d9", "#d62728",
            "#e88080", "#ff7f0e", "#ffb070", "#9467bd"])
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("e⁺e⁻ pairs per incident neutron")
    ax.set_title(f"Raw pair production by source\n"
                 f"(wall total = {y_wall/y_he3:,.0f}× the He-3 IPC yield!)")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    ax = axes[1, 0]   # KE spectra + trigger curve
    for name, c in [("He3 IPC", "#4a90d9"), ("Al wall", "#d62728"),
                    ("CFRP(C)", "#ff7f0e")]:
        mc = src_mc[name]
        kes = np.concatenate([mc["ke1"], mc["ke2"]])
        ax.hist(kes, bins=np.linspace(0, 21, 85), histtype="step",
                density=True, color=c, lw=1.8, label=f"{name} e± KE")
    ke_ax = np.linspace(0, 21, 200)
    ax2 = ax.twinx()
    ax2.plot(ke_ax, p_trig(ke_ax), "k--", lw=2, label="P(trigger | KE, aimed)")
    ax2.set_ylabel("Per-particle trigger probability")
    ax2.set_ylim(0, 1.05)
    ax.set_xlabel("e± kinetic energy [MeV]")
    ax.set_ylabel("Probability density")
    ax.set_title("Why wall pairs can't double-trigger:\n"
                 "≤7.7 MeV shared by two legs vs 4.1 MeV threshold each")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]   # stacked double-trigger spectrum
    bot = np.zeros(len(a_cen))
    for h, lbl, c in [(b_he3, f"He-3 IPC (N={b_he3.sum():,.0f})", "#9dc3e6"),
                      (b_al, f"Al-wall dbl-trig (N={b_al.sum():,.1f})", "#f4b8b8"),
                      (b_comb, f"combinatorial (N={b_comb.sum():,.0f})", "#cccccc")]:
        ax.bar(a_cen, h, width=4, bottom=bot, color=c, edgecolor="none", label=lbl)
        bot += h
    ax.bar(a_cen, s_x17, width=4, bottom=bot, color="#e84040",
           label=f"X17 UNSCALED (N={s_x17.sum():.0f})")
    ax.set_xlabel("Reco opening angle [deg]")
    ax.set_ylabel(f"Pairs / {RUN_DAYS} d / 4°")
    ax.set_title(f"Double-trigger angle spectrum with all backgrounds\n"
                 f"Z = {z_fixed:.2f}σ (shapes known), "
                 f"{z_prof3:.2f}σ (bkg fit from data)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("IPC Background Decomposition — He-3 vs target-material captures "
                 "+ combinatorics\n(toy model; nuclear data from "
                 "ENDF/IAEA-PGAA/Wolfs 1989; trigger curve from Geant4)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ipc_background_study.png")
    fig.savefig(out, dpi=130)
    print(f"\n[Output] Saved {out}")


if __name__ == "__main__":
    main()

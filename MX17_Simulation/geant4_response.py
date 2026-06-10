"""
Geant4-derived detector response for the MX17 fast-MC
=====================================================
Loads the JSON summary written by

    MX17_Full_Geant/scripts/analyze_pairs.py --export-response <file>.json

and provides:

  smear_direction(direction, ke_mev)
      Energy-dependent multiple-scattering smear: draws a space angle ψ from
      the measured P(ψ | KE) table for the chosen direction estimator and
      rotates the true direction by ψ at a uniform azimuth.  Applying this at
      particle creation reproduces both the Geant4 acceptance (the particle
      really travels along the scattered direction) and the opening-angle
      resolution *including its positive bias*.

  constrained_pair_mass(ke_em, dir_em, ke_ep, dir_ep)  ← RECOMMENDED
      Invariant mass with the E-sum kinematic constraint E1+E2 = S = 20.58 MeV.
      Samples raw LS deposits edep_1, edep_2 and uses only their ratio
      x = edep_1/(edep_1+edep_2) to infer the sharing fraction; sets
      E1 = x·S, E2 = (1−x)·S.  Avoids the noisy absolute energy scale entirely.
      σ68 ≈ 1.4 MeV for X17 (vs 2.7 MeV unconstrained, 4.4 MeV linear-corrected).

  reco_energy(ke_mev, pid)
      Samples a calorimeter deposit from the measured P(edep | E_total) table
      and (optionally) applies the linear upstream-loss correction
      E_est = (edep + b) / (1 - a).  NOTE: the linear correction amplifies
      noise by 1/(1−a) ≈ 2×; prefer constrained_pair_mass() for invariant-mass
      reconstruction.

  reco_pair_mass(ke_em, dir_em, ke_ep, dir_ep)
      Unconstrained invariant mass from individually corrected energies.
      Kept for backward compatibility; constrained_pair_mass() is better.

Estimator choices for the direction tables (see JSON 'direction_error'):
  'first'   direction at the first MM drift-gas hit — what an ideal MM
            measurement gives (recommended default; PCA fit adds <1°)
  'fit'     PCA straight-line fit of the drift hits
  'vline'   true vertex → first MM hit (vertex-constrained pointing)
  'nomline' target centre → first MM hit (realistic vertex constraint)

Random numbers use the global np.random state, matching MX17_Simulator's
seeding convention (np.random.seed in worker batches).
"""

import json
import numpy as np

_ME_MEV    = 0.511
_E_SUM_MEV = 20.58   # ³He(n,γ*)⁴He* transition; E(e-)+E(e+) ≈ this for all events


def _build_row_samplers(counts, min_counts=20):
    """Per-row CDFs of a 2D count table; sparse rows borrow the nearest
    populated row so sampling never fails. Returns (cdfs, ok_any)."""
    counts = np.asarray(counts, dtype=float)
    sums   = counts.sum(axis=1)
    good   = np.where(sums >= min_counts)[0]
    cdfs   = np.zeros_like(counts)
    if len(good) == 0:
        return cdfs, False
    for i in range(counts.shape[0]):
        j = i if sums[i] >= min_counts else good[np.argmin(np.abs(good - i))]
        cdfs[i] = np.cumsum(counts[j]) / sums[j]
    return cdfs, True


def _sample_from_cdf(cdf_row, edges):
    """Inverse-CDF sample with uniform spread inside the chosen bin."""
    u = np.random.uniform()
    i = int(np.searchsorted(cdf_row, u))
    i = min(i, len(edges) - 2)
    return edges[i] + np.random.uniform() * (edges[i + 1] - edges[i])


class Geant4Response:
    """Detector-response sampler backed by the Geant4 analysis JSON."""

    def __init__(self, path, direction_estimator="first",
                 energy_method="ls", energy_corrected=True):
        with open(path) as fp:
            d = json.load(fp)
        self.meta = d["meta"]

        de = d["direction_error"]
        self.ke_edges  = np.asarray(de["ke_bin_edges"])
        self.psi_edges = np.asarray(de["psi_bin_edges"])
        if direction_estimator not in de["estimators"]:
            raise KeyError(f"estimator '{direction_estimator}' not in response "
                           f"file; available: {list(de['estimators'])}")
        self.estimator = direction_estimator
        self._psi_cdfs, ok = _build_row_samplers(
            de["estimators"][direction_estimator]["counts"])
        if not ok:
            raise ValueError("direction-error table is empty")

        er = d["energy_response"]
        self.e_edges    = np.asarray(er["e_total_bin_edges"])
        self.edep_edges = np.asarray(er["edep_bin_edges"])
        self.energy_method    = energy_method        # 'ls' or 'all'
        self.energy_corrected = energy_corrected
        self._edep_cdfs = {}
        for key, table in er["tables"].items():
            self._edep_cdfs[key] = _build_row_samplers(table)[0]
        self.loss_fit = er["loss_fit_a_b"]

        self.validation = d.get("validation", {})

    # ── direction smearing ────────────────────────────────────────────────
    def sample_psi_deg(self, ke_mev):
        """Draw a space angle ψ [deg] between true and measured direction."""
        ki = int(np.clip(np.searchsorted(self.ke_edges, ke_mev) - 1,
                         0, len(self.ke_edges) - 2))
        return _sample_from_cdf(self._psi_cdfs[ki], self.psi_edges)

    def smear_direction(self, direction, ke_mev):
        """Rotate `direction` by a sampled ψ at uniform azimuth (unit vector in,
        unit vector out)."""
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        psi = np.radians(self.sample_psi_deg(ke_mev))
        # orthonormal basis around d
        a = np.array([1.0, 0.0, 0.0])
        if abs(d[0]) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        out = (np.cos(psi) * d
               + np.sin(psi) * (np.cos(phi) * e1 + np.sin(phi) * e2))
        return out / np.linalg.norm(out)

    # ── calorimetric energy ───────────────────────────────────────────────
    def _table_key(self, pid):
        side = "ep" if pid in ("positron", "e+") else "em"
        return f"{self.energy_method}_{side}"

    def sample_edep(self, ke_mev, pid="electron"):
        """Sample the scintillator deposit [MeV] given true KE (conditional on
        the particle having reached the calorimeter — geometry handles that)."""
        e_tot = ke_mev + _ME_MEV
        ei = int(np.clip(np.searchsorted(self.e_edges, e_tot) - 1,
                         0, len(self.e_edges) - 2))
        return _sample_from_cdf(self._edep_cdfs[self._table_key(pid)][ei],
                                self.edep_edges)

    def reco_energy(self, ke_mev, pid="electron"):
        """Reconstructed total energy estimate [MeV]: sampled deposit, with
        the linear upstream-loss correction if energy_corrected."""
        edep = self.sample_edep(ke_mev, pid)
        fit  = self.loss_fit.get(self._table_key(pid))
        if self.energy_corrected and fit is not None:
            a, b = fit
            if abs(1.0 - a) > 0.01:
                return max((edep + b) / (1.0 - a), _ME_MEV)
        return max(edep + _ME_MEV, _ME_MEV)

    # ── invariant mass ────────────────────────────────────────────────────
    def constrained_pair_mass(self, ke_em, dir_em, ke_ep, dir_ep,
                              S=_E_SUM_MEV):
        """E-sum–constrained e+e- invariant mass (recommended).

        Uses only the measured sharing fraction x = edep_em/(edep_em+edep_ep)
        from raw LS deposits and fixes E1+E2 = S = 20.58 MeV.  This bypasses
        the noisy absolute energy scale; σ68 ≈ 1.4 MeV for X17.

        The constraint is valid for both X17 and IPC: regardless of the true
        pair mass, energy conservation gives E1+E2 = S − T_rec(⁴He) ≈ S to
        within 57 keV (factor ~25 below resolution).
        """
        d1 = self.sample_edep(ke_em, "electron")
        d2 = self.sample_edep(ke_ep, "positron")
        denom = d1 + d2
        if denom < 1e-6:
            return 0.0
        x  = d1 / denom
        E1 = x * S
        E2 = (1.0 - x) * S
        v1 = np.asarray(dir_em, float); v1 = v1 / np.linalg.norm(v1)
        v2 = np.asarray(dir_ep, float); v2 = v2 / np.linalg.norm(v2)
        cos_th = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        p1 = np.sqrt(max(E1**2 - _ME_MEV**2, 0.0))
        p2 = np.sqrt(max(E2**2 - _ME_MEV**2, 0.0))
        m2 = 2.0 * _ME_MEV**2 + 2.0 * (E1 * E2 - p1 * p2 * cos_th)
        return float(np.sqrt(max(m2, 0.0)))

    def reco_pair_mass(self, ke_em, dir_em, ke_ep, dir_ep):
        """Unconstrained e+e- invariant mass from individually corrected energies.
        Kept for backward compatibility; constrained_pair_mass() is recommended."""
        e1 = self.reco_energy(ke_em, "electron")
        e2 = self.reco_energy(ke_ep, "positron")
        p1 = np.sqrt(max(e1**2 - _ME_MEV**2, 0.0))
        p2 = np.sqrt(max(e2**2 - _ME_MEV**2, 0.0))
        d1 = np.asarray(dir_em, float); d1 = d1 / np.linalg.norm(d1)
        d2 = np.asarray(dir_ep, float); d2 = d2 / np.linalg.norm(d2)
        cos_th = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
        m2 = 2 * _ME_MEV**2 + 2 * (e1 * e2 - p1 * p2 * cos_th)
        return float(np.sqrt(max(m2, 0.0)))

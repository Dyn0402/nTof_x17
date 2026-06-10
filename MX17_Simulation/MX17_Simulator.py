"""
Micromegas Detector Simulation with Scintillator Trigger Stack
==============================================================
Simulates the 4-arm MX17 detector geometry (n-TOF X17 experiment).
Beam axis = +Y (matching Geant4 frame). Four arms at ±X and ±Z (transverse plane).
He-3 target capsule at origin (r=1.5 cm, 5 cm long along Y beam axis).

Detector stack per arm (from origin outward):
  MM window at detector_distance, active area detector_size (38×34 cm)
  [mm_drift_gap = 3 cm scored gas region]
  [gap_mm_to_scint ≈ 2.77 cm: PCB + air gap]
  Trigger scint wall (48×48 cm, 3 mm plastic scint)   ← "SiPM wall"

  Standard order (stack_order='standard'):
    [gap_scint_to_liq ≈ 3.44 cm: air + CFRP/Al liners]
    Liquid scintillator layer 1 (45×45 cm, 2 cm LAB)
    [gap_ls1_to_ls2 ≈ 2.26 cm: CFRP + liner structure]
    Liquid scintillator layer 2 (45×45 cm, 2 cm LAB)
    [gap_ls2_to_backscint ≈ 2.30 cm: CFRP + air gap]
    Back plastic scintillator: 2 bars × (20×30 cm), 2 cm thick, 0.3 cm gap

  Back-scint-first order (stack_order='back_scint_first'):
    [gap_scint_to_backscint ≈ 3.15 cm: air gap]
    Back plastic scintillator: 2 bars × (20×30 cm), 2 cm thick, 0.3 cm gap
    [gap_backscint_to_ls1 ≈ 2.36 cm: air + CFRP/Al liners]
    Liquid scintillator layer 1 (45×45 cm, 2 cm LAB)
    [gap_ls1_to_ls2 ≈ 2.26 cm]
    Liquid scintillator layer 2 (45×45 cm, 2 cm LAB)

Geometry matches the full Geant4 simulation (MX17_Full_Geant/SimConfig.hh).

Trigger logic
-------------
  single    : scint_wall AND trigger_second_layer both fired on ≥1 side
  double    : single fired on ≥2 sides  ← primary analysis trigger
  mm_any    : ≥1 MM panel hit
  mm_double : ≥2 MM panels hit

  trigger_second_layer:
    'liq_scint_1' in standard config (default)
    'back_scint'  in back_scint_first config

Key metrics (from pair_stats after run())
-----------------------------------------
  trigger_miss_fraction        : fraction of signal pairs (both hit MM) that
                                  fail the scint_double trigger
  calorimetry_incomplete_fraction : fraction of signal pairs (both hit MM)
                                  where ≥1 particle misses back_scint on its arm

Source
------
  Default: all particles originate at the target (origin).
  he_capsule_source=True: origins sampled uniformly within the He-3 capsule
  cylinder (r=he_radius_cm, half-length=he_half_length_cm along Y = beam).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional
import warnings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ME_MEV = 0.511   # electron rest-mass energy [MeV]

# Starting at 1eV = 1337.5 us
EFFICIENCY_SCALE_FACTOR = 0.3  # Alberto scaled down by this to account for detector efficiency
IPC_PER_PULSE = 1.12e-02 / EFFICIENCY_SCALE_FACTOR  # Number of IPCs expected per pulse in above stated energy range.
X17_FRACTION_OF_IPC = 0.025   # X17 rate = IPC_PER_PULSE * X17_FRACTION_OF_IPC
X17_PER_PULSE = IPC_PER_PULSE * X17_FRACTION_OF_IPC
PULSES_PER_DAY = 1.929e4  # Number of pulses expected per day
RUN_DAYS = 30  # Number of days we expect to run
TOTAL_PULSES = int(RUN_DAYS * PULSES_PER_DAY)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """A single simulated particle."""
    pid: str
    origin: np.ndarray
    direction: np.ndarray
    t0: float
    event_id: int = -1
    particle_id: int = -1
    pair_id: int = -1
    source: str = 'random'
    true_inv_mass: float = 0.0   # MeV; set by physics spectra, 0 = unknown
    ke_mev: float = 0.0          # MeV; lab kinetic energy, 0 = unknown


@dataclass
class Hit:
    """A reconstructed hit on a detector panel."""
    detector_id: int
    true_pos: np.ndarray
    reco_pos: np.ndarray
    true_time: float
    reco_time: float
    direction: np.ndarray
    pid: str
    event_id: int
    particle_id: int
    pair_id: int
    source: str
    # 'mm', 'scint_wall', 'liq_scint_1', 'liq_scint_2', 'back_scint'
    detector_type: str = 'mm'


# ---------------------------------------------------------------------------
# Detector Geometry
# ---------------------------------------------------------------------------

class Detector:
    """
    A flat rectangular detector panel.

    Orientation by side index (beam along +Y, matching Geant4 frame):
      0: +X arm  (normal toward origin = -X)
      1: -X arm
      2: +Z arm
      3: -Z arm

    offset_u: shift of the panel centre along the local u-axis [cm].
              Used for the two back-scint bars that are displaced ±u_off
              from the arm centreline.

    det_type: 'mm', 'scint_wall', 'liq_scint_1', 'liq_scint_2', 'back_scint'
    """

    SIDE_NORMALS = {
        0: np.array([-1.0, 0.0, 0.0]),
        1: np.array([ 1.0, 0.0, 0.0]),
        2: np.array([ 0.0, 0.0,-1.0]),
        3: np.array([ 0.0, 0.0, 1.0]),
    }
    SIDE_NAMES = {0: '+X', 1: '-X', 2: '+Z', 3: '-Z'}

    def __init__(self, side: int, distance: float,
                 size_u: float = 40.0, size_v: float = 40.0,
                 det_type: str = 'mm', offset_u: float = 0.0):
        self.side     = side
        self.distance = distance
        self.size_u   = size_u
        self.size_v   = size_v
        self.det_type = det_type
        self.offset_u = offset_u   # lateral shift from arm centreline along u [cm]
        self.normal   = self.SIDE_NORMALS[side]
        self.name     = self.SIDE_NAMES[side]
        self.center   = -self.normal * distance

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """Ray–plane intersection. Returns (hit_3d, local_2d) or (None, None).
        local_2d is in bar-local coordinates (u relative to bar centre)."""
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-9:
            return None, None
        t = np.dot(self.normal, self.center - origin) / denom
        if t <= 0:
            return None, None
        hit_3d = origin + t * direction
        local_u, local_v = self._local_axes()
        delta = hit_3d - self.center
        u = np.dot(delta, local_u) - self.offset_u
        v = np.dot(delta, local_v)
        if abs(u) > self.size_u / 2.0 or abs(v) > self.size_v / 2.0:
            return None, None
        return hit_3d, np.array([u, v])

    def _local_axes(self):
        # v = +Y (beam direction) for all arms — matches Geant4 vHat convention
        v_axis = np.array([0.0, 1.0, 0.0])
        u_axis = np.cross(v_axis, self.normal)
        u_axis /= np.linalg.norm(u_axis)
        return u_axis, v_axis

    def center_3d(self):
        return self.center.copy()


# ---------------------------------------------------------------------------
# Spectra / Angular Distributions
# ---------------------------------------------------------------------------

class AngularSpectrum:
    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError


class GaussianSpectrum(AngularSpectrum):
    def __init__(self, mean_deg: float = 120.0, sigma_deg: float = 5.0):
        self.mean  = mean_deg
        self.sigma = sigma_deg

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mean, self.sigma, n)


class FlatSpectrum(AngularSpectrum):
    def __init__(self, min_deg: float = 0.0, max_deg: float = 180.0):
        self.min = min_deg
        self.max = max_deg

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.min, self.max, n)


class IsotropicSpectrum(AngularSpectrum):
    def sample(self, n: int) -> np.ndarray:
        samples = []
        while len(samples) < n:
            theta  = np.random.uniform(0, 180, n * 2)
            prob   = np.sin(np.radians(theta))
            accept = np.random.uniform(0, 1, n * 2) < prob
            samples.extend(theta[accept].tolist())
        return np.array(samples[:n])


class HistogramSpectrum(AngularSpectrum):
    def __init__(self, bin_edges: np.ndarray, counts: np.ndarray):
        self.bin_edges = bin_edges
        probs     = counts / counts.sum()
        self.cdf  = np.concatenate([[0], np.cumsum(probs)])

    def sample(self, n: int) -> np.ndarray:
        u       = np.random.uniform(0, 1, n)
        indices = np.clip(np.searchsorted(self.cdf, u) - 1, 0, len(self.bin_edges) - 2)
        lo = self.bin_edges[indices]
        hi = self.bin_edges[indices + 1]
        return lo + np.random.uniform(0, 1, n) * (hi - lo)


class ExponentialSpectrum(AngularSpectrum):
    def __init__(self, scale_deg: float = 40.0,
                 min_deg: float = 0.0, max_deg: float = 180.0):
        self.scale = scale_deg
        self.min   = min_deg
        self.max   = max_deg
        self._lo   = np.exp(-min_deg / scale_deg)
        self._hi   = np.exp(-max_deg / scale_deg)

    def sample(self, n: int) -> np.ndarray:
        u = np.random.uniform(0, 1, n)
        return -self.scale * np.log(self._lo - u * (self._lo - self._hi))


class X17PhysicsSpectrum(AngularSpectrum):
    """
    Full relativistic kinematics for X17 → e+e- via ³He(n,γ*)⁴He* transition.
    Fixed invariant mass m_X17; isotropic parent emission + isotropic rest-frame decay.
    Matches Geant4 X17PrimaryGenerator::GeneratePair exactly.
    """
    def __init__(self, m_x17_mev: float = 16.8, E_transition_mev: float = 20.58):
        self.m_x17_mev        = m_x17_mev
        self.E_transition_mev = E_transition_mev

    def sample_pairs(self, n: int):
        """Return list of (dir_em, dir_ep, true_mass_mev, opening_angle_deg,
        ke_em_mev, ke_ep_mev)."""
        result = []
        for _ in range(n):
            dir_em, dir_ep, mass, ke_em, ke_ep = _generate_pair_kinematics(
                self.m_x17_mev, self.E_transition_mev)
            cos_th = float(np.clip(np.dot(dir_em, dir_ep), -1.0, 1.0))
            result.append((dir_em, dir_ep, mass,
                           np.degrees(np.arccos(cos_th)), ke_em, ke_ep))
        return result

    def sample(self, n: int) -> np.ndarray:
        """Backward-compatible: return opening angles only."""
        return np.array([p[3] for p in self.sample_pairs(n)])


class IPCPhysicsSpectrum(AngularSpectrum):
    """
    Full relativistic kinematics for IPC γ* → e+e- via ³He(n,γ*)⁴He* transition.
    Invariant mass sampled from dN/dMee ∝ 1/Mee on [2mₑ, E_transition].
    Matches Geant4 X17PrimaryGenerator::GenerateIPC exactly.
    """
    def __init__(self, E_transition_mev: float = 20.58):
        self.E_transition_mev = E_transition_mev

    def sample_pairs(self, n: int):
        """Return list of (dir_em, dir_ep, true_mass_mev, opening_angle_deg,
        ke_em_mev, ke_ep_mev)."""
        result = []
        for _ in range(n):
            mass = _sample_ipc_mass(self.E_transition_mev)
            dir_em, dir_ep, _, ke_em, ke_ep = _generate_pair_kinematics(
                mass, self.E_transition_mev)
            cos_th = float(np.clip(np.dot(dir_em, dir_ep), -1.0, 1.0))
            result.append((dir_em, dir_ep, mass,
                           np.degrees(np.arccos(cos_th)), ke_em, ke_ep))
        return result

    def sample(self, n: int) -> np.ndarray:
        """Backward-compatible: return opening angles only."""
        return np.array([p[3] for p in self.sample_pairs(n)])


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All tuneable parameters in one place."""

    # MM geometry  (Geant4 SimConfig.hh: mm_distance=25 cm, 38×34 cm face)
    detector_distance: float = 25.0
    detector_size: tuple[float, float] = (38.0, 34.0)   # cm, (u, v along beam)

    # Scintillator trigger stack — distances derived from full Geant4 slab layout
    mm_drift_gap:         float = 3.0                          # cm, active drift gas (30 mm)
    gap_mm_to_scint:      float = 2.77                         # cm, drift back → scint centre
    scint_wall_size:      tuple[float, float] = (50, 50)   # cm
    scint_wall_thickness: float = 0.3                          # cm, 3 mm PVT
    gap_scint_to_liq:     float = 3.44                         # cm, scint centre → LS-1 centre
    liq_scint_size:       tuple[float, float] = (45.0, 45.0)   # cm
    liq_scint_thickness:  float = 2.0                          # cm, LS-1 (LAB)

    # LS-2 (second liquid scintillator layer, same face size as LS-1)
    liq_scint_2_size:      tuple[float, float] = (45.0, 45.0)  # cm
    liq_scint_2_thickness: float = 2.0                          # cm, LS-2 (LAB)
    gap_ls1_to_ls2:        float = 2.26                         # cm, LS-1 centre → LS-2 centre
                                                                # (CFRP wall + inner CFRP + Al liner)

    # Back plastic scintillators (2 bars per arm, side-by-side in u)
    # From Geant4: 2 × (20×30 cm), 2 cm thick, 0.3 cm gap between wrapped bars
    back_scint_size_u:    float = 20.0   # each bar: u [cm]
    back_scint_size_v:    float = 30.0   # each bar: v along beam [cm]
    back_scint_thickness: float = 2.0    # cm
    back_scint_gap:       float = 0.3    # cm, gap between the two bars in u

    # Stack ordering controls where back_scint sits relative to LS layers
    # 'standard'         : scint_wall → LS-1 → LS-2 → back_scint  (Geant4 default)
    # 'back_scint_first' : scint_wall → back_scint → LS-1 → LS-2
    stack_order: str = 'standard'

    # Centre-to-centre gaps that change with stack_order:
    #   standard mode: LS-2 centre → back_scint centre
    #     = ls2_half(1.0) + CFRP3(0.2) + gap(0.1) + bs_half(1.0) = 2.3 cm
    gap_ls2_to_backscint: float = 2.30

    #   back_scint_first mode: scint_wall centre → back_scint centre
    #     = scint_half(0.15) + air(2.0) + bs_half(1.0) = 3.15 cm
    gap_scint_to_backscint: float = 3.15

    #   back_scint_first mode: back_scint centre → LS-1 centre
    #     = bs_half(1.0) + air(0.1) + CFRP(0.2) + liner(0.064) + ls1_half(1.0) = 2.36 cm
    gap_backscint_to_ls1: float = 2.36

    # Trigger: which detector type (together with scint_wall) defines a "single"
    # 'liq_scint_1' for standard config, 'back_scint' for back_scint_first config
    trigger_second_layer: str = 'liq_scint_1'

    # He-3 capsule as particle source (cylinder axis = Y = beam direction)
    # Capsule = cylinder (r, ±he_half_length_cm) + two r-radius hemispherical end caps.
    # Source: STEP file "MASTINU X17 HPRV 00 01 (Cylinder D20 L40 mm)": r=10 mm, cyl L=40 mm.
    # If False, all particles originate at the geometric origin.
    he_capsule_source:  bool  = False
    he_radius_cm:       float = 1.0    # bore radius [cm]  (D=20 mm per STEP)
    he_half_length_cm:  float = 2.0    # half-length of cylinder section [cm]  (L=40 mm per STEP)

    # Resolution (MM only)
    spatial_resolution: float = 0.5   # cm, 1-sigma Gaussian smear per axis
    time_resolution:    float = 5.0   # ns, 1-sigma Gaussian smear

    # Invariant mass parameterisation (back-scint coincidence sample)
    # Used as FALLBACK when no Geant4 response file is configured below.
    inv_mass_mean:  float = 16.8   # MeV, Gaussian centre (X17 mass)
    inv_mass_sigma: float = 4.0    # MeV, Gaussian width (detector resolution)

    # ── Geant4-derived detector response (geant4_response.py) ────────────
    # Path to the JSON written by analyze_pairs.py --export-response.
    # When set:
    #   • pair directions are smeared at creation with the energy-dependent
    #     multiple-scattering ψ(KE) tables (affects acceptance AND the
    #     reconstructed opening angles, including their positive bias);
    #   • the both-back-scint invariant mass is computed from sampled
    #     calorimeter energies + smeared directions instead of the flat
    #     Gaussian inv_mass_sigma above.
    g4_response_path:       Optional[str] = None
    g4_direction_estimator: str  = 'first'  # 'first'|'fit'|'vline'|'nomline'
    g4_smear_directions:    bool = True
    g4_mass_from_response:  bool = True
    g4_energy_method:       str  = 'ls'     # 'ls' (LS only) | 'all' (all scint)
    g4_energy_corrected:    bool = True     # apply upstream-loss correction
    # Use the E-sum kinematic constraint E1+E2=20.58 MeV for invariant-mass reco.
    # Fixes E1+E2=S and measures only the sharing fraction x from the LS deposits;
    # σ68 ≈ 1.4 MeV vs 2.7 MeV unconstrained.  Set False to revert to old method.
    g4_use_constraint:      bool = True

    # Timing
    coincidence_window: float = 20.0  # ns
    time_spread:        float = 200.0 # ns, duration of one readout event

    # Event counts (Poisson means per event)
    n_events:            int   = 100
    n_random:            float = 5.0
    n_background_pairs:  float = 10.0
    n_signal:            float = 2.0

    # Pairing
    allow_same_detector_pairs: bool = True

    # MM hit merging
    merge_hits:              bool  = True
    merge_spatial_threshold: float = 1.0   # cm
    merge_time_threshold:    float = 10.0  # ns

    # Spectra
    signal_spectrum: AngularSpectrum = field(
        default_factory=lambda: GaussianSpectrum(mean_deg=120.0, sigma_deg=5.0)
    )
    background_spectrum: AngularSpectrum = field(
        default_factory=lambda: ExponentialSpectrum(scale_deg=40.0)
    )

    random_pid_weights: dict = field(default_factory=lambda: {
        'electron': 0.3, 'positron': 0.3, 'photon': 0.3, 'proton': 0.1,
    })

    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Particle Generator
# ---------------------------------------------------------------------------

def _random_direction() -> np.ndarray:
    phi       = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def _direction_at_angle(reference: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.radians(angle_deg)
    ref  = reference / np.linalg.norm(reference)
    perp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, perp)) > 0.9:
        perp = np.array([0.0, 1.0, 0.0])
    perp -= np.dot(perp, ref) * ref
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(ref, perp)
    phi = np.random.uniform(0, 2 * np.pi)
    d = (np.cos(angle_rad) * ref
         + np.sin(angle_rad) * (np.cos(phi) * perp + np.sin(phi) * perp2))
    return d / np.linalg.norm(d)


def _lorentz_boost(p3_rest: np.ndarray, E_rest: float,
                   beta: float, gamma: float, n_hat: np.ndarray
                   ) -> tuple[np.ndarray, float]:
    """Boost a 4-momentum (p3_rest, E_rest) by velocity beta along n_hat."""
    p_par     = np.dot(p3_rest, n_hat)
    p_perp    = p3_rest - p_par * n_hat
    E_lab     = gamma * (E_rest + beta * p_par)
    p_par_lab = gamma * (p_par  + beta * E_rest)
    return p_par_lab * n_hat + p_perp, E_lab


def _generate_pair_kinematics(m_parent_mev: float,
                               E_transition_mev: float
                               ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate physical e+e- directions from parent → e+e- via 4He* de-excitation.

    Physics (matching Geant4 X17PrimaryGenerator.cc):
      • 4He* is at rest in the lab; it emits the parent isotropically with
        momentum p = sqrt(E_transition² - m_parent²).
      • Parent decays back-to-back in its rest frame (random isotropic axis).
      • Both leptons are boosted to the lab.

    Returns (dir_em, dir_ep, m_parent_mev, ke_em_mev, ke_ep_mev)
    where ke_* are the lab-frame kinetic energies.
    """
    me       = _ME_MEV
    p_parent = np.sqrt(max(E_transition_mev**2 - m_parent_mev**2, 0.0))
    beta     = p_parent / E_transition_mev
    gamma    = E_transition_mev / m_parent_mev
    n_hat    = _random_direction()          # isotropic parent emission direction

    E_e  = m_parent_mev / 2.0
    p_e  = np.sqrt(max(E_e**2 - me**2, 0.0))
    d    = _random_direction()              # random decay axis in rest frame

    p3_em, E_em = _lorentz_boost( p_e * d, E_e, beta, gamma, n_hat)
    p3_ep, E_ep = _lorentz_boost(-p_e * d, E_e, beta, gamma, n_hat)

    mag_em = np.linalg.norm(p3_em)
    mag_ep = np.linalg.norm(p3_ep)
    dir_em = p3_em / mag_em if mag_em > 1e-10 else  d
    dir_ep = p3_ep / mag_ep if mag_ep > 1e-10 else -d

    return dir_em, dir_ep, m_parent_mev, max(E_em - me, 0.0), max(E_ep - me, 0.0)


def _sample_ipc_mass(E_transition_mev: float) -> float:
    """
    Sample e+e- invariant mass from dN/dMee ∝ 1/Mee on [2mₑ, E_transition].
    Inverse-CDF: Mee = 2mₑ · (E_transition / 2mₑ)^U,  U ~ Uniform(0,1).
    Matches Geant4 X17PrimaryGenerator::GenerateIPC.
    """
    return 2.0 * _ME_MEV * (E_transition_mev / (2.0 * _ME_MEV)) ** np.random.uniform()


def _build_detector_planes(cfg: SimConfig) -> list[Detector]:
    """Build the full detector-plane list for one configuration.

    Single source of truth used by BOTH the serial path
    (MicromegasSimulation._build_detectors) and the parallel path
    (_run_event_batch).  These previously used different conventions
    (back-face vs front-face planes), giving ~40% different MM acceptance
    between n_workers=1 and n_workers>1.

    Each plane sits at the BACK face of its active volume (furthest from the
    target), so a hit requires the track to traverse the full active
    thickness.
    """
    d_mm    = cfg.detector_distance
    d_scint = d_mm + cfg.mm_drift_gap + cfg.gap_mm_to_scint

    if cfg.stack_order == 'standard':
        d_ls1  = d_scint + cfg.gap_scint_to_liq
        d_ls2  = d_ls1   + cfg.gap_ls1_to_ls2
        d_back = d_ls2   + cfg.gap_ls2_to_backscint
    else:   # 'back_scint_first'
        d_back = d_scint + cfg.gap_scint_to_backscint
        d_ls1  = d_back  + cfg.gap_backscint_to_ls1
        d_ls2  = d_ls1   + cfg.gap_ls1_to_ls2

    # Back scint bar u-offsets: bars are ±u_off from the arm centreline
    u_off = (cfg.back_scint_size_u + cfg.back_scint_gap) / 2.0

    d_mm_plane    = d_mm    + cfg.mm_drift_gap
    d_scint_plane = d_scint + cfg.scint_wall_thickness    / 2.0
    d_ls1_plane   = d_ls1   + cfg.liq_scint_thickness     / 2.0
    d_ls2_plane   = d_ls2   + cfg.liq_scint_2_thickness   / 2.0
    d_back_plane  = d_back  + cfg.back_scint_thickness    / 2.0

    detectors = []
    for side in range(4):
        detectors.append(Detector(side, d_mm_plane,    *cfg.detector_size,    'mm'))
        detectors.append(Detector(side, d_scint_plane,  *cfg.scint_wall_size,  'scint_wall'))
        detectors.append(Detector(side, d_ls1_plane,    *cfg.liq_scint_size,   'liq_scint_1'))
        detectors.append(Detector(side, d_ls2_plane,    *cfg.liq_scint_2_size, 'liq_scint_2'))
        # Two back-scint bars per arm, displaced ±u_off in u
        detectors.append(Detector(side, d_back_plane,
                                  cfg.back_scint_size_u, cfg.back_scint_size_v,
                                  'back_scint', offset_u=+u_off))
        detectors.append(Detector(side, d_back_plane,
                                  cfg.back_scint_size_u, cfg.back_scint_size_v,
                                  'back_scint', offset_u=-u_off))
    return detectors


# ── Geant4 response loading (cached per path/estimator) ─────────────────────
_g4_response_cache: dict = {}


def _get_g4_response(cfg: SimConfig):
    """Load (and cache) the Geant4Response configured in cfg, or None."""
    path = getattr(cfg, 'g4_response_path', None)
    if not path:
        return None
    key = (path, cfg.g4_direction_estimator,
           cfg.g4_energy_method, cfg.g4_energy_corrected)
    if key not in _g4_response_cache:
        from geant4_response import Geant4Response
        _g4_response_cache[key] = Geant4Response(
            path,
            direction_estimator=cfg.g4_direction_estimator,
            energy_method=cfg.g4_energy_method,
            energy_corrected=cfg.g4_energy_corrected)
    return _g4_response_cache[key]


def _pair_mass_estimate(cfg: SimConfig, resp, ha: 'Hit', hb: 'Hit',
                        p_ke: dict, true_m: float) -> float:
    """Reconstructed pair mass: Geant4 calorimeter response when available,
    otherwise the legacy flat Gaussian around the true mass."""
    if resp is not None and cfg.g4_mass_from_response:
        ke_a = p_ke.get(ha.particle_id, 0.0)
        ke_b = p_ke.get(hb.particle_id, 0.0)
        if ke_a > 0.0 and ke_b > 0.0:
            # order as (e-, e+); tables differ only mildly but keeps convention
            if ha.pid == 'positron':
                ha, hb = hb, ha
                ke_a, ke_b = ke_b, ke_a
            if cfg.g4_use_constraint:
                return resp.constrained_pair_mass(ke_a, ha.direction,
                                                  ke_b, hb.direction)
            return resp.reco_pair_mass(ke_a, ha.direction, ke_b, hb.direction)
    return float(np.random.normal(true_m, cfg.inv_mass_sigma))


class ParticleGenerator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._g4resp = _get_g4_response(cfg)
        self._event_counter    = 0
        self._particle_counter = 0
        self._pair_counter     = 0
        self.true_signal_angles: list[float] = []
        self.true_bg_angles:    list[float] = []

    def _next_event_id(self):
        eid = self._event_counter; self._event_counter += 1; return eid

    def _next_particle_id(self):
        pid = self._particle_counter; self._particle_counter += 1; return pid

    def _next_pair_id(self):
        pair_id = self._pair_counter; self._pair_counter += 1; return pair_id

    def _random_pid(self):
        pids    = list(self.cfg.random_pid_weights.keys())
        weights = np.array(list(self.cfg.random_pid_weights.values()))
        weights /= weights.sum()
        return np.random.choice(pids, p=weights)

    def _maybe_smear(self, direction: np.ndarray, ke_mev: float) -> np.ndarray:
        """Apply Geant4 multiple-scattering direction smearing if configured.

        Physically the dominant scatter happens in the target wall (~16 mm
        from the vertex), so the particle's actual trajectory through the
        detectors IS the smeared direction — smearing at creation gives both
        the correct acceptance and the correct reconstructed angles.
        """
        if self._g4resp is None or not self.cfg.g4_smear_directions:
            return direction
        if ke_mev <= 0.0:
            return direction
        return self._g4resp.smear_direction(direction, ke_mev)

    def _sample_origin(self) -> np.ndarray:
        """Return a particle origin: target centre or uniform inside the He-3 capsule.

        Capsule = cylinder (radius R, half-length Lh) + two hemispherical end caps.
        Rejection sampling from bounding cylinder; acceptance rate ~89%.
        """
        if not self.cfg.he_capsule_source:
            return np.zeros(3)
        R  = self.cfg.he_radius_cm
        Lh = self.cfg.he_half_length_cm
        while True:
            r   = R * np.sqrt(np.random.uniform(0, 1))
            phi = np.random.uniform(0, 2 * np.pi)
            y   = np.random.uniform(-(Lh + R), Lh + R)
            abs_y = abs(y)
            if abs_y <= Lh or r**2 + (abs_y - Lh)**2 <= R**2:
                return np.array([r * np.cos(phi), y, r * np.sin(phi)])

    def generate_event(self, event_t0: float = 0.0) -> list[Particle]:
        particles = []

        for _ in range(np.random.poisson(self.cfg.n_random)):
            t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
            particles.append(Particle(
                pid=self._random_pid(), origin=self._sample_origin(),
                direction=_random_direction(), t0=t0,
                event_id=self._next_event_id(), particle_id=self._next_particle_id(),
                pair_id=-1, source='random',
            ))

        n_bg = np.random.poisson(self.cfg.n_background_pairs)
        if n_bg > 0:
            bg_spec = self.cfg.background_spectrum
            if hasattr(bg_spec, 'sample_pairs'):
                pairs = bg_spec.sample_pairs(n_bg)
                self.true_bg_angles.extend([p[3] for p in pairs])
                for d1, d2, true_mass, angle, ke_em, ke_ep in pairs:
                    eid        = self._next_event_id()
                    t0         = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                    origin     = self._sample_origin()
                    bg_pair_id = self._next_pair_id()
                    d1, d2 = self._maybe_smear(d1, ke_em), self._maybe_smear(d2, ke_ep)
                    particles.append(Particle('electron', origin, d1, t0, eid,
                                              self._next_particle_id(), bg_pair_id,
                                              'background_pair', true_inv_mass=true_mass,
                                              ke_mev=ke_em))
                    particles.append(Particle('positron', origin, d2, t0, eid,
                                              self._next_particle_id(), bg_pair_id,
                                              'background_pair', true_inv_mass=true_mass,
                                              ke_mev=ke_ep))
            else:
                angles = bg_spec.sample(n_bg)
                self.true_bg_angles.extend(angles.tolist())
                for angle in angles:
                    eid        = self._next_event_id()
                    t0         = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                    origin     = self._sample_origin()
                    d1         = _random_direction()
                    d2         = _direction_at_angle(d1, angle)
                    bg_pair_id = self._next_pair_id()
                    particles.append(Particle('electron', origin, d1, t0, eid,
                                              self._next_particle_id(), bg_pair_id, 'background_pair'))
                    particles.append(Particle('positron', origin, d2, t0, eid,
                                              self._next_particle_id(), bg_pair_id, 'background_pair'))

        n_sig = np.random.poisson(self.cfg.n_signal)
        if n_sig > 0:
            sig_spec = self.cfg.signal_spectrum
            if hasattr(sig_spec, 'sample_pairs'):
                pairs = sig_spec.sample_pairs(n_sig)
                self.true_signal_angles.extend([p[3] for p in pairs])
                for d1, d2, true_mass, angle, ke_em, ke_ep in pairs:
                    eid         = self._next_event_id()
                    t0          = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                    origin      = self._sample_origin()
                    sig_pair_id = self._next_pair_id()
                    d1, d2 = self._maybe_smear(d1, ke_em), self._maybe_smear(d2, ke_ep)
                    particles.append(Particle('electron', origin, d1, t0, eid,
                                              self._next_particle_id(), sig_pair_id,
                                              'signal', true_inv_mass=true_mass,
                                              ke_mev=ke_em))
                    particles.append(Particle('positron', origin, d2, t0, eid,
                                              self._next_particle_id(), sig_pair_id,
                                              'signal', true_inv_mass=true_mass,
                                              ke_mev=ke_ep))
            else:
                angles = sig_spec.sample(n_sig)
                self.true_signal_angles.extend(angles.tolist())
                for angle in angles:
                    eid         = self._next_event_id()
                    t0          = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                    origin      = self._sample_origin()
                    d1          = _random_direction()
                    d2          = _direction_at_angle(d1, angle)
                    sig_pair_id = self._next_pair_id()
                    particles.append(Particle('electron', origin, d1, t0, eid,
                                              self._next_particle_id(), sig_pair_id, 'signal'))
                    particles.append(Particle('positron', origin, d2, t0, eid,
                                              self._next_particle_id(), sig_pair_id, 'signal'))

        return particles

    def generate_all(self) -> list[Particle]:
        all_particles = []
        for i in range(self.cfg.n_events):
            all_particles.extend(self.generate_event(i * self.cfg.time_spread))
        return all_particles


# ---------------------------------------------------------------------------
# Propagation & Reconstruction
# ---------------------------------------------------------------------------

class Propagator:
    """Straight-line propagation from origin to all detector planes."""

    def __init__(self, detectors: list[Detector], cfg: SimConfig):
        self.detectors = detectors
        self.cfg       = cfg

    def _collect_intersections(self, particle: Particle):
        results = []
        for det in self.detectors:
            hit_3d, local_2d = det.intersect(particle.origin, particle.direction)
            if local_2d is None:
                continue
            t = np.linalg.norm(hit_3d - particle.origin)
            results.append((t, det, hit_3d, local_2d))
        results.sort(key=lambda x: x[0])
        return results

    def _build_hit(self, particle: Particle, det: Detector,
                   hit_3d: np.ndarray, local_2d: np.ndarray) -> Hit:
        travel_dist = np.linalg.norm(hit_3d - particle.origin)
        true_time   = particle.t0 + travel_dist / 30.0   # c ~ 30 cm/ns
        reco_pos    = local_2d + np.random.normal(0, self.cfg.spatial_resolution, 2)
        reco_time   = true_time + np.random.normal(0, self.cfg.time_resolution)
        return Hit(
            detector_id=det.side,
            true_pos=local_2d,
            reco_pos=reco_pos,
            true_time=true_time,
            reco_time=reco_time,
            direction=particle.direction.copy(),
            pid=particle.pid,
            event_id=particle.event_id,
            particle_id=particle.particle_id,
            pair_id=particle.pair_id,
            source=particle.source,
            detector_type=det.det_type,
        )

    def propagate(self, particle: Particle) -> list[Hit]:
        return [self._build_hit(particle, det, hit_3d, local_2d)
                for _, det, hit_3d, local_2d in self._collect_intersections(particle)]


# ---------------------------------------------------------------------------
# Trigger Functions
# ---------------------------------------------------------------------------

def scint_single_sides(hits: list[Hit],
                       second_layer: str = 'liq_scint_1') -> set[int]:
    """Side IDs where scint_wall AND second_layer both have ≥1 hit."""
    sw  = {h.detector_id for h in hits if h.detector_type == 'scint_wall'}
    ly2 = {h.detector_id for h in hits if h.detector_type == second_layer}
    return sw & ly2


def event_passes_scint_single(hits: list[Hit],
                              second_layer: str = 'liq_scint_1') -> bool:
    return len(scint_single_sides(hits, second_layer)) >= 1


def event_passes_scint_double(hits: list[Hit],
                              second_layer: str = 'liq_scint_1') -> bool:
    return len(scint_single_sides(hits, second_layer)) >= 2


def mm_fired_sides(hits: list[Hit]) -> set[int]:
    return {h.detector_id for h in hits if h.detector_type == 'mm'}


def event_passes_mm_any(hits: list[Hit]) -> bool:
    return len(mm_fired_sides(hits)) >= 1


def event_passes_mm_double(hits: list[Hit]) -> bool:
    return len(mm_fired_sides(hits)) >= 2


def _classify_pair_source(ha: Hit, hb: Hit) -> str:
    if (ha.source == 'signal' and hb.source == 'signal'
            and ha.pair_id == hb.pair_id and ha.pair_id != -1):
        return 'signal'
    if (ha.source == 'background_pair' and hb.source == 'background_pair'
            and ha.pair_id == hb.pair_id and ha.pair_id != -1):
        return 'background_pair'
    return 'random'


def find_coincident_pairs(hits: list[Hit], cfg: SimConfig) -> list[tuple[Hit, Hit]]:
    """Find all pairs of MM hits within the coincidence window."""
    pairs = []
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            ha, hb = hits[i], hits[j]
            if ha.particle_id == hb.particle_id:
                continue
            if not cfg.allow_same_detector_pairs and ha.detector_id == hb.detector_id:
                continue
            if abs(ha.reco_time - hb.reco_time) <= cfg.coincidence_window:
                pairs.append((ha, hb))
    return pairs


def merge_hits(hits: list[Hit], cfg: SimConfig) -> list[Hit]:
    """
    Merge nearby hits on the same detector panel (greedy single-pass).
    Groups by (detector_id, detector_type) so planes on the same side are
    never merged with each other.
    """
    if not hits:
        return hits

    by_det: dict[tuple, list[Hit]] = {}
    for h in hits:
        by_det.setdefault((h.detector_id, h.detector_type), []).append(h)

    merged: list[Hit] = []
    for (det_id, det_type), det_hits in by_det.items():
        det_hits.sort(key=lambda h: h.reco_time)
        used = [False] * len(det_hits)
        for i, seed in enumerate(det_hits):
            if used[i]:
                continue
            cluster = [seed]
            used[i] = True
            for j in range(i + 1, len(det_hits)):
                if used[j]:
                    continue
                other = det_hits[j]
                if other.reco_time - seed.reco_time > cfg.merge_time_threshold:
                    break
                if np.linalg.norm(other.reco_pos - seed.reco_pos) < cfg.merge_spatial_threshold:
                    cluster.append(other)
                    used[j] = True
            if len(cluster) == 1:
                merged.append(seed)
            else:
                merged_pos  = np.mean([h.reco_pos  for h in cluster], axis=0)
                merged_time = np.mean([h.reco_time for h in cluster])
                priority    = {'signal': 0, 'background_pair': 1, 'random': 2}
                best        = min(cluster, key=lambda h: priority[h.source])
                merged.append(Hit(
                    detector_id=det_id,
                    true_pos=seed.true_pos,
                    reco_pos=merged_pos,
                    true_time=seed.true_time,
                    reco_time=merged_time,
                    direction=best.direction,
                    pid=best.pid,
                    event_id=best.event_id,
                    particle_id=best.particle_id,
                    pair_id=best.pair_id,
                    source=best.source,
                    detector_type=det_type,
                ))
    return merged


def angular_separation_from_hits(ha: Hit, hb: Hit) -> float:
    cos_angle = np.clip(np.dot(ha.direction, hb.direction), -1, 1)
    return np.degrees(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# Main Simulation Runner
# ---------------------------------------------------------------------------

class MicromegasSimulation:
    """
    Top-level simulation object. Build, run, and analyse in one place.

    Primary triggers (scintillator-based)
    --------------------------------------
    'single'    scint_wall AND trigger_second_layer on ≥1 side
    'double'    single on ≥2 sides  ← default analysis trigger

    Comparison triggers (MM-only)
    ------------------------------------------------------
    'mm_any'    any MM panel hit
    'mm_double' any two MM panels hit

    Key attributes after run()
    --------------------------
    pair_stats               dict with efficiency and acceptance fractions
    scint_trigger_stats      per-track efficiency dict
    angular_separations_double / _single / _mm_any / _mm_double
    pair_sources_double / ...
    hits                     merged MM hits (serial mode only)
    """

    def __init__(self, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

        self.detectors  = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)
        self.generator  = ParticleGenerator(self.cfg)

        self.particles:     list[Particle] = []
        self.hits:          list[Hit] = []
        self.hits_premerge: list[Hit] = []

        self.coincident_pairs_single:    list[tuple[Hit, Hit]] = []
        self.coincident_pairs_double:    list[tuple[Hit, Hit]] = []
        self.angular_separations_single: np.ndarray = np.array([])
        self.angular_separations_double: np.ndarray = np.array([])
        self.pair_sources_single: list[str] = []
        self.pair_sources_double: list[str] = []

        self.coincident_pairs_mm_any:     list[tuple[Hit, Hit]] = []
        self.coincident_pairs_mm_double:  list[tuple[Hit, Hit]] = []
        self.angular_separations_mm_any:    np.ndarray = np.array([])
        self.angular_separations_mm_double: np.ndarray = np.array([])
        self.pair_sources_mm_any:    list[str] = []
        self.pair_sources_mm_double: list[str] = []

        self.angular_separations_both_bs: np.ndarray = np.array([])
        self.pair_sources_both_bs:        list[str]  = []
        self.inv_masses_both_bs:          np.ndarray = np.array([])

        self.true_signal_angles: np.ndarray = np.array([])
        self.true_bg_angles:     np.ndarray = np.array([])
        self.scint_trigger_stats: dict = {}
        self.pair_stats:          dict = {}

    # Backward-compat aliases pointing to the default (scint double) trigger
    @property
    def coincident_pairs(self):
        return self.coincident_pairs_double

    @property
    def angular_separations(self):
        return self.angular_separations_double

    @property
    def pair_sources(self):
        return self.pair_sources_double

    # ------------------------------------------------------------------
    def _build_detectors(self) -> list[Detector]:
        return _build_detector_planes(self.cfg)

    def _get_stack_distances(self):
        """Return (d_mm, d_scint, d_ls1, d_ls2, d_back) in cm."""
        cfg     = self.cfg
        d_mm    = cfg.detector_distance
        d_scint = d_mm + cfg.mm_drift_gap + cfg.gap_mm_to_scint
        if cfg.stack_order == 'standard':
            d_ls1  = d_scint + cfg.gap_scint_to_liq
            d_ls2  = d_ls1   + cfg.gap_ls1_to_ls2
            d_back = d_ls2   + cfg.gap_ls2_to_backscint
        else:
            d_back = d_scint + cfg.gap_scint_to_backscint
            d_ls1  = d_back  + cfg.gap_backscint_to_ls1
            d_ls2  = d_ls1   + cfg.gap_ls1_to_ls2
        return d_mm, d_scint, d_ls1, d_ls2, d_back

    def update_distance(self, new_distance: float):
        self.cfg.detector_distance = new_distance
        self.detectors  = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)

    def update_resolution(self, spatial_cm: float = None, time_ns: float = None):
        if spatial_cm is not None:
            self.cfg.spatial_resolution = spatial_cm
        if time_ns is not None:
            self.cfg.time_resolution = time_ns
        self.propagator = Propagator(self.detectors, self.cfg)

    # ------------------------------------------------------------------
    def run(self, n_workers: int = 1):
        """
        Simulate n_events readout windows.

        n_workers > 1 uses multiprocessing (hits/particles not stored in that mode).
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        n    = self.cfg.n_events
        mode = 'serial' if n_workers == 1 else f'{n_workers} workers'
        print(f"[Sim] Running {n} events ({mode})...")

        self.particles     = []
        self.hits          = []
        self.hits_premerge = []

        # ------------------------------------------------------------------
        # Serial path
        # ------------------------------------------------------------------
        if n_workers == 1:
            angles_single, sources_single, cpairs_single = [], [], []
            angles_double, sources_double, cpairs_double = [], [], []
            angles_mm_any,    sources_mm_any,    cpairs_mm_any    = [], [], []
            angles_mm_double, sources_mm_double, cpairs_mm_double = [], [], []
            angles_both_bs, sources_both_bs, masses_both_bs       = [], [], []

            n_mm_raw = n_hits_away = n_mm_final = 0
            n_sig_prod = n_sig_det = n_sig_trig = n_sig_both_bs = 0
            n_bg_prod  = n_bg_det  = 0
            n_trk_mm = n_trk_sw = n_trk_scint_single = n_trk_back_scint = 0

            sl   = self.cfg.trigger_second_layer
            resp = _get_g4_response(self.cfg)

            iterator = range(n)
            if _tqdm is not None:
                iterator = _tqdm(iterator, desc='Simulating', unit='evt',
                                 dynamic_ncols=True)

            for i_evt in iterator:
                event_t0      = i_evt * self.cfg.time_spread
                evt_particles = self.generator.generate_event(event_t0)
                self.particles.extend(evt_particles)

                evt_mm_raw_hits    = []
                evt_scint_raw_hits = []
                p_mm_sides:  dict[int, set]   = {}   # particle_id → set of MM sides
                p_bs_sides:  dict[int, set]   = {}   # particle_id → set of back_scint sides
                p_true_mass: dict[int, float] = {}   # particle_id → true invariant mass (MeV)
                p_ke:        dict[int, float] = {}   # particle_id → true KE (MeV)

                for p in evt_particles:
                    p_hits = self.propagator.propagate(p)
                    mm_h   = [h for h in p_hits if h.detector_type == 'mm']
                    sc_h   = [h for h in p_hits if h.detector_type != 'mm']
                    evt_mm_raw_hits.extend(mm_h)
                    evt_scint_raw_hits.extend(sc_h)

                    mm_s  = {h.detector_id for h in mm_h}
                    sw_s  = {h.detector_id for h in sc_h if h.detector_type == 'scint_wall'}
                    ly2_s = {h.detector_id for h in sc_h if h.detector_type == sl}
                    bs_s  = {h.detector_id for h in sc_h if h.detector_type == 'back_scint'}

                    p_mm_sides[p.particle_id]  = mm_s
                    p_bs_sides[p.particle_id]  = bs_s
                    p_ke[p.particle_id]        = p.ke_mev
                    if p.true_inv_mass != 0.0:
                        p_true_mass[p.particle_id] = p.true_inv_mass

                    for side in mm_s:
                        n_trk_mm += 1
                        if side in sw_s:
                            n_trk_sw += 1
                        if side in sw_s and side in ly2_s:
                            n_trk_scint_single += 1
                        if side in bs_s:
                            n_trk_back_scint += 1

                self.hits_premerge.extend(evt_mm_raw_hits)
                n_mm_raw += len(evt_mm_raw_hits)

                if self.cfg.merge_hits:
                    evt_mm_hits  = merge_hits(evt_mm_raw_hits, self.cfg)
                    n_hits_away += len(evt_mm_raw_hits) - len(evt_mm_hits)
                else:
                    evt_mm_hits = evt_mm_raw_hits
                self.hits.extend(evt_mm_hits)
                n_mm_final += len(evt_mm_hits)

                evt_pairs  = find_coincident_pairs(evt_mm_hits, self.cfg)
                evt_angles = [angular_separation_from_hits(a, b) for a, b in evt_pairs]
                evt_src    = [_classify_pair_source(a, b)        for a, b in evt_pairs]

                for (ha, hb), ang, src in zip(evt_pairs, evt_angles, evt_src):
                    if (p_bs_sides.get(ha.particle_id, set()) & p_mm_sides.get(ha.particle_id, set())
                            and p_bs_sides.get(hb.particle_id, set()) & p_mm_sides.get(hb.particle_id, set())):
                        angles_both_bs.append(ang)
                        sources_both_bs.append(src)
                        true_m = p_true_mass.get(ha.particle_id,
                                 p_true_mass.get(hb.particle_id, self.cfg.inv_mass_mean))
                        masses_both_bs.append(
                            _pair_mass_estimate(self.cfg, resp, ha, hb, p_ke, true_m))

                evt_all    = evt_mm_hits + evt_scint_raw_hits
                dbl_fired  = event_passes_scint_double(evt_all, sl)

                if event_passes_scint_single(evt_all, sl):
                    cpairs_single.extend(evt_pairs)
                    angles_single.extend(evt_angles); sources_single.extend(evt_src)
                if dbl_fired:
                    cpairs_double.extend(evt_pairs)
                    angles_double.extend(evt_angles); sources_double.extend(evt_src)
                if event_passes_mm_any(evt_mm_hits):
                    cpairs_mm_any.extend(evt_pairs)
                    angles_mm_any.extend(evt_angles); sources_mm_any.extend(evt_src)
                if event_passes_mm_double(evt_mm_hits):
                    cpairs_mm_double.extend(evt_pairs)
                    angles_mm_double.extend(evt_angles); sources_mm_double.extend(evt_src)

                # Pair detection efficiency
                detected = {h.particle_id for h in evt_mm_hits}
                ptmap: dict[int, set] = {}
                for p in evt_particles:
                    if p.pair_id != -1:
                        ptmap.setdefault(p.pair_id, set()).add(p.particle_id)

                for pid, ptids in ptmap.items():
                    src      = next(p.source for p in evt_particles if p.pair_id == pid)
                    both_mm  = ptids.issubset(detected)
                    # both particles have back_scint hit on the same side as their MM hit
                    both_bs  = all(
                        bool(p_mm_sides.get(p_id, set()) & p_bs_sides.get(p_id, set()))
                        for p_id in ptids
                    )
                    if src == 'signal':
                        n_sig_prod += 1
                        if both_mm:
                            n_sig_det += 1
                            if dbl_fired:
                                n_sig_trig += 1
                            if both_bs:
                                n_sig_both_bs += 1
                    elif src == 'background_pair':
                        n_bg_prod += 1
                        if both_mm:
                            n_bg_det += 1

            self.coincident_pairs_single    = cpairs_single
            self.coincident_pairs_double    = cpairs_double
            self.coincident_pairs_mm_any    = cpairs_mm_any
            self.coincident_pairs_mm_double = cpairs_mm_double
            self.true_signal_angles = np.array(self.generator.true_signal_angles)
            self.true_bg_angles     = np.array(self.generator.true_bg_angles)
            results = [dict(
                angles_single=angles_single, sources_single=sources_single,
                angles_double=angles_double, sources_double=sources_double,
                angles_mm_any=angles_mm_any, sources_mm_any=sources_mm_any,
                angles_mm_double=angles_mm_double, sources_mm_double=sources_mm_double,
                angles_both_bs=angles_both_bs, sources_both_bs=sources_both_bs,
                masses_both_bs=masses_both_bs,
                n_mm_raw=n_mm_raw, n_hits_away=n_hits_away, n_mm_final=n_mm_final,
                n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
                n_sig_trig=n_sig_trig, n_sig_both_bs=n_sig_both_bs,
                n_bg_prod=n_bg_prod,   n_bg_det=n_bg_det,
                n_trk_mm=n_trk_mm,     n_trk_sw=n_trk_sw,
                n_trk_scint_single=n_trk_scint_single,
                n_trk_back_scint=n_trk_back_scint,
                true_signal_angles=self.generator.true_signal_angles,
                true_bg_angles=self.generator.true_bg_angles,
            )]

        # ------------------------------------------------------------------
        # Parallel path
        # ------------------------------------------------------------------
        else:
            import concurrent.futures
            from concurrent.futures import as_completed

            chunk_size = max(1, n // (n_workers * 20))
            chunks, offset, chunk_idx = [], 0, 0
            main_seed = self.cfg.seed
            while offset < n:
                size = min(chunk_size, n - offset)
                seed = ((main_seed + chunk_idx * 7919) % (2**31)
                        if main_seed is not None else None)
                chunks.append((self.cfg, size, offset, seed))
                offset    += size
                chunk_idx += 1

            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_event_batch, c): c for c in chunks}
                it = ((_tqdm(as_completed(futures), total=len(futures),
                             desc='Simulating', unit='chunk', dynamic_ncols=True)
                       if _tqdm is not None else as_completed(futures)))
                for fut in it:
                    results.append(fut.result())

            self.coincident_pairs_single    = []
            self.coincident_pairs_double    = []
            self.coincident_pairs_mm_any    = []
            self.coincident_pairs_mm_double = []
            self.true_signal_angles = np.array(
                sum((r['true_signal_angles'] for r in results), []))
            self.true_bg_angles = np.array(
                sum((r['true_bg_angles'] for r in results), []))

        # ------------------------------------------------------------------
        # Aggregate results (common to both paths)
        # ------------------------------------------------------------------
        def _cat(key):
            return sum((r[key] for r in results), [])

        self.angular_separations_single    = np.array(_cat('angles_single'))
        self.pair_sources_single           = _cat('sources_single')
        self.angular_separations_double    = np.array(_cat('angles_double'))
        self.pair_sources_double           = _cat('sources_double')
        self.angular_separations_mm_any    = np.array(_cat('angles_mm_any'))
        self.pair_sources_mm_any           = _cat('sources_mm_any')
        self.angular_separations_mm_double = np.array(_cat('angles_mm_double'))
        self.pair_sources_mm_double        = _cat('sources_mm_double')
        self.angular_separations_both_bs   = np.array(_cat('angles_both_bs'))
        self.pair_sources_both_bs          = _cat('sources_both_bs')
        self.inv_masses_both_bs            = np.array(_cat('masses_both_bs'))

        n_mm_raw      = sum(r['n_mm_raw']         for r in results)
        n_hits_away   = sum(r['n_hits_away']       for r in results)
        n_mm_final    = sum(r['n_mm_final']        for r in results)
        n_sig_prod    = sum(r['n_sig_prod']        for r in results)
        n_sig_det     = sum(r['n_sig_det']         for r in results)
        n_sig_trig    = sum(r['n_sig_trig']        for r in results)
        n_sig_both_bs = sum(r['n_sig_both_bs']     for r in results)
        n_bg_prod     = sum(r['n_bg_prod']         for r in results)
        n_bg_det      = sum(r['n_bg_det']          for r in results)
        n_trk_mm      = sum(r['n_trk_mm']          for r in results)
        n_trk_sw      = sum(r['n_trk_sw']          for r in results)
        n_trk_ss      = sum(r['n_trk_scint_single'] for r in results)
        n_trk_bs      = sum(r['n_trk_back_scint']  for r in results)

        self.pair_stats = {
            'signal_produced':               n_sig_prod,
            'signal_detected':               n_sig_det,
            'signal_triggered':              n_sig_trig,
            'signal_both_backscint':         n_sig_both_bs,
            'signal_efficiency':             n_sig_det     / n_sig_prod if n_sig_prod else 0.0,
            'trigger_efficiency':            n_sig_trig    / n_sig_det  if n_sig_det  else 0.0,
            'trigger_miss_fraction':    1 - (n_sig_trig    / n_sig_det  if n_sig_det  else 0.0),
            'calorimetry_complete_fraction': n_sig_both_bs / n_sig_det  if n_sig_det  else 0.0,
            'calorimetry_incomplete_fraction':
                1 - (n_sig_both_bs / n_sig_det if n_sig_det else 0.0),
            'bg_produced':   n_bg_prod,
            'bg_detected':   n_bg_det,
            'bg_efficiency': n_bg_det / n_bg_prod if n_bg_prod else 0.0,
        }
        self.scint_trigger_stats = {
            'n_tracks_mm':               n_trk_mm,
            'n_tracks_scint_wall':       n_trk_sw,
            'n_tracks_scint_single':     n_trk_ss,
            'n_tracks_back_scint':       n_trk_bs,
            'scint_wall_efficiency':     n_trk_sw / n_trk_mm if n_trk_mm else 0.0,
            'scint_single_efficiency':   n_trk_ss / n_trk_mm if n_trk_mm else 0.0,
            'back_scint_efficiency':     n_trk_bs / n_trk_mm if n_trk_mm else 0.0,
            'liq_scint_given_scint_wall': n_trk_ss / n_trk_sw if n_trk_sw else 0.0,
        }

        print(f"  MM hits (raw)              : {n_mm_raw}")
        if self.cfg.merge_hits:
            print(f"  MM hits merged away        : {n_hits_away}")
        print(f"  MM hits (final)            : {n_mm_final}")
        print(f"  Pairs (scint single)       : {len(self.angular_separations_single)}")
        print(f"  Pairs (scint double)       : {len(self.angular_separations_double)}")
        print(f"  Pairs (MM any)             : {len(self.angular_separations_mm_any)}")
        print(f"  Pairs (MM double)          : {len(self.angular_separations_mm_double)}")
        st = self.scint_trigger_stats
        print(f"  Track eff: MM → scint wall : {st['scint_wall_efficiency']*100:.1f}%")
        sl_label = self.cfg.trigger_second_layer
        print(f"  Track eff: MM → {sl_label:<12s}: {st['scint_single_efficiency']*100:.1f}%")
        if sl_label != 'back_scint':
            print(f"  Track eff: MM → back_scint : {st['back_scint_efficiency']*100:.1f}%")
        print("[Sim] Done.")

    def run_display_sample(self, n_sample: int = 5_000):
        """Run a short serial batch to populate self.hits for display plots."""
        import copy
        tmp_cfg = copy.copy(self.cfg)
        tmp_cfg.n_events = n_sample
        tmp_gen = ParticleGenerator(tmp_cfg)

        self.hits          = []
        self.hits_premerge = []
        self.particles     = []

        for i in range(n_sample):
            evt_particles = tmp_gen.generate_event(i * tmp_cfg.time_spread)
            self.particles.extend(evt_particles)
            evt_mm_raw = []
            for p in evt_particles:
                evt_mm_raw.extend(
                    h for h in self.propagator.propagate(p) if h.detector_type == 'mm'
                )
            self.hits_premerge.extend(evt_mm_raw)
            self.hits.extend(
                merge_hits(evt_mm_raw, self.cfg) if self.cfg.merge_hits else evt_mm_raw
            )
        print(f"[Sim] Display sample: {len(self.hits)} MM hits from {n_sample} events.")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_geometry(self, ax=None):
        """
        Top-down XZ view of all detector layers (to scale).
        Beam runs along +Y (into page). Back scint bars drawn individually.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        cfg = self.cfg
        d_mm, d_scint, d_ls1, d_ls2, d_back = self._get_stack_distances()
        u_off = (cfg.back_scint_size_u + cfg.back_scint_gap) / 2.0

        def _lbl(name, sz, thick):
            return (f'{name}  {sz[0]:.0f}×{sz[1]:.0f} cm (u×v), t={thick:.1f} cm')

        # (front_dist, half_u, thickness, color, alpha, label)
        rect_layers = [
            (d_mm,    cfg.detector_size[0]   / 2, cfg.mm_drift_gap,
             '#2196F3', 0.55, _lbl('MM',         cfg.detector_size,    cfg.mm_drift_gap)),
            (d_scint, cfg.scint_wall_size[0] / 2, cfg.scint_wall_thickness,
             '#FF9800', 0.75, _lbl('Scint wall',  cfg.scint_wall_size,  cfg.scint_wall_thickness)),
            (d_ls1,   cfg.liq_scint_size[0]  / 2, cfg.liq_scint_thickness,
             '#4CAF50', 0.60, _lbl('LS-1',        cfg.liq_scint_size,   cfg.liq_scint_thickness)),
            (d_ls2,   cfg.liq_scint_2_size[0]/ 2, cfg.liq_scint_2_thickness,
             '#009688', 0.55, _lbl('LS-2',        cfg.liq_scint_2_size, cfg.liq_scint_2_thickness)),
        ]

        from matplotlib.patches import Patch
        legend_handles = []

        for dist, half_u, thick, color, alpha, label in rect_layers:
            rect_params = [
                dict(xy=(dist,            -half_u), width=thick,      height=2 * half_u),
                dict(xy=(-(dist + thick), -half_u), width=thick,      height=2 * half_u),
                dict(xy=(-half_u,          dist),   width=2 * half_u, height=thick),
                dict(xy=(-half_u,        -(dist + thick)), width=2 * half_u, height=thick),
            ]
            for rp in rect_params:
                ax.add_patch(mpatches.Rectangle(
                    **rp, linewidth=0.4, edgecolor=color,
                    facecolor=color, alpha=alpha, zorder=2))
            legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=alpha, label=label))

        # Back scint: two bars per arm, with gap
        bs_color = '#E91E63'
        bs_thick = cfg.back_scint_thickness
        bar_hu   = cfg.back_scint_size_u / 2.0

        def _add_backscint_bars(ax, dist, half_u_bar, thick, u_off, color, alpha):
            """Draw two back-scint bars per side."""
            lo = u_off - half_u_bar   # inner edge (gap side)
            hi = u_off + half_u_bar   # outer edge
            # +X arm
            ax.add_patch(mpatches.Rectangle((dist, lo),    thick, 2*half_u_bar,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            ax.add_patch(mpatches.Rectangle((dist, -hi),   thick, 2*half_u_bar,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            # -X arm
            ax.add_patch(mpatches.Rectangle((-(dist+thick), lo),   thick, 2*half_u_bar,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            ax.add_patch(mpatches.Rectangle((-(dist+thick), -hi),  thick, 2*half_u_bar,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            # +Z arm
            ax.add_patch(mpatches.Rectangle((lo,   dist),   2*half_u_bar, thick,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            ax.add_patch(mpatches.Rectangle((-hi,   dist),   2*half_u_bar, thick,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            # -Z arm
            ax.add_patch(mpatches.Rectangle((lo,   -(dist+thick)), 2*half_u_bar, thick,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))
            ax.add_patch(mpatches.Rectangle((-hi,   -(dist+thick)), 2*half_u_bar, thick,
                facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.4, zorder=2))

        _add_backscint_bars(ax, d_back, bar_hu, bs_thick, u_off, bs_color, 0.75)
        legend_handles.append(Patch(
            facecolor=bs_color, edgecolor=bs_color, alpha=0.75,
            label=f'Back scint  2×{cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f} cm'
                  f', t={bs_thick:.0f} cm'))

        if cfg.he_capsule_source:
            cap_circle = mpatches.Circle((0, 0), cfg.he_radius_cm,
                                         facecolor='lightblue', edgecolor='steelblue',
                                         alpha=0.7, linewidth=1.2, zorder=3)
            ax.add_patch(cap_circle)
            legend_handles.append(Patch(facecolor='lightblue', edgecolor='steelblue', alpha=0.7,
                                        label=f'He-3 capsule (r={cfg.he_radius_cm} cm, '
                                              f'L=±{cfg.he_half_length_cm} cm)'))

        lim = (d_back + bs_thick + u_off + bar_hu + 2) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Z [cm]')
        order_tag = cfg.stack_order.replace('_', ' ')
        ax.set_title(f'Detector Layout (XZ top-down, beam +Y into page, to scale)\nstack_order={order_tag}')
        ax.legend(handles=legend_handles, loc='upper left', fontsize=6.5)
        ax.grid(True, alpha=0.3, zorder=0)

        mm_dets = [d for d in self.detectors if d.det_type == 'mm']
        sw_dets = [d for d in self.detectors if d.det_type == 'scint_wall']
        ls_dets = [d for d in self.detectors if d.det_type == 'liq_scint_1']
        bs_dets = [d for d in self.detectors if d.det_type == 'back_scint']
        frac_mm = calculate_solid_angle_coverage(mm_dets, n_samples=5_000)
        frac_sw = calculate_solid_angle_coverage(sw_dets, n_samples=5_000)
        frac_ls = calculate_solid_angle_coverage(ls_dets, n_samples=5_000)
        frac_bs = calculate_solid_angle_coverage(bs_dets, n_samples=5_000)
        box_style = dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round')
        ax.annotate(
            f'MM solid angle:         {frac_mm*100:.1f}%\n'
            f'Scint wall solid angle: {frac_sw*100:.1f}%',
            xy=(0.04, 0.04), xycoords='axes fraction',
            ha='left', va='bottom', fontsize=8, bbox=box_style,
        )
        ax.annotate(
            f'LS-1 solid angle:       {frac_ls*100:.1f}%\n'
            f'Back scint solid angle: {frac_bs*100:.1f}%',
            xy=(0.96, 0.04), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=8, bbox=box_style,
        )
        return ax

    def plot_hits(self, ax=None):
        """Hit positions on each MM panel."""
        if not self.hits:
            warnings.warn("No hits to plot. Run sim.run() first.")
            return
        if ax is None:
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        else:
            axes = ax

        source_colors = {
            'random': 'gray', 'background_pair': 'steelblue', 'signal': 'red',
        }
        det_names = ['+X', '-X', '+Z', '-Z']

        for det_id in range(4):
            ax_ = axes[det_id]
            for src, col in source_colors.items():
                xs = [h.reco_pos[0] for h in self.hits
                      if h.detector_id == det_id and h.source == src]
                ys = [h.reco_pos[1] for h in self.hits
                      if h.detector_id == det_id and h.source == src]
                if xs:
                    ax_.scatter(xs, ys, s=4, alpha=0.5, color=col, label=src)
            half_u = self.cfg.detector_size[0] / 2
            half_v = self.cfg.detector_size[1] / 2
            rect = mpatches.Rectangle(
                (-half_u, -half_v), self.cfg.detector_size[0], self.cfg.detector_size[1],
                linewidth=1.5, edgecolor='black', facecolor='none')
            ax_.add_patch(rect)
            ax_.set_xlim(-half_u * 1.2, half_u * 1.2)
            ax_.set_ylim(-half_v * 1.2, half_v * 1.2)
            ax_.set_title(f'MM {det_names[det_id]}')
            ax_.set_xlabel('u [cm]')
            ax_.set_ylabel('v [cm]')
            ax_.set_aspect('equal')
            if det_id == 0:
                ax_.legend(fontsize=7)

        plt.tight_layout()
        return axes

    def plot_angular_separation(self, bins=36, ax=None, show_components=True,
                                trigger='double'):
        """
        Plot reconstructed angular separation for time-coincident MM pairs.

        trigger : 'single', 'double' (default), 'mm_any', 'mm_double'
        """
        _data = {
            'single':    (self.angular_separations_single,    self.pair_sources_single,
                          'Scint single (≥1 side)'),
            'double':    (self.angular_separations_double,    self.pair_sources_double,
                          'Scint double (≥2 sides)'),
            'mm_any':    (self.angular_separations_mm_any,    self.pair_sources_mm_any,
                          'MM any hit'),
            'mm_double': (self.angular_separations_mm_double, self.pair_sources_mm_double,
                          'MM double (≥2 sides)'),
        }
        angles, sources, trigger_label = _data[trigger]

        if len(angles) == 0:
            warnings.warn(f"No angular separations for trigger='{trigger}'. "
                          "Run sim.run() first.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        bin_edges = np.linspace(0, 180, bins + 1)

        if show_components:
            source_styles = {
                'random':          ('gray',      'Random combinatorial'),
                'background_pair': ('steelblue', 'Correlated background'),
                'signal':          ('red',       'Signal (e+e- ~120°)'),
            }
            angs   = np.array(angles)
            srcs   = np.array(sources)
            bottom = np.zeros(bins)
            for src, (col, label) in source_styles.items():
                mask = srcs == src
                if mask.any():
                    counts, _ = np.histogram(angs[mask], bins=bin_edges)
                    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                           bottom=bottom, color=col, alpha=0.7, label=label, align='edge')
                    bottom += counts
        else:
            ax.hist(angles, bins=bin_edges, color='steelblue', alpha=0.7,
                    label='All coincident pairs')

        ax.axvline(120, color='red', ls='--', lw=1.5, label='120° signal')
        ax.set_xlabel('Angular separation [°]')
        ax.set_ylabel('Pairs / bin')
        ax.set_title(f'Reconstructed Angular Separation\n(Trigger: {trigger_label})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        srcs_arr = np.array(sources)
        n_sig = int((srcs_arr == 'signal').sum())
        n_bg  = len(angles) - n_sig
        ax.text(0.97, 0.97,
                f'Signal:     {n_sig:,}\nBackground: {n_bg:,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.85))
        return ax

    def plot_invmass(self, bins=30, mass_range=(0.0, 30.0)):
        """
        Stacked bar histogram of reconstructed invariant mass for pairs where
        both particles hit the back-scintillators, broken down by source
        (signal / IPC background / random combinatorial).  Style mirrors
        plot_angular_separation.
        """
        masses  = self.inv_masses_both_bs
        sources = np.array(self.pair_sources_both_bs)

        if len(masses) == 0:
            warnings.warn("No back-scint coincident pairs found. Run sim.run() first.")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        bin_edges = np.linspace(mass_range[0], mass_range[1], bins + 1)

        source_styles = {
            'random':          ('gray',      'Random combinatorial'),
            'background_pair': ('steelblue', 'IPC background'),
            'signal':          ('red',       'X17 signal'),
        }
        bottom = np.zeros(bins)
        for src, (col, label) in source_styles.items():
            mask = sources == src
            if mask.any():
                counts, _ = np.histogram(masses[mask], bins=bin_edges)
                ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                       bottom=bottom, color=col, alpha=0.7, label=label, align='edge')
                bottom += counts

        ax.axvline(self.cfg.inv_mass_mean, color='red', ls='--', lw=1.5,
                   label=f'm_X17 = {self.cfg.inv_mass_mean:.1f} MeV')
        ax.set_xlabel('Reconstructed invariant mass  [MeV]')
        ax.set_ylabel('Pairs / bin')
        ax.set_title(
            f'Reconstructed Invariant Mass  —  back-scint coincidence\n'
            f'Mass model: Gaussian({self.cfg.inv_mass_mean:.1f}, '
            f'{self.cfg.inv_mass_sigma:.1f}) MeV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        n_sig = int((sources == 'signal').sum())
        n_bg  = len(masses) - n_sig
        ax.text(0.97, 0.97,
                f'Signal:     {n_sig:,}\nBackground: {n_bg:,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.85))
        return fig

    def plot_invmass_vs_angle(self, angle_bins=36, mass_bins=30,
                              mass_range=(0.0, 30.0)):
        """
        2D scatter/histogram of reconstructed invariant mass vs reconstructed
        relative angle for pairs where both particles hit the back-scintillators.

        Left panel  : X17 signal
        Right panel : IPC background
        Colorbar shows raw counts.
        """
        angles  = self.angular_separations_both_bs
        sources = np.array(self.pair_sources_both_bs)
        masses  = self.inv_masses_both_bs

        if len(angles) == 0:
            warnings.warn("No back-scint coincident pairs found. Run sim.run() first.")
            return None

        angle_edges = np.linspace(0, 180, angle_bins + 1)
        mass_edges  = np.linspace(mass_range[0], mass_range[1], mass_bins + 1)
        angle_cen   = 0.5 * (angle_edges[:-1] + angle_edges[1:])
        mass_cen    = 0.5 * (mass_edges[:-1]  + mass_edges[1:])

        src_specs = [
            ('signal',          'X17 signal'),
            ('background_pair', 'IPC background'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, (src, label) in zip(axes, src_specs):
            mask = sources == src
            n    = int(mask.sum())

            if n > 0:
                h, _, _ = np.histogram2d(
                    angles[mask], masses[mask],
                    bins=[angle_edges, mass_edges])
                im = ax.pcolormesh(angle_cen, mass_cen, h.T,
                                   cmap='viridis', vmin=0)
                plt.colorbar(im, ax=ax, label='Counts')

            ax.axhline(self.cfg.inv_mass_mean, color='yellow', lw=1.2, ls='--',
                       label=f'm_X17 = {self.cfg.inv_mass_mean:.1f} MeV')
            ax.axvline(120, color='orange', lw=1.0, ls=':', label='120° signal')
            ax.set_xlabel('Reconstructed relative angle  [°]')
            ax.set_ylabel('Reconstructed invariant mass  [MeV]')
            ax.set_xlim(0, 180)
            ax.set_ylim(mass_range)
            ax.set_title(f'{label}  (N = {n:,})\nboth particles hit back-scint')
            ax.legend(fontsize=8, loc='upper left')

        fig.suptitle(
            f'Invariant mass vs relative angle  —  back-scint coincidence\n'
            f'Mass model: Gaussian({self.cfg.inv_mass_mean:.1f}, '
            f'{self.cfg.inv_mass_sigma:.1f}) MeV  |  '
            f'MM dist = {self.cfg.detector_distance} cm',
            fontsize=11,
        )
        plt.tight_layout()
        return fig

    def plot_trigger_comparison(self, bins=36):
        """Four-panel comparison: scint single, scint double, MM any, MM double."""
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, trig in zip(axes, ['single', 'double', 'mm_any', 'mm_double']):
            self.plot_angular_separation(bins=bins, ax=ax, trigger=trig)
        fig.suptitle('Trigger Scenario Comparison', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_mm_vs_scint_comparison(self, bins=36):
        """Side-by-side: MM double trigger vs scint double trigger."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self.plot_angular_separation(bins=bins, ax=axes[0], trigger='mm_double')
        self.plot_angular_separation(bins=bins, ax=axes[1], trigger='double')

        n_mm  = len(self.angular_separations_mm_double)
        n_sc  = len(self.angular_separations_double)
        ratio = n_sc / n_mm if n_mm > 0 else 0.0
        ls = self.cfg.liq_scint_size
        mm = self.cfg.detector_size
        sw = self.cfg.scint_wall_size
        fig.suptitle(
            f'MM double vs Scint double  |  pair ratio: {ratio*100:.1f}%\n'
            f'liq scint: {ls[0]:.0f}×{ls[1]:.0f} cm  '
            f'MM: {mm[0]:.0f}×{mm[1]:.0f} cm  '
            f'scint wall: {sw[0]:.0f}×{sw[1]:.0f} cm',
            fontsize=11,
        )
        plt.tight_layout()
        return fig

    def plot_true_vs_reconstructed(self, bins=36, normalized=False, trigger=None):
        """True generated spectra vs reconstructed under one or more triggers."""
        if trigger is None:
            trigger = ['double', 'mm_double']
        if isinstance(trigger, str):
            trigger = [trigger]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bin_edges = np.linspace(0, 180, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        panel_cfg = [
            ('Signal',               self.true_signal_angles, 'red',       'signal'),
            ('Correlated Background', self.true_bg_angles,    'steelblue', 'background_pair'),
        ]
        trig_styles = {
            'single':    ('-',  2.0, 'scint single'),
            'double':    ('--', 2.5, 'scint double'),
            'mm_any':    (':',  2.0, 'MM any'),
            'mm_double': ('-.', 2.0, 'MM double'),
        }
        data_map = {
            'single':    (self.angular_separations_single,    self.pair_sources_single),
            'double':    (self.angular_separations_double,    self.pair_sources_double),
            'mm_any':    (self.angular_separations_mm_any,    self.pair_sources_mm_any),
            'mm_double': (self.angular_separations_mm_double, self.pair_sources_mm_double),
        }
        ylabel = 'Probability density' if normalized else 'Count'

        for ax, (title, true_angles, color, src_key) in zip(axes, panel_cfg):
            if len(true_angles):
                true_counts, _ = np.histogram(true_angles, bins=bin_edges)
                true_vals = (true_counts / (true_counts.sum() * bin_width)
                             if normalized and true_counts.sum() > 0
                             else true_counts.astype(float))
                ax.bar(bin_edges[:-1], true_vals, width=bin_width,
                       color=color, alpha=0.35, align='edge',
                       label=f'True generated  (N={len(true_angles)})')

            for trig in trigger:
                ls, lw, trig_name = trig_styles.get(trig, ('-', 1.5, trig))
                angles, sources = data_map[trig]
                reco = np.array([a for a, s in zip(angles, sources) if s == src_key])
                if len(reco):
                    reco_counts, _ = np.histogram(reco, bins=bin_edges)
                    reco_vals = (reco_counts / (reco_counts.sum() * bin_width)
                                 if normalized and reco_counts.sum() > 0
                                 else reco_counts.astype(float))
                    ax.step(bin_edges[:-1], reco_vals, where='post',
                            color=color, lw=lw, ls=ls,
                            label=f'Reco ({trig_name})  N={len(reco)}')

            if src_key == 'signal':
                ax.axvline(120, color='black', ls=':', lw=1, alpha=0.5)
            ax.set_title(f'{title}: True vs Reconstructed')
            ax.set_xlabel('Angular separation [°]')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        norm_tag = ' (shape normalized)' if normalized else ''
        fig.suptitle(f'True Generated vs Reconstructed{norm_tag}', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_scint_efficiency(self):
        """
        Bar chart: per-track acceptance at each stage of the detector stack.

        Bars shown:
          MM hit → + scint wall → + trigger layer (LS-1 or back_scint) → + back_scint
        In back_scint_first mode, the 'trigger layer' bar and 'back_scint' bar
        are the same detector, so they will be equal.
        """
        st = self.scint_trigger_stats
        if not st:
            warnings.warn("No stats available. Run sim.run() first.")
            return

        cfg    = self.cfg
        sl     = cfg.trigger_second_layer
        _, d_scint, d_ls1, d_ls2, d_back = self._get_stack_distances()

        if sl == 'liq_scint_1':
            trig_label = f'+ {sl}\n({cfg.liq_scint_size[0]:.0f}×{cfg.liq_scint_size[1]:.0f} cm)'
            trig_count = st['n_tracks_scint_single']
        elif sl == 'back_scint':
            trig_label = f'+ back_scint [trig]\n({cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f} cm×2)'
            trig_count = st['n_tracks_scint_single']
        else:
            trig_label = f'+ {sl}'
            trig_count = st['n_tracks_scint_single']

        sw = cfg.scint_wall_size
        labels = [
            f'MM hit\n({cfg.detector_size[0]:.0f}×{cfg.detector_size[1]:.0f} cm)',
            f'+ scint wall\n({sw[0]:.0f}×{sw[1]:.0f} cm)',
            trig_label,
            f'+ back_scint\n({cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f} cm×2)',
        ]
        values = [
            st['n_tracks_mm'],
            st['n_tracks_scint_wall'],
            trig_count,
            st['n_tracks_back_scint'],
        ]
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='k', lw=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)

        n_mm = st['n_tracks_mm']
        effs = [
            1.0,
            st['scint_wall_efficiency'],
            st['scint_single_efficiency'],
            st['back_scint_efficiency'],
        ]
        for bar, eff in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f'{eff*100:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

        order_tag = cfg.stack_order.replace('_', ' ')
        ax.set_ylabel('Track count')
        ax.set_title(f'Per-track scintillator acceptance ({order_tag})\n'
                     f'(same-side coincidence required)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def plot_summary(self):
        """
        Summary figure:
          Top row    : geometry (left) | scint single trigger (right)
          Bottom row : scint double | MM any | MM double
        """
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(14, 9.5))
        gs  = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1.3, 1.0],
                       hspace=0.3, wspace=0.38, bottom=0.05, top=0.88, left=0.05, right=0.99)

        ax_geo    = fig.add_subplot(gs[0, 0:3])
        ax_single = fig.add_subplot(gs[0, 3:6])
        ax_bot    = [fig.add_subplot(gs[1, i*2:(i+1)*2]) for i in range(3)]

        self.plot_geometry(ax=ax_geo)
        self.plot_angular_separation(ax=ax_single, trigger='single')
        for ax, trig in zip(ax_bot, ['double', 'mm_any', 'mm_double']):
            self.plot_angular_separation(ax=ax, trigger=trig)

        fig.suptitle(
            f'Micromegas Simulation Summary\n'
            f'MM dist={self.cfg.detector_distance} cm | '
            f'σ_x={self.cfg.spatial_resolution} cm | '
            f'σ_t={self.cfg.time_resolution} ns | '
            f'window={self.cfg.coincidence_window} ns',
            fontsize=11,
        )
        return fig

    def summary_stats(self):
        """Print a full text summary."""
        ps = getattr(self, 'pair_stats', None)
        st = getattr(self, 'scint_trigger_stats', {})

        print("=" * 65)
        print("Simulation Summary")
        print("=" * 65)
        cfg = self.cfg
        print(f"  Events simulated       : {cfg.n_events}")
        print(f"  Particles generated    : {len(self.particles)}")
        print(f"  MM hits (raw)          : {len(self.hits_premerge)}")
        if cfg.merge_hits:
            print(f"  MM hits (merged final) : {len(self.hits)}")
        print(f"  True signal angles     : {len(self.true_signal_angles)}")
        print(f"  True bg angles         : {len(self.true_bg_angles)}")

        d_mm, d_scint, d_ls1, d_ls2, d_back = self._get_stack_distances()
        print(f"\n  --- Detector stack geometry ({cfg.stack_order}) ---")
        print(f"  MM face at             : {d_mm:.2f} cm  "
              f"({cfg.detector_size[0]:.0f}×{cfg.detector_size[1]:.0f} cm, u×v)")
        print(f"  Scint wall at          : {d_scint:.2f} cm  "
              f"({cfg.scint_wall_size[0]:.0f}×{cfg.scint_wall_size[1]:.0f} cm)")
        print(f"  LS-1 at                : {d_ls1:.2f} cm  "
              f"({cfg.liq_scint_size[0]:.0f}×{cfg.liq_scint_size[1]:.0f} cm)")
        print(f"  LS-2 at                : {d_ls2:.2f} cm  "
              f"({cfg.liq_scint_2_size[0]:.0f}×{cfg.liq_scint_2_size[1]:.0f} cm)")
        print(f"  Back scint at          : {d_back:.2f} cm  "
              f"(2×{cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f} cm, "
              f"gap={cfg.back_scint_gap:.1f} cm)")
        print(f"  Trigger second layer   : {cfg.trigger_second_layer}")
        if cfg.he_capsule_source:
            print(f"  He capsule source      : r={cfg.he_radius_cm} cm, "
                  f"L=±{cfg.he_half_length_cm} cm")

        print("\n  --- Per-track scintillator acceptance ---")
        if st:
            n_mm = st.get('n_tracks_mm', 0)
            print(f"  Tracks through MM                : {n_mm:,}")
            n_sw = st.get('n_tracks_scint_wall', 0)
            print(f"  + scint wall (same side)         : {n_sw:,}"
                  f"  ({st.get('scint_wall_efficiency',0)*100:.1f}%)")
            n_ss = st.get('n_tracks_scint_single', 0)
            sl   = cfg.trigger_second_layer
            print(f"  + {sl:<22s}: {n_ss:,}"
                  f"  ({st.get('scint_single_efficiency',0)*100:.1f}%)")
            n_bs = st.get('n_tracks_back_scint', 0)
            print(f"  + back_scint (same side)         : {n_bs:,}"
                  f"  ({st.get('back_scint_efficiency',0)*100:.1f}%)")

        for label, angles, sources in [
            ('Scint single trigger', self.angular_separations_single,    self.pair_sources_single),
            ('Scint double trigger', self.angular_separations_double,    self.pair_sources_double),
            ('MM any trigger',       self.angular_separations_mm_any,    self.pair_sources_mm_any),
            ('MM double trigger',    self.angular_separations_mm_double, self.pair_sources_mm_double),
        ]:
            total = len(angles)
            srcs  = np.array(sources) if total else np.array([])
            print(f"\n  --- {label} ---")
            print(f"  Coincident pairs    : {total}")
            for src in ['signal', 'background_pair', 'random']:
                nn  = int((srcs == src).sum()) if total else 0
                pct = 100 * nn / total if total else 0
                if nn > 0:
                    print(f"    {src:<22s}: {nn:5d}  ({pct:.1f}%)")
            n_sig = int((srcs == 'signal').sum()) if total else 0
            print(f"  Signal / Background : {n_sig} / {total - n_sig}")

        if ps:
            print("-" * 65)
            print("  Signal pair acceptance")
            print(f"  Both hit MM        : {ps['signal_detected']:4d}/{ps['signal_produced']:4d}"
                  f"  →  {ps['signal_efficiency']*100:.1f}%")
            print(f"  + scint_double trig: {ps['signal_triggered']:4d}/{ps['signal_detected']:4d}"
                  f"  →  {ps['trigger_efficiency']*100:.1f}%  "
                  f"(miss {ps['trigger_miss_fraction']*100:.1f}%)")
            print(f"  + both back_scint  : {ps['signal_both_backscint']:4d}/{ps['signal_detected']:4d}"
                  f"  →  {ps['calorimetry_complete_fraction']*100:.1f}%  "
                  f"(incomplete {ps['calorimetry_incomplete_fraction']*100:.1f}%)")
            print(f"  BG pair efficiency : {ps['bg_detected']:4d}/{ps['bg_produced']:4d}"
                  f"  →  {ps['bg_efficiency']*100:.1f}%")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Parallel batch worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _run_event_batch(args) -> dict:
    cfg, n_batch, event_offset, seed = args
    if seed is not None:
        np.random.seed(seed)

    detectors  = _build_detector_planes(cfg)
    propagator = Propagator(detectors, cfg)
    generator  = ParticleGenerator(cfg)
    sl         = cfg.trigger_second_layer
    resp       = _get_g4_response(cfg)

    angles_single, sources_single = [], []
    angles_double, sources_double = [], []
    angles_mm_any,    sources_mm_any    = [], []
    angles_mm_double, sources_mm_double = [], []
    angles_both_bs, sources_both_bs, masses_both_bs = [], [], []
    n_mm_raw = n_hits_away = n_mm_final = 0
    n_sig_prod = n_sig_det = n_sig_trig = n_sig_both_bs = 0
    n_bg_prod  = n_bg_det  = 0
    n_trk_mm = n_trk_sw = n_trk_scint_single = n_trk_back_scint = 0

    for i in range(n_batch):
        event_t0      = (event_offset + i) * cfg.time_spread
        evt_particles = generator.generate_event(event_t0)

        evt_mm_raw    = []
        evt_scint_raw = []
        p_mm_sides:  dict[int, set]   = {}
        p_bs_sides:  dict[int, set]   = {}
        p_true_mass: dict[int, float] = {}
        p_ke:        dict[int, float] = {}

        for p in evt_particles:
            p_hits = propagator.propagate(p)
            mm_h   = [h for h in p_hits if h.detector_type == 'mm']
            sc_h   = [h for h in p_hits if h.detector_type != 'mm']
            evt_mm_raw.extend(mm_h)
            evt_scint_raw.extend(sc_h)

            mm_s  = {h.detector_id for h in mm_h}
            sw_s  = {h.detector_id for h in sc_h if h.detector_type == 'scint_wall'}
            ly2_s = {h.detector_id for h in sc_h if h.detector_type == sl}
            bs_s  = {h.detector_id for h in sc_h if h.detector_type == 'back_scint'}

            p_mm_sides[p.particle_id]  = mm_s
            p_bs_sides[p.particle_id]  = bs_s
            p_ke[p.particle_id]        = p.ke_mev
            if p.true_inv_mass != 0.0:
                p_true_mass[p.particle_id] = p.true_inv_mass

            for side in mm_s:
                n_trk_mm += 1
                if side in sw_s:
                    n_trk_sw += 1
                if side in sw_s and side in ly2_s:
                    n_trk_scint_single += 1
                if side in bs_s:
                    n_trk_back_scint += 1

        n_mm_raw += len(evt_mm_raw)

        if cfg.merge_hits:
            evt_mm_hits  = merge_hits(evt_mm_raw, cfg)
            n_hits_away += len(evt_mm_raw) - len(evt_mm_hits)
        else:
            evt_mm_hits = evt_mm_raw
        n_mm_final += len(evt_mm_hits)

        evt_pairs  = find_coincident_pairs(evt_mm_hits, cfg)
        evt_angles = [angular_separation_from_hits(a, b) for a, b in evt_pairs]
        evt_src    = [_classify_pair_source(a, b)        for a, b in evt_pairs]

        for (ha, hb), ang, src in zip(evt_pairs, evt_angles, evt_src):
            if (p_bs_sides.get(ha.particle_id, set()) & p_mm_sides.get(ha.particle_id, set())
                    and p_bs_sides.get(hb.particle_id, set()) & p_mm_sides.get(hb.particle_id, set())):
                angles_both_bs.append(ang)
                sources_both_bs.append(src)
                true_m = p_true_mass.get(ha.particle_id,
                         p_true_mass.get(hb.particle_id, cfg.inv_mass_mean))
                masses_both_bs.append(
                    _pair_mass_estimate(cfg, resp, ha, hb, p_ke, true_m))

        evt_all   = evt_mm_hits + evt_scint_raw
        dbl_fired = event_passes_scint_double(evt_all, sl)

        if event_passes_scint_single(evt_all, sl):
            angles_single.extend(evt_angles); sources_single.extend(evt_src)
        if dbl_fired:
            angles_double.extend(evt_angles); sources_double.extend(evt_src)
        if event_passes_mm_any(evt_mm_hits):
            angles_mm_any.extend(evt_angles); sources_mm_any.extend(evt_src)
        if event_passes_mm_double(evt_mm_hits):
            angles_mm_double.extend(evt_angles); sources_mm_double.extend(evt_src)

        detected = {h.particle_id for h in evt_mm_hits}
        ptmap: dict[int, set] = {}
        for p in evt_particles:
            if p.pair_id != -1:
                ptmap.setdefault(p.pair_id, set()).add(p.particle_id)

        for pid, ptids in ptmap.items():
            src     = next(p.source for p in evt_particles if p.pair_id == pid)
            both_mm = ptids.issubset(detected)
            both_bs = all(
                bool(p_mm_sides.get(p_id, set()) & p_bs_sides.get(p_id, set()))
                for p_id in ptids
            )
            if src == 'signal':
                n_sig_prod += 1
                if both_mm:
                    n_sig_det += 1
                    if dbl_fired:
                        n_sig_trig += 1
                    if both_bs:
                        n_sig_both_bs += 1
            elif src == 'background_pair':
                n_bg_prod += 1
                if both_mm:
                    n_bg_det += 1

    return dict(
        angles_single=angles_single, sources_single=sources_single,
        angles_double=angles_double, sources_double=sources_double,
        angles_mm_any=angles_mm_any, sources_mm_any=sources_mm_any,
        angles_mm_double=angles_mm_double, sources_mm_double=sources_mm_double,
        angles_both_bs=angles_both_bs, sources_both_bs=sources_both_bs,
        masses_both_bs=masses_both_bs,
        true_signal_angles=generator.true_signal_angles,
        true_bg_angles=generator.true_bg_angles,
        n_mm_raw=n_mm_raw, n_hits_away=n_hits_away, n_mm_final=n_mm_final,
        n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
        n_sig_trig=n_sig_trig,   n_sig_both_bs=n_sig_both_bs,
        n_bg_prod=n_bg_prod,     n_bg_det=n_bg_det,
        n_trk_mm=n_trk_mm,       n_trk_sw=n_trk_sw,
        n_trk_scint_single=n_trk_scint_single,
        n_trk_back_scint=n_trk_back_scint,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def calculate_solid_angle_coverage(detectors: list[Detector],
                                   n_samples: int = 1_000_000) -> float:
    """Estimate fraction of 4π sr covered by a list of detectors (Monte Carlo)."""
    origin = np.zeros(3)
    hits   = 0
    for _ in range(n_samples):
        direction = _random_direction()
        for det in detectors:
            _, local_2d = det.intersect(origin, direction)
            if local_2d is not None:
                hits += 1
                break
    return hits / n_samples

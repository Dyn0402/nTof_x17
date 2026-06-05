"""
Micromegas Detector Simulation with Scintillator Trigger Stack
==============================================================
Simulates the 4-arm MX17 detector geometry (n-TOF X17 experiment).
Beam axis = +Y. Four identical detector arms at ±X and ±Y (transverse plane).
He-3 target at origin.

Detector stack per arm (from origin outward):
  MM window at detector_distance, active area detector_size (38×34 cm)
  [mm_drift_gap = 3 cm scored gas region]
  [gap_mm_to_scint ≈ 2.77 cm: PCB + air gap]
  Trigger scint wall (48×48 cm, 3 mm plastic scint)
  [gap_scint_to_liq ≈ 3.44 cm: air gap + CFRP/Al liners]
  Liquid scintillator layer 1 (45×45 cm, 2 cm LAB)

Geometry matches the full Geant4 simulation (MX17_Full_Geant/SimConfig.hh):
  mm_distance = 22 cm, MM face 38×34 cm, scint wall 48×48 cm, LS 45×45 cm

Trigger logic
-------------
  single    : scint_wall AND liq_scint_1 both fired on at least one side
  double    : single fired on at least two sides  <- primary analysis trigger
  mm_any    : at least one MM panel hit (comparison)
  mm_double : at least two MM panels hit (comparison)

Coincident pairs for angular separation are always formed from MM hits
(best position/direction reconstruction). Scint triggers gate which
events contribute pairs; they are not merged and carry no position info.

Particle types:
  random background  : single e/e+, photons, protons (uncorrelated)
  correlated bg pairs: coincident pairs with a configurable angular spectrum
  signal             : coincident e+/e- pairs with ~120 deg separation (Gaussian)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional
import warnings


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
    detector_type: str = 'mm'   # 'mm', 'scint_wall', 'liq_scint_1'


# ---------------------------------------------------------------------------
# Detector Geometry
# ---------------------------------------------------------------------------

class Detector:
    """
    A flat rectangular detector panel.

    Orientation by side index (beam along +Z in Python frame):
      0: +X arm  (normal toward origin = -X)
      1: -X arm
      2: +Y arm
      3: -Y arm

    det_type: 'mm', 'scint_wall', or 'liq_scint_1'
    """

    SIDE_NORMALS = {
        0: np.array([-1.0, 0.0, 0.0]),
        1: np.array([ 1.0, 0.0, 0.0]),
        2: np.array([ 0.0,-1.0, 0.0]),
        3: np.array([ 0.0, 1.0, 0.0]),
    }
    SIDE_NAMES = {0: '+X', 1: '-X', 2: '+Y', 3: '-Y'}

    def __init__(self, side: int, distance: float,
                 size_u: float = 40.0, size_v: float = 40.0,
                 det_type: str = 'mm'):
        self.side     = side
        self.distance = distance
        self.size_u   = size_u   # horizontal extent (visible in XY top-down view), cm
        self.size_v   = size_v   # vertical extent (into the page / Z direction), cm
        self.det_type = det_type
        self.normal   = self.SIDE_NORMALS[side]
        self.name     = self.SIDE_NAMES[side]
        self.center   = -self.normal * distance

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """Ray–plane intersection. Returns (hit_3d, local_2d) or (None, None)."""
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-9:
            return None, None
        t = np.dot(self.normal, self.center - origin) / denom
        if t <= 0:
            return None, None
        hit_3d = origin + t * direction
        local_u, local_v = self._local_axes()
        delta = hit_3d - self.center
        u = np.dot(delta, local_u)
        v = np.dot(delta, local_v)
        if abs(u) > self.size_u / 2.0 or abs(v) > self.size_v / 2.0:
            return None, None
        return hit_3d, np.array([u, v])

    def _local_axes(self):
        up = np.array([1.0, 0.0, 0.0]) if abs(self.normal[0]) < 0.9 \
             else np.array([0.0, 1.0, 0.0])
        v_axis = np.cross(self.normal, up)
        v_axis /= np.linalg.norm(v_axis)
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


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All tuneable parameters in one place."""

    # MM geometry  (Geant4 SimConfig.hh: mm_distance=22 cm, 38×34 cm face)
    detector_distance: float = 22.0                  # cm, origin to MM front face
    detector_size: tuple[float, float] = (38.0, 34.0)  # cm, (horiz/u, along-beam/v)

    # Scintillator trigger stack — distances derived from full Geant4 slab layout
    # PlasticScint centre: 22 + 57.7 mm = 27.77 cm from origin
    # LS-1 centre:         22 + 92.1 mm = 31.21 cm from origin
    mm_drift_gap:         float = 3.0                          # cm, active drift gas region (30 mm)
    gap_mm_to_scint:      float = 2.77                         # cm, drift back → scint centre (PCB + air + tape)
    scint_wall_size:      tuple[float, float] = (48.0, 48.0)   # cm, (horiz/u, along-beam/v)
    scint_wall_thickness: float = 0.3                          # cm, plastic scint slab (3 mm PVT)
    gap_scint_to_liq:     float = 3.44                         # cm, scint centre → LS-1 centre (air + CFRP/Al)
    liq_scint_size:       tuple[float, float] = (45.0, 45.0)   # cm, (horiz/u, along-beam/v)
    liq_scint_thickness:  float = 2.0                          # cm, LAB layer (2 cm per layer)

    # Resolution (MM only)
    spatial_resolution: float = 0.5   # cm, 1-sigma Gaussian smear per axis
    time_resolution:    float = 5.0   # ns, 1-sigma Gaussian smear

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


class ParticleGenerator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._event_counter   = 0
        self._particle_counter = 0
        self._pair_counter    = 0
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

    def generate_event(self, event_t0: float = 0.0) -> list[Particle]:
        particles = []

        for _ in range(np.random.poisson(self.cfg.n_random)):
            t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
            particles.append(Particle(
                pid=self._random_pid(), origin=np.zeros(3),
                direction=_random_direction(), t0=t0,
                event_id=self._next_event_id(), particle_id=self._next_particle_id(),
                pair_id=-1, source='random',
            ))

        n_bg = np.random.poisson(self.cfg.n_background_pairs)
        if n_bg > 0:
            angles = self.cfg.background_spectrum.sample(n_bg)
            self.true_bg_angles.extend(angles.tolist())
            for angle in angles:
                eid = self._next_event_id()
                t0  = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                d1  = _random_direction()
                d2  = _direction_at_angle(d1, angle)
                bg_pair_id = self._next_pair_id()
                particles.append(Particle('electron', np.zeros(3), d1, t0, eid,
                                          self._next_particle_id(), bg_pair_id, 'background_pair'))
                particles.append(Particle('positron', np.zeros(3), d2, t0, eid,
                                          self._next_particle_id(), bg_pair_id, 'background_pair'))

        n_sig = np.random.poisson(self.cfg.n_signal)
        if n_sig > 0:
            angles = self.cfg.signal_spectrum.sample(n_sig)
            self.true_signal_angles.extend(angles.tolist())
            for angle in angles:
                eid = self._next_event_id()
                t0  = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                d1  = _random_direction()
                d2  = _direction_at_angle(d1, angle)
                sig_pair_id = self._next_pair_id()
                particles.append(Particle('electron', np.zeros(3), d1, t0, eid,
                                          self._next_particle_id(), sig_pair_id, 'signal'))
                particles.append(Particle('positron', np.zeros(3), d2, t0, eid,
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

def scint_single_sides(hits: list[Hit]) -> set[int]:
    """Side IDs where scint_wall AND liq_scint_1 both have at least one hit."""
    sw = {h.detector_id for h in hits if h.detector_type == 'scint_wall'}
    ls = {h.detector_id for h in hits if h.detector_type == 'liq_scint_1'}
    return sw & ls


def event_passes_scint_single(hits: list[Hit]) -> bool:
    """True if scint single trigger fired on at least one side."""
    return len(scint_single_sides(hits)) >= 1


def event_passes_scint_double(hits: list[Hit]) -> bool:
    """True if scint single trigger fired on at least two sides."""
    return len(scint_single_sides(hits)) >= 2


def mm_fired_sides(hits: list[Hit]) -> set[int]:
    """Side IDs with at least one MM hit."""
    return {h.detector_id for h in hits if h.detector_type == 'mm'}


def event_passes_mm_any(hits: list[Hit]) -> bool:
    return len(mm_fired_sides(hits)) >= 1


def event_passes_mm_double(hits: list[Hit]) -> bool:
    """True if any two MM panels were hit."""
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
    'single'    scint_wall AND liq_scint_1 on at least one side
    'double'    single on at least two sides  <- default analysis trigger

    Comparison triggers (MM-only, for acceptance studies)
    ------------------------------------------------------
    'mm_any'    any MM panel hit
    'mm_double' any two MM panels hit

    Pairs are always formed from MM hits. Scint triggers gate events only.

    Key attributes after run()
    --------------------------
    angular_separations_double / _single / _mm_any / _mm_double
    pair_sources_double / ...
    scint_trigger_stats  per-track efficiency dict
    pair_stats           pair detection efficiency dict
    hits                 merged MM hits (serial mode only)
    """

    def __init__(self, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

        self.detectors  = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)
        self.generator  = ParticleGenerator(self.cfg)

        self.particles:    list[Particle] = []
        self.hits:         list[Hit] = []   # merged MM hits (serial mode)
        self.hits_premerge: list[Hit] = []  # raw MM hits before merging (serial)

        # Scintillator triggers (primary)
        self.coincident_pairs_single:    list[tuple[Hit, Hit]] = []
        self.coincident_pairs_double:    list[tuple[Hit, Hit]] = []
        self.angular_separations_single: np.ndarray = np.array([])
        self.angular_separations_double: np.ndarray = np.array([])
        self.pair_sources_single: list[str] = []
        self.pair_sources_double: list[str] = []

        # MM triggers (comparison)
        self.coincident_pairs_mm_any:     list[tuple[Hit, Hit]] = []
        self.coincident_pairs_mm_double:  list[tuple[Hit, Hit]] = []
        self.angular_separations_mm_any:    np.ndarray = np.array([])
        self.angular_separations_mm_double: np.ndarray = np.array([])
        self.pair_sources_mm_any:    list[str] = []
        self.pair_sources_mm_double: list[str] = []

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
        cfg     = self.cfg
        d_scint = cfg.detector_distance + cfg.mm_drift_gap + cfg.gap_mm_to_scint
        d_liq   = d_scint + cfg.gap_scint_to_liq
        detectors = []
        for side in range(4):
            detectors.append(Detector(side, cfg.detector_distance,
                                      *cfg.detector_size,  'mm'))
            detectors.append(Detector(side, d_scint,
                                      *cfg.scint_wall_size, 'scint_wall'))
            detectors.append(Detector(side, d_liq,
                                      *cfg.liq_scint_size,  'liq_scint_1'))
        return detectors

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

        self.particles      = []
        self.hits           = []
        self.hits_premerge  = []

        # ------------------------------------------------------------------
        # Serial path
        # ------------------------------------------------------------------
        if n_workers == 1:
            angles_single, sources_single, cpairs_single = [], [], []
            angles_double, sources_double, cpairs_double = [], [], []
            angles_mm_any,    sources_mm_any,    cpairs_mm_any    = [], [], []
            angles_mm_double, sources_mm_double, cpairs_mm_double = [], [], []
            n_mm_raw = n_hits_away = n_mm_final = 0
            n_sig_prod = n_sig_det = n_bg_prod = n_bg_det = 0
            n_trk_mm = n_trk_sw = n_trk_scint_single = 0

            iterator = range(n)
            if _tqdm is not None:
                iterator = _tqdm(iterator, desc='Simulating', unit='evt',
                                 dynamic_ncols=True)

            for i_evt in iterator:
                event_t0      = i_evt * self.cfg.time_spread
                evt_particles = self.generator.generate_event(event_t0)
                self.particles.extend(evt_particles)

                # Propagate per-particle to enable per-track efficiency tracking
                evt_mm_raw_hits   = []
                evt_scint_raw_hits = []
                for p in evt_particles:
                    p_hits = self.propagator.propagate(p)
                    mm_h   = [h for h in p_hits if h.detector_type == 'mm']
                    sc_h   = [h for h in p_hits if h.detector_type != 'mm']
                    evt_mm_raw_hits.extend(mm_h)
                    evt_scint_raw_hits.extend(sc_h)

                    # Per-track scint acceptance (pre-merge, per particle)
                    mm_s = {h.detector_id for h in mm_h}
                    sw_s = {h.detector_id for h in sc_h if h.detector_type == 'scint_wall'}
                    ls_s = {h.detector_id for h in sc_h if h.detector_type == 'liq_scint_1'}
                    for side in mm_s:
                        n_trk_mm += 1
                        if side in sw_s:
                            n_trk_sw += 1
                        if side in sw_s and side in ls_s:
                            n_trk_scint_single += 1

                self.hits_premerge.extend(evt_mm_raw_hits)
                n_mm_raw += len(evt_mm_raw_hits)

                # Merge MM hits only; scint hits are boolean triggers
                if self.cfg.merge_hits:
                    evt_mm_hits  = merge_hits(evt_mm_raw_hits, self.cfg)
                    n_hits_away += len(evt_mm_raw_hits) - len(evt_mm_hits)
                else:
                    evt_mm_hits = evt_mm_raw_hits
                self.hits.extend(evt_mm_hits)
                n_mm_final += len(evt_mm_hits)

                # Coincident MM pairs (used by all triggers)
                evt_pairs  = find_coincident_pairs(evt_mm_hits, self.cfg)
                evt_angles = [angular_separation_from_hits(a, b) for a, b in evt_pairs]
                evt_src    = [_classify_pair_source(a, b)        for a, b in evt_pairs]

                # All hits for scint trigger checking
                evt_all = evt_mm_hits + evt_scint_raw_hits

                if event_passes_scint_single(evt_all):
                    cpairs_single.extend(evt_pairs)
                    angles_single.extend(evt_angles); sources_single.extend(evt_src)
                if event_passes_scint_double(evt_all):
                    cpairs_double.extend(evt_pairs)
                    angles_double.extend(evt_angles); sources_double.extend(evt_src)
                if event_passes_mm_any(evt_mm_hits):
                    cpairs_mm_any.extend(evt_pairs)
                    angles_mm_any.extend(evt_angles); sources_mm_any.extend(evt_src)
                if event_passes_mm_double(evt_mm_hits):
                    cpairs_mm_double.extend(evt_pairs)
                    angles_mm_double.extend(evt_angles); sources_mm_double.extend(evt_src)

                # Pair detection efficiency (both particles of a pair hit MM)
                detected = {h.particle_id for h in evt_mm_hits}
                ptmap: dict[int, set] = {}
                for p in evt_particles:
                    if p.pair_id != -1:
                        ptmap.setdefault(p.pair_id, set()).add(p.particle_id)
                for pid, ptids in ptmap.items():
                    src  = next(p.source for p in evt_particles if p.pair_id == pid)
                    both = ptids.issubset(detected)
                    if src == 'signal':
                        n_sig_prod += 1; n_sig_det += int(both)
                    elif src == 'background_pair':
                        n_bg_prod  += 1; n_bg_det  += int(both)

            self.coincident_pairs_single   = cpairs_single
            self.coincident_pairs_double   = cpairs_double
            self.coincident_pairs_mm_any   = cpairs_mm_any
            self.coincident_pairs_mm_double = cpairs_mm_double
            self.true_signal_angles = np.array(self.generator.true_signal_angles)
            self.true_bg_angles     = np.array(self.generator.true_bg_angles)
            results = [dict(
                angles_single=angles_single, sources_single=sources_single,
                angles_double=angles_double, sources_double=sources_double,
                angles_mm_any=angles_mm_any, sources_mm_any=sources_mm_any,
                angles_mm_double=angles_mm_double, sources_mm_double=sources_mm_double,
                n_mm_raw=n_mm_raw, n_hits_away=n_hits_away, n_mm_final=n_mm_final,
                n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
                n_bg_prod=n_bg_prod,   n_bg_det=n_bg_det,
                n_trk_mm=n_trk_mm, n_trk_sw=n_trk_sw,
                n_trk_scint_single=n_trk_scint_single,
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

            self.coincident_pairs_single   = []
            self.coincident_pairs_double   = []
            self.coincident_pairs_mm_any   = []
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

        n_mm_raw   = sum(r['n_mm_raw']   for r in results)
        n_hits_away = sum(r['n_hits_away'] for r in results)
        n_mm_final = sum(r['n_mm_final'] for r in results)
        n_sig_prod = sum(r['n_sig_prod'] for r in results)
        n_sig_det  = sum(r['n_sig_det']  for r in results)
        n_bg_prod  = sum(r['n_bg_prod']  for r in results)
        n_bg_det   = sum(r['n_bg_det']   for r in results)
        n_trk_mm   = sum(r['n_trk_mm']   for r in results)
        n_trk_sw   = sum(r['n_trk_sw']   for r in results)
        n_trk_ss   = sum(r['n_trk_scint_single'] for r in results)

        self.pair_stats = {
            'signal_produced':   n_sig_prod,
            'signal_detected':   n_sig_det,
            'signal_efficiency': n_sig_det / n_sig_prod if n_sig_prod else 0.0,
            'bg_produced':       n_bg_prod,
            'bg_detected':       n_bg_det,
            'bg_efficiency':     n_bg_det / n_bg_prod   if n_bg_prod  else 0.0,
        }
        self.scint_trigger_stats = {
            'n_tracks_mm':              n_trk_mm,
            'n_tracks_scint_wall':      n_trk_sw,
            'n_tracks_scint_single':    n_trk_ss,
            'scint_wall_efficiency':    n_trk_sw / n_trk_mm if n_trk_mm else 0.0,
            'scint_single_efficiency':  n_trk_ss / n_trk_mm if n_trk_mm else 0.0,
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
        print(f"  Track eff: MM → scint singl: {st['scint_single_efficiency']*100:.1f}%")
        print("[Sim] Done.")

    def run_display_sample(self, n_sample: int = 5_000):
        """
        Run a short serial batch to populate self.hits for display plots.
        Useful after a parallel run(), which skips storing individual hits.
        Does not touch any statistics arrays from the main run.
        """
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
                    h for h in self.propagator.propagate(p)
                    if h.detector_type == 'mm'
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
        Top-down XY view of all detector layers.
        Rectangles are drawn to scale in both the radial (thickness) and
        lateral (size_u) dimensions. size_v (into the page) is labelled but
        cannot be shown in a 2-D top-down projection.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        cfg     = self.cfg
        d_mm    = cfg.detector_distance
        d_scint = d_mm + cfg.mm_drift_gap + cfg.gap_mm_to_scint
        d_liq   = d_scint + cfg.gap_scint_to_liq

        # (front_dist, half_u, thickness, color, alpha, legend_label)
        def _lbl(name, sz, thick):
            return (f'{name}  {sz[0]:.0f}×{sz[1]:.0f} cm (u×v)'
                    f',  t={thick:.1f} cm')

        layers = [
            (d_mm,    cfg.detector_size[0]   / 2, cfg.mm_drift_gap,
             '#2196F3', 0.55, _lbl('MM',          cfg.detector_size,   cfg.mm_drift_gap)),
            (d_scint, cfg.scint_wall_size[0] / 2, cfg.scint_wall_thickness,
             '#FF9800', 0.70, _lbl('Scint wall',   cfg.scint_wall_size, cfg.scint_wall_thickness)),
            (d_liq,   cfg.liq_scint_size[0]  / 2, cfg.liq_scint_thickness,
             '#4CAF50', 0.65, _lbl('Liq scint 1',  cfg.liq_scint_size,  cfg.liq_scint_thickness)),
        ]

        # For each layer draw 4 filled rectangles (one per side).
        # Side 0 (+X): rectangle x ∈ [d, d+t],   y ∈ [-h, +h]
        # Side 1 (-X): rectangle x ∈ [-(d+t), -d], y ∈ [-h, +h]
        # Side 2 (+Y): rectangle y ∈ [d, d+t],   x ∈ [-h, +h]
        # Side 3 (-Y): rectangle y ∈ [-(d+t), -d], x ∈ [-h, +h]
        from matplotlib.patches import Patch
        legend_handles = []

        for dist, half_u, thick, color, alpha, label in layers:
            rect_params = [
                dict(xy=(dist,          -half_u), width=thick,      height=2 * half_u),
                dict(xy=(-(dist + thick), -half_u), width=thick,    height=2 * half_u),
                dict(xy=(-half_u,        dist),   width=2 * half_u, height=thick),
                dict(xy=(-half_u,        -(dist + thick)), width=2 * half_u, height=thick),
            ]
            for rp in rect_params:
                ax.add_patch(mpatches.Rectangle(
                    **rp, linewidth=0.4, edgecolor=color,
                    facecolor=color, alpha=alpha, zorder=2))
            legend_handles.append(
                Patch(facecolor=color, edgecolor=color, alpha=alpha, label=label))

        ax.plot(0, 0, 'k*', ms=12, zorder=3)
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='k', markersize=10,
                       linewidth=0, label='Target'))

        lim = (d_liq + cfg.liq_scint_thickness + max(cfg.liq_scint_size) / 2 + 2) * 1.08
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_title('Detector Layout (XY top-down, to scale)')
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, zorder=0)

        mm_dets = [d for d in self.detectors if d.det_type == 'mm']
        ls_dets = [d for d in self.detectors if d.det_type == 'liq_scint_1']
        frac_mm = calculate_solid_angle_coverage(mm_dets, n_samples=1_000)
        frac_ls = calculate_solid_angle_coverage(ls_dets, n_samples=1_000)
        ax.annotate(
            f'MM solid angle:        {frac_mm*100:.1f}%\n'
            f'Liq scint solid angle: {frac_ls*100:.1f}%',
            xy=(0.04, 0.04), xycoords='axes fraction',
            ha='left', va='bottom', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round'),
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
        det_names = ['+X', '-X', '+Y', '-Y']

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

        # Counts textbox
        srcs_arr = np.array(sources)
        n_sig = int((srcs_arr == 'signal').sum())
        n_bg  = len(angles) - n_sig
        ax.text(0.97, 0.97,
                f'Signal:     {n_sig:,}\nBackground: {n_bg:,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.85))
        return ax

    def plot_trigger_comparison(self, bins=36):
        """Four-panel comparison: scint single, scint double, MM any, MM double."""
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, trig in zip(axes, ['single', 'double', 'mm_any', 'mm_double']):
            self.plot_angular_separation(bins=bins, ax=ax, trigger=trig)
        fig.suptitle('Trigger Scenario Comparison', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_mm_vs_scint_comparison(self, bins=36):
        """
        Side-by-side: MM double trigger vs scint double trigger.
        Quantifies acceptance loss due to finite liq_scint size.
        """
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

    def plot_true_vs_reconstructed(self, bins=36, normalized=False,
                                   trigger=None):
        """
        True generated spectra vs reconstructed under one or more triggers.

        trigger : str or list of str; defaults to ['double', 'mm_double']
        """
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
        Bar chart showing per-track scintillator acceptance at each stage.
        Illustrates how much MM acceptance is lost at the scint wall and
        at the (smaller) liquid scintillator.
        """
        st = self.scint_trigger_stats
        if not st:
            warnings.warn("No stats available. Run sim.run() first.")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        sw = self.cfg.scint_wall_size
        ls = self.cfg.liq_scint_size
        labels = [
            f'MM hit\n({self.cfg.detector_size[0]:.0f}×{self.cfg.detector_size[1]:.0f} cm)',
            f'+ scint wall\n({sw[0]:.0f}×{sw[1]:.0f} cm)',
            f'+ liq scint 1\n({ls[0]:.0f}×{ls[1]:.0f} cm)',
        ]
        values = [
            st['n_tracks_mm'],
            st['n_tracks_scint_wall'],
            st['n_tracks_scint_single'],
        ]
        colors = ['#2196F3', '#FF9800', '#4CAF50']
        bars   = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='k', lw=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)

        n_mm = st['n_tracks_mm']
        if n_mm > 0:
            for bar, eff in zip(bars,
                                [1.0, st['scint_wall_efficiency'],
                                 st['scint_single_efficiency']]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f'{eff*100:.1f}%', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')

        ax.set_ylabel('Track count')
        ax.set_title('Per-track scintillator acceptance\n(same-side coincidence required)')
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
        fig = plt.figure(figsize=(18, 10))
        gs  = GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.38)

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

        print("\n  --- Detector stack geometry ---")
        d_scint = cfg.detector_distance + cfg.mm_drift_gap + cfg.gap_mm_to_scint
        d_liq   = d_scint + cfg.gap_scint_to_liq
        print(f"  MM face at             : {cfg.detector_distance:.1f} cm  "
              f"({cfg.detector_size[0]:.0f}×{cfg.detector_size[1]:.0f} cm, u×v)")
        print(f"  Scint wall at          : {d_scint:.1f} cm  "
              f"({cfg.scint_wall_size[0]:.0f}×{cfg.scint_wall_size[1]:.0f} cm, u×v)")
        print(f"  Liq scint 1 at         : {d_liq:.1f} cm  "
              f"({cfg.liq_scint_size[0]:.0f}×{cfg.liq_scint_size[1]:.0f} cm, u×v)")

        print("\n  --- Per-track scintillator acceptance ---")
        if st:
            n_mm = st.get('n_tracks_mm', 0)
            print(f"  Tracks through MM                : {n_mm:,}")
            n_sw = st.get('n_tracks_scint_wall', 0)
            print(f"  + scint wall (same side)         : {n_sw:,}"
                  f"  ({st.get('scint_wall_efficiency',0)*100:.1f}%)")
            n_ss = st.get('n_tracks_scint_single', 0)
            print(f"  + liq scint 1 (scint single)     : {n_ss:,}"
                  f"  ({st.get('scint_single_efficiency',0)*100:.1f}%)")
            print(f"  liq scint eff | scint wall hit   : "
                  f"{st.get('liq_scint_given_scint_wall',0)*100:.1f}%")

        for label, angles, sources in [
            ('Scint single trigger', self.angular_separations_single,   self.pair_sources_single),
            ('Scint double trigger', self.angular_separations_double,   self.pair_sources_double),
            ('MM any trigger',       self.angular_separations_mm_any,   self.pair_sources_mm_any),
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
            print("  MM pair detection efficiency")
            print(f"  Signal     : {ps['signal_detected']:4d}/{ps['signal_produced']:4d}"
                  f"  →  {ps['signal_efficiency']*100:.1f}%")
            print(f"  Background : {ps['bg_detected']:4d}/{ps['bg_produced']:4d}"
                  f"  →  {ps['bg_efficiency']*100:.1f}%")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Parallel batch worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _run_event_batch(args) -> dict:
    cfg, n_batch, event_offset, seed = args
    if seed is not None:
        np.random.seed(seed)

    d_scint = cfg.detector_distance + cfg.mm_drift_gap + cfg.gap_mm_to_scint
    d_liq   = d_scint + cfg.gap_scint_to_liq
    detectors = []
    for side in range(4):
        detectors.append(Detector(side, cfg.detector_distance,
                                  *cfg.detector_size,   'mm'))
        detectors.append(Detector(side, d_scint,
                                  *cfg.scint_wall_size, 'scint_wall'))
        detectors.append(Detector(side, d_liq,
                                  *cfg.liq_scint_size,  'liq_scint_1'))

    propagator = Propagator(detectors, cfg)
    generator  = ParticleGenerator(cfg)

    angles_single, sources_single = [], []
    angles_double, sources_double = [], []
    angles_mm_any,    sources_mm_any    = [], []
    angles_mm_double, sources_mm_double = [], []
    n_mm_raw = n_hits_away = n_mm_final = 0
    n_sig_prod = n_sig_det = n_bg_prod = n_bg_det = 0
    n_trk_mm = n_trk_sw = n_trk_scint_single = 0

    for i in range(n_batch):
        event_t0      = (event_offset + i) * cfg.time_spread
        evt_particles = generator.generate_event(event_t0)

        evt_mm_raw    = []
        evt_scint_raw = []
        for p in evt_particles:
            p_hits = propagator.propagate(p)
            mm_h   = [h for h in p_hits if h.detector_type == 'mm']
            sc_h   = [h for h in p_hits if h.detector_type != 'mm']
            evt_mm_raw.extend(mm_h)
            evt_scint_raw.extend(sc_h)

            mm_s = {h.detector_id for h in mm_h}
            sw_s = {h.detector_id for h in sc_h if h.detector_type == 'scint_wall'}
            ls_s = {h.detector_id for h in sc_h if h.detector_type == 'liq_scint_1'}
            for side in mm_s:
                n_trk_mm += 1
                if side in sw_s:
                    n_trk_sw += 1
                if side in sw_s and side in ls_s:
                    n_trk_scint_single += 1

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

        evt_all = evt_mm_hits + evt_scint_raw

        if event_passes_scint_single(evt_all):
            angles_single.extend(evt_angles); sources_single.extend(evt_src)
        if event_passes_scint_double(evt_all):
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
            src  = next(p.source for p in evt_particles if p.pair_id == pid)
            both = ptids.issubset(detected)
            if src == 'signal':
                n_sig_prod += 1; n_sig_det += int(both)
            elif src == 'background_pair':
                n_bg_prod  += 1; n_bg_det  += int(both)

    return dict(
        angles_single=angles_single, sources_single=sources_single,
        angles_double=angles_double, sources_double=sources_double,
        angles_mm_any=angles_mm_any, sources_mm_any=sources_mm_any,
        angles_mm_double=angles_mm_double, sources_mm_double=sources_mm_double,
        true_signal_angles=generator.true_signal_angles,
        true_bg_angles=generator.true_bg_angles,
        n_mm_raw=n_mm_raw, n_hits_away=n_hits_away, n_mm_final=n_mm_final,
        n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
        n_bg_prod=n_bg_prod,   n_bg_det=n_bg_det,
        n_trk_mm=n_trk_mm, n_trk_sw=n_trk_sw,
        n_trk_scint_single=n_trk_scint_single,
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

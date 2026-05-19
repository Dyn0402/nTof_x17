"""
Micromegas Detector Simulation
================================
Simulates a 4-detector Micromegas setup surrounding a central target.
Detectors are 40x40 cm flat panels arranged in a square geometry.

Particle types:
  - Random background: single e/e+, photons, protons (uncorrelated in time)
  - Correlated background: coincident pairs with angular distribution from a background spectrum
  - Signal: coincident e+/e- pairs with angular separation ~120° (Gaussian)

All particles originate at the target (origin). Each travels in a straight line
and registers at most one hit (the geometry ensures this when detector_distance
>= detector_size / 2).

Reconstruction: hits are smeared in space and time to emulate detector resolution.
Analysis: builds mock angular separation distributions for time-coincident hits
          under two trigger scenarios — any detector hit, or two adjacent detectors hit.
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
    pid: str            # 'electron', 'positron', 'photon', 'proton'
    origin: np.ndarray  # 3D origin
    direction: np.ndarray  # unit 3-vector
    t0: float           # true emission time [ns]
    event_id: int = -1
    particle_id: int = -1
    pair_id: int = -1        # shared by both particles in a correlated pair (-1 if unpaired)
    source: str = 'random'  # 'random', 'background_pair', 'signal'


@dataclass
class Hit:
    """A reconstructed hit on a detector panel."""
    detector_id: int
    true_pos: np.ndarray    # true 2D hit position on detector face [cm]
    reco_pos: np.ndarray    # smeared 2D hit position [cm]
    true_time: float        # true arrival time [ns]
    reco_time: float        # smeared arrival time [ns]
    direction: np.ndarray   # original 3D direction of particle
    pid: str
    event_id: int
    particle_id: int
    pair_id: int
    source: str


# ---------------------------------------------------------------------------
# Detector Geometry
# ---------------------------------------------------------------------------

class Detector:
    """
    A flat 40x40 cm Micromegas panel.

    The detector sits at a given distance from the origin, facing inward.
    Orientation is defined by which side of the square it is on:
      0: +X face  (normal pointing in -X direction toward origin)
      1: -X face
      2: +Y face
      3: -Y face
    """

    SIDE_NORMALS = {
        0: np.array([-1.0, 0.0, 0.0]),
        1: np.array([ 1.0, 0.0, 0.0]),
        2: np.array([ 0.0,-1.0, 0.0]),
        3: np.array([ 0.0, 1.0, 0.0]),
    }

    SIDE_NAMES = {0: '+X', 1: '-X', 2: '+Y', 3: '-Y'}

    def __init__(self, side: int, distance: float, size: float = 40.0):
        self.side = side
        self.distance = distance
        self.size = size
        self.normal = self.SIDE_NORMALS[side]
        self.name = self.SIDE_NAMES[side]
        self.center = -self.normal * distance

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """
        Find where a ray (origin + t*direction) hits this detector plane.
        Returns (hit_3d, local_2d) or (None, None) if no valid intersection.
        """
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

        half = self.size / 2.0
        if abs(u) > half or abs(v) > half:
            return None, None

        return hit_3d, np.array([u, v])

    def _local_axes(self):
        if abs(self.normal[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
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
    """Base class for angular separation spectra."""
    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError


class GaussianSpectrum(AngularSpectrum):
    """Sharp Gaussian peak — for signal."""
    def __init__(self, mean_deg: float = 120.0, sigma_deg: float = 5.0):
        self.mean = mean_deg
        self.sigma = sigma_deg

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mean, self.sigma, n)


class FlatSpectrum(AngularSpectrum):
    """Flat (uniform) distribution over [min, max] degrees."""
    def __init__(self, min_deg: float = 0.0, max_deg: float = 180.0):
        self.min = min_deg
        self.max = max_deg

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.min, self.max, n)


class IsotropicSpectrum(AngularSpectrum):
    """
    Angular separation for isotropic pairs: p(theta) ~ sin(theta),
    peaking near 90 degrees.
    """
    def sample(self, n: int) -> np.ndarray:
        samples = []
        while len(samples) < n:
            theta = np.random.uniform(0, 180, n * 2)
            prob = np.sin(np.radians(theta))
            accept = np.random.uniform(0, 1, n * 2) < prob
            samples.extend(theta[accept].tolist())
        return np.array(samples[:n])


class HistogramSpectrum(AngularSpectrum):
    """User-defined spectrum from histogram bins."""
    def __init__(self, bin_edges: np.ndarray, counts: np.ndarray):
        self.bin_edges = bin_edges
        probs = counts / counts.sum()
        self.cdf = np.concatenate([[0], np.cumsum(probs)])

    def sample(self, n: int) -> np.ndarray:
        u = np.random.uniform(0, 1, n)
        indices = np.searchsorted(self.cdf, u) - 1
        indices = np.clip(indices, 0, len(self.bin_edges) - 2)
        lo = self.bin_edges[indices]
        hi = self.bin_edges[indices + 1]
        return lo + np.random.uniform(0, 1, n) * (hi - lo)


class ExponentialSpectrum(AngularSpectrum):
    """
    Falling exponential background: p(theta) ~ exp(-theta / scale_deg),
    sampled over [min_deg, max_deg] via inverse-CDF.
    """
    def __init__(self, scale_deg: float = 40.0,
                 min_deg: float = 0.0, max_deg: float = 180.0):
        self.scale = scale_deg
        self.min = min_deg
        self.max = max_deg
        self._lo = np.exp(-min_deg / scale_deg)
        self._hi = np.exp(-max_deg / scale_deg)

    def sample(self, n: int) -> np.ndarray:
        u = np.random.uniform(0, 1, n)
        return -self.scale * np.log(self._lo - u * (self._lo - self._hi))


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All tuneable parameters in one place."""
    # Geometry
    detector_distance: float = 15.0     # cm, center-to-detector
    detector_size: float = 40.0         # cm, active area side

    # Resolution (smearing)
    spatial_resolution: float = 0.5     # cm, 1-sigma Gaussian smear per axis
    time_resolution: float = 5.0        # ns, 1-sigma Gaussian smear

    # Timing
    coincidence_window: float = 20.0    # ns, window to call hits coincident
    time_spread: float = 200.0          # ns, duration of a single readout event

    # Event structure — counts are Poisson means *per event*
    n_events: int = 100
    n_random: float = 5.0              # mean random single-particle background per event
    n_background_pairs: float = 10.0   # mean correlated background pairs per event
    n_signal: float = 2.0              # mean signal e+/e- pairs per event

    # Pairing
    allow_same_detector_pairs: bool = True

    # Hit merging
    merge_hits: bool = True
    merge_spatial_threshold: float = 1.0   # cm
    merge_time_threshold: float = 10.0     # ns

    # Spectra
    signal_spectrum: AngularSpectrum = field(
        default_factory=lambda: GaussianSpectrum(mean_deg=120.0, sigma_deg=5.0)
    )
    background_spectrum: AngularSpectrum = field(
        default_factory=lambda: ExponentialSpectrum(scale_deg=40.0)
    )

    # Random particle types for single-particle background
    random_pid_weights: dict = field(default_factory=lambda: {
        'electron': 0.3,
        'positron': 0.3,
        'photon': 0.3,
        'proton': 0.1,
    })

    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Particle Generator
# ---------------------------------------------------------------------------

def _random_direction() -> np.ndarray:
    """Sample a uniformly random direction on the sphere."""
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.array([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta])


def _direction_at_angle(reference: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return a direction making a given angle with `reference`, random azimuth."""
    angle_rad = np.radians(angle_deg)
    ref = reference / np.linalg.norm(reference)
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
        self._event_counter = 0
        self._particle_counter = 0
        self._pair_counter = 0
        # Accumulate true generated angular separations for all events
        self.true_signal_angles: list[float] = []
        self.true_bg_angles: list[float] = []

    def _next_event_id(self):
        eid = self._event_counter
        self._event_counter += 1
        return eid

    def _next_particle_id(self):
        pid = self._particle_counter
        self._particle_counter += 1
        return pid

    def _next_pair_id(self):
        pair_id = self._pair_counter
        self._pair_counter += 1
        return pair_id

    def _random_pid(self):
        pids = list(self.cfg.random_pid_weights.keys())
        weights = np.array(list(self.cfg.random_pid_weights.values()))
        weights /= weights.sum()
        return np.random.choice(pids, p=weights)

    def generate_event(self, event_t0: float = 0.0) -> list[Particle]:
        """Generate all particles for a single readout event."""
        particles = []

        # --- Random single-particle background (uncorrelated) ---
        n_rand = np.random.poisson(self.cfg.n_random)
        for _ in range(n_rand):
            eid = self._next_event_id()
            ptid = self._next_particle_id()
            t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
            particles.append(Particle(
                pid=self._random_pid(),
                origin=np.zeros(3),
                direction=_random_direction(),
                t0=t0,
                event_id=eid,
                particle_id=ptid,
                pair_id=-1,
                source='random',
            ))

        # --- Correlated background pairs ---
        n_bg = np.random.poisson(self.cfg.n_background_pairs)
        if n_bg > 0:
            angles = self.cfg.background_spectrum.sample(n_bg)
            self.true_bg_angles.extend(angles.tolist())
            for angle in angles:
                eid = self._next_event_id()
                t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                d1 = _random_direction()
                d2 = _direction_at_angle(d1, angle)
                bg_pair_id = self._next_pair_id()
                particles.append(Particle('electron', np.zeros(3), d1, t0, eid,
                                          self._next_particle_id(), bg_pair_id, 'background_pair'))
                particles.append(Particle('positron', np.zeros(3), d2, t0, eid,
                                          self._next_particle_id(), bg_pair_id, 'background_pair'))

        # --- Signal e+/e- pairs ---
        n_sig = np.random.poisson(self.cfg.n_signal)
        if n_sig > 0:
            angles = self.cfg.signal_spectrum.sample(n_sig)
            self.true_signal_angles.extend(angles.tolist())
            for angle in angles:
                eid = self._next_event_id()
                t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
                d1 = _random_direction()
                d2 = _direction_at_angle(d1, angle)
                sig_pair_id = self._next_pair_id()
                particles.append(Particle('electron', np.zeros(3), d1, t0, eid,
                                          self._next_particle_id(), sig_pair_id, 'signal'))
                particles.append(Particle('positron', np.zeros(3), d2, t0, eid,
                                          self._next_particle_id(), sig_pair_id, 'signal'))

        return particles

    def generate_all(self) -> list[Particle]:
        all_particles = []
        for i in range(self.cfg.n_events):
            event_t0 = i * self.cfg.time_spread
            all_particles.extend(self.generate_event(event_t0))
        return all_particles


# ---------------------------------------------------------------------------
# Propagation & Reconstruction
# ---------------------------------------------------------------------------

class Propagator:
    """
    Propagates particles to detectors and applies smearing.
    Straight-line track model (no magnetic field).
    """

    def __init__(self, detectors: list[Detector], cfg: SimConfig):
        self.detectors = detectors
        self.cfg = cfg

    def _collect_intersections(self, particle: Particle):
        """Find all detector intersections, sorted by distance from origin."""
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
        true_time = particle.t0 + travel_dist / 30.0  # c ≈ 30 cm/ns
        reco_pos = local_2d + np.random.normal(0, self.cfg.spatial_resolution, 2)
        reco_time = true_time + np.random.normal(0, self.cfg.time_resolution)
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
        )

    def propagate(self, particle: Particle) -> list[Hit]:
        """Propagate from origin — record all detector intersections."""
        return [self._build_hit(particle, det, hit_3d, local_2d)
                for _, det, hit_3d, local_2d in self._collect_intersections(particle)]


# ---------------------------------------------------------------------------
# Analysis: Coincidence, Triggers & Angular Separation
# ---------------------------------------------------------------------------

# Adjacent detector pairs in the square geometry (non-opposite sides).
# Opposite pairs (0↔1 = +X↔-X, 2↔3 = +Y↔-Y) are NOT adjacent.
ADJACENT_DETECTOR_PAIRS = frozenset([
    frozenset([0, 2]),  # +X and +Y
    frozenset([0, 3]),  # +X and -Y
    frozenset([1, 2]),  # -X and +Y
    frozenset([1, 3]),  # -X and -Y
])


def event_passes_single_trigger(hits: list[Hit]) -> bool:
    """Return True if hits exist on exactly one detector (no inter-detector coincidence)."""
    return len({h.detector_id for h in hits}) == 1


def event_passes_adjacent_trigger(hits: list[Hit]) -> bool:
    """Return True if at least one pair of adjacent detectors both have hits."""
    hit_det_ids = {h.detector_id for h in hits}
    return any(pair.issubset(hit_det_ids) for pair in ADJACENT_DETECTOR_PAIRS)


def _classify_pair_source(ha: Hit, hb: Hit) -> str:
    """
    Label a coincident pair by its truth source.
    Correlated pairs (signal / background_pair) require the same pair_id so that
    hits from two *different* background pairs in the same event are not
    misclassified as a correlated pair — they are combinatorial background.
    """
    if (ha.source == 'signal' and hb.source == 'signal'
            and ha.pair_id == hb.pair_id and ha.pair_id != -1):
        return 'signal'
    if (ha.source == 'background_pair' and hb.source == 'background_pair'
            and ha.pair_id == hb.pair_id and ha.pair_id != -1):
        return 'background_pair'
    return 'random'


def find_coincident_pairs(hits: list[Hit], cfg: SimConfig) -> list[tuple[Hit, Hit]]:
    """
    Find all pairs of hits within the coincidence window.
    Same-particle hits are always excluded (they'd give spurious 0° entries).
    """
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
    Merge hits that are too close together on the same detector in both
    space and time (greedy single-pass clustering).
    """
    if not hits:
        return hits

    by_det: dict[int, list[Hit]] = {}
    for h in hits:
        by_det.setdefault(h.detector_id, []).append(h)

    merged_hits: list[Hit] = []

    for det_id, det_hits in by_det.items():
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
                merged_hits.append(seed)
            else:
                merged_pos  = np.mean([h.reco_pos  for h in cluster], axis=0)
                merged_time = np.mean([h.reco_time for h in cluster])
                source_priority = {'signal': 0, 'background_pair': 1, 'random': 2}
                best = min(cluster, key=lambda h: source_priority[h.source])
                merged_hits.append(Hit(
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
                ))

    return merged_hits


def angular_separation_from_hits(ha: Hit, hb: Hit) -> float:
    """Compute angular separation between two hits using their 3D directions."""
    cos_angle = np.clip(np.dot(ha.direction, hb.direction), -1, 1)
    return np.degrees(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# Main Simulation Runner
# ---------------------------------------------------------------------------

class MicromegasSimulation:
    """
    Top-level simulation object. Build, run, and analyse in one place.

    After run(), results are available under two trigger scenarios:
      - 'any'  : all events with at least one detector hit
      - 'adj'  : events where at least one pair of adjacent detectors both fire

    Attributes (per trigger)
    ------------------------
    angular_separations_any / _single / _adj : np.ndarray of angular separations [deg]
    pair_sources_any / _single / _adj        : list of truth labels per pair
    coincident_pairs_any / _single / _adj    : list of (Hit, Hit) tuples

    True generated spectra (before detector acceptance)
    ---------------------------------------------------
    true_signal_angles : np.ndarray of generated signal pair opening angles
    true_bg_angles     : np.ndarray of generated background pair opening angles
    """

    def __init__(self, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

        self.detectors = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)
        self.generator = ParticleGenerator(self.cfg)

        self.particles: list[Particle] = []
        self.hits: list[Hit] = []
        self.hits_premerge: list[Hit] = []

        # Per-trigger result storage
        self.coincident_pairs_any: list[tuple[Hit, Hit]] = []
        self.coincident_pairs_single: list[tuple[Hit, Hit]] = []
        self.coincident_pairs_adj: list[tuple[Hit, Hit]] = []
        self.angular_separations_any: np.ndarray = np.array([])
        self.angular_separations_single: np.ndarray = np.array([])
        self.angular_separations_adj: np.ndarray = np.array([])
        self.pair_sources_any: list[str] = []
        self.pair_sources_single: list[str] = []
        self.pair_sources_adj: list[str] = []

        # True generated spectra
        self.true_signal_angles: np.ndarray = np.array([])
        self.true_bg_angles: np.ndarray = np.array([])

    # Backward-compat aliases (point to 'any' trigger results)
    @property
    def coincident_pairs(self):
        return self.coincident_pairs_any

    @property
    def angular_separations(self):
        return self.angular_separations_any

    @property
    def pair_sources(self):
        return self.pair_sources_any

    def _build_detectors(self):
        return [
            Detector(side=i,
                     distance=self.cfg.detector_distance,
                     size=self.cfg.detector_size)
            for i in range(4)
        ]

    def update_distance(self, new_distance: float):
        self.cfg.detector_distance = new_distance
        self.detectors = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)

    def update_resolution(self, spatial_cm: float = None, time_ns: float = None):
        if spatial_cm is not None:
            self.cfg.spatial_resolution = spatial_cm
        if time_ns is not None:
            self.cfg.time_resolution = time_ns
        self.propagator = Propagator(self.detectors, self.cfg)

    def run(self, n_workers: int = 1):
        """
        Simulate n_events readout windows.

        Parameters
        ----------
        n_workers : int
            Number of parallel worker processes (default 1 = serial).
            Parallel mode skips storing per-event Hit/Particle objects
            (self.hits and self.particles will be empty) to avoid pickling
            overhead across processes.
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        n = self.cfg.n_events
        mode = 'serial' if n_workers == 1 else f'{n_workers} workers'
        print(f"[Sim] Running {n} events ({mode})...")

        self.particles     = []
        self.hits          = []
        self.hits_premerge = []

        # ------------------------------------------------------------------
        # Serial path — stores Hit/Particle objects, event-level progress bar
        # ------------------------------------------------------------------
        if n_workers == 1:
            angles_any,    sources_any,    cpairs_any    = [], [], []
            angles_single, sources_single, cpairs_single = [], [], []
            angles_adj,    sources_adj,    cpairs_adj    = [], [], []
            n_hits_raw = n_hits_away = n_hits_final = 0
            n_sig_prod = n_sig_det = n_bg_prod = n_bg_det = 0

            iterator = range(n)
            if _tqdm is not None:
                iterator = _tqdm(iterator, desc='Simulating', unit='evt',
                                 dynamic_ncols=True)

            for i_evt in iterator:
                event_t0      = i_evt * self.cfg.time_spread
                evt_particles = self.generator.generate_event(event_t0)
                self.particles.extend(evt_particles)

                evt_hits = [h for p in evt_particles
                            for h in self.propagator.propagate(p)]
                self.hits_premerge.extend(evt_hits)
                n_hits_raw += len(evt_hits)

                if self.cfg.merge_hits:
                    evt_merged   = merge_hits(evt_hits, self.cfg)
                    n_hits_away += len(evt_hits) - len(evt_merged)
                else:
                    evt_merged = evt_hits
                self.hits.extend(evt_merged)
                n_hits_final += len(evt_merged)

                evt_pairs  = find_coincident_pairs(evt_merged, self.cfg)
                evt_angles = [angular_separation_from_hits(a, b) for a, b in evt_pairs]
                evt_src    = [_classify_pair_source(a, b)        for a, b in evt_pairs]

                if evt_merged:
                    cpairs_any.extend(evt_pairs)
                    angles_any.extend(evt_angles); sources_any.extend(evt_src)
                if event_passes_single_trigger(evt_merged):
                    cpairs_single.extend(evt_pairs)
                    angles_single.extend(evt_angles); sources_single.extend(evt_src)
                if event_passes_adjacent_trigger(evt_merged):
                    cpairs_adj.extend(evt_pairs)
                    angles_adj.extend(evt_angles); sources_adj.extend(evt_src)

                detected = {h.particle_id for h in evt_merged}
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

            self.coincident_pairs_any    = cpairs_any
            self.coincident_pairs_single = cpairs_single
            self.coincident_pairs_adj    = cpairs_adj
            self.true_signal_angles = np.array(self.generator.true_signal_angles)
            self.true_bg_angles     = np.array(self.generator.true_bg_angles)
            results = [dict(
                angles_any=angles_any,       sources_any=sources_any,
                angles_single=angles_single, sources_single=sources_single,
                angles_adj=angles_adj,       sources_adj=sources_adj,
                n_hits_raw=n_hits_raw, n_hits_away=n_hits_away,
                n_hits_final=n_hits_final,
                n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
                n_bg_prod=n_bg_prod,   n_bg_det=n_bg_det,
            )]

        # ------------------------------------------------------------------
        # Parallel path — chunk events across workers, chunk-level progress
        # ------------------------------------------------------------------
        else:
            import concurrent.futures
            from concurrent.futures import as_completed

            # Aim for ~20 chunks per worker for reasonable progress granularity
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
                if _tqdm is not None:
                    it = _tqdm(as_completed(futures), total=len(futures),
                               desc='Simulating', unit='chunk', dynamic_ncols=True)
                else:
                    it = as_completed(futures)
                for fut in it:
                    results.append(fut.result())

            # Parallel mode: hits/particles not stored
            self.coincident_pairs_any    = []
            self.coincident_pairs_single = []
            self.coincident_pairs_adj    = []
            self.true_signal_angles = np.array(
                sum((r['true_signal_angles'] for r in results), []))
            self.true_bg_angles = np.array(
                sum((r['true_bg_angles']     for r in results), []))

        # ------------------------------------------------------------------
        # Merge and store analysis results (common to both paths)
        # ------------------------------------------------------------------
        def _cat(key):
            return sum((r[key] for r in results), [])

        self.angular_separations_any    = np.array(_cat('angles_any'))
        self.pair_sources_any           = _cat('sources_any')
        self.angular_separations_single = np.array(_cat('angles_single'))
        self.pair_sources_single        = _cat('sources_single')
        self.angular_separations_adj    = np.array(_cat('angles_adj'))
        self.pair_sources_adj           = _cat('sources_adj')

        n_hits_raw   = sum(r['n_hits_raw']   for r in results)
        n_hits_away  = sum(r['n_hits_away']  for r in results)
        n_hits_final = sum(r['n_hits_final'] for r in results)
        n_sig_prod   = sum(r['n_sig_prod']   for r in results)
        n_sig_det    = sum(r['n_sig_det']    for r in results)
        n_bg_prod    = sum(r['n_bg_prod']    for r in results)
        n_bg_det     = sum(r['n_bg_det']     for r in results)

        self.pair_stats = {
            'signal_produced':   n_sig_prod,
            'signal_detected':   n_sig_det,
            'signal_efficiency': n_sig_det / n_sig_prod if n_sig_prod else 0.0,
            'bg_produced':       n_bg_prod,
            'bg_detected':       n_bg_det,
            'bg_efficiency':     n_bg_det / n_bg_prod   if n_bg_prod  else 0.0,
        }

        print(f"  Hits (raw)             : {n_hits_raw}")
        if self.cfg.merge_hits:
            print(f"  Hits merged away       : {n_hits_away}")
        print(f"  Hits (final)           : {n_hits_final}")
        print(f"  Pairs (any trigger)    : {len(self.angular_separations_any)}")
        print(f"  Pairs (single trigger) : {len(self.angular_separations_single)}")
        print(f"  Pairs (adj trigger)    : {len(self.angular_separations_adj)}")
        print("[Sim] Done.")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_geometry(self, ax=None):
        """Top-down view of detector layout."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        d = self.cfg.detector_distance
        s = self.cfg.detector_size / 2

        colors = ['steelblue', 'tomato', 'seagreen', 'goldenrod']
        positions = [
            ([d, d], [-s, s]),
            ([-d, -d], [-s, s]),
            ([-s, s], [d, d]),
            ([-s, s], [-d, -d]),
        ]
        names = ['+X panel', '-X panel', '+Y panel', '-Y panel']

        for i, ((x0, x1), (y0, y1)) in enumerate(positions):
            ax.plot([x0, x1], [y0, y1], lw=4, color=colors[i], label=names[i])

        ax.plot(0, 0, 'k*', ms=12, label='Target')
        ax.set_xlim(-d * 1.5, d * 1.5)
        ax.set_ylim(-d * 1.5, d * 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_title(f'Detector Layout (distance={self.cfg.detector_distance} cm)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        frac = calculate_solid_angle_coverage(self.detectors, n_samples=1_000)
        ax.annotate(f'Solid angle coverage: {frac*100:.1f}%',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        return ax

    def plot_hits(self, ax=None):
        """Show hit positions on each detector panel."""
        if not self.hits:
            warnings.warn("No hits to plot. Run sim.run() first.")
            return
        if ax is None:
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        else:
            axes = ax

        source_colors = {
            'random': 'gray',
            'background_pair': 'steelblue',
            'signal': 'red',
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

            half = self.cfg.detector_size / 2
            rect = mpatches.Rectangle((-half, -half), self.cfg.detector_size,
                                      self.cfg.detector_size,
                                      linewidth=1.5, edgecolor='black',
                                      facecolor='none')
            ax_.add_patch(rect)
            ax_.set_xlim(-half * 1.2, half * 1.2)
            ax_.set_ylim(-half * 1.2, half * 1.2)
            ax_.set_title(f'Detector {det_names[det_id]}')
            ax_.set_xlabel('u [cm]')
            ax_.set_ylabel('v [cm]')
            ax_.set_aspect('equal')
            if det_id == 0:
                ax_.legend(fontsize=7)

        plt.tight_layout()
        return axes

    def plot_angular_separation(self, bins=36, ax=None, show_components=True,
                                trigger='any'):
        """
        Plot reconstructed angular separation for time-coincident pairs.

        Parameters
        ----------
        trigger : 'any', 'single', or 'adj'
            Which trigger scenario to display.
        """
        _angles_map = {
            'any':    self.angular_separations_any,
            'single': self.angular_separations_single,
            'adj':    self.angular_separations_adj,
        }
        _sources_map = {
            'any':    self.pair_sources_any,
            'single': self.pair_sources_single,
            'adj':    self.pair_sources_adj,
        }
        _label_map = {
            'any':    'Any Detector Hit',
            'single': 'Single Detector Hit',
            'adj':    'Two Adjacent Detectors Hit',
        }
        angles = _angles_map[trigger]
        sources = _sources_map[trigger]

        if len(angles) == 0:
            warnings.warn("No angular separations. Run sim.run() first.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        bin_edges = np.linspace(0, 180, bins + 1)
        trigger_label = _label_map[trigger]

        if show_components:
            source_styles = {
                'random':           ('gray',      'Random combinatorial'),
                'background_pair':  ('steelblue', 'Correlated background'),
                'signal':           ('red',       'Signal (e+e- ~120°)'),
            }
            angs = np.array(angles)
            srcs = np.array(sources)
            bottom = np.zeros(bins)
            for src, (col, label) in source_styles.items():
                mask = srcs == src
                if mask.any():
                    counts, _ = np.histogram(angs[mask], bins=bin_edges)
                    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                           bottom=bottom, color=col, alpha=0.7, label=label,
                           align='edge')
                    bottom += counts
        else:
            ax.hist(angles, bins=bin_edges, color='steelblue',
                    alpha=0.7, label='All coincident pairs')

        ax.axvline(120, color='red', ls='--', lw=1.5, label='120° signal')
        ax.set_xlabel('Angular separation [°]')
        ax.set_ylabel('Pairs / bin')
        ax.set_title(f'Reconstructed Angular Separation\n(Trigger: {trigger_label})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_trigger_comparison(self, bins=36):
        """Side-by-side comparison of all three trigger scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        self.plot_angular_separation(bins=bins, ax=axes[0], trigger='any')
        self.plot_angular_separation(bins=bins, ax=axes[1], trigger='single')
        self.plot_angular_separation(bins=bins, ax=axes[2], trigger='adj')
        fig.suptitle('Trigger Scenario Comparison', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_true_vs_reconstructed(self, bins=36, normalized=False):
        """
        Compare true generated angular distributions (before detector acceptance)
        against reconstructed distributions from detector hits, for both signal
        and background, under both trigger scenarios.

        Parameters
        ----------
        normalized : bool
            If True, normalize all histograms to unit area so shape differences
            are visible independently of overall count differences.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bin_edges = np.linspace(0, 180, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        panel_cfg = [
            ('Signal', self.true_signal_angles, 'red', 'signal'),
            ('Correlated Background', self.true_bg_angles, 'steelblue', 'background_pair'),
        ]

        ylabel = 'Probability density' if normalized else 'Count'

        for ax, (title, true_angles, color, src_key) in zip(axes, panel_cfg):
            # True generated spectrum
            if len(true_angles):
                true_counts, _ = np.histogram(true_angles, bins=bin_edges)
                if normalized and true_counts.sum() > 0:
                    true_vals = true_counts / (true_counts.sum() * bin_width)
                else:
                    true_vals = true_counts.astype(float)
                ax.bar(bin_edges[:-1], true_vals, width=bin_width,
                       color=color, alpha=0.35, align='edge',
                       label=f'True generated  (N={len(true_angles)})')

            # Reconstructed under each trigger
            trig_cfgs = [
                ('any',    self.angular_separations_any,    self.pair_sources_any,    '-',  2.0, 'any-hit'),
                ('single', self.angular_separations_single, self.pair_sources_single, ':',  2.0, 'single'),
                ('adj',    self.angular_separations_adj,    self.pair_sources_adj,    '--', 2.0, 'adjacent'),
            ]
            for trigger, angles, sources, ls, lw, trig_name in trig_cfgs:
                reco = np.array([a for a, s in zip(angles, sources) if s == src_key])
                label = f'Reco, {trig_name} trigger  (N={len(reco)})'
                if len(reco):
                    reco_counts, _ = np.histogram(reco, bins=bin_edges)
                    if normalized and reco_counts.sum() > 0:
                        reco_vals = reco_counts / (reco_counts.sum() * bin_width)
                    else:
                        reco_vals = reco_counts.astype(float)
                    ax.step(bin_edges[:-1], reco_vals, where='post',
                            color=color, lw=lw, ls=ls, label=label)

            if src_key == 'signal':
                ax.axvline(120, color='black', ls=':', lw=1, alpha=0.5)
            ax.set_title(f'{title}: True vs Reconstructed')
            ax.set_xlabel('Angular separation [°]')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        norm_tag = ' (shape normalized)' if normalized else ''
        fig.suptitle(f'True Generated vs Reconstructed Angular Distributions{norm_tag}',
                     fontsize=12)
        plt.tight_layout()
        return fig

    def plot_summary(self, trigger='any'):
        """4-panel summary figure."""
        fig = plt.figure(figsize=(14, 10))
        ax_geo = fig.add_subplot(2, 2, 1)
        ax_ang = fig.add_subplot(2, 2, 2)
        ax_hits_row = [fig.add_subplot(2, 4, 5 + i) for i in range(4)]

        self.plot_geometry(ax=ax_geo)
        self.plot_angular_separation(ax=ax_ang, trigger=trigger)
        self.plot_hits(ax=ax_hits_row)

        fig.suptitle(
            f'Micromegas Simulation Summary\n'
            f'dist={self.cfg.detector_distance} cm | '
            f'σ_x={self.cfg.spatial_resolution} cm | '
            f'σ_t={self.cfg.time_resolution} ns | '
            f'window={self.cfg.coincidence_window} ns',
            fontsize=11
        )
        plt.tight_layout()
        return fig

    def summary_stats(self):
        """Print a text summary including both trigger scenarios."""
        n_raw = len(self.hits_premerge) if hasattr(self, 'hits_premerge') else '?'
        n_merged = n_raw - len(self.hits) if isinstance(n_raw, int) else '?'
        ps = getattr(self, 'pair_stats', None)

        print("=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        print(f"  Events simulated    : {self.cfg.n_events}")
        print(f"  Particles generated : {len(self.particles)}")
        print(f"  Hits (raw)          : {n_raw}")
        if self.cfg.merge_hits:
            print(f"  Hits merged away    : {n_merged}")
        print(f"  Hits (final)        : {len(self.hits)}")
        print(f"  True signal angles  : {len(self.true_signal_angles)}")
        print(f"  True bg angles      : {len(self.true_bg_angles)}")

        for label, angles, sources in [
            ('Any-hit trigger',    self.angular_separations_any,    self.pair_sources_any),
            ('Single-det trigger', self.angular_separations_single, self.pair_sources_single),
            ('Adjacent trigger',   self.angular_separations_adj,    self.pair_sources_adj),
        ]:
            total = len(angles)
            srcs = np.array(sources) if total else np.array([])
            print(f"\n  --- {label} ---")
            print(f"  Coincident pairs    : {total}")
            for src in ['signal', 'background_pair', 'random']:
                n = int((srcs == src).sum()) if total else 0
                pct = 100 * n / total if total else 0
                if n > 0:
                    print(f"    {src:<22s}: {n:5d}  ({pct:.1f}%)")
            n_sig = int((srcs == 'signal').sum()) if total else 0
            print(f"  Signal / Background : {n_sig} / {total - n_sig}")

        if ps:
            print("-" * 60)
            print("  Pair detection efficiency (any-hit trigger)")
            print(f"  Signal     : {ps['signal_detected']:4d}/{ps['signal_produced']:4d}"
                  f"  →  {ps['signal_efficiency']*100:.1f}%")
            print(f"  Background : {ps['bg_detected']:4d}/{ps['bg_produced']:4d}"
                  f"  →  {ps['bg_efficiency']*100:.1f}%")
        print("=" * 60)


def _run_event_batch(args) -> dict:
    """
    Module-level worker for multiprocessing.
    Runs a batch of events and returns summary results — no Hit or Particle
    objects are returned so inter-process data transfer stays fast.
    """
    cfg, n_batch, event_offset, seed = args
    if seed is not None:
        np.random.seed(seed)

    detectors = [Detector(side=i, distance=cfg.detector_distance, size=cfg.detector_size)
                 for i in range(4)]
    propagator = Propagator(detectors, cfg)
    generator  = ParticleGenerator(cfg)

    angles_any,    sources_any    = [], []
    angles_single, sources_single = [], []
    angles_adj,    sources_adj    = [], []
    n_hits_raw = n_hits_away = n_hits_final = 0
    n_sig_prod = n_sig_det = n_bg_prod = n_bg_det = 0

    for i in range(n_batch):
        event_t0     = (event_offset + i) * cfg.time_spread
        evt_particles = generator.generate_event(event_t0)
        evt_hits      = [h for p in evt_particles for h in propagator.propagate(p)]
        n_hits_raw   += len(evt_hits)

        if cfg.merge_hits:
            evt_merged   = merge_hits(evt_hits, cfg)
            n_hits_away += len(evt_hits) - len(evt_merged)
        else:
            evt_merged = evt_hits
        n_hits_final += len(evt_merged)

        evt_pairs  = find_coincident_pairs(evt_merged, cfg)
        evt_angles = [angular_separation_from_hits(a, b) for a, b in evt_pairs]
        evt_src    = [_classify_pair_source(a, b)        for a, b in evt_pairs]

        if evt_merged:
            angles_any.extend(evt_angles);    sources_any.extend(evt_src)
        if event_passes_single_trigger(evt_merged):
            angles_single.extend(evt_angles); sources_single.extend(evt_src)
        if event_passes_adjacent_trigger(evt_merged):
            angles_adj.extend(evt_angles);    sources_adj.extend(evt_src)

        detected = {h.particle_id for h in evt_merged}
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
        angles_any=angles_any,       sources_any=sources_any,
        angles_single=angles_single, sources_single=sources_single,
        angles_adj=angles_adj,       sources_adj=sources_adj,
        true_signal_angles=generator.true_signal_angles,
        true_bg_angles=generator.true_bg_angles,
        n_hits_raw=n_hits_raw, n_hits_away=n_hits_away, n_hits_final=n_hits_final,
        n_sig_prod=n_sig_prod, n_sig_det=n_sig_det,
        n_bg_prod=n_bg_prod,   n_bg_det=n_bg_det,
    )


def calculate_solid_angle_coverage(detectors: list[Detector], n_samples: int = 1_000_000) -> float:
    """
    Estimate the fraction of the full solid angle (4π sr) covered by the
    detector array via Monte Carlo sampling.
    """
    origin = np.zeros(3)
    hits = 0
    for _ in range(n_samples):
        direction = _random_direction()
        for det in detectors:
            _, local_2d = det.intersect(origin, direction)
            if local_2d is not None:
                hits += 1
                break
    return hits / n_samples

"""
Micromegas Detector Simulation
================================
Simulates a 4-detector Micromegas setup surrounding a central target.
Detectors are 40x40 cm flat panels arranged in a square geometry.

Particle types:
  - Random background: single e/e+, photons, protons (uncorrelated in time)
  - Correlated background: coincident pairs with angular distribution from a background spectrum
  - Signal: coincident e+/e- pairs with angular separation ~120° (Gaussian)

Reconstruction: hits are smeared in space and time to emulate detector resolution.
Analysis: builds mock angular separation distributions for time-coincident hits.
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
    origin: np.ndarray  # 3D origin (should be [0,0,0] for target)
    direction: np.ndarray  # unit 3-vector
    t0: float           # true emission time [ns]
    event_id: int = -1
    particle_id: int = -1   # unique per particle; hits from same particle share this
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
        """
        Parameters
        ----------
        side     : int, 0-3, which face of the square
        distance : float, center-to-detector distance [cm]
        size     : float, detector active area side length [cm]
        """
        self.side = side
        self.distance = distance
        self.size = size
        self.normal = self.SIDE_NORMALS[side]
        self.name = self.SIDE_NAMES[side]

        # Center position of the detector panel
        self.center = -self.normal * distance  # e.g. side 0 normal=-X, center at +X

    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """
        Find where a ray (origin + t*direction) hits this detector plane.

        Returns (hit_3d, local_2d) or (None, None) if no valid intersection.
        local_2d is the 2D coordinate within the detector face [-size/2, size/2]^2.
        """
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-9:
            return None, None  # parallel to panel

        # Plane equation: normal · (p - center) = 0  =>  t = normal·(center-origin) / normal·direction
        t = np.dot(self.normal, self.center - origin) / denom
        if t <= 0:
            return None, None  # behind origin

        hit_3d = origin + t * direction

        # Project onto detector local 2D coordinates
        # Build two orthogonal axes in the plane
        local_u, local_v = self._local_axes()
        delta = hit_3d - self.center
        u = np.dot(delta, local_u)
        v = np.dot(delta, local_v)

        half = self.size / 2.0
        if abs(u) > half or abs(v) > half:
            return None, None  # outside active area

        return hit_3d, np.array([u, v])

    def _local_axes(self):
        """Return two orthogonal unit vectors spanning the detector face."""
        # Choose a consistent up vector
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
    """Flat (uniform) distribution over [min, max] degrees — simple background."""
    def __init__(self, min_deg: float = 0.0, max_deg: float = 180.0):
        self.min = min_deg
        self.max = max_deg

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.min, self.max, n)


class IsotropicSpectrum(AngularSpectrum):
    """
    Background angular separation for isotropic pairs.
    Two isotropic particles: the angular separation distribution
    goes as sin(theta), peaking near 90 degrees.
    """
    def sample(self, n: int) -> np.ndarray:
        # Sample via rejection: p(theta) ~ sin(theta)
        samples = []
        while len(samples) < n:
            theta = np.random.uniform(0, 180, n * 2)
            prob = np.sin(np.radians(theta))
            accept = np.random.uniform(0, 1, n * 2) < prob
            samples.extend(theta[accept].tolist())
        return np.array(samples[:n])


class HistogramSpectrum(AngularSpectrum):
    """User-defined spectrum from histogram bins. Easy to swap in later."""
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

    Parameters
    ----------
    scale_deg : float
        Decay constant in degrees. Smaller = steeper fall. Default 40°.
    min_deg   : float
        Lower bound of the distribution. Default 0°.
    max_deg   : float
        Upper bound of the distribution. Default 180°.
    """
    def __init__(self, scale_deg: float = 40.0,
                 min_deg: float = 0.0, max_deg: float = 180.0):
        self.scale = scale_deg
        self.min = min_deg
        self.max = max_deg
        # Normalisation constants for inverse-CDF on truncated exponential
        self._lo = np.exp(-min_deg / scale_deg)
        self._hi = np.exp(-max_deg / scale_deg)

    def sample(self, n: int) -> np.ndarray:
        # Inverse-CDF of truncated exponential
        u = np.random.uniform(0, 1, n)
        return -self.scale * np.log(self._lo - u * (self._lo - self._hi))



class ExponentialSpectrum(AngularSpectrum):
    """
    Falling exponential background: p(theta) ~ exp(-theta / scale_deg),
    sampled over [min_deg, max_deg] via inverse-CDF.

    Parameters
    ----------
    scale_deg : float
        Decay constant in degrees. Smaller = steeper fall. Default 40 deg.
    min_deg   : float
        Lower bound of the distribution. Default 0 deg.
    max_deg   : float
        Upper bound of the distribution. Default 180 deg.
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
    time_spread: float = 200.0          # ns, duration of a single readout event [ns]

    # Event structure — counts are Poisson means *per event*
    n_events: int = 100                 # number of readout events to simulate
    n_random: float = 50.0             # mean random single-particle background per event
    n_background_pairs: float = 10.0   # mean correlated background pairs per event
    n_signal: float = 2.0              # mean signal e+/e- pairs per event

    # Pairing
    allow_same_detector_pairs: bool = True  # Allow pairs from the same detector

    # Hit merging: two hits on the same detector are merged if closer than these thresholds
    merge_hits: bool = True
    merge_spatial_threshold: float = 1.0   # cm — merge if distance < this
    merge_time_threshold: float = 10.0     # ns — merge if |dt| < this

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

    # Seed
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
    """
    Return a direction that makes a given angle with `reference`,
    with a random azimuth around reference.
    """
    angle_rad = np.radians(angle_deg)

    # Build orthonormal basis around reference
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
        weights = list(self.cfg.random_pid_weights.values())
        weights = np.array(weights) / sum(weights)
        return np.random.choice(pids, p=weights)

    def generate_event(self, event_t0: float = 0.0) -> list[Particle]:
        """
        Generate all particles for a single readout event.

        Particle counts are drawn from Poisson distributions using the
        configured means. event_t0 offsets all times so events don't overlap.
        Each particle gets a unique particle_id so hits from the same particle
        can always be identified and excluded from pairing.

        Parameters
        ----------
        event_t0 : float
            Absolute time offset for this event [ns].
        """
        particles = []

        # --- Random single-particle background ---
        n_rand = np.random.poisson(self.cfg.n_random)
        for _ in range(n_rand):
            eid = self._next_event_id()
            ptid = self._next_particle_id()
            pid = self._random_pid()
            t0 = event_t0 + np.random.uniform(0, self.cfg.time_spread)
            particles.append(Particle(
                pid=pid,
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
        """Generate particles across all events."""
        all_particles = []
        for i in range(self.cfg.n_events):
            event_t0 = i * self.cfg.time_spread  # non-overlapping event windows
            all_particles.extend(self.generate_event(event_t0))
        return all_particles


# ---------------------------------------------------------------------------
# Propagation & Reconstruction
# ---------------------------------------------------------------------------

class Propagator:
    """
    Propagates particles to detectors and applies smearing.
    Uses a simple straight-line track model (no magnetic field).
    """

    def __init__(self, detectors: list[Detector], cfg: SimConfig):
        self.detectors = detectors
        self.cfg = cfg

    def propagate(self, particle: Particle) -> list[Hit]:
        """Try to intersect a particle with all detectors. Return hits."""
        hits = []
        for det in self.detectors:
            hit_3d, local_2d = det.intersect(particle.origin, particle.direction)
            if local_2d is None:
                continue

            # Compute travel distance → time (assume relativistic, v~c)
            # c ≈ 30 cm/ns
            travel_dist = np.linalg.norm(hit_3d - particle.origin)
            true_time = particle.t0 + travel_dist / 30.0

            # Smear position and time
            smear_pos = np.random.normal(0, self.cfg.spatial_resolution, 2)
            smear_t = np.random.normal(0, self.cfg.time_resolution)
            reco_pos = local_2d + smear_pos
            reco_time = true_time + smear_t

            hits.append(Hit(
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
            ))
        return hits


# ---------------------------------------------------------------------------
# Analysis: Coincidence & Angular Separation
# ---------------------------------------------------------------------------

def find_coincident_pairs(hits: list[Hit], cfg: SimConfig) -> list[tuple[Hit, Hit]]:
    """
    Find all pairs of hits within the coincidence window.

    Same-particle pairs (hits sharing a particle_id) are always excluded —
    these are just one particle crossing two detector panels and would
    produce spurious 0° entries.

    By default only pairs hits on *different* detectors. Set
    cfg.allow_same_detector_pairs = True to also include pairs where both
    hits landed on the same panel (useful for studying in-detector
    conversions or delta rays).

    Returns list of (hit_a, hit_b) tuples.
    """
    pairs = []
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            ha, hb = hits[i], hits[j]
            # Always reject hits from the same particle (would give 0°)
            if ha.particle_id == hb.particle_id:
                continue
            if not cfg.allow_same_detector_pairs and ha.detector_id == hb.detector_id:
                continue
            dt = abs(ha.reco_time - hb.reco_time)
            if dt <= cfg.coincidence_window:
                pairs.append((ha, hb))
    return pairs


def merge_hits(hits: list[Hit], cfg: SimConfig) -> list[Hit]:
    """
    Merge hits that are too close together on the same detector in both
    space and time — emulating the finite spatial and temporal resolution
    of the readout strips/pads.

    Algorithm: greedy single-pass clustering.
      - Sort hits by detector then reco_time.
      - For each unmerged hit, collect all subsequent hits on the same
        detector within (merge_time_threshold, merge_spatial_threshold).
      - Replace the cluster with one merged hit whose reco_pos and
        reco_time are the centroid of the cluster.
      - The merged hit inherits source/pid from the earliest hit.
        If any constituent is 'signal', the merged hit is labelled 'signal'.

    Parameters
    ----------
    hits : list[Hit]
        Reconstructed hits from the propagator (already smeared).
    cfg  : SimConfig
        Must have merge_spatial_threshold [cm] and merge_time_threshold [ns].

    Returns
    -------
    list[Hit] with merged hits removed / replaced.
    """
    if not hits:
        return hits

    # Group by detector
    by_det: dict[int, list[Hit]] = {}
    for h in hits:
        by_det.setdefault(h.detector_id, []).append(h)

    merged_hits: list[Hit] = []

    for det_id, det_hits in by_det.items():
        # Sort by reco_time for efficient time-windowed clustering
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
                # Early exit: hits are time-sorted, so once dt exceeds threshold
                # no further hits in this detector can be in the cluster
                if other.reco_time - seed.reco_time > cfg.merge_time_threshold:
                    break
                spatial_dist = np.linalg.norm(other.reco_pos - seed.reco_pos)
                if spatial_dist < cfg.merge_spatial_threshold:
                    cluster.append(other)
                    used[j] = True

            if len(cluster) == 1:
                merged_hits.append(seed)
            else:
                # Build merged hit: centroid position and time
                merged_pos  = np.mean([h.reco_pos  for h in cluster], axis=0)
                merged_time = np.mean([h.reco_time for h in cluster])
                # Truth position/time: keep earliest (seed, already time-sorted)
                # Source: promote to most specific
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
    """
    Compute angular separation between two hits using their 3D directions.
    For a more realistic reconstruction you'd use hit positions on detectors;
    here we use the true direction smeared implicitly by spatial resolution.
    """
    cos_angle = np.clip(np.dot(ha.direction, hb.direction), -1, 1)
    return np.degrees(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# Main Simulation Runner
# ---------------------------------------------------------------------------

class MicromegasSimulation:
    """
    Top-level simulation object. Build, run, and analyse in one place.

    Example
    -------
    >>> cfg = SimConfig(detector_distance=15.0, n_signal=500)
    >>> sim = MicromegasSimulation(cfg)
    >>> sim.run()
    >>> sim.plot_angular_separation()
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
        self.coincident_pairs: list[tuple[Hit, Hit]] = []
        self.angular_separations: np.ndarray = np.array([])
        self.pair_sources: list[str] = []

    def _build_detectors(self):
        return [
            Detector(side=i,
                     distance=self.cfg.detector_distance,
                     size=self.cfg.detector_size)
            for i in range(4)
        ]

    def update_distance(self, new_distance: float):
        """Move all detectors closer/further from the target."""
        self.cfg.detector_distance = new_distance
        self.detectors = self._build_detectors()
        self.propagator = Propagator(self.detectors, self.cfg)

    def update_resolution(self, spatial_cm: float = None, time_ns: float = None):
        """Update detector resolutions."""
        if spatial_cm is not None:
            self.cfg.spatial_resolution = spatial_cm
        if time_ns is not None:
            self.cfg.time_resolution = time_ns
        self.propagator = Propagator(self.detectors, self.cfg)

    def run(self):
        """
        Simulate n_events readout windows. For each event:
          1. Generate particles (Poisson-sampled counts).
          2. Propagate to detectors and smear hits.
          3. Optionally merge nearby hits on the same detector.
          4. Find time-coincident pairs within the event.
          5. Compute angular separations.
        Results are accumulated across all events.
        """
        print(f"[Sim] Running {self.cfg.n_events} events...")
        self.particles = []
        self.hits = []
        self.hits_premerge = []   # for diagnostics
        self.coincident_pairs = []
        all_angles = []
        all_sources = []

        n_merged_total = 0

        # Pair-detection counters: how many correlated pairs were produced,
        # and how many had *both* particles register at least one hit.
        n_signal_pairs_produced = 0
        n_signal_pairs_detected = 0
        n_bg_pairs_produced = 0
        n_bg_pairs_detected = 0

        for i_evt in range(self.cfg.n_events):
            print(f'Event {i_evt+1}/{self.cfg.n_events}')
            event_t0 = i_evt * self.cfg.time_spread

            # 1. Generate
            evt_particles = self.generator.generate_event(event_t0)
            self.particles.extend(evt_particles)

            # 2. Propagate
            evt_hits = []
            for p in evt_particles:
                evt_hits.extend(self.propagator.propagate(p))
            self.hits_premerge.extend(evt_hits)

            # 3. Merge
            if self.cfg.merge_hits:
                evt_hits_merged = merge_hits(evt_hits, self.cfg)
                n_merged_total += len(evt_hits) - len(evt_hits_merged)
            else:
                evt_hits_merged = evt_hits
            self.hits.extend(evt_hits_merged)

            # 4. Coincident pairs — scoped to this event's hits only
            evt_pairs = find_coincident_pairs(evt_hits_merged, self.cfg)
            self.coincident_pairs.extend(evt_pairs)

            # 5. Angular separations
            for ha, hb in evt_pairs:
                all_angles.append(angular_separation_from_hits(ha, hb))
                if ha.source == 'signal' and hb.source == 'signal':
                    all_sources.append('signal')
                elif ha.source == 'background_pair' and hb.source == 'background_pair':
                    all_sources.append('background_pair')
                else:
                    all_sources.append('random')

            # 6. Count produced and detected pairs for signal and background.
            #    A pair is "detected" if both of its particles left at least one
            #    hit after merging (identified via particle_id -> pair_id mapping).
            detected_particle_ids = {h.particle_id for h in evt_hits_merged}

            # Build a map: pair_id -> set of particle_ids in that pair
            pair_to_particles: dict[int, set] = {}
            for p in evt_particles:
                if p.pair_id == -1:
                    continue
                pair_to_particles.setdefault(p.pair_id, set()).add(p.particle_id)

            for pair_id, ptids in pair_to_particles.items():
                # Determine source from any particle in the pair
                src = next(p.source for p in evt_particles if p.pair_id == pair_id)
                both_detected = ptids.issubset(detected_particle_ids)
                if src == 'signal':
                    n_signal_pairs_produced += 1
                    if both_detected:
                        n_signal_pairs_detected += 1
                elif src == 'background_pair':
                    n_bg_pairs_produced += 1
                    if both_detected:
                        n_bg_pairs_detected += 1

            # Print some stats per event
            print(f"   Event Particles: {len(evt_particles)}")
            print(f"   Event Hits: {len(evt_hits)}")
            print(f"   Event Coincident Pairs: {len(evt_pairs)}")

        self.angular_separations = np.array(all_angles)
        self.pair_sources = all_sources

        # Store pair detection summary for use in summary_stats / external access
        self.pair_stats = {
            'signal_produced':    n_signal_pairs_produced,
            'signal_detected':    n_signal_pairs_detected,
            'signal_efficiency':  n_signal_pairs_detected / n_signal_pairs_produced
                                   if n_signal_pairs_produced else 0.0,
            'bg_produced':        n_bg_pairs_produced,
            'bg_detected':        n_bg_pairs_detected,
            'bg_efficiency':      n_bg_pairs_detected / n_bg_pairs_produced
                                   if n_bg_pairs_produced else 0.0,
        }

        print(f"  Total particles      : {len(self.particles)}")
        print(f"  Total hits (raw)     : {len(self.hits_premerge)}")
        if self.cfg.merge_hits:
            print(f"  Hits merged away     : {n_merged_total}")
        print(f"  Total hits (final)   : {len(self.hits)}")
        print(f"  Coincident pairs     : {len(self.coincident_pairs)}")
        print(f"[Sim] Done.")

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
            ([d, d], [-s, s]),             # +X: vertical line at x=d
            ([-d, -d], [-s, s]),           # -X
            ([-s, s], [d, d]),             # +Y
            ([-s, s], [-d, -d]),           # -Y
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

        source_colors = {'random': 'gray', 'background_pair': 'steelblue', 'signal': 'red'}
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

    def plot_angular_separation(self, bins=36, ax=None, show_components=True):
        """
        Plot the reconstructed angular separation distribution for
        time-coincident pairs, optionally broken down by truth source.
        """
        if len(self.angular_separations) == 0:
            warnings.warn("No angular separations. Run sim.run() first.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        bin_edges = np.linspace(0, 180, bins + 1)

        if show_components:
            source_styles = {
                'random':           ('gray',       'Random background'),
                'background_pair':  ('steelblue', 'Correlated background'),
                'signal':           ('red',       'Signal (e+e- ~120°)'),
            }
            angs = np.array(self.angular_separations)
            srcs = np.array(self.pair_sources)
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
            ax.hist(self.angular_separations, bins=bin_edges, color='steelblue',
                    alpha=0.7, label='All coincident pairs')

        ax.axvline(120, color='red', ls='--', lw=1.5, label='120° signal')
        ax.set_xlabel('Angular separation [°]')
        ax.set_ylabel('Pairs / bin')
        ax.set_title('Reconstructed Angular Separation (coincident pairs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_summary(self):
        """4-panel summary figure."""
        fig = plt.figure(figsize=(14, 10))
        ax_geo = fig.add_subplot(2, 2, 1)
        ax_ang = fig.add_subplot(2, 2, 2)
        ax_hits_row = [fig.add_subplot(2, 4, 5 + i) for i in range(4)]

        self.plot_geometry(ax=ax_geo)
        self.plot_angular_separation(ax=ax_ang)
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
        """Print a text summary."""
        total = len(self.coincident_pairs)
        srcs = np.array(self.pair_sources)
        n_raw  = len(self.hits_premerge) if hasattr(self, 'hits_premerge') else '?'
        n_merged = n_raw - len(self.hits) if isinstance(n_raw, int) else '?'
        ps = getattr(self, 'pair_stats', None)
        print("=" * 55)
        print("Simulation Summary")
        print("=" * 55)
        print(f"  Events simulated    : {self.cfg.n_events}")
        print(f"  Particles generated : {len(self.particles)}")
        print(f"  Hits (raw)          : {n_raw}")
        if self.cfg.merge_hits:
            print(f"  Hits merged away    : {n_merged}")
        print(f"  Hits (final)        : {len(self.hits)}")
        print(f"  Coincident pairs    : {total}")
        for src in ['signal', 'background_pair', 'random']:
            n = (srcs == src).sum()
            pct = 100 * n / total if total else 0
            print(f"    {src:<20s}: {n:5d}  ({pct:.1f}%)")
        print(f"  Signal/Background   : "
              f"{(srcs=='signal').sum()} / {(srcs!='signal').sum()}")
        if ps:
            print("-" * 55)
            print("  Pair detection efficiency (both particles hit a detector)")
            print(f"  Signal     : {ps['signal_detected']:4d} / {ps['signal_produced']:4d} produced"
                  f"  →  {ps['signal_efficiency']*100:.1f}%")
            print(f"  Background : {ps['bg_detected']:4d} / {ps['bg_produced']:4d} produced"
                  f"  →  {ps['bg_efficiency']*100:.1f}%")
        print("=" * 55)


def calculate_solid_angle_coverage(detectors: list[Detector], n_samples: int = 1_000_000) -> float:
    """
    Estimate the fraction of the full solid angle (4π sr) covered by the
    detector array, as seen from the origin (target position).

    Uses Monte Carlo sampling: throw random directions uniformly over the
    sphere and count what fraction hits at least one detector's active area.

    Parameters
    ----------
    detectors : list[Detector]
        The detector panels to test against.
    n_samples : int
        Number of random directions to sample. 1M gives ~0.1% statistical
        precision. Increase for higher accuracy.

    Returns
    -------
    float
        Fractional solid angle coverage in [0, 1].
        Multiply by 4π to get coverage in steradians.

    Example
    -------
    >>> sim = MicromegasSimulation(cfg)
    >>> frac = calculate_solid_angle_coverage(sim.detectors)
    >>> print(f"Coverage: {frac*100:.2f}%  ({frac*4*np.pi:.4f} sr)")
    """
    origin = np.zeros(3)
    hits = 0
    for _ in range(n_samples):
        direction = _random_direction()
        for det in detectors:
            _, local_2d = det.intersect(origin, direction)
            if local_2d is not None:
                hits += 1
                break  # count direction once even if it hits multiple panels
    return hits / n_samples
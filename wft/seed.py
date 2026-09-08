"""
Seeding: the one place hits are allowed in.

Hits answer 'which events and which strips carry a track' — a question about
detection, which is what the analyzer's trigger is for. They do not answer
'where and at what angle', which is what the waveform fit is for. Everything
this module returns is a *set of channels*; no hit time crosses the boundary.

The clustering reproduces the production selection so that detection semantics
(and therefore efficiency) stay comparable with the hits chain:
per-plane relative significance floor, then spatial clustering with the
production gap threshold, largest cluster kept
(``cosmic_micro_tpc_analysis._fit_single_axis``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

SIG_REL_FLOOR = 0.10       # per-plane, relative to that plane's strongest strip
GAP_THRESHOLD_MM = 12.0    # production spatial clustering gap
MIN_STRIPS = 3
SPARK_VETO_HITS = 50       # full-detector discharge: drop the event
N_CANDIDATES = 3           # clusters offered to the fit per plane
PITCH_MM = 398.58 / 512    # 0.7784 — strip pitch, for the hot-run bridge below
#: Let a run of hot strips be crossed for free when clustering the clean ones
#: (``_clean_cluster_labels``). Off, on the argument in that docstring: making
#: every band free to cross re-welds the noise column onto the track the rule
#: exists to separate. On D/run_145 the two settings are also indistinguishable
#: in practice (2 658 vs 2 650 of 3 600 triggers seeded, every fit metric equal
#: to three decimals), so this is a correctness choice, not a tuned one.
#: Scanned with ``WFT_HOT_BRIDGE=1``.
HOT_BRIDGE = os.environ.get('WFT_HOT_BRIDGE', '0') == '1'


@dataclass
class Seed:
    """Candidate cluster for one plane of one event."""
    channels: np.ndarray
    n_strips: int
    n_dropped: int          # strips in competing clusters
    amp_sum: float
    n_raw: int              # strips before the significance floor


def apply_significance_floor(df: pd.DataFrame, rel: float = SIG_REL_FLOOR) -> pd.DataFrame:
    """Keep strips with significance >= rel x that plane's strongest strip in
    that event (the 2026-07-25 fix: coherent noise otherwise inflates
    multiplicity, steals cluster membership and fakes the spark veto)."""
    if not rel or 'significance' not in df.columns:
        return df
    sig = df['significance'].to_numpy()
    if not np.isfinite(sig).any():
        return df
    mx = df.groupby(['eventId', 'feu'])['significance'].transform('max')
    return df[sig >= rel * mx.to_numpy()].copy()


def _cluster_labels(pos: np.ndarray, gap_mm: float) -> np.ndarray:
    """Spatial cluster label per strip, from sorted positions."""
    if len(pos) == 0:
        return np.zeros(0, dtype=int)
    return np.concatenate([[0], np.cumsum(np.diff(pos) > gap_mm)])


def _clean_cluster_labels(pos: np.ndarray, hot_mask: np.ndarray,
                          gap_mm: float, bridge: bool = HOT_BRIDGE) -> np.ndarray:
    """Cluster the CLEAN strips only, with the ordinary ``gap_mm``.

    Deleting the hot strips and re-applying the production gap threshold is
    the whole rule, and it is self-tuning in a way worth spelling out: a hot
    run NARROWER than ``gap_mm`` (~15 strips) leaves a sub-threshold hole, so
    a track crossing it stays one cluster for free; a run WIDER than that
    opens a real gap and the clean strips on either side become separate
    candidates. Which is the right split — a band that wide is the noise
    column itself, and clean strips beyond it are a different object, not the
    far half of one track. Both candidates are offered to the fit anyway
    (``n_candidates``), so nothing is lost if the far side was real.

    ``bridge=True`` instead subtracts the hot run's pitch-width from the gap,
    making every band free to cross. Measured on D/run_145 and NOT kept: see
    the module note in HANDOFF_HOT_WILDCARD_TUNING.md.

    Returns labels over ``pos[~hot_mask]``, which is sorted because ``pos`` is.
    """
    cp = pos[~hot_mask]
    if len(cp) == 0:
        return np.zeros(0, dtype=int)
    d = np.diff(cp)
    if bridge and len(d):
        hp = pos[hot_mask]
        if len(hp):
            n_between = (np.searchsorted(hp, cp[1:], 'left')
                         - np.searchsorted(hp, cp[:-1], 'right'))
            d = d - n_between * PITCH_MM
    return np.concatenate([[0], np.cumsum(d > gap_mm)])


def seed_candidates(pos: np.ndarray, channels: np.ndarray, amps: np.ndarray,
                    gap_mm: float = GAP_THRESHOLD_MM,
                    min_strips: int = MIN_STRIPS,
                    n_candidates: int = 1, hot=None) -> list:
    """Spatial clusters of one plane, ranked by CLEAN (non-``hot``) strip
    count.

    ``n_candidates > 1`` returns the runners-up as well, so the caller can let
    the waveform fit decide which cluster is the muon. That matters: "largest
    cluster wins" picks the wrong charge in ~5 % of events, and when it does,
    the true track sits a median of 37 mm outside the fit window (measured on
    det3 — see mx_june_wft/DET3_GATE_2026-07-29.md).

    ``hot`` (HANDOFF_D_NOISY_CHANNELS.md's wildcard spec, item 1: "never seed
    on a flagged channel") is a set/array of channel numbers that must not, by
    themselves, qualify a cluster as a seed. When it is given, **the clusters
    themselves are formed from the clean strips** (``_clean_cluster_labels``)
    rather than being formed from all strips and then judged on their clean
    count. A cluster must hold ``min_strips`` CLEAN strips to be admitted, and
    candidates are ranked by that count.

    Forming the clusters this way is the part that matters, and it is not the
    same as counting afterwards. On D, whose hot channels are whole connector
    runs at ~35x their neighbours' occupancy, 12 mm gap-clustering bridges
    *through* a hot band and welds a noise column onto a real track a
    centimetre away; the merged object is then judged as one cluster. Judged
    on clean count it is admitted (it has the real track's clean strips) and
    the fit gets a window spanning both — measured at chi2/dof 62 against 25
    for an uncontaminated one. Clustering on the clean strips separates them,
    while a band narrower than ``gap_mm`` still leaves a sub-threshold hole, so
    a track that genuinely crosses one stays intact (``_clean_cluster_labels``).

    Cluster MEMBERSHIP still includes the hot strips: ``Seed.channels`` spans
    the clean cluster's full positional extent, hot strips included, so a
    track is never split or holed for having crossed one, and
    ``wft.io.extract_window`` pads outward from those endpoints as usual. The
    hot strips arrive at the fit down-weighted (``wft.model.prep_plane`` reads
    the same bundle's ``hot`` list), which is where their influence is capped
    (item 3).

    With ``hot`` empty or None this is exactly the historical behaviour:
    clusters over all strips, ranked and admitted on raw count.

    **Read this before turning it on in production.** Measured on D/run_145
    over 3 600 triggers, this rule and the count-after-clustering rule it
    replaces come out the same on every fit metric. It is the better-defined
    of the two -- it cannot hand the fit a window that is mostly hot strips,
    and the tests pin that -- but it is not a measured improvement, and no
    seeding rule tried so far makes D's SURVIVING fits better: paired per
    event, when the mask changes which cluster is seeded the fit gets WORSE
    (chi2/dof 13.1 -> 20.6 on the clean stratum, p0 moving a median 4.8 mm).
    What actually improves D is using the same classification as a downstream
    cut on the frozen products, with no re-reconstruction at all --
    ``sept26_prelim_analysis/hot_seed_strata.py`` and ``k_robustness``'s
    ``no_hotstrip`` variant. Seeding on it is the October question.
    """
    good = np.isfinite(pos)
    pos, channels, amps = pos[good], channels[good], amps[good]
    if len(pos) < min_strips:
        return []
    o = np.argsort(pos)
    pos, channels, amps = pos[o], channels[o], amps[o]
    hot_mask = (np.isin(channels, list(hot)) if hot is not None and len(hot)
               else np.zeros(len(channels), bool))

    if hot_mask.any():
        lab = _clean_cluster_labels(pos, hot_mask, gap_mm)
        cpos = pos[~hot_mask]
        counts = np.bincount(lab, minlength=int(lab.max()) + 1 if len(lab) else 0)
        order = np.argsort(counts)[::-1][:max(1, n_candidates)]
        out = []
        for c in order:
            if counts[c] < min_strips:
                continue
            m = lab == c
            # the seed spans the clean cluster, hot strips inside included
            span = (pos >= cpos[m].min()) & (pos <= cpos[m].max())
            out.append(Seed(channels=channels[span].astype(np.int64),
                            n_strips=int(span.sum()),
                            n_dropped=int((~span).sum()),
                            amp_sum=float(amps[span].sum()),
                            n_raw=int(len(pos))))
        return out

    lab = _cluster_labels(pos, gap_mm)
    n_labels = int(lab.max()) + 1 if len(lab) else 0
    counts = np.bincount(lab, minlength=n_labels)
    order = np.argsort(counts)[::-1][:max(1, n_candidates)]
    out = []
    for c in order:
        if counts[c] < min_strips:
            continue
        m = lab == c
        out.append(Seed(channels=channels[m].astype(np.int64),
                        n_strips=int(m.sum()), n_dropped=int((~m).sum()),
                        amp_sum=float(amps[m].sum()), n_raw=int(len(pos))))
    return out


def seed_plane(pos: np.ndarray, channels: np.ndarray, amps: np.ndarray,
               gap_mm: float = GAP_THRESHOLD_MM,
               min_strips: int = MIN_STRIPS, hot=None) -> Optional[Seed]:
    """Largest (clean-strip-ranked) spatial cluster of one plane, from
    already-floored hits."""
    c = seed_candidates(pos, channels, amps, gap_mm, min_strips, 1, hot=hot)
    return c[0] if c else None


def seeds_from_hits(df_hits: pd.DataFrame, pos_maps: Dict[int, np.ndarray],
                    feu_x: int, feu_y: int, rel_floor: float = SIG_REL_FLOOR,
                    spark_veto: Optional[int] = SPARK_VETO_HITS,
                    n_candidates: int = N_CANDIDATES,
                    hot: Optional[Dict[str, object]] = None) -> Dict[int, dict]:
    """Build per-event seeds for both planes from a combined hits DataFrame.

    ``hot``: optional ``{'x': [...], 'y': [...]}`` channel numbers, passed to
    ``seed_candidates`` (see there — "never seed on a flagged channel").
    Typically ``cal.hot`` off the calibration bundle in use.

    Returns {eventId: {'x': [Seed], 'y': [Seed], 'n_hits': int,
                       'spark': bool}} — a list per plane, ranked by (clean)
    strip count, for the waveform fit to choose between (see
    wft.reco.fit_plane_candidates).
    """
    df = df_hits[df_hits['feu'].isin((feu_x, feu_y))]
    df = apply_significance_floor(df, rel_floor)
    out: Dict[int, dict] = {}
    if len(df) == 0:
        return out
    hot = hot or {}
    counts = df.groupby('eventId').size()
    for eid, g in df.groupby('eventId'):
        n_hits = int(counts.loc[eid])
        rec = {'x': [], 'y': [], 'n_hits': n_hits,
               'spark': bool(spark_veto is not None and n_hits > spark_veto)}
        if not rec['spark']:
            for plane, feu in (('x', feu_x), ('y', feu_y)):
                gp = g[g['feu'] == feu]
                if len(gp) == 0:
                    continue
                ch = gp['channel'].to_numpy().astype(int)
                rec[plane] = seed_candidates(pos_maps[feu][ch], ch,
                                             gp['amplitude'].to_numpy(),
                                             n_candidates=n_candidates,
                                             hot=hot.get(plane))
        out[int(eid)] = rec
    return out

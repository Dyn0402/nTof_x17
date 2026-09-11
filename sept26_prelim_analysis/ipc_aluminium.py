#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipc_aluminium.py -- the pair continuum the capsule makes, line by line.

    python -m sept26_prelim_analysis.ipc_aluminium
    python -m sept26_prelim_analysis.ipc_aluminium --write

WHY THIS IS A SEPARATE MODULE NOW.  The aluminium estimate in
:mod:`ipc_channels` was one line of arithmetic: *fraction of captures with a
hard primary* x *alpha_pair(E1, 7.73 MeV)* x *fraction beyond 109 deg*, with
the first factor carried as a factor-ten bracket because nobody had pulled the
branchings.  The branchings are now pulled (``data/nuclear/``), and having them
does not just tighten that number -- **it says the number was measuring the
wrong lines.**

THE THREE THINGS THE LINE LIST CHANGES.

1.  **The two hard primaries are M1, not E1.**  27Al(g.s.) is 5/2+ and s-wave
    capture makes the 7725 keV state 2+ or 3+.  The 7724.0 keV primary feeds
    the 3+ ground state of 28Al and the 7693.4 keV one feeds the 2+ level at
    30.6 keV -- both POSITIVE parity, so both transitions are M1 (or E2, which
    is within 4 % of M1 in everything this module computes).  M1 converts less
    (alpha_pair 2.2e-3 against E1's 3.1e-3 at 7.7 MeV) and converts more
    collimated.  Assuming E1 for them, as the old estimate did, overstates
    their wide-angle pair yield by 3.0x.

2.  **The wide-angle yield is dominated by the 2.3-4.3 MeV primaries, which
    ARE E1.**  Those feed the negative-parity levels -- 2960.1 keV to the 2-
    at 4765, 3033.9 to the 3- at 4691, 4133.4 to the 3- at 3591, 4259.5 to the
    4- at 3465, and so on -- and together they are ~40 % of all captures.
    Individually each converts less than a 7.7 MeV photon does; collectively
    they beat the hard lines by a factor of a few, because a 3 MeV pair is far
    less collimated (19 % beyond 109 deg, against 6 % for a 7.7 MeV M1).

3.  **So the Al background is a LOW-ENERGY background, and that is good news
    for anyone who can measure the pair energy and bad news for anyone who
    cannot.**  The wide-angle Al pairs carry 2-4 MeV between them, not 7.7 and
    certainly not 20.6.  A magnet, a calorimeter, or a range measurement
    separates them trivially.  Opening angle alone does not: see
    :func:`shape_comparison`.

WHAT THE INPUTS ARE.  ``data/nuclear/README.md`` has the provenance and the one
normalisation trap.  In brief: intensities per capture are IAEA PGAA partial
cross sections divided by sigma_0 = 0.231 b, and the level assignments (which
gamma leaves which level, and each level's J-pi) are EGAF's.  Two completeness
numbers, both computed in :func:`line_list` rather than asserted: the placed
prompt lines carry 86 % of sigma_0 x S_n in gamma energy, and the identified
primaries carry 81 % of the captures.

AND THE WALL IS NOT ONLY ALUMINIUM.  :func:`capsule_lines` runs the same
machinery over ``12C(n,g)13C``.  Carbon is 11 % of the wall's captures and
14 % of its wide-angle pairs, because both of its strong primaries -- 4945 keV
to the 1/2- ground state and 1262 keV to the 3/2- at 3685 -- are E1 and soft.
Everything downstream (:func:`shape_comparison`, :func:`rate_comparison`) works
on the capsule, both species together.

WHAT IS STILL MISSING is in ``IPC_MISSING.md`` and in :func:`missing`.  The
short version: neutron transport in the capsule wall (the rate table's own
numbers say a thermal neutron scatters about twice in the carbon fibre before
it captures, which no formula here models), the multipolarity of the secondary
cascade, and the hydrogen content of the carbon-fibre binder.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import ipc_born as IB  # noqa: E402

SCHEMA = 'sept26_prelim/ipc_aluminium/1'

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nuclear')

# --------------------------------------------------------------------------- #
# nuclear inputs -- every one with its source
# --------------------------------------------------------------------------- #
SN_AL28_KEV = 7725.10        # neutron separation energy of 28Al, EGAF
SIGMA0_AL_B = 0.231          # 27Al(n,g) thermal, 1981MuZQ via EGAF (0.231(6))
SIGMA0_AL_B_ERR = 0.006
SIGMA0_C_B = 0.00353         # 12C(n,g) thermal
SN_C13_KEV = 4946.31         # neutron separation energy of 13C

#: 27Al is 5/2+ and thermal capture is s-wave, so the 7725 keV capture state is
#: 2+ or 3+ -- positive parity either way, which is the only thing the
#: multipole assignment below actually uses.
CAPTURE_PARITY = +1

# --------------------------------------------------------------------------- #
# capsule geometry -- read off the header of results_3He, not re-derived
# --------------------------------------------------------------------------- #
#: Path-averaged areal densities over the 4 cm sphere, atoms per barn.  For the
#: 0.5 mm Al shell the header's "equivalent thickness" of 2.05 mm is exactly
#: the 4t a parallel beam sees crossing a thin spherical shell twice, so these
#: are already the right numbers to multiply a cross section by.
N_AL_ATB = 1.235e-2
N_CF_ATB = 9.660e-2          # carbon only; the binder's hydrogen is not given
N_HE3_ATB = 3.158e-2
SIGMA_NP_HE3_B = 5333.0      # 3He(n,p)3H at 25.3 meV
SIGMA_NG_HE3_UB = 55.0       # 3He(n,g)4He at 25.3 meV

#: What the December-2025 rate table reports for the thermal bin (0.01-0.1 eV),
#: per pulse.  Carried so this module's own numbers can be held against it.
TABLE_NEUTRONS = 3.11e6
TABLE_GC_CAPTURES = 5.38e4
TABLE_HE3_CAPTURES = 4.37
TABLE_GC_NEL = 5.78e6        # elastic scatters in the capsule, same bin

#: Elastic cross sections at thermal, for the scattering argument only.
SIGMA_EL_AL_B = 1.5
SIGMA_EL_C_B = 4.75

#: Radiation lengths, g/cm2, and the capsule's own thicknesses in g/cm2.
X0_AL_GCM2, X0_C_GCM2 = 24.01, 42.70
T_AL_GCM2 = 0.05 * 2.699     # 0.5 mm of aluminium
T_CF_GCM2 = 0.12 * 1.60      # 1.2 mm of carbon fibre, 1.6 g/cm3
T_WALL_GCM2 = T_AL_GCM2 + T_CF_GCM2
#: One effective radiation length for the two-layer wall, weighted by mass.
X0_WALL_GCM2 = T_WALL_GCM2 / (T_AL_GCM2 / X0_AL_GCM2 + T_CF_GCM2 / X0_C_GCM2)


# --------------------------------------------------------------------------- #
# the line list
# --------------------------------------------------------------------------- #
def _num(s):
    m = re.match(r'^\s*([0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?)', s)
    return float(m.group(1)) if m else np.nan


def read_egaf(nuclide: str = '28AL') -> tuple:
    """``(levels, transitions, primaries)`` from an EGAF ENSDF file.

    ``levels`` is one row per level with its energy and J-pi as written;
    ``transitions`` is one row per placed gamma with the level it leaves.  The
    file's LAST level record is the capture state, so the gammas under it are
    exactly the primaries -- that list, not an energy coincidence, is how a
    line gets called primary below.  Its intensities are relative and
    renormalised to 100 and are deliberately NOT returned: only the energies
    are, because the absolute intensities come from PGAA.
    """
    path = os.path.join(DATA, f'{nuclide}_EGAF.ens')
    levels, trans = [], []
    cur = None
    for raw in open(path, encoding='latin-1'):
        line = raw.rstrip('\r\n')
        if len(line) < 20 or line[0:5].strip() != nuclide:
            continue
        kind = line[5:8].strip()
        if kind == 'L':
            cur = dict(e_lev=_num(line[9:19]), jpi=line[21:39].strip())
            levels.append(cur)
        elif kind == 'G' and cur is not None:
            trans.append(dict(e_from=cur['e_lev'], e_gam=_num(line[9:19])))
    lv = pd.DataFrame(levels).dropna(subset=['e_lev']).reset_index(drop=True)
    tr = pd.DataFrame(trans).dropna(subset=['e_gam']).reset_index(drop=True)
    cap_e = lv.e_lev.iloc[-1]
    prim = tr[np.isclose(tr.e_from, cap_e)].e_gam.to_numpy()
    return lv, tr, prim


def read_pgaa(product: str = '28-Al') -> pd.DataFrame:
    """Prompt gamma lines with their **partial cross sections in barns**."""
    rows = []
    with open(os.path.join(DATA, 'pgaa_lines_subset.tsv'), encoding='latin-1') as f:
        for raw in f:
            f7 = raw.rstrip('\n').split('\t')
            if len(f7) < 8 or f7[1].strip() != product:
                continue
            rows.append(dict(e_gam=float(f7[4]), sigma_b=float(f7[6]),
                             prompt=f7[3].strip() == 'p'))
    return pd.DataFrame(rows)


def _parity(jpi: str) -> int:
    """+1, -1 or 0 (unknown) from an ENSDF J-pi string.

    ENSDF writes things like ``3+``, ``2-``, ``(2+,3)``, ``(1 TO 4+)``.  A
    parity is returned only when the string carries exactly one sign and no
    ambiguity that could flip it; everything else is 0 and is carried as
    'unassigned' all the way to the report rather than being guessed.
    """
    s = jpi.strip()
    if not s:
        return 0
    has_p, has_m = '+' in s, '-' in s
    if has_p == has_m:                    # neither, or both -> ambiguous
        return 0
    if '(' in s and (',' in s or 'TO' in s.upper()):
        # a multi-valued assignment: safe only if every listed value would
        # carry the same parity, which we do not attempt to prove.
        return 0
    return 1 if has_p else -1


def line_list(nuclide: str = '28AL', product: str = '28-Al',
              sn_kev: float = SN_AL28_KEV, sigma0_b: float = SIGMA0_AL_B,
              capture_parity: int = CAPTURE_PARITY) -> pd.DataFrame:
    """Every prompt gamma of one capture reaction, with a multipole assignment.

    One row per line: energy, intensity per capture, whether it is a primary
    (i.e. leaves the capture state), the final level's J-pi where known, and
    the resulting E1 / M1 assignment.  The assignment rule is parity alone:

        capture state is positive parity (27Al 5/2+ + s-wave neutron)
        final level negative  ->  E1
        final level positive  ->  M1 (E2 is within 4 % of M1 here)
        final level unknown   ->  'unassigned', and carried as a bracket

    Secondaries are left unassigned by construction: assigning them needs the
    parity of BOTH ends and a decision about mixing, and the bracket between
    all-E1 and all-M1 turns out to be narrower than the E0 uncertainty on the
    3He side, so buying it down is not where the effort belongs.
    """
    lv, tr, prim_e = read_egaf(nuclide)
    pg = read_pgaa(product)
    pg = pg[pg.prompt].copy()
    pg['intensity'] = pg.sigma_b / sigma0_b

    # a line is primary iff EGAF places it under the capture state.  1 keV is
    # loose against the two catalogues' energy precision (they agree to
    # ~0.1 keV on the strong lines) and tight against the 28Al level spacing.
    jpi, dlev, is_prim = [], [], []
    for eg in pg.e_gam:
        prim = bool(np.min(np.abs(prim_e - eg)) < 1.0)
        ef = sn_kev - eg
        j = (lv.e_lev - ef).abs().idxmin()
        d = abs(lv.e_lev[j] - ef)
        is_prim.append(prim)
        jpi.append(lv.jpi[j] if (prim and d < 3.0) else '')
        dlev.append(d)
    pg['is_primary'] = is_prim
    pg['final_jpi'] = jpi
    pg['level_match_keV'] = dlev
    pg['final_parity'] = [_parity(s) for s in pg.final_jpi]
    pg['multipole'] = np.where(
        ~pg.is_primary, 'unassigned',
        np.where(pg.final_parity == -capture_parity, 'E1',
                 np.where(pg.final_parity == capture_parity, 'M1',
                          'unassigned')))
    pg.attrs['sigma0_b'] = sigma0_b
    pg.attrs['sn_keV'] = sn_kev
    # completeness, computed rather than asserted
    pg.attrs['energy_completeness'] = float(
        (pg.sigma_b * pg.e_gam).sum() / (sigma0_b * sn_kev))
    pg.attrs['primary_completeness'] = float(
        pg.loc[pg.is_primary, 'intensity'].sum())
    return pg.sort_values('e_gam', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the pair continuum this line list makes
# --------------------------------------------------------------------------- #
#: Below this the pair conversion coefficient is negligible and the two-track
#: topology is not reconstructable anyway; kept as a constant so the cut is
#: visible rather than buried in a comparison.
E_MIN_MEV = 1.5


def pair_yield(lines: pd.DataFrame, assume: str = 'M1') -> pd.DataFrame:
    """Per-line pair yield and shape.  ``assume`` covers unassigned lines.

    Adds, for each line above the pair threshold, the conversion coefficient
    at that transition energy for that line's multipole, and the pairs per
    capture it contributes.  The returned frame keeps every line so the sums
    below can be broken down any way the report wants.
    """
    d = lines.copy()
    d['w_MeV'] = d.e_gam / 1000.0
    d['mult_used'] = np.where(d.multipole == 'unassigned', assume, d.multipole)
    keep = d.w_MeV > E_MIN_MEV
    ap = np.zeros(len(d))
    for i in np.flatnonzero(keep.to_numpy()):
        ap[i] = IB.alpha_pair(d.mult_used.iloc[i], float(d.w_MeV.iloc[i]))
    d['alpha_pair'] = ap
    d['pairs_per_capture'] = d.intensity * d.alpha_pair
    return d


def spectrum(lines: pd.DataFrame, assume: str = 'M1', bins=None) -> np.ndarray:
    """The capture reaction's whole pair spectrum, ``dN/dtheta``, normalised.

    The sum over lines of each line's Born curve at its own transition energy,
    weighted by that line's contribution to the pair yield.  This -- not a
    fraction beyond some angle -- is the thing to compare against a measured
    opening-angle distribution.
    """
    if bins is None:
        bins = IB.THETA_BINS
    d = pair_yield(lines, assume)
    d = d[d.pairs_per_capture > 0]
    tot = np.zeros(len(bins) - 1)
    for _, r in d.iterrows():
        tot += r.pairs_per_capture * IB.grid_spectrum(r.mult_used, r.w_MeV, bins)
    s = (tot * np.diff(bins)).sum()
    return tot / s if s > 0 else tot


def yield_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Pairs per capture, and where they come from, under both assumptions."""
    rows = []
    for assume in ('M1', 'E1'):
        d = pair_yield(lines, assume)
        y = spectrum(lines, assume)
        tot = d.pairs_per_capture.sum()
        hard = d.loc[d.w_MeV > 7.0, 'pairs_per_capture'].sum()
        mid = d.loc[(d.w_MeV > 2.0) & (d.w_MeV <= 5.0),
                    'pairs_per_capture'].sum()
        rows.append(dict(
            unassigned_taken_as=assume,
            pairs_per_capture=tot,
            frac_from_above_7MeV=hard / tot,
            frac_from_2_to_5MeV=mid / tot,
            median_deg=float(np.interp(
                0.5, IB.spectrum_table(y).frac_in_bin.cumsum(),
                IB.THETA_MID)),
            frac_gt109=IB.frac_above(y, 109.0),
            pairs_gt109_per_capture=tot * IB.frac_above(y, 109.0)))
    return pd.DataFrame(rows)


def top_lines(lines: pd.DataFrame, assume: str = 'M1',
              n: int = 14) -> pd.DataFrame:
    """The lines that actually make the wide-angle pairs, ranked."""
    d = pair_yield(lines, assume)
    d = d[d.pairs_per_capture > 0].copy()
    fr = []
    for _, r in d.iterrows():
        y = IB.grid_spectrum(r.mult_used, r.w_MeV)
        fr.append(IB.frac_above(y, 109.0))
    d['frac_gt109'] = fr
    d['pairs_gt109_per_capture'] = d.pairs_per_capture * d.frac_gt109
    d['share_of_gt109'] = (d.pairs_gt109_per_capture
                           / d.pairs_gt109_per_capture.sum())
    cols = ['e_gam', 'intensity', 'final_jpi', 'multipole', 'mult_used',
            'alpha_pair', 'frac_gt109', 'pairs_gt109_per_capture',
            'share_of_gt109']
    return d.sort_values('pairs_gt109_per_capture',
                         ascending=False).head(n)[cols].reset_index(drop=True)


def capsule_lines() -> dict:
    """Both capture reactions in the capsule wall, keyed by label.

    Carbon is not a rounding correction.  12C captures at 3.53 mb against
    aluminium's 231 mb, but the fibre carries 7.8x the areal density, so it is
    ~12 % of the wall's captures -- and per capture it is WORSE, because its
    two strong primaries (4945 keV to the 1/2- ground state, 1262 keV to the
    3/2- at 3685) are both E1 and both soft.  It contributes about a fifth of
    the capsule's wide-angle pairs.

    The 12C entrance channel is 0+ x 1/2+ = 1/2+, so the same positive
    capture parity drives the assignment.
    """
    return {
        '27Al': dict(lines=line_list('28AL', '28-Al', SN_AL28_KEV,
                                     SIGMA0_AL_B, +1),
                     sigma0_b=SIGMA0_AL_B, n_atb=N_AL_ATB),
        '12C': dict(lines=line_list('13C', '13-C', SN_C13_KEV,
                                    SIGMA0_C_B, +1),
                    sigma0_b=SIGMA0_C_B, n_atb=N_CF_ATB),
    }


def capsule_weights(caps: dict = None) -> dict:
    """Each species' share of the wall's captures, at any energy (all 1/v)."""
    caps = caps or capsule_lines()
    raw = {k: v['n_atb'] * v['sigma0_b'] for k, v in caps.items()}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


def capsule_summary(assume: str = 'M1') -> pd.DataFrame:
    """Per species and combined: captures, pairs, and where the pairs sit."""
    caps = capsule_lines()
    wt = capsule_weights(caps)
    rows = []
    for k, v in caps.items():
        d = pair_yield(v['lines'], assume)
        y = spectrum(v['lines'], assume)
        tot = d.pairs_per_capture.sum()
        rows.append(dict(
            species=k, share_of_captures=wt[k],
            pairs_per_capture=tot,
            median_deg=float(np.interp(
                0.5, np.cumsum(y * np.diff(IB.THETA_BINS)), IB.THETA_MID)),
            frac_gt109=IB.frac_above(y, 109.0),
            gt109_per_capture=tot * IB.frac_above(y, 109.0)))
    d = pd.DataFrame(rows)
    d['share_of_gt109'] = (d.share_of_captures * d.gt109_per_capture)
    d['share_of_gt109'] /= d.share_of_gt109.sum()
    return d


def capsule_spectrum(assume: str = 'M1', bins=None) -> np.ndarray:
    """The whole wall's pair spectrum, weighted by each species' pair yield."""
    if bins is None:
        bins = IB.THETA_BINS
    caps = capsule_lines()
    wt = capsule_weights(caps)
    acc = np.zeros(len(bins) - 1)
    for k, v in caps.items():
        y = spectrum(v['lines'], assume, bins)
        n = pair_yield(v['lines'], assume).pairs_per_capture.sum()
        acc += wt[k] * n * y
    return acc / (acc * np.diff(bins)).sum()


def capsule_gt109_per_capture(assume: str = 'M1') -> float:
    """Wide-angle pairs per capture ANYWHERE in the wall, Al and C together."""
    S = capsule_summary(assume)
    return float((S.share_of_captures * S.gt109_per_capture).sum())


# --------------------------------------------------------------------------- #
# multiple scattering in the wall the pair is born inside
# --------------------------------------------------------------------------- #
def msc_theta0_deg(e_mev, t_gcm2, x0_gcm2: float = None) -> np.ndarray:
    """Highland plane-projected RMS scattering angle, in degrees.

    Vectorised over both arguments.  Written out rather than imported because
    it is the whole argument of :func:`shape_comparison` and a reader should be
    able to check it in one line.  Beyond ~1 rad the Gaussian it parametrises
    stops being a description of anything, which is why :func:`smear_wall`
    reports how often that happens instead of silently clipping.
    """
    if x0_gcm2 is None:
        x0_gcm2 = X0_WALL_GCM2
    e_mev = np.asarray(e_mev, float)
    t_gcm2 = np.asarray(t_gcm2, float)
    p = np.sqrt(np.clip(e_mev ** 2 - IB.M_E ** 2, 1e-6, None))
    x = np.clip(t_gcm2 / x0_gcm2, 1e-9, None)
    return np.degrees(13.6 / p * np.sqrt(x) * (1 + 0.038 * np.log(x)))


def csda_range_gcm2(e_mev) -> np.ndarray:
    """Katz-Penfold practical range for electrons, g/cm2, low-Z absorber.

        R = 0.412 E^(1.265 - 0.0954 ln E)   below 2.5 MeV
        R = 0.530 E - 0.106                 above

    Two decimal places is all this needs to do: the question it answers is
    binary -- does the track get out of a 0.33 g/cm2 wall -- and the answer is
    only interesting within a factor of two of 0.8 MeV.
    """
    e = np.clip(np.asarray(e_mev, float), 1e-3, None)
    low = 0.412 * e ** (1.265 - 0.0954 * np.log(e))
    high = 0.530 * e - 0.106
    return np.where(e < 2.5, low, high)


def smear_wall(kind: str, w: float, n: int = 200_000, seed: int = 51,
               t_gcm2: float = None, x0_gcm2: float = None,
               born_inside: bool = True, bins=None) -> tuple:
    """One Born pair spectrum, as it leaves the capsule wall it was born in.

    Returns ``(density, escape_fraction, unreliable_fraction)``.  Three effects,
    in the order they matter:

    ``born_inside=False`` is the helium case: the pair is made in the gas and
    crosses the whole wall rather than a random part of it.

    **Escape.**  The wall is 0.33 g/cm2 of aluminium plus carbon fibre, which
    is the practical range of a 0.8 MeV electron.  A pair born in the middle of
    it with 2 MeV to share loses one track more often than not, and a lost
    track is a lost pair.  This is the only thing on this page that makes the
    aluminium background *smaller*, and it bites hardest on exactly the
    2-4 MeV lines that dominate the production.

    **Scattering.**  What survives is scattered by a 2-D Gaussian of Highland
    width about its own direction, with the birth depth uniform through the
    wall.  It broadens the Al continuum and moves its median up by ~7 deg; the
    20.6 MeV 3He pair, which only crosses the wall, hardly notices.

    **Where this stops being a calculation.**  ``unreliable_fraction`` is the
    weight for which the Highland angle came out above 1 rad, where a Gaussian
    is not a description of multiple scattering and the right answer needs
    transport.  It is reported rather than hidden; if it is large the curve is
    an indication, not a prediction.

    Deliberately only the CAPSULE.  The gas, the chamber windows and the drift
    volume are Geant's job downstream, and folding them twice would be worse
    than not folding them at all.
    """
    if bins is None:
        bins = IB.THETA_BINS
    if t_gcm2 is None:
        t_gcm2 = T_WALL_GCM2
    rng = np.random.default_rng(seed)
    d = IB.sample(kind, n, w, seed=seed)
    e1 = d.e_plus.to_numpy()
    e2 = d.e_minus.to_numpy()
    th = np.radians(d.theta_deg.to_numpy())
    wt = d.weight.to_numpy()
    # Wall still ahead of the pair.  An aluminium pair is BORN in the wall, so
    # that is uniform on [0, t]; a helium pair is born in the gas and crosses
    # all of it.  Getting this wrong understates the helium smearing by a
    # factor sqrt(2) in the Highland angle, which is small but free to fix.
    t_rem = (rng.uniform(0.0, 1.0, len(wt)) * t_gcm2 if born_inside
             else np.full(len(wt), t_gcm2))

    escapes = ((csda_range_gcm2(e1) > t_rem) & (csda_range_gcm2(e2) > t_rem))
    esc = float(wt[escapes].sum() / wt.sum())

    v1 = np.stack([np.zeros_like(th), np.zeros_like(th), np.ones_like(th)], 1)
    v2 = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    big = np.zeros(len(wt), bool)
    for v, e in ((v1, e1), (v2, e2)):
        t0 = msc_theta0_deg(e, t_rem, x0_gcm2)
        big |= t0 > 57.3
        t0 = np.radians(np.clip(t0, 0, 57.3))
        a = rng.normal(0, 1, len(e)) * t0
        b = rng.normal(0, 1, len(e)) * t0
        ref = np.tile([0.0, 1.0, 0.0], (len(e), 1))
        u1 = np.cross(v, ref)
        u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
        u2 = np.cross(v, u1)
        v += a[:, None] * u1 + b[:, None] * u2
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = np.degrees(np.arccos(np.clip((v1 * v2).sum(1), -1, 1)))
    m = escapes
    h, _ = np.histogram(out[m], bins=bins, weights=wt[m])
    tot = h.sum()
    y = h / (tot * np.diff(bins)) if tot > 0 else h
    unrel = float(wt[m & big].sum() / wt[m].sum()) if wt[m].sum() > 0 else 1.0
    return y, esc, unrel


def shape_comparison(assume: str = 'M1') -> pd.DataFrame:
    """The capsule against the gas, at birth and after the wall.

    Four columns, all normalised densities on the 1 deg axis, plus attributes
    carrying the escape fraction and the fraction of the smeared weight for
    which the Gaussian is not trustworthy.  The capsule column is aluminium and
    carbon fibre together, weighted by their captures.
    """
    from sept26_prelim_analysis import ipc_channels as IC
    bins = IB.THETA_BINS
    caps = capsule_lines()
    wt = capsule_weights(caps)
    out = {'theta_mid': IB.THETA_MID}
    out['capsule_birth'] = capsule_spectrum(assume, bins)
    t = IC.thermal_channels()
    f = dict(zip(t.channel, t.share_of_pairs))
    out['he3_birth'] = sum(f[k] * IB.grid_spectrum(k, IB.E_TRANSITION, bins)
                           for k in ('M1', 'E0'))

    # after the wall: the capsule pair is born in it, the 3He pair crosses it
    acc = np.zeros(len(bins) - 1)
    esc_w = unrel_w = norm = 0.0
    n_smeared = 0
    for sp, v in caps.items():
        d = pair_yield(v['lines'], assume)
        d = d[d.pairs_per_capture > 0].copy()
        # only the lines that carry the spectrum get the (sampled, slow)
        # treatment; the tail below 0.1 % of a species' yield cannot move a
        # 1 deg bin.
        d = d[d.pairs_per_capture > 1e-3 * d.pairs_per_capture.sum()]
        n_smeared += len(d)
        for _, r in d.iterrows():
            y, esc, unrel = smear_wall(r.mult_used, r.w_MeV, n=120_000,
                                       bins=bins)
            w = wt[sp] * r.pairs_per_capture
            acc += w * esc * y
            esc_w += w * esc
            unrel_w += w * esc * unrel
            norm += w
    out['capsule_after_wall'] = acc / (acc * np.diff(bins)).sum()

    he = np.zeros(len(bins) - 1)
    he_esc = 0.0
    for k in ('M1', 'E0'):
        y, esc, _ = smear_wall(k, IB.E_TRANSITION, n=400_000,
                               born_inside=False, bins=bins)
        he += f[k] * esc * y
        he_esc += f[k] * esc
    out['he3_after_wall'] = he / (he * np.diff(bins)).sum()

    r = pd.DataFrame(out)
    r.attrs['capsule_escape'] = esc_w / norm
    r.attrs['he3_escape'] = he_esc
    r.attrs['capsule_unreliable'] = unrel_w / esc_w if esc_w else 1.0
    r.attrs['lines_smeared'] = int(n_smeared)
    return r


# --------------------------------------------------------------------------- #
# how many of each there are
# --------------------------------------------------------------------------- #
def _sphere_absorption(n_bar_atb: float, sigma_b: float) -> float:
    """Absorption probability of a uniform beam on a uniform sphere.

    ``n_bar`` is the mass-over-area column, which for a sphere is exactly the
    beam-averaged one.  The chord at impact parameter ``x = r/R`` is
    ``1.5 n_bar sqrt(1 - x^2)``, and the answer is that, exponentiated and
    averaged over the disc.  For the 500 atm 3He cell at thermal the optical
    depth on axis is ~200, so this returns 1 to four decimals -- which is the
    entire point of computing it.
    """
    x = np.linspace(0, 1, 4001)
    tau = 1.5 * n_bar_atb * sigma_b * np.sqrt(np.clip(1 - x ** 2, 0, None))
    return float(np.trapezoid(2 * x * (1 - np.exp(-tau)), x))


def bookkeeping(en_ev: float = 0.0316) -> pd.DataFrame:
    """Captures per neutron entering the cell, computed rather than quoted.

    ``en_ev`` defaults to the log-centre of the rate table's thermal bin so the
    rows can be held against it.  Everything is 1/v-scaled from 25.3 meV, which
    is exact for all four cross sections involved anywhere below ~1 keV.
    """
    v = np.sqrt(0.0253 / en_ev)          # 1/v factor from the thermal point
    s_al, s_c = SIGMA0_AL_B * v, SIGMA0_C_B * v
    s_np, s_ng = SIGMA_NP_HE3_B * v, SIGMA_NG_HE3_UB * 1e-6 * v

    p_abs = _sphere_absorption(N_HE3_ATB, s_np)
    # with the gas black, only the entrance half of the wall sees the beam
    shadow = 0.5 + 0.5 * (1.0 - p_abs)
    rows = [
        dict(what='27Al(n,g) in the capsule wall',
             per_neutron_single_pass=N_AL_ATB * s_al,
             per_neutron=N_AL_ATB * s_al * shadow),
        dict(what='12C(n,g) in the carbon fibre',
             per_neutron_single_pass=N_CF_ATB * s_c,
             per_neutron=N_CF_ATB * s_c * shadow),
        dict(what='3He(n,p) -- absorbs the beam, makes no pairs',
             per_neutron_single_pass=min(N_HE3_ATB * s_np, np.inf),
             per_neutron=p_abs),
        dict(what='3He(n,g) -- the signal channel',
             per_neutron_single_pass=N_HE3_ATB * s_ng,
             per_neutron=p_abs * s_ng / s_np),
    ]
    d = pd.DataFrame(rows)
    d.attrs['p_abs_he3'] = p_abs
    d.attrs['shadow'] = shadow
    d.attrs['en_ev'] = en_ev
    return d


def rate_comparison(assume: str = 'M1',
                    en_ev: float = 0.0316) -> pd.DataFrame:
    """Wide-angle Al pairs per wide-angle 3He pair, three ways.

    The three differ only in how many 3He radiative captures there are, and
    that single number moves the answer by two orders of magnitude.  It is the
    dominant uncertainty on this page and it is not a nuclear one.
    """
    from sept26_prelim_analysis import ipc_channels as IC
    bk = bookkeeping(en_ev)
    al_per_cap_gt109 = capsule_gt109_per_capture(assume)

    t = IC.thermal_channels()
    he_pairs_per_radcap = float(t.sigma_pair_ub.sum()) / IC.SIGMA_NGAMMA_UB
    he_gt109_per_radcap = float((t.sigma_pair_ub * t.frac_gt109).sum()
                                / IC.SIGMA_NGAMMA_UB)

    al_cap = float(bk.loc[bk.what.str.startswith('27Al'), 'per_neutron'].iloc[0]
                   + bk.loc[bk.what.str.startswith('12C'), 'per_neutron'].iloc[0])
    al_cap_1pass = float(
        bk.loc[bk.what.str.startswith('27Al'), 'per_neutron_single_pass'].iloc[0]
        + bk.loc[bk.what.str.startswith('12C'), 'per_neutron_single_pass'].iloc[0])
    he_rad_shielded = float(
        bk.loc[bk.what.str.startswith('3He(n,g)'), 'per_neutron'].iloc[0])
    he_rad_thin = float(
        bk.loc[bk.what.str.startswith('3He(n,g)'), 'per_neutron_single_pass'].iloc[0])

    rows = [
        dict(variant='the rate table as published',
             al_captures=TABLE_GC_CAPTURES / TABLE_NEUTRONS,
             he3_radiative=TABLE_HE3_CAPTURES / TABLE_NEUTRONS,
             note='GC-captures and He3-captures straight out of results_3He'),
        dict(variant='the table\'s capsule, a self-shielded gas',
             al_captures=TABLE_GC_CAPTURES / TABLE_NEUTRONS,
             he3_radiative=he_rad_shielded,
             note='keeps the table\'s transport in the wall, fixes the gas'),
        dict(variant='analytic, single pass, self-shielded gas',
             al_captures=al_cap,
             he3_radiative=he_rad_shielded,
             note='no scattering in the wall at all -- a floor, not an estimate'),
    ]
    r = pd.DataFrame(rows)
    r['al_gt109_per_neutron'] = r.al_captures * al_per_cap_gt109
    r['he3_gt109_per_neutron'] = r.he3_radiative * he_gt109_per_radcap
    r['ratio'] = r.al_gt109_per_neutron / r.he3_gt109_per_neutron
    r.attrs['al_gt109_per_capture'] = al_per_cap_gt109
    r.attrs['he_gt109_per_radcap'] = he_gt109_per_radcap
    r.attrs['he_pairs_per_radcap'] = he_pairs_per_radcap
    r.attrs['al_cap_1pass'] = al_cap_1pass
    r.attrs['he_rad_thin'] = he_rad_thin
    return r


def scattering_check() -> pd.DataFrame:
    """Why the analytic capsule number is a floor, in the table's own numbers."""
    rows = [
        dict(quantity='elastic optical depth of the Al shell',
             value=N_AL_ATB * SIGMA_EL_AL_B),
        dict(quantity='elastic optical depth of the carbon fibre',
             value=N_CF_ATB * SIGMA_EL_C_B),
        dict(quantity='elastic scatters per neutron, the two together',
             value=N_AL_ATB * SIGMA_EL_AL_B + N_CF_ATB * SIGMA_EL_C_B),
        dict(quantity='the same, as the rate table reports it (GC-nel/neutrons)',
             value=TABLE_GC_NEL / TABLE_NEUTRONS),
        dict(quantity='capsule captures per neutron, analytic single pass',
             value=float(bookkeeping().per_neutron_single_pass.iloc[0]
                         + bookkeeping().per_neutron_single_pass.iloc[1])),
        dict(quantity='the same, as the rate table reports it',
             value=TABLE_GC_CAPTURES / TABLE_NEUTRONS),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def missing() -> pd.DataFrame:
    """What this module does not know, ranked by how much it could move.

    Kept as data, not prose, so the report renders it and it cannot quietly
    drift out of date relative to the numbers above it.
    """
    rows = [
        ('3He self-shielding in the rate table', 'x140',
         'results_3He appears to compute 3He radiative captures with a '
         'thin-target formula, but the 500 atm cell is optically thick to '
         'thermal neutrons (tau ~ 200). If so every expected IPC and X17 '
         'yield in that table is high by ~2 orders of magnitude, and the '
         'Al-to-gas ratio here is correspondingly worse. Needs confirming '
         'with whoever produced the table -- this module cannot tell whether '
         'the code applies it elsewhere.'),
        ('neutron transport in the capsule wall', 'x1-6',
         'the table reports ~1.9 elastic scatters per neutron in the capsule, '
         'so a thermal neutron random-walks in the carbon fibre before it '
         'captures. The analytic single-pass number here is a floor; the '
         'table\'s own GC-captures is 6x higher. Only a transport run '
         '(Geant4/MCNP, the real capsule geometry) settles it.'),
        ('hydrogen in the carbon-fibre binder', 'x1-3',
         'the carbon itself is now in the calculation -- 11 % of the wall\'s '
         'captures and 14 % of its wide-angle pairs -- but the geometry header '
         'gives a carbon areal density and nothing else. Epoxy is ~5 wt% H and '
         '1H captures at 0.333 b against carbon\'s 3.5 mb, so even a little '
         'binder can outweigh all the carbon. The 2223 keV line is soft but '
         'not below the pair threshold.'),
        ('multipolarity of the secondary cascade', 'x1.4',
         'primaries are assigned from the final level\'s parity; secondaries '
         'are not, and are carried as an all-M1 to all-E1 bracket. Closing it '
         'means parities at both ends plus mixing ratios -- EGAF has some of '
         'that and this module does not read it yet.'),
        ('Coulomb (Z = 13) corrections to the Born form', '~10 %',
         'alpha*Z = 0.095 for aluminium, against 0.015 for helium. The Born '
         'pair spectrum is still good but no longer exact, and the correction '
         'grows towards wide angles and towards asymmetric energy sharing. A '
         'Dirac-Coulomb IPC code (or the published Z-dependence tables) would '
         'settle it; the 3He numbers do not need it.'),
        ('acceptance for pairs born in the wall', 'unknown',
         'everything here is production. A pair born in the capsule wall is '
         '2 cm off the gas centre, so the pointing and vertex cuts treat it '
         'differently from a gas pair -- possibly much better. That is an '
         'acceptance question for the Geant chain, and it is the most likely '
         'place for this background to shrink.'),
        ('external conversion in the wall', 'unknown',
         'a 7.7 MeV capture photon converting IN the aluminium makes a real '
         'pair too, and there are ~500 times more photons than internal '
         'pairs. Those pairs are born collimated (theta ~ m/E) so they should '
         'not reach 109 deg, but they will dominate the small-angle region '
         'the IPC control sample lives in, and MSC in the wall moves some of '
         'them. Not modelled here at all.'),
        ('the 2 eV to 5.9 keV window', 'n/a below 2 eV',
         'the invariance argument uses 1/v for both 27Al and 3He, which holds '
         'below the first 27Al resonance at 5903 eV (34 us of flight). '
         'Anything reaching back to tens of microseconds -- it does not here, '
         'the flash veto is at 1 ms -- would need the resonance region.'),
    ]
    return pd.DataFrame(rows, columns=['what is missing', 'how much it moves',
                                       'why it matters'])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--assume', default='M1', choices=('M1', 'E1'))
    a = ap.parse_args()

    L = line_list()
    print(f'27Al(n,g) LINE LIST -- {len(L)} prompt lines, sigma_0 = '
          f'{SIGMA0_AL_B} b')
    print(f'  gamma energy carried by the placed lines : '
          f'{100 * L.attrs["energy_completeness"]:.0f} % of S_n')
    print(f'  captures carried by identified primaries : '
          f'{100 * L.attrs["primary_completeness"]:.0f} %')
    print(f'  primaries assigned E1 / M1 / unassigned  : '
          f'{(L.multipole == "E1").sum()} / {(L.multipole == "M1").sum()} / '
          f'{(L.multipole == "unassigned").sum()}\n')

    T = top_lines(L, a.assume)
    print('THE LINES THAT MAKE THE WIDE-ANGLE PAIRS')
    print(T.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    Y = yield_summary(L)
    print('\nPAIR YIELD PER 27Al CAPTURE')
    print(Y.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    B = bookkeeping()
    print(f'\nCAPTURE BOOKKEEPING at En = {B.attrs["en_ev"]:.4f} eV '
          f'(3He absorbs {100 * B.attrs["p_abs_he3"]:.2f} % of what enters)')
    print(B.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    S = scattering_check()
    print('\nWHY THE ANALYTIC CAPSULE NUMBER IS A FLOOR')
    print(S.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    CS = capsule_summary(a.assume)
    R = rate_comparison(a.assume)
    print('\nTHE WALL IS ALUMINIUM AND CARBON FIBRE, AND CARBON IS NOT SMALL')
    print(CS.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    print('\nWIDE-ANGLE CAPSULE PAIRS PER WIDE-ANGLE 3He PAIR')
    print(R.drop(columns=['note']).to_string(
        index=False, float_format=lambda x: f'{x:.4g}'))
    for _, r in R.iterrows():
        print(f'    {r.variant:<44s} {r.note}')

    from sept26_prelim_analysis import ipc_channels as IC
    E = IC.energy_invariance()
    print('\nDOES ANY OF THIS MOVE WITH TIME OF FLIGHT?  '
          '(the capsule-to-gas column too: both absorbers are 1/v)')
    print(E.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'  Born spectrum across the whole window: total variation '
          f'{E.attrs["spectrum_tv_across_window"]:.2e}')

    print('\nWHAT IS MISSING')
    for _, r in missing().iterrows():
        print(f'  [{r["how much it moves"]:>10s}]  {r["what is missing"]}')

    if a.write:
        from sept26_prelim_analysis import paths
        od = paths.out('ipc')
        L.to_csv(od / 'al_line_list.csv', index=False)
        T.to_csv(od / 'al_top_lines.csv', index=False)
        Y.to_csv(od / 'al_yield_summary.csv', index=False)
        B.to_csv(od / 'al_bookkeeping.csv', index=False)
        S.to_csv(od / 'al_scattering_check.csv', index=False)
        R.to_csv(od / 'al_rate_comparison.csv', index=False)
        E.to_csv(od / 'al_energy_invariance.csv', index=False)
        missing().to_csv(od / 'al_missing.csv', index=False)
        CS.to_csv(od / 'al_capsule_summary.csv', index=False)
        sc = shape_comparison(a.assume)
        sc.to_csv(od / 'al_shape_comparison.csv', index=False)
        print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

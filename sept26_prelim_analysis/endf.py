#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
endf.py -- just enough ENDF-6 to read a cross section off an evaluation.

    python -m sept26_prelim_analysis.endf            # what is staged, and a check

WHY WRITE THIS RATHER THAN INSTALL SOMETHING.  The GANIL study needs one thing
from nuclear data: sigma(E) for a named reaction on a named nuclide, between
1 and 40 MeV.  That is MF = 3, which is a TAB1 record, which is forty lines of
parsing.  A general ENDF library would be a dependency, a version to pin and a
second thing to explain in the provenance chain; forty lines that only claim to
read MF = 3 are auditable in one sitting.  Anything beyond MF = 3 raises rather
than guessing.

WHAT MF = 3 IS.  One section per reaction (MT), each of them:

    line 1   ZA, AWR, ...
    line 2   QM, QI, 0, LR, NR, NP        -- QI is minus the level energy for
                                             an inelastic level, which is how
                                             :func:`levels` recovers them
    then     NR interpolation ranges, then NP (E, sigma) pairs, three per line

Numbers are in the ENDF float format -- ``1.234567+5`` means 1.234567e5, with
the exponent's ``e`` left out to save a column.  That is the only real trap.

THE MT NUMBERS THIS PACKAGE USES:

    1    total                       102  (n,gamma)
    2    elastic                     103  (n,p)
    4    total inelastic             104  (n,d)
    16   (n,2n)                      107  (n,alpha)
    51-90  inelastic to discrete level 1-40  -- the pair background at GANIL
    91   inelastic to the continuum

Staged data is ``data/nuclear/*.mf3.endf``: the MF = 3 records of the
ENDF/B-VIII.0 neutron sublibrary, verbatim, with every other file section
stripped so the repository carries half a megabyte rather than seven.
"""
from __future__ import annotations

import functools
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCHEMA = 'sept26_prelim/endf/1'

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nuclear')

#: Staged evaluations, by the short name the rest of the package uses.
FILES = {
    'Al27': 'n_1325_13-Al-27.mf3.endf',
    'C12': 'n_0625_6-C-12.mf3.endf',
    'He3': 'n_0225_2-He-3.mf3.endf',
}

#: The reactions named rather than numbered, for readable call sites.
MT = {'total': 1, 'elastic': 2, 'inelastic': 4, 'n2n': 16,
      'capture': 102, 'np': 103, 'nd': 104, 'nalpha': 107,
      'continuum_inelastic': 91}

_FLOAT = re.compile(r'^\s*([+-]?\d*\.?\d*)([+-]\d+)?\s*$')


def _f(s: str) -> float:
    """One ENDF number.  ``1.234567+5`` is 1.234567e5; blank is 0."""
    s = s.strip()
    if not s:
        return 0.0
    m = _FLOAT.match(s)
    if m and m.group(2):
        return float(m.group(1) + 'e' + m.group(2))
    return float(s)


@functools.lru_cache(maxsize=8)
def _sections(name: str) -> dict:
    """``{MT: [line, ...]}`` for one staged evaluation.

    Cached: the aluminium evaluation is a quarter of a megabyte and the GANIL
    scan asks for a cross section a few thousand times, so re-reading it each
    time turned a two-second figure into a ten-minute one.
    """
    path = os.path.join(DATA, FILES[name])
    out: dict = {}
    for raw in open(path, encoding='latin-1'):
        line = raw.rstrip('\r\n')
        if len(line) < 75:
            continue
        mf = line[70:72].strip()
        mt = line[72:75].strip()
        if mf != '3' or not mt or mt == '0':
            continue
        out.setdefault(int(mt), []).append(line)
    if not out:
        raise ValueError(f'no MF=3 records in {path}')
    return out


@functools.lru_cache(maxsize=512)
def _xs_arrays(name: str, mt: int) -> tuple:
    """``(E, sigma, QM, QI)`` for one reaction, parsed once per process."""
    sec = _sections(name)
    if mt not in sec:
        raise KeyError(f'{name} has no MF=3 MT={mt}; has '
                       f'{sorted(sec)[:12]}...')
    lines = sec[mt]
    qm, qi = _f(lines[1][0:11]), _f(lines[1][11:22])
    # ENDF puts NR in columns 45-55 and NP in 56-66.  Getting them the wrong
    # way round is silent, so they are read once and never re-derived.
    nr, np_ = int(lines[1][44:55]), int(lines[1][55:66])
    n_int_lines = (nr * 2 + 5) // 6
    vals = []
    for line in lines[2 + n_int_lines:]:
        for c in range(0, 66, 11):
            t = line[c:c + 11]
            if t.strip():
                vals.append(_f(t))
    vals = vals[:2 * np_]
    return (np.array(vals[0::2], float), np.array(vals[1::2], float), qm, qi)


def xs(name: str, mt) -> pd.DataFrame:
    """``sigma(E)`` for one reaction: columns ``E_eV`` and ``sigma_b``.

    ``mt`` may be a number or one of the keys of :data:`MT`.  The frame carries
    ``QI`` in ``.attrs`` -- for an inelastic level that is minus the level
    energy in eV, which is where :func:`levels` gets them from.
    """
    if isinstance(mt, str):
        mt = MT[mt]
    e, sig, qm, qi = _xs_arrays(name, int(mt))
    d = pd.DataFrame(dict(E_eV=e, sigma_b=sig))
    d.attrs.update(nuclide=name, MT=mt, QM_eV=qm, QI_eV=qi)
    return d


def sigma_at(name: str, mt, e_ev) -> np.ndarray:
    """Interpolate a cross section onto ``e_ev`` (linear, 0 outside)."""
    if isinstance(mt, str):
        mt = MT[mt]
    e, sig, _, _ = _xs_arrays(name, int(mt))
    return np.interp(np.asarray(e_ev, float), e, sig, left=0.0, right=0.0)


@functools.lru_cache(maxsize=8)
def _levels_cached(name: str) -> pd.DataFrame:
    return _levels(name)


def levels(name: str) -> pd.DataFrame:
    """Cached wrapper -- see :func:`_levels`."""
    return _levels_cached(name)


def _levels(name: str) -> pd.DataFrame:
    """Every discrete inelastic level of one nuclide, with its energy.

    ``MT = 51 + i`` is the (i+1)-th excited state and its ``QI`` is minus that
    state's excitation energy, so the level scheme falls straight out of the
    cross-section file with no second source to reconcile.
    """
    sec = _sections(name)
    rows = []
    for mt in sorted(m for m in sec if 51 <= m <= 90):
        d = xs(name, mt)
        rows.append(dict(MT=mt, level=mt - 50,
                         e_level_MeV=-d.attrs['QI_eV'] * 1e-6,
                         e_max_MeV=d.E_eV.max() * 1e-6,
                         sigma_peak_b=float(d.sigma_b.max()),
                         e_at_peak_MeV=float(
                             d.E_eV[d.sigma_b.idxmax()] * 1e-6)))
    return pd.DataFrame(rows)


def available(name: str) -> list:
    return sorted(_sections(name))


def main() -> int:
    for name in FILES:
        mts = available(name)
        print(f'{name}: {len(mts)} MF=3 sections, MT = '
              f'{mts[:14]}{" ..." if len(mts) > 14 else ""}')
        for label in ('total', 'capture'):
            try:
                d = xs(name, label)
            except KeyError:
                continue
            e = np.array([1e6, 14e6])
            print(f'    {label:<8s} {d.E_eV.min():.2e}-{d.E_eV.max():.2e} eV, '
                  f'{len(d)} points; sigma(1 MeV) = '
                  f'{sigma_at(name, label, e)[0]:.4g} b, '
                  f'sigma(14 MeV) = {sigma_at(name, label, e)[1]:.4g} b')
    print('\nAl-27 discrete inelastic levels (the GANIL gamma source):')
    print(levels('Al27').head(12).to_string(
        index=False, float_format=lambda x: f'{x:.4g}'))
    print('\nC-12:')
    print(levels('C12').head(4).to_string(
        index=False, float_format=lambda x: f'{x:.4g}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

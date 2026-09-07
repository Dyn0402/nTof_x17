#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paths.py -- where this analysis reads and writes, on whichever machine.

No script in this package hard-codes ``/media/dylan/data/x17``.  Everything
resolves through here, so moving the tree (or running on the desktop, or on a
condor worker with the data staged somewhere else) is one environment variable
rather than a sweep of edits.

Resolution, for each root, in order:

  1. its own environment variable, if set -- explicit, wins everywhere
  2. the laptop default

**Nothing here is created and nothing is silently invented.**  ``root()`` and
``require()`` raise if the path is absent, with the variable name and the path
that was tried in the message.  A script that fails because the data is not
staged should say exactly that, at the top, not fail three functions deep on an
empty glob.  The one exception is :func:`out`, which *does* create -- output
directories are ours to make.

The CERN-side paths are here too, as strings rather than ``Path``.  They are
not mounted on this machine; they exist so that a script that builds an
``rsync``/``xrdcp`` command or a condor job spells them the same way as every
other script.

    python -m sept26_prelim_analysis.paths      # what resolves, and what exists
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Local roots
# --------------------------------------------------------------------------- #
_DEFAULTS = {
    'X17_ROOT': Path('/media/dylan/data/x17'),
}

#: Every local root: name -> (env var, default, one-line description).
#: Paths relative to X17_ROOT are resolved against whatever it turns out to be.
_ROOTS = {
    'x17':        ('X17_ROOT',        None,                'the data tree'),
    'beam_july':  ('X17_BEAM_JULY',   'beam_july',         'July/August n_TOF campaign'),
    'runs':       ('X17_RUNS',        'beam_july/runs',    'DREAM runs, per run/sub-run'),
    'analysis':   ('X17_ANALYSIS',    'beam_july/analysis', 'staged products pulled from CERN'),
    'out':        ('X17_SEPT26_OUT',  'sept26_prelim',     'THIS analysis writes here'),
}


def root(name: str = 'x17') -> Path:
    """Resolve one root, or raise saying which variable would fix it.

    >>> root('out')                    # doctest: +SKIP
    PosixPath('/media/dylan/data/x17/sept26_prelim')
    """
    try:
        env, rel, what = _ROOTS[name]
    except KeyError:
        raise KeyError(f'unknown root {name!r}; known: {", ".join(_ROOTS)}') from None

    override = os.environ.get(env)
    if override:
        p = Path(override).expanduser()
    elif rel is None:
        p = _DEFAULTS[env]
    else:
        p = root('x17') / rel

    if not p.is_dir():
        raise FileNotFoundError(
            f'{name} root ({what}) does not exist: {p}\n'
            f'  set ${env} to point at it, or stage the tree there.')
    return p


def require(path, what: str = '') -> Path:
    """Return ``path`` if it exists, else raise naming what was wanted.

    Use at the top of a script for every input it cannot run without, so a
    missing stage-1 product is one clear line and not a confusing empty result
    forty seconds in.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f'missing {what or "input"}: {p}\n'
            f'  nothing in this package creates it -- see STATUS.md "Resume here".')
    return p


def out(*parts: str) -> Path:
    """A directory under the analysis output root, created if needed.

    The one function here that creates.  Output directories are ours; input
    trees are not.
    """
    env = os.environ.get('X17_SEPT26_OUT')
    base = Path(env).expanduser() if env else _DEFAULTS['X17_ROOT'] / 'sept26_prelim'
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures(stage: str) -> Path:
    """``<out>/<stage>/figures`` -- where a stage's PNG+CSV pairs go."""
    return out(stage, 'figures')


# --------------------------------------------------------------------------- #
# CERN-side paths -- strings, not Paths: not mounted here
# --------------------------------------------------------------------------- #
#: DREAM runs, sub-runs, waveforms and the n_TOF slim products.
EOS_JULY = os.environ.get(
    'X17_EOS_JULY', '/eos/experiment/ntof/data/x17/july_beam')
#: n_TOF official processing output, one flat ``run<N>.root`` per run.
EOS_NTOF_DONE = os.environ.get(
    'X17_EOS_NTOF_DONE', '/eos/experiment/ntof/processing/official/done')
#: User space -- 2 TB, and where campaign products belong.  AFS work is nearly
#: full (~26 GB free of 100 GB), so nothing large goes there.  See STATUS.md C2.
EOS_USER = os.environ.get('X17_EOS_USER', '/eos/user/d/dneff')

#: ssh target.  GSSAPI, so it needs a live ticket (``kinit dneff@CERN.CH``).
LXPLUS = os.environ.get('X17_LXPLUS', 'lxplus')


def eos_slim(run: str, subrun: str) -> str:
    """The n_TOF slim directory for one sub-run, on EOS.

    ``run`` is the DREAM run (``run_145``), ``subrun`` the sub-run name
    (``stat090_0000``).  The ROOT file inside is
    ``ntof_hits_<run>_<subrun>_<ntof run>.root`` -- the n_TOF run number is not
    derivable from the DREAM one, so glob for it rather than building it.
    """
    return f'{EOS_JULY}/runs/{run}/{subrun}/ntof_hits'


def eos_subrun(run: str, subrun: str, kind: str = '') -> str:
    """One sub-run on EOS; ``kind`` is ``decoded_root``, ``combined_hits_root``,
    ``hits_root``, ``raw_daq_data`` or ``''`` for the sub-run directory."""
    base = f'{EOS_JULY}/runs/{run}/{subrun}'
    return f'{base}/{kind}' if kind else base


if __name__ == '__main__':
    print('local roots')
    for name, (env, rel, what) in _ROOTS.items():
        mark, detail = 'OK     ', ''
        try:
            p = root(name)
        except FileNotFoundError as exc:
            mark, p = 'MISSING', str(exc).splitlines()[0].split(': ', 1)[-1]
        if os.environ.get(env):
            detail = f'  (${env})'
        print(f'  {name:<10} {mark} {p}{detail}')
        print(f'  {"":<10}         {what}')
    print('\noutput root (created on use)')
    print(f'  out()      {out()}')
    print('\nCERN-side (not mounted here)')
    for k, v in (('EOS_JULY', EOS_JULY), ('EOS_NTOF_DONE', EOS_NTOF_DONE),
                 ('EOS_USER', EOS_USER), ('LXPLUS', LXPLUS)):
        print(f'  {k:<14} {v}')

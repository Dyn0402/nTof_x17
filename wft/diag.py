"""
wft.diag — the one way to run the reconstruction with anything other than its own algorithm.

There is one reconstruction (RERUN_PLAN B1, Dylan 2026-10-26: "a single algorithm for reconstruction, not a handful of
options cherry-picked in each dataset").  Its behaviour is fixed by module constants in ``wft.model``, ``wft.seed``,
``wft.reco`` and ``wft.io`` plus the calibration bundle; what differs between chambers and datasets is only what a
bundle (or the data's own record, e.g. zero suppression) says.  Nothing is read from the environment.

Studies still need to switch a piece off or move a constant (held-out arms, overlays, replays of old passes).  They do
it here, and only here: ``WFT_DIAG="model.SAT_SOLVE=0,reco.TWO_TRACK=0"`` (module.NAME=value, comma-separated), or
in-process ``wft.diag.override('reco', TWO_TRACK=False)`` before the workers start.  Every override is recorded in
:data:`ACTIVE`, copied into each table's meta (``diagnostic_overrides``), and a production stage refuses to run with
any (``rerun/stage_job.py``).

The retired ``WFT_*`` switches are still understood, as aliases of the same overrides, so the study scripts that set
them keep working and are visibly marked as diagnostics.  A legacy variable set to the algorithm's own value is a
no-op and is not recorded.
"""
import os
import warnings

#: legacy environment name -> (module, constant, parser)
_b = lambda s: str(s).strip().lower() in ('1', 'true', 'yes', 'on')
_drop_flat = lambda s: {'1': 'both', 'both': 'both', 'left': 'left'}.get(str(s).strip())
LEGACY = {
    # wft.model
    'WFT_MODEL_FRAC': ('model', 'MODEL_FRAC', float),
    'WFT_SAT_SOLVE': ('model', 'SAT_SOLVE', _b),
    'WFT_COL_GUARD': ('model', 'COL_GUARD', float),
    'WFT_RC_T0_INTERP': ('model', 'RC_T0_INTERP', _b),
    'WFT_ZS_CENSOR': ('model', 'ZS_CENSOR', _b),
    'WFT_ZS_BOUND': ('model', 'ZS_BOUND', float),
    'WFT_HOT_NOISE_INFLATION': ('model', 'HOT_NOISE_INFLATION', float),
    # wft.seed
    'WFT_SIG_FLOOR_LOCAL_MM': ('seed', 'SIG_FLOOR_LOCAL_MM', float),
    'WFT_SIG_FLOOR_LOCAL_MODE': ('seed', 'SIG_FLOOR_LOCAL_MODE', str),
    'WFT_HOT_BRIDGE': ('seed', 'HOT_BRIDGE', _b),
    'WFT_DEAD_SEED': ('seed', 'DEAD_SEED', _b),
    'WFT_HEADON_SIG': ('seed', 'HEADON_SIG', float),
    'WFT_SPLIT_GAP_MM': ('seed', 'SPLIT_GAP_MM', float),
    # wft.io
    'WFT_CM_MASK': ('io', 'CM_MASK', int),
    # wft.reco
    'WFT_TWO_SIDED': ('reco', 'TWO_SIDED', _b),
    'WFT_PIVOT_START': ('reco', 'PIVOT_START', _b),
    'WFT_TAN_MAX': ('reco', 'TAN_MAX', float),
    'WFT_TWO_TRACK_FIT': ('reco', 'TWO_TRACK', _b),
    'WFT_TWO_TRACK_F': ('reco', 'TWO_TRACK_F', float),
    'WFT_TWO_TRACK_F_CORROB': ('reco', 'TWO_TRACK_F_CORROB', float),
    'WFT_TWO_TRACK_WIDTH_MM': ('reco', 'TWO_TRACK_WIDTH_MARGIN', float),
    'WFT_TWO_TRACK_T0': ('reco', 'TWO_TRACK_T0', str),
    'WFT_TWO_TRACK_RESID_Z': ('reco', 'TWO_TRACK_RESID_Z', float),
    'WFT_TWO_TRACK_ALL_CANDIDATES': ('reco', 'TWO_TRACK_SELECTED_ONLY', lambda s: not _b(s)),
    'WFT_TWO_TRACK_SCALE': ('reco', 'TWO_TRACK_SCALE', str),
    'WFT_TWO_TRACK_SEARCH': ('reco', 'TWO_TRACK_SEARCH', str),
    'WFT_PAIR_SELECT': ('reco', 'PAIR_SELECT', _b),
    'WFT_HEADON_RESCUE': ('reco', 'HEADON_RESCUE', _b),
    'WFT_HEADON_RESCUE_Z': ('reco', 'HEADON_RESCUE_Z', float),
    'WFT_DROP_FLAT_HITS': ('reco', 'DROP_FLAT_HITS', _drop_flat),
}
#: variables that never change the reconstruction (paths, output format, replay ladders, the c2 < c1 gate's escape
#: for reports): not overrides, not recorded.
NEUTRAL = {'WFT_REPO', 'WFT_CODE', 'WFT_DATA', 'WFT_OUT', 'WFT_RND_DIR', 'WFT_BEAM_BASE', 'WFT_BEAM_ANALYSIS',
           'WFT_BENCH_ANALYSIS', 'WFT_DOC', 'WFT_DOC_BUNDLE', 'WFT_DOC_FIGDIR', 'WFT_FIGS', 'WFT_EMIT_CANDIDATES',
           'WFT_TWO_TRACK_F_LADDER', 'WFT_ALLOW_INVERTED_KERNEL', 'WFT_BEAM_MIN_STRIPS', 'WFT_DIAG',
           # read by nothing any more (wft_beam only echoes them): kept quiet so old wrappers do not warn
           'WFT_PRESCAN', 'WFT_CHI2DOF_BAD'}

#: {'module.NAME': value} of every override in effect in this process
ACTIVE: dict = {}
_MODULES: dict = {}


def _parse_value(s):
    s = str(s).strip()
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return None if s.lower() == 'none' else s


def _requested(module: str) -> dict:
    out = {}
    for k, (m, name, parse) in LEGACY.items():
        if m == module and k in os.environ:
            out[name] = parse(os.environ[k])
    for item in filter(None, (s.strip() for s in os.environ.get('WFT_DIAG', '').split(','))):
        key, _, val = item.partition('=')
        m, _, name = key.strip().partition('.')
        if m == module:
            out[name] = _parse_value(val)
    unknown = {k for k in os.environ if k.startswith('WFT_')} - set(LEGACY) - NEUTRAL
    if unknown and module == 'model':
        warnings.warn(f'wft.diag: unknown WFT_* variables ignored: {sorted(unknown)}')
    return out


def apply(module: str, g: dict) -> None:
    """Called once at the end of each wft module's import with its globals()."""
    _MODULES[module] = g
    for name, val in _requested(module).items():
        _set(module, g, name, val)


def _set(module, g, name, val):
    if name not in g:
        raise KeyError(f'wft.diag: wft.{module} has no constant {name!r}')
    old = g[name]
    if isinstance(old, bool) and not isinstance(val, bool):
        val = bool(val)
    elif isinstance(old, float) and isinstance(val, int) and not isinstance(val, bool):
        val = float(val)
    if val == old and f'{module}.{name}' not in ACTIVE:
        return
    g[name] = val
    ACTIVE[f'{module}.{name}'] = val


def override(module: str, **kw) -> None:
    """In-process override, e.g. ``override('reco', TWO_TRACK=False)``; the module must already be imported."""
    import importlib
    g = _MODULES.get(module) or vars(importlib.import_module(f'wft.{module}'))
    for name, val in kw.items():
        _set(module, g, name, val)


def require_production(what: str = 'this stage') -> None:
    """Raise if any override is in effect: production runs the algorithm, not a variant."""
    if ACTIVE:
        raise RuntimeError(f'{what} is a production stage but wft runs with diagnostic overrides {ACTIVE}; '
                           f'unset them (WFT_DIAG / retired WFT_* switches) or run it as a study')

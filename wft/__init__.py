"""
wft — waveform-first tracking for the MX17 micro-TPC chambers.

Reconstructs position, angle and drift depth by fitting the raw (strip x sample)
waveform picture with a forward model: the charge arriving in each 60 ns slice
of drift, folded through the measured per-plane impulse response and the
resistive charge-sharing kernel.

This exists because per-strip *hit times* cannot be used for geometry on
resistive strips — they are aggregates of shared charge and compress the drift
ladder by 20-30 % (~4 deg too steep), for every time estimator. See
``RECONSTRUCTION_BASIS.md`` at the repo root.

Hits enter this package in exactly one place — ``wft.seed``, which uses them to
decide *which strips and which events* to look at. Nothing downstream of the
seed sees a hit time.

Typical use::

    from wft import CalibrationBundle, reconstruct_run
    cal = CalibrationBundle.load(path)          # per detector AND run condition
    table = reconstruct_run(cfg, cal)           # -> parquet, one row per event

Origin: mx_june_cosmic_qa/waveform_first_threading/ (study + validation),
report WAVEFORM_FIRST_THREADING.md.
"""
from .calib import CalibrationBundle          # noqa: F401
from .reco import (                            # noqa: F401
    PlaneFit, fit_plane, fit_event, reconstruct_run, RECO_COLUMNS,
)

__all__ = ['CalibrationBundle', 'PlaneFit', 'fit_plane', 'fit_event',
           'reconstruct_run', 'RECO_COLUMNS']

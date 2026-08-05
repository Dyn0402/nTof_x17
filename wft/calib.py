"""
Calibration bundle: everything the forward model needs for one detector under
one set of run conditions.

A bundle is a directory:

    <name>/
        bundle.json     hypers, v, geometry, provenance
        arrays.npz      per-plane impulse templates, per-channel gain maps

**A bundle used outside the conditions it was fitted for is a silent error** —
the sharing kernel, the template and v are all detector- and condition-specific
(det4's kY is 2.36 against det3's 1.375; v moves from 12 to 39 um/ns across the
drift-HV scan). Hence `conditions` is a required, recorded field and
`check_conditions()` is called by the reconstruction driver.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import numpy as np

# hyper names in the order the calibration optimiser uses them
HYPER_NAMES = ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')


def _git_commit(path: str) -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=path, capture_output=True, text=True,
                              timeout=10).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


@dataclass
class CalibrationBundle:
    """Per-detector, per-condition calibration for the forward model."""

    # --- model calibration ---
    hyper: Dict[str, float]                 # c1, c2, kY, tau_s, sigma_s, sigma_p0, Dp
    v_drift: float                          # um/ns
    grid: np.ndarray                        # template time grid [ns]
    tmpl: Dict[str, np.ndarray]             # per-plane impulse response
    gain: Dict[str, np.ndarray]             # per-channel gain (512), 1.0 = unmeasured
    dt_xy: Dict[int, float] = field(default_factory=dict)   # t0x - t0y by ftst diff

    # --- geometry / DAQ ---
    pitch_mm: float = 0.78
    sample_ns: float = 60.0
    n_depth_bins: int = 18                  # K: 60 ns charge bins in the fit basis
    sat_adc: float = 3550.0

    # --- sharing-kernel form: 'delay' (legacy) | 'lp' (RC-dispersed copy,
    # the H4-beam-measured structure; tau_s becomes the RC constant) ---
    share_mode: str = 'delay'

    # --- identity and provenance (recorded, and checked at reco time) ---
    detector: str = ''                      # e.g. 'mx17_3'
    run_key: str = ''                       # qa_config key the fit was done on
    conditions: Dict[str, object] = field(default_factory=dict)  # amp/drift HV, gas...
    provenance: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def save(self, path: str, note: str = '') -> str:
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, 'arrays.npz'),
            grid=self.grid, tmpl_x=self.tmpl['x'], tmpl_y=self.tmpl['y'],
            gain_x=self.gain['x'], gain_y=self.gain['y'])
        prov = dict(self.provenance)
        prov.setdefault('code_commit', _git_commit(os.path.dirname(
            os.path.abspath(__file__))))
        if note:
            prov['note'] = note
        self.provenance = prov
        meta = dict(hyper={k: float(v) for k, v in self.hyper.items()},
                    v_drift=float(self.v_drift),
                    dt_xy={str(k): float(v) for k, v in self.dt_xy.items()},
                    pitch_mm=self.pitch_mm, sample_ns=self.sample_ns,
                    n_depth_bins=self.n_depth_bins, sat_adc=self.sat_adc,
                    share_mode=self.share_mode,
                    detector=self.detector, run_key=self.run_key,
                    conditions=self.conditions, provenance=prov)
        with open(os.path.join(path, 'bundle.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> 'CalibrationBundle':
        with open(os.path.join(path, 'bundle.json')) as f:
            m = json.load(f)
        z = np.load(os.path.join(path, 'arrays.npz'))
        return cls(hyper=m['hyper'], v_drift=m['v_drift'],
                   grid=z['grid'], tmpl={'x': z['tmpl_x'], 'y': z['tmpl_y']},
                   gain={'x': z['gain_x'], 'y': z['gain_y']},
                   dt_xy={int(k): v for k, v in m.get('dt_xy', {}).items()},
                   pitch_mm=m.get('pitch_mm', 0.78),
                   sample_ns=m.get('sample_ns', 60.0),
                   n_depth_bins=m.get('n_depth_bins', 18),
                   sat_adc=m.get('sat_adc', 3550.0),
                   share_mode=m.get('share_mode', 'delay'),
                   detector=m.get('detector', ''), run_key=m.get('run_key', ''),
                   conditions=m.get('conditions', {}),
                   provenance=m.get('provenance', {}))

    # ------------------------------------------------- legacy R&D artifacts
    @classmethod
    def from_legacy(cls, wf_dir: str, detector: str = '', run_key: str = '',
                    conditions: Optional[dict] = None,
                    hyper_file: str = 'hyper_v2.json') -> 'CalibrationBundle':
        """Build a bundle from the 2026-07-25/26 R&D products in a
        ``<Analysis>/.../waveform_first`` directory (templates_perplane.npz,
        gainmap.npz, dt_xy.json, hyper_v2.json). Numerics are taken verbatim —
        this is how det3/det2/det4 keep their validated calibration."""
        tz = np.load(os.path.join(wf_dir, 'templates_perplane.npz'))
        gain_path = os.path.join(wf_dir, 'gainmap.npz')
        if os.path.exists(gain_path):
            gz = np.load(gain_path)
            gain = {'x': gz['gain_x'], 'y': gz['gain_y']}
        else:
            # only det3 has a measured map; the ablation study found the 1.4 %
            # channel-gain spread changes per-event angles by less than the
            # statistical noise, so unit gains are the right default elsewhere
            gain = {'x': np.ones(512), 'y': np.ones(512)}
        with open(os.path.join(wf_dir, hyper_file)) as f:
            hj = json.load(f)
        dt = {}
        dt_path = os.path.join(wf_dir, 'dt_xy.json')
        if os.path.exists(dt_path):
            with open(dt_path) as f:
                dt = {int(k): float(v) for k, v in json.load(f).items()}
        hyper = {k: float(hj[k]) for k in HYPER_NAMES if k in hj}
        hyper.setdefault('kY', 1.0)
        return cls(hyper=hyper, v_drift=float(hj['v']),
                   grid=tz['grid'], tmpl={'x': tz['tmpl_x'], 'y': tz['tmpl_y']},
                   gain=gain, dt_xy=dt,
                   detector=detector, run_key=run_key,
                   conditions=conditions or {},
                   provenance=dict(source=wf_dir, hyper_file=hyper_file,
                                   n_train=hj.get('n_train'),
                                   chi2=hj.get('chi2'),
                                   imported='legacy waveform_first R&D'))

    # ------------------------------------------------------------- checks
    def check_conditions(self, conditions: dict, strict: bool = False) -> list:
        """Compare run conditions against the ones the bundle was fitted for.
        Returns a list of human-readable mismatches (empty = fine)."""
        bad = []
        for k, want in self.conditions.items():
            got = conditions.get(k)
            if got is None:
                continue
            if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                if abs(float(want) - float(got)) > 1e-6:
                    bad.append(f'{k}: bundle {want} vs run {got}')
            elif want != got:
                bad.append(f'{k}: bundle {want!r} vs run {got!r}')
        if bad and strict:
            raise ValueError('calibration bundle does not match run conditions: '
                             + '; '.join(bad))
        return bad

    def summary(self) -> str:
        h = self.hyper
        return (f"{self.detector or '?'} / {self.run_key or '?'} "
                f"[{self.share_mode}]: "
                f"v={self.v_drift:.2f} um/ns, c1={h['c1']:.3f}, c2={h['c2']:.3f}, "
                f"kY={h.get('kY', 1.0):.3f}, tau_s={h['tau_s']:.0f} ns, "
                f"sigma_s={h['sigma_s']:.0f} ns, sigma_p0={h['sigma_p0']:.3f} mm, "
                f"Dp={h['Dp']:.4f}  [{self.provenance.get('code_commit', '?')}]")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/gain_map.py

Thin loader over garfield_sim/results/hv_equivalence.json that converts a
Micromegas resist/mesh voltage in one Ar/isobutane mixture to the voltage of
another mixture giving the SAME simulated gas gain (Garfield++/Magboltz).

Used to put HV scans taken in different quencher fractions (e.g. run_44 in
95/5 vs run_38/41 in 90/10) on a common x-axis. The default reference is
95/5 (the JSON's reference_gas), so 90/10 voltages map onto the 95/5 scale.

Gain model per mixture: ln G = a + b V + c2 V^2 (logquad fit, R^2 >= 0.997).
Equivalence is done accurately by matching gain (invert the reference
logquad), not via the linearised analytic map.

  from gain_map import GainMap
  gm = GainMap(pressure='CERN_450m')
  v95 = gm.to_ref_voltage('Ar/Iso 90/10', 560.0)   # -> ~484.5
"""
import json
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = os.path.normpath(os.path.join(
    _HERE, '..', 'garfield_sim', 'results', 'hv_equivalence.json'))


def gas_to_key(gas):
    """'Ar/Iso 90/10' | 'Ar/iC4H10 95/5' -> 'Ar_iC4H10_90_10'."""
    m = re.search(r'(\d+)\s*/\s*(\d+)', gas or '')
    if not m:
        raise ValueError(f'cannot parse Ar/iso ratio from gas string {gas!r}')
    return f'Ar_iC4H10_{m.group(1)}_{m.group(2)}'


class GainMap:
    def __init__(self, pressure='CERN_450m', path=_JSON):
        with open(path) as f:
            d = json.load(f)
        if pressure not in d['pressures']:
            raise KeyError(f'{pressure!r} not in {list(d["pressures"])}')
        self.pressure = pressure
        self.ref_gas = d['reference_gas']                       # Ar_iC4H10_95_5
        self.ref_range = tuple(d['reference_voltage_range_V'])  # (400, 490)
        self._mix = d['pressures'][pressure]['mixtures']

    # -- gain model ---------------------------------------------------------
    def _fit(self, key):
        if key not in self._mix:
            raise KeyError(f'{key!r} not simulated; have {list(self._mix)}')
        return self._mix[key]['fit_logquad']       # {'a','b','c2'}

    def ln_gain(self, gas, V):
        f = self._fit(gas_to_key(gas))
        V = np.asarray(V, dtype=float)
        return f['a'] + f['b'] * V + f['c2'] * V * V

    def gain(self, gas, V):
        return np.exp(self.ln_gain(gas, V))

    # -- equivalence --------------------------------------------------------
    def to_ref_voltage(self, gas, V):
        """Reference-gas (95/5) voltage giving the same gain as `gas` at V.

        Accurate gain match: invert the reference logquad ln G = a+bV+c2 V^2.
        Returns float or np.ndarray matching the shape of V. NaN where the
        target gain lies outside the reference fit's reach (no real root).
        """
        key = gas_to_key(gas)
        if key == self.ref_gas:            # already the reference mixture
            return np.asarray(V, dtype=float) if np.ndim(V) else float(V)
        target_lnG = self.ln_gain(gas, V)
        rf = self._fit(self.ref_gas)
        a, b, c2 = rf['a'], rf['b'], rf['c2']
        # c2 V^2 + b V + (a - lnG) = 0 ; take the increasing (physical) root.
        disc = b * b - 4.0 * c2 * (a - target_lnG)
        disc = np.where(disc < 0, np.nan, disc)
        v = (-b + np.sqrt(disc)) / (2.0 * c2)
        return float(v) if np.ndim(V) == 0 else v

    def ref_range_extrapolated(self, v_ref):
        """Bool mask: reference voltage outside the simulated 95/5 span."""
        lo, hi = self.ref_range
        v = np.asarray(v_ref, dtype=float)
        return (v < lo) | (v > hi)


if __name__ == '__main__':
    # Sanity check vs published HV_EQUIVALENCE.md table (Saclay: 90/10 475<->95/5 400).
    for press, expect in (('Saclay_160m', {475: 400, 565: 490}),
                          ('CERN_450m', {473: 400, 562: 490})):
        gm = GainMap(pressure=press)
        print(f'{press}:')
        for v9010 in sorted(expect):
            v95 = gm.to_ref_voltage('Ar/Iso 90/10', float(v9010))
            print(f'  90/10 {v9010} V -> 95/5 {v95:6.1f} V '
                  f'(table ~{expect[v9010]})')

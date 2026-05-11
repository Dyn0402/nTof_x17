#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DreamConfig.py

Parser for DREAM DAQ configuration files (.cfg).

The format uses space-separated tokens; lines beginning with '#' are full-line
comments and the '#' character also introduces inline comments on data lines.
Wildcard 'Feu *' settings apply to all FEUs; FEU-specific entries
('Feu <N> ...') override them for that FEU.

Key parameters extracted
------------------------
n_samples    : int           Sys NbOfSamples
trig_type    : str           Sys DaqRun Trig  ('Ext', 'Slf', 'Cst', 'Exp')
daq_mode     : str           Sys DaqRun Mode  ('Raw', 'ZS', 'Ped', 'CMN')
rd_clk_div   : float         Feu * DrmClk RdClk_Div
wr_clk_div   : float         Feu * DrmClk WrClk_Div
ns_per_sample: float         derived: WrClk_Div × TRIG_CLOCK_NS

Sample period derivation
------------------------
The DREAM TrigClock runs at 100 MHz (10 ns period). The write clock that
controls the sampling rate is TrigClock / WrClk_Div, so:

    ns_per_sample = WrClk_Div × 10 ns

Examples: WrClk_Div=2 → 20 ns/sample ; WrClk_Div=6 → 60 ns/sample.
"""

import os
from pathlib import Path
from typing import Optional


TRIG_CLOCK_NS = 10.0   # TrigClock period in ns (100 MHz DREAM reference clock)


class DreamConfig:
    """Parse a DREAM .cfg file and expose key run parameters."""

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = str(cfg_path)

        self.n_samples:  Optional[int]   = None
        self.trig_type:  Optional[str]   = None
        self.daq_mode:   Optional[str]   = None
        self.rd_clk_div: Optional[float] = None
        self.wr_clk_div: Optional[float] = None

        self._parse()

    def _parse(self) -> None:
        with open(self.cfg_path) as f:
            for raw_line in f:
                # Strip inline comment, then leading/trailing whitespace
                line = raw_line.split('#')[0].strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 3:
                    continue

                match tokens[0]:
                    case 'Sys':
                        self._parse_sys(tokens)
                    case 'Feu':
                        self._parse_feu(tokens)

    def _parse_sys(self, tokens: list) -> None:
        if tokens[1] == 'NbOfSamples' and len(tokens) >= 3:
            self.n_samples = int(tokens[2])
        elif tokens[1:3] == ['DaqRun', 'Trig'] and len(tokens) >= 4:
            self.trig_type = tokens[3]
        elif tokens[1:3] == ['DaqRun', 'Mode'] and len(tokens) >= 4:
            self.daq_mode = tokens[3]

    def _parse_feu(self, tokens: list) -> None:
        # tokens[1] is FEU id or '*'; tokens[2] is subsystem
        if len(tokens) < 5 or tokens[2] != 'DrmClk':
            return
        if tokens[3] == 'RdClk_Div':
            self.rd_clk_div = float(tokens[4])
        elif tokens[3] == 'WrClk_Div':
            self.wr_clk_div = float(tokens[4])

    @property
    def ns_per_sample(self) -> Optional[float]:
        """Sample period in nanoseconds (WrClk_Div × 10 ns TrigClock period)."""
        return self.wr_clk_div * TRIG_CLOCK_NS if self.wr_clk_div is not None else None

    def __repr__(self) -> str:
        return (
            f'DreamConfig('
            f'n_samples={self.n_samples}, '
            f'trig={self.trig_type}, '
            f'mode={self.daq_mode}, '
            f'rd_clk_div={self.rd_clk_div}, '
            f'wr_clk_div={self.wr_clk_div}, '
            f'ns_per_sample={self.ns_per_sample}'
            f')'
        )


def find_dream_config(subrun_dir: str | Path) -> Optional[DreamConfig]:
    """
    Search <subrun_dir>/raw_daq_data/ for the first .cfg file and return a
    parsed DreamConfig, or None if no file is found.
    """
    raw_dir = Path(subrun_dir) / 'raw_daq_data'
    if not raw_dir.is_dir():
        return None
    cfg_files = sorted(f for f in raw_dir.iterdir() if f.suffix == '.cfg')
    if not cfg_files:
        return None
    return DreamConfig(cfg_files[0])

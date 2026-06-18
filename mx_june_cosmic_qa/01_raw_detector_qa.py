#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_raw_detector_qa.py

Step 1 of the June cosmic-bench QA: raw detector-side QA for the mx17 strips.
Reuses the plotting functions in cosmic_bench_analysis/detector_qa.py, just
reconfigured for this run via qa_config.

Products (written to output/<run>/raw_detector_qa/):
  - hits_vs_channel.png        strip occupancy per FEU (X=7, Y=8)
  - hits_vs_position.png       hits vs physical strip position [mm]
  - hits_vs_time.png           event rate vs time over the run
  - amplitude_vs_time.png      mean hit amplitude vs time (gain stability)
  - hit_position_scatter.png   earliest-arrival (x,y) per event
  - amplitude_map_earliest.png 2D mean amplitude, earliest pair
  - amplitude_map_time_paired.png  2D mean amplitude, time-paired hits

Run headless; everything is saved to disk (no interactive windows).
"""

import matplotlib
matplotlib.use('Agg')  # headless: save figures, do not block on plt.show()

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import os
import uproot
import detector_qa as dq
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig


def _load_hits_for_det():
    """Load combined hits for CFG.MX17_FEUS and map with CFG.DET_NAME."""
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    hit_files = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                       if f.endswith('.root') and '_datrun_' in f)
    sources = [f'{CFG.combined_hits_dir}{f}:hits' for f in hit_files]
    df = uproot.concatenate(sources, library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    df = cm._map_strip_positions(df, det)
    print(f'Loaded {len(df):,} hits over {df["eventId"].nunique():,} events '
          f'({CFG.DET_NAME}, FEUs {CFG.MX17_FEUS})')
    return df


def main():
    out_dir = CFG.out_dir('raw_detector_qa')

    # Point the reused module at this run (its plot titles read these globals).
    dq.BASE_PATH = CFG.BASE_PATH
    dq.RUN = CFG.RUN
    dq.SUB_RUN = CFG.SUB_RUN
    dq.MX17_FEUS = CFG.MX17_FEUS
    dq.RUN_CONFIG_PATH = CFG.run_config_path
    dq.MAP_CSV_PATH = CFG.MAP_CSV_PATH
    dq.OUT_DIR = out_dir

    # NOTE: dq.load_hits() hardcodes detector 'mx17_1'; load here with the
    # configured detector so multi-detector runs (mx17_2 on FEU 7/8) map correctly.
    df = _load_hits_for_det()

    # --- detector-side raw QA ---
    dq.plot_hits_vs_channel(df, out_dir=out_dir)
    dq.plot_hits_vs_position(df, out_dir=out_dir)
    dq.plot_hits_vs_time(df, out_dir=out_dir)
    dq.plot_amplitude_vs_time(df, out_dir=out_dir)

    # --- reconstructed hit positions (earliest-arrival pair per event) ---
    pos_df = dq.get_hit_positions(df)
    print(f'Reconstructed {len(pos_df):,} events with both X and Y hits')
    if pos_df.empty:
        print('  (no events with BOTH X and Y hits — skipping position/amplitude maps)')
    else:
        dq.plot_hit_position_scatter(pos_df, out_dir=out_dir)
        dq.plot_amplitude_map_earliest(pos_df, out_dir=out_dir)

        # --- time-paired amplitude map (samples full track length) ---
        print('Building time-paired hit positions ...')
        paired_df = dq.get_hit_positions_time_paired(df)
        n_ev = paired_df['event_id'].nunique() if not paired_df.empty else 0
        print(f'  {len(paired_df):,} time-paired (x,y) points from {n_ev:,} events')
        if not paired_df.empty:
            dq.plot_amplitude_map_time_paired(paired_df, out_dir=out_dir)

    print(f'\nRaw detector QA written to: {out_dir}')


if __name__ == '__main__':
    main()

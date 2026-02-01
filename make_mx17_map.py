#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 01 2:45 PM 2026
Created in PyCharm
Created as nTof_x17_DAQ/make_mx17_map.py

@author: Dylan Neff, dylan
"""

import numpy as np
import pandas as pd


def main():
    make_m4_mapping()
    print('donzo')


def make_m4_mapping():
    """
    Make mapping csv for the M4 connectors on the MX17 40x40 detectors.
    Returns:

    """
    out_csv_name = 'mx17_m4_map.csv'
    channels_per_connector = 64
    x0 = 0  # mm
    y0 = 0  # mm
    x_pitch_mm = 0.780
    y_pitch_mm = 0.780
    x_connectors = 8
    y_connectors = 8

    # Make csv file with mapping. Connector number, channel number, x pos mm, y pos mm
    mapping_rows = []
    for conn_x in range(x_connectors):
        for chan_x in range(channels_per_connector):
            channel_num = conn_x * channels_per_connector + chan_x
            x_pos = round(x0 + (conn_x * channels_per_connector + chan_x) * x_pitch_mm, 4)
            mapping_rows.append({
                'connector': conn_x + 1,
                'channel': chan_x,
                'axis': 'x',
                'channel_num': channel_num,
                'x_position_mm': x_pos,
                'y_position_mm': y0
            })
    for conn_y in range(y_connectors):
        for chan_y in range(channels_per_connector):
            channel_num = conn_y * channels_per_connector + chan_y
            y_pos = round(y0 + (conn_y * channels_per_connector + chan_y) * y_pitch_mm, 4)
            mapping_rows.append({
                'connector': conn_y + 1,
                'channel': chan_y,
                'axis': 'y',
                'channel_num': channel_num,
                'x_position_mm': x0,
                'y_position_mm': y_pos
            })
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(out_csv_name, index=False)


if __name__ == '__main__':
    main()

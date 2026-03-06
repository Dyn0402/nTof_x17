#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 23 9:04 AM 2026
Created in PyCharm
Created as nTof_x17/make_run_table.py

@author: Dylan Neff, dylan
"""

import os
import json
import pandas as pd
from plot_beam_hits import get_run_time

import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials


def main():
    # run_dir = '/media/dylan/data/x17/feb_beam/runs/'
    run_dir = '/eos/experiment/ntof/data/x17/feb_beam/runs/'
    run_cfg_name = 'run_config.json'
    csv_out_path = f'{run_dir}run_table.csv'
    cred_file = '/afs/cern.ch/user/d/dneff/creds/ntof-x17-776cc528cb62.json'
    sheet_id = "1YGOsxWJ4xaqU8YSr-17SluhPxHsLE2jhn6QnOk59WCE"
    tab_name = "Json_Run_Summary"

    creds = Credentials.from_service_account_file(cred_file, scopes=[
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ])

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    print(sh.title)
    sh = sh.worksheet(tab_name)

    df = []
    for run in os.listdir(run_dir):
        if not os.path.isdir(os.path.join(run_dir, run)):
            continue
        print(f'\nProcessing run {run}')
        run_config_path = os.path.join(run_dir, run, run_cfg_name)
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        if run_config['run_name'] != run:
            print(f'Run name in {run_cfg_name} does not match directory name: {run_config["run_name"]} != {run}')

        trig_type = os.path.basename(run_config['dream_daq_info']['daq_config_template_path'])

        trig_type_map = {
            'Tcm_Mx17_Feb_test.cfg': 'PS',
            'Self_Tcm_MM_Mx17_Feb_test.cfg': 'Self',
            'Tcm_Mx17_SiPM_trig.cfg': 'Scintillator',
        }

        trig_type = trig_type_map.get(trig_type, 'Other')

        det = run_config['detectors'][0]
        drift_gap = det.get('drift_gap', 30)  # mm
        if ' mm' in drift_gap:
            drift_gap = int(drift_gap.split(' ')[0])
        frame_type = det.get('frame_type', 'aluminum')
        distance_from_target = det.get('distance_from_target', 20)

        sub_run_dirs = [x for x in os.listdir(os.path.join(run_dir, run))]

        sub_runs = run_config['sub_runs']
        n_runs = len(sub_runs)
        total_run_time, resist_hvs, drift_hvs = 0, [], []
        resist_hv_channel, drift_hv_channel = ['2', '0'], ['5', '0']
        for sub_run in sub_runs:
            sub_run_name = sub_run['sub_run_name']
            if sub_run_name not in sub_run_dirs:
                continue
            try:
                sub_run_time = get_run_time(run_dir, run, sub_run_name)
            except Exception:
                continue
            total_run_time += sub_run_time
            try:
                resist_hv = sub_run['hvs'][resist_hv_channel[0]][resist_hv_channel[1]]
                resist_hvs.append(resist_hv)
            except KeyError:
                print(f'No resist HV found for run {run} subrun {sub_run}')
            try:
                drift_hv = -sub_run['hvs'][drift_hv_channel[0]][drift_hv_channel[1]]
                drift_hvs.append(drift_hv)
            except KeyError:
                pass

        average_subrun_time = total_run_time / n_runs
        if len(resist_hvs) == 0:
            resist_hv_range = [None, None]
        elif max(resist_hvs) == 0:
            resist_hv_range = [0, 0]
        else:
            resist_hvs = [hv for hv in resist_hvs if hv > 0]  # Remove all 0s
            resist_hv_range = [min(resist_hvs), max(resist_hvs)]

        # Get unique drift hvs and sort from lowest to highest
        drift_hvs = sorted(drift_hvs, key=lambda x: abs(x))
        drift_hvs = list(set(drift_hvs))

        run_row = {
            'run': int(run_config['run_name'].split('_')[-1]),
            'start_time': run_config['start_time'],
            'gas': run_config['gas'],
            'beam_type': run_config['beam_type'],
            'target_type': run_config['target_type'],
            'trigger_type': trig_type,
            'drift_gap (mm)': drift_gap,
            'frame_type': frame_type,
            'distance_from_target (cm)': distance_from_target,
            'average_subrun_time (min)': average_subrun_time / 60,
            'total_run_time (h)': total_run_time / 3600,
            'resist_hv_range (V)': resist_hv_range,
            'drift_hv_range (V)': drift_hvs,
        }

        df.append(run_row)
    df = pd.DataFrame(df)
    # Sort by run number
    df = df.sort_values(by='run')
    print(df)

    df.to_csv(csv_out_path, index=False)
    set_with_dataframe(sh, df)  # df is your pandas DataFrame

    print('donzo')



if __name__ == '__main__':
    main()

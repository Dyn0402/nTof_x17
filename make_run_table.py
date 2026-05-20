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
    run_dir = '/eos/experiment/ntof/data/x17/may_beam/runs/'
    run_cfg_name = 'run_config.json'
    csv_out_path = f'{run_dir}run_table.csv'
    cred_file = '/afs/cern.ch/user/d/dneff/creds/ntof-x17-776cc528cb62.json'
    # sheet_id = "10wyBo0X1NHgaT1eFw8WhN5VEAtIhZlUT-AbBVSVW6Uk"  # Feb
    sheet_id = "15wrV7EEeFTRH8oFbr-_WXRgdZNJfOeitJR4K97UgVaA"  # May
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
        if not run.startswith('run_'):
            continue
        print(f'\nProcessing run {run}')
        run_config_path = os.path.join(run_dir, run, run_cfg_name)
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        if run_config['run_name'] != run:
            print(f'Run name in {run_cfg_name} does not match directory name: {run_config["run_name"]} != {run}')

        trig_type = os.path.basename(run_config['dream_daq_info']['daq_config_template_path'])

        trig_type_map = {
            'Tcm_Mx17_May.cfg': 'PS',
            'Cosmics_Mx17_May.cfg': 'Cosmics',
            'Self_Trig_det3_QA.cfg': 'Self',
            'Self_Trig_QA.cfg': 'Self',
            'Tcm_Mx17_May_Coinc.cfg': 'Scintillator',
        }

        trig_type = trig_type_map.get(trig_type, 'Other')

        # Build per-detector info from config, reading HV channels dynamically
        detectors = {}
        for det_cfg in run_config.get('detectors', []):
            if det_cfg.get('det_type') != 'mx17':
                continue
            det_name = det_cfg['name']
            drift_gap = det_cfg.get('drift_gap', 30)
            if isinstance(drift_gap, str) and ' mm' in drift_gap:
                drift_gap = float(drift_gap.split(' ')[0])
            hv_channels = det_cfg.get('hv_channels', {})
            detectors[det_name] = {
                'drift_gap': drift_gap,
                'frame_type': det_cfg.get('frame_type', 'aluminum'),
                'distance_from_target': det_cfg.get('distance_from_target', 20),
                'resist_ch': hv_channels.get('resist'),  # e.g. [card, channel]
                'drift_ch': hv_channels.get('drift'),
                'resist_hvs': [],
                'drift_hvs': [],
                'resist_imons': [],   # imon (A) from hv_monitor.csv across all subruns
                'resist_vmons': [],   # stable vmon (V) for HV fallback
            }

        sub_run_dirs = [x for x in os.listdir(os.path.join(run_dir, run))]

        sub_runs = run_config['sub_runs']
        n_subruns = len(sub_runs)
        total_run_time = 0
        for sub_run in sub_runs:
            sub_run_name = sub_run['sub_run_name']
            if sub_run_name not in sub_run_dirs:
                continue
            try:
                sub_run_time = get_run_time(run_dir, run, sub_run_name)
            except Exception:
                continue
            total_run_time += sub_run_time
            hvs = sub_run.get('hvs', {})
            for det_name, det_info in detectors.items():
                if det_info['resist_ch']:
                    card, ch = det_info['resist_ch']
                    hv = hvs.get(str(card), {}).get(str(ch))
                    if hv is not None:
                        det_info['resist_hvs'].append(hv)
                    else:
                        print(f'No resist HV found for {det_name} in run {run} subrun {sub_run_name}')
                if det_info['drift_ch']:
                    card, ch = det_info['drift_ch']
                    hv = hvs.get(str(card), {}).get(str(ch))
                    if hv is not None:
                        det_info['drift_hvs'].append(-hv)

        # Second pass: read hv_monitor.csv for current stats and HV fallback
        for sub_run in sub_runs:
            sub_run_name = sub_run['sub_run_name']
            hv_file = os.path.join(run_dir, run, sub_run_name, 'hv_monitor.csv')
            if not os.path.exists(hv_file):
                continue
            try:
                hv_df = pd.read_csv(hv_file)
            except Exception:
                continue
            for det_name, det_info in detectors.items():
                if not det_info['resist_ch']:
                    continue
                card, ch = det_info['resist_ch']
                col_imon = f'{card}:{ch} imon'
                col_vmon = f'{card}:{ch} vmon'
                if col_imon in hv_df.columns:
                    det_info['resist_imons'].extend(hv_df[col_imon].dropna().tolist())
                if col_vmon in hv_df.columns:
                    stable_mask = hv_df[col_vmon].rolling(window=5, center=True).std() < 0.5
                    det_info['resist_vmons'].extend(hv_df.loc[stable_mask, col_vmon].dropna().tolist())

        run_row = {
            'run': int(run_config['run_name'].split('_')[-1]),
            'start_time': run_config['start_time'],
            'gas': run_config['gas'],
            'beam_type': run_config['beam_type'],
            'target_type': run_config['target_type'],
            'trigger_type': trig_type,
            'number of subruns': n_subruns,
            'average_subrun_time (min)': total_run_time / n_subruns / 60,
            'total_run_time (h)': total_run_time / 3600,
        }

        for det_name, det_info in detectors.items():
            resist_hvs = det_info['resist_hvs']
            drift_hvs = det_info['drift_hvs']

            if len(resist_hvs) == 0:
                # Fallback: infer HV range from stable monitoring data
                stable_hvs = [v for v in det_info['resist_vmons'] if v > 0]
                if stable_hvs:
                    resist_hv_range = [min(stable_hvs), max(stable_hvs)]
                else:
                    resist_hv_range = [None, None]
            elif max(resist_hvs) == 0:
                resist_hv_range = [0, 0]
            else:
                resist_hvs = [hv for hv in resist_hvs if hv > 0]
                resist_hv_range = [min(resist_hvs), max(resist_hvs)]

            drift_hvs = sorted(set(drift_hvs), key=lambda x: abs(x))

            imons_ua = [x * 1e6 for x in det_info['resist_imons']]
            current_min = round(min(imons_ua), 3) if imons_ua else None
            current_avg = round(sum(imons_ua) / len(imons_ua), 3) if imons_ua else None
            current_max = round(max(imons_ua), 3) if imons_ua else None

            run_row[f'{det_name} drift_gap (mm)'] = det_info['drift_gap']
            run_row[f'{det_name} frame_type'] = det_info['frame_type']
            run_row[f'{det_name} distance_from_target (cm)'] = det_info['distance_from_target']
            run_row[f'{det_name} resist_hv_range (V)'] = resist_hv_range
            run_row[f'{det_name} drift_hv_range (V)'] = drift_hvs
            run_row[f'{det_name} resist_current_min (uA)'] = current_min
            run_row[f'{det_name} resist_current_avg (uA)'] = current_avg
            run_row[f'{det_name} resist_current_max (uA)'] = current_max

        df.append(run_row)
    df = pd.DataFrame(df)
    # Sort by run number
    df = df.sort_values(by='run').reset_index(drop=True)
    print(df)

    df.to_csv(csv_out_path, index=False)
    set_with_dataframe(sh, df)  # df is your pandas DataFrame

    print('donzo')



if __name__ == '__main__':
    main()

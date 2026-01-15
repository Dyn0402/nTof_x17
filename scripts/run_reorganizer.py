#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 15 2:16 PM 2026
Created in PyCharm
Created as nTof_x17/run_reorganizer.py

@author: Dylan Neff, dylan
"""

import os

def main():
    runs_path = '/media/dylan/data/x17/dream_run/'

    raw_dream_dir_name = 'raw_dream'
    raw_dream_file_types = ['.fdf', '.cfg_cpy', '.cfg', '.log']

    for dir_name in os.listdir(runs_path):
        print(f'\nChecking {dir_name}...')
        if not dir_name.startswith('run_'):
            print(f'Skipping {dir_name}, not a run directory.')
            continue
        run_dir_path = os.path.join(runs_path, dir_name)
        # Check if there's a raw_dream
        raw_dream_path = os.path.join(run_dir_path, raw_dream_dir_name)
        if not os.path.isdir(raw_dream_path):
            os.mkdir(raw_dream_path)
        for file_name in os.listdir(run_dir_path):
            if any(file_name.endswith(ext) for ext in raw_dream_file_types):
                src_path = os.path.join(run_dir_path, file_name)
                dst_path = os.path.join(raw_dream_path, file_name)
                os.rename(src_path, dst_path)
                print(f'Moved {file_name} to {raw_dream_dir_name} in {dir_name}')

    print('donzo')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 27 14:11 2025
Created in PyCharm
Created as nTof_x17/decode_runs

@author: Dylan Neff, dn277127
"""

import os

BASE_DIR = "/local/home/dn277127/x17/dream_run"
DECODER_PATH = '/local/home/dn277127/CLionProjects/beam_test_2023/decode/decode'
CONVERTER_PATH = '/local/home/dn277127/CLionProjects/beam_test_2023/decode/convert_vec_tree_to_array'


def main():
    # runs = ['run_60', 'run_61', 'run_62']  # List of runs to decode
    runs = ['run_84', 'run_85', 'run_86']  # List of runs to decode
    for run in runs:
        # run = 'run_50'
        # run = 'test_tcm'
        # Find all .fdf files in the run directory
        run_dir = os.path.join(BASE_DIR, run)
        fdf_files = [f for f in os.listdir(run_dir) if f.endswith('.fdf')]
        print(f'Found {len(fdf_files)} .fdf files in {run_dir}')

        # Iterate through and decode each .fdf file
        for fdf_file in fdf_files:
            fdf_path = os.path.join(run_dir, fdf_file)
            print(f'Decoding {fdf_path}...')
            root_path = fdf_path.replace('.fdf', '.root')
            decode_command = f'{DECODER_PATH} {fdf_path} {root_path}'
            print(f'Running decode: {decode_command}...')
            os.system(decode_command)
            print(f'Converting {root_path} to final format...')
            array_path = root_path.replace('.root', '_array.root')
            convert_command = f'{CONVERTER_PATH} {root_path} {array_path}'
            print(f'Running convert: {convert_command}...')
            os.system(convert_command)

    print('donzo')


if __name__ == '__main__':
    main()

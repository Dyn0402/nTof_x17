#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 09 5:10 PM 2026
Created in PyCharm
Created as nTof_x17/compute_dream_trig_mask.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Given a set of channels, compute the mask for the DREAM trigger.
    """

    masks = [  # Orientation is which way the strips travel, not which coordinate they give
        {'orientation': 'y', 'strips': np.arange(314, 324)},
        {'orientation': 'x', 'strips': np.arange(486, 494)},
        {'orientation': 'x', 'strips': np.arange(255, 257)}
    ]

    strips_per_dream = 64
    feus = {'x': 4, 'y': 5}
    generate_dream_masks(masks, feus, strips_per_dream)

    print('donzo')


def generate_dream_masks(masks, feus, strips_per_dream=64, total_dreams=8, suppress_empty=True):
    # Initialize config: { feu_id: { dream_id: { 8: val, 9: val } } }
    config = {feu_id: {d: {8: 0, 9: 0} for d in range(total_dreams)} for feu_id in feus.values()}

    for mask_set in masks:
        feu_id = feus[mask_set['orientation']]
        for strip in mask_set['strips']:
            dream_idx = int(strip // strips_per_dream)
            channel = int((strip % strips_per_dream) + 1)

            if channel <= 32:
                # Register 8: bit 31 is ch 1, bit 0 is ch 32
                bit_pos = 32 - channel
                config[feu_id][dream_idx][8] |= (1 << bit_pos)
            else:
                # Register 9: bit 0 is ch 33, bit 31 is ch 64
                bit_pos = channel - 33
                config[feu_id][dream_idx][9] |= (1 << bit_pos)

    for feu_id in sorted(config.keys()):
        for dream_id in range(total_dreams):
            reg8_val = config[feu_id][dream_id][8]
            reg9_val = config[feu_id][dream_id][9]

            # # If suppress is ON and both registers are empty, skip this DREAM chip
            # if suppress_empty and reg8_val == 0 and reg9_val == 0:
            #     continue

            for reg_num, val in [(8, reg8_val), (9, reg9_val)]:
                # Optional: further suppress individual register lines if val == 0
                # if suppress_empty and val == 0: continue
                # If suppress is ON register is empty, skip this entry
                if suppress_empty and val == 0:
                    continue

                word_high = (val >> 16) & 0xFFFF
                word_low = val & 0xFFFF

                print(f"Feu {feu_id} Dream {dream_id}  {reg_num} 0x{word_high:04X} 0x{word_low:04X} 0x0000 0x0000")

if __name__ == '__main__':
    main()

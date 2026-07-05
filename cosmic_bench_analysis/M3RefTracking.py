#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 7:26 PM 2024
Created in PyCharm
Created as saclay_micromegas/M3RefTracking.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

import uproot
import awkward as ak


class M3RefTracking:
    def __init__(self, ray_dir, file_nums='all', variables=None, single_track=True, trigger_list=None, chi2_cut=5.0,
                 min_nclus=3):
        """
        :param chi2_cut: per-coordinate cut on Chi2X and Chi2Y (unweighted sum of squared
            residuals in mm^2). Recommended M3 v2 value: 5.0.
        :param min_nclus: minimum clusters per coordinate (NClusX, NClusY) for a track to
            count as good. Recommended M3 v2 value: 3 (requires a genuine 3-4 layer fit and
            drops 2-point-per-coordinate fits, which are only ~38% within 5 mm of the DUT vs
            ~85% for full fits). Set to 0 or None to disable. Only applied when the rays were
            produced by tracking v2 (older files have no NClusX/NClusY branches).
        """
        self.ray_dir = ray_dir
        self.file_nums = file_nums
        self.single_track = single_track
        self.trigger_list = trigger_list

        self.chi2_cut = chi2_cut
        self.min_nclus = min_nclus
        # self.detector_xy_extent_cuts = {'x': [-250, 250], 'y': [-250, 250]}
        self.detector_xy_extent_cuts = {'x': [-500, 500], 'y': [-500, 500]}
        if variables is None:
            self.variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X',
                              'Chi2Y', 'NClusX', 'NClusY']
        else:
            self.variables = variables

        self.ray_data = get_ray_data(ray_dir, file_nums, self.variables)
        # NClusX/NClusY exist only in v2-reprocessed rays; older files silently lack them, so
        # keep self.variables to what was actually loaded and gate the NClus cut on presence.
        self.variables = list(ak.fields(self.ray_data))
        self.has_nclus = 'NClusX' in self.variables and 'NClusY' in self.variables
        if self.min_nclus and not self.has_nclus:
            print(f'M3RefTracking: NClusX/NClusY absent (pre-v2 rays in {ray_dir}); skipping the '
                  f'NClus>={self.min_nclus} cut -- reprocess with tracking v2 to enable it.')

        if self.trigger_list is not None:
            self.filter_on_trigger_list()
        if single_track:
            self.get_single_track_events()

    @property
    def track_vars(self):
        """Per-track (jagged) branches present in the loaded data -- masked together to stay aligned."""
        return [v for v in ('X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y', 'NClusX', 'NClusY')
                if v in self.variables]

    def good_track_mask(self):
        """Per-track boolean: passes the recommended chi2 (+ NClus, when available) recipe."""
        mask = (self.ray_data['Chi2X'] < self.chi2_cut) & (self.ray_data['Chi2Y'] < self.chi2_cut)
        if self.has_nclus and self.min_nclus:
            mask = mask & (self.ray_data['NClusX'] >= self.min_nclus) & (self.ray_data['NClusY'] >= self.min_nclus)
        return mask

    def get_xy_positions(self, z, event_list=None, multi_track_events=False, one_track=True):
        if multi_track_events:
            return get_xy_positions_multi_track_events(self.ray_data, z, event_list, one_track)
        else:
            return get_xy_positions(self.ray_data, z, event_list)

    def get_xy_angles(self, event_list=None):
        return get_xy_angles(self.ray_data, event_list)

    def get_traversing_triggers(self, z, x_bounds, y_bounds, expansion_factor=1):
        """
        Get the event numbers of events that traverse the detector, given by the x and y bounds at altitude z.
        :param z: mm Altitude at which to get the traversing events.
        :param x_bounds: mm Tuple of x bounds of detector at altitude z.
        :param y_bounds: mm Tuple of y bounds of detector at altitude z.
        :param expansion_factor: Factor to expand the bounds by.
        :return: List of event numbers that traverse the detector.
        """
        x_positions, y_positions, event_nums = self.get_xy_positions(z)
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        x_min, x_max = x_min - (x_max - x_min) * expansion_factor, x_max + (x_max - x_min) * expansion_factor
        y_min, y_max = y_min - (y_max - y_min) * expansion_factor, y_max + (y_max - y_min) * expansion_factor
        mask = (x_min < x_positions) & (x_positions < x_max) & (y_min < y_positions) & (y_positions < y_max)
        return event_nums[mask]

    def cut_on_chi2(self, chi2_cut):
        chi2_x, chi2_y = self.ray_data['Chi2X'], self.ray_data['Chi2Y']
        mask = (chi2_x < chi2_cut) & (chi2_y < chi2_cut)
        if self.has_nclus and self.min_nclus:
            mask = mask & (self.ray_data['NClusX'] >= self.min_nclus) & (self.ray_data['NClusY'] >= self.min_nclus)
        for var in self.track_vars:
            self.ray_data[var] = self.ray_data[var][mask]

    def cut_on_det_size(self):
        x_up, x_down, y_up, y_down = [self.ray_data[x_i] for x_i in ['X_Up', 'X_Down', 'Y_Up', 'Y_Down']]
        x_min, x_max, y_min, y_max = self.detector_xy_extent_cuts['x'] + self.detector_xy_extent_cuts['y']
        mask = ((x_min < x_up) & (x_up < x_max) & (x_min < x_down) & (x_down < x_max) &
                (y_min < y_up) & (y_up < y_max) & (y_min < y_down) & (y_down < y_max))
        n_tracks = int(ak.sum(ak.num(mask, axis=1)))  # len(mask) is the EVENT count, not the track count
        print(f'Cutting on detector size: {np.sum(mask)} / {n_tracks} tracks remain, '
              f'{np.sum(mask) / n_tracks * 100:.2f}% (over {len(mask)} events)')
        for var in self.track_vars:
            self.ray_data[var] = self.ray_data[var][mask]

    def get_single_track_events(self):
        """
        Keep events with exactly one good track (recommended recipe: both chi2 < chi2_cut and,
        for v2 rays, NClusX/NClusY >= min_nclus) inside the detector area, and flatten to that
        one track per event.
        """
        self.cut_on_det_size()
        recipe = f'chi2<{self.chi2_cut:g}' + (f' & NClus>={self.min_nclus}' if self.has_nclus and self.min_nclus
                                              else '')
        num_zero = np.sum(ak.num(self.ray_data['Chi2X'], axis=1) == 0)
        num_multi = np.sum(ak.num(self.ray_data['Chi2X'], axis=1) > 1)
        print(f'Pre-cut, Found {num_zero} events with 0 tracks ({num_zero / len(self.ray_data["Chi2X"]) * 100:.2f}%), '
              f'{num_multi} events with >1 tracks ({num_multi / len(self.ray_data["Chi2X"]) * 100:.2f}%)')
        good = self.good_track_mask()
        num_good_tracks = ak.sum(good, axis=1)
        num_zero = np.sum(num_good_tracks == 0)
        num_multi = np.sum(num_good_tracks > 1)
        print(f'Recipe [{recipe}]: {num_zero} events with 0 good tracks ({num_zero / len(num_good_tracks) * 100:.2f}%), '
              f'{num_multi} events with >1 good tracks ({num_multi / len(num_good_tracks) * 100:.2f}%)')
        event_mask = num_good_tracks == 1
        self.ray_data = self.ray_data[event_mask]
        good = good[event_mask]

        # Exactly one good track per kept event: select it directly. (Cannot pick the global
        # min-chi2 track -- a 2-point fit has chi2~0 but fails the NClus cut.)
        for var in self.track_vars:
            self.ray_data[var] = ak.ravel(self.ray_data[var][good])

    def filter_on_trigger_list(self):
        """
        Filter the ray data on the trigger list.
        Sort trigger list and ray data by event number first to make filtering faster.
        :return:
        """
        if self.trigger_list is None:
            return self.ray_data

        # Sort trigger list and ray data by event number
        trigger_list = np.array(self.trigger_list)
        sort_idx = np.argsort(trigger_list)
        trigger_list = trigger_list[sort_idx]
        sort_idx = np.argsort(self.ray_data['evn'])
        for var in self.variables:
            self.ray_data[var] = self.ray_data[var][sort_idx]

        # Filter ray data on trigger list
        mask = np.isin(self.ray_data['evn'], trigger_list)
        for var in self.variables:
            self.ray_data[var] = self.ray_data[var][mask]

    def remove_duplicate_events(self):
        """
        Remove duplicate events based on 'evn' in self.ray_data (Awkward Array).
        Keeps the first occurrence of each event number.
        """
        evn = ak.to_numpy(self.ray_data["evn"])  # Convert to flat NumPy array for uniqueness check
        unique_triggers = np.unique(evn)

        print(f'Removing duplicate events: {len(evn) - len(unique_triggers)} duplicates found of {len(evn)}.')

        if self.trigger_list is not None:
            # Keep only unique triggers that are in the original trigger list
            unique_triggers = np.intersect1d(unique_triggers, np.array(self.trigger_list))
        self.trigger_list = unique_triggers
        self.filter_on_trigger_list()

    def plot_xy(self, z, event_list=None, multi_track_events=False, one_track=True, plt_type='scatter', bins=50):
        x, y, event_nums = self.get_xy_positions(z, event_list, multi_track_events, one_track)
        if plt_type == 'scatter' or plt_type == 'both':
            fig, ax = plt.subplots()
            ax.scatter(x, y, alpha=0.5)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(f'xy Positions at z={z:.2f} mm')
            fig.tight_layout()
        if plt_type == '2D_hist' or plt_type == 'both':
            fig, ax = plt.subplots()
            h = ax.hist2d(x, y, bins=bins, range=[self.detector_xy_extent_cuts['x'],
                                                 self.detector_xy_extent_cuts['y']], cmap='viridis')
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(f'xy Positions at z={z:.2f} mm')
            fig.tight_layout()


def get_ray_data(ray_dir, file_nums='all', variables=None):
    if variables is None:
        variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y',
                     'NClusX', 'NClusY']

    def read_file(file_name):
        if not file_name.endswith('_rays.root'):
            return None

        if isinstance(file_nums, list):
            file_num = int(file_name.split('_')[-2])
            if file_num not in file_nums:
                return None

        with uproot.open(f'{ray_dir}{file_name}') as file:
            tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
            tree = file[tree_name]  # Get tree with max ;# at end
            # NClusX/NClusY exist only in v2-reprocessed rays; read only branches present so
            # older files still load (all files in one ray_dir have the same schema).
            avail = [v for v in variables if v in tree.keys()]
            new_data = tree.arrays(avail, library='ak')
            return new_data

    # List of ROOT files in the directory
    root_files = os.listdir(ray_dir)

    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for parallel file processing
        # Use tqdm for a progress bar and map the read_file function across all root_files
        for file_data in tqdm(executor.map(read_file, root_files), total=len(root_files)):
            if file_data is not None:
                data.append(file_data)
    data = ak.concatenate(data, axis=0)

    return data


def get_xy_positions_multi_track_events(ray_data, z, event_list=None, one_track=True):
    if isinstance(ray_data, ak.highlevel.Array):  # If ray data is awkward array, convert relevant entries to dict of
        variables = ['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down']  # numpy arrays
        ray_data_hold = ray_data
        ray_data = {}
        for var in variables:
            ray_data[var] = ak.to_numpy(ray_data_hold[var])

    mask = np.full(ray_data['evn'].size, True)
    if event_list is not None:
        mask = np.isin(ray_data['evn'], event_list)
    if one_track:
        one_track_mask = np.array([x.size == 1 for x in ray_data['X_Up']])
        mask = mask & one_track_mask

    z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
    x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
    y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]
    event_nums = ray_data['evn'][mask]

    x_up, x_down = np.array([x[0] for x in x_up]), np.array([x[0] for x in x_down])
    y_up, y_down = np.array([y[0] for y in y_up]), np.array([y[0] for y in y_down])

    # Calculate the interpolation factors
    t = (z - z_up) / (z_down - z_up)
    t = np.mean(t)

    # Interpolate the x and y positions
    x_positions = x_up + t * (x_down - x_up)
    y_positions = y_up + t * (y_down - y_up)

    return x_positions, y_positions, event_nums


def get_xy_positions(ray_data, z, event_list=None):
    if isinstance(ray_data, ak.highlevel.Array):  # If ray data is awkward array, convert relevant entries to dict of
        variables = ['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down']  # numpy arrays
        ray_data_hold = ray_data
        ray_data = {}
        for var in variables:
            ray_data[var] = ak.to_numpy(ray_data_hold[var])

    mask = np.full(ray_data['evn'].size, True)
    if event_list is not None:
        mask = np.isin(ray_data['evn'], event_list)

    z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
    x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
    y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]
    event_nums = ray_data['evn'][mask]

    # Calculate the interpolation factors
    t = (z - z_up) / (z_down - z_up)
    t = np.mean(t)

    # Interpolate the x and y positions
    x_positions = x_up + t * (x_down - x_up)
    y_positions = y_up + t * (y_down - y_up)

    return x_positions, y_positions, event_nums


def get_xy_angles(ray_data, event_list=None):
    if isinstance(ray_data, ak.highlevel.Array):  # If ray data is awkward array, convert relevant entries to dict of
        variables = ['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down']  # numpy arrays
        ray_data_hold = ray_data
        ray_data = {}
        for var in variables:
            ray_data[var] = ak.to_numpy(ray_data_hold[var])

    mask = np.full(ray_data['evn'].size, True)
    if event_list is not None:
        mask = np.isin(ray_data['evn'], event_list)

    z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
    x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
    y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]
    event_nums = ray_data['evn'][mask]

    # Calculate the angles
    delta_z = z_down - z_up
    x_angles = np.arctan((x_down - x_up) / delta_z)
    y_angles = np.arctan((y_down - y_up) / delta_z)

    return x_angles, y_angles, event_nums

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 01 3:10 PM 2026
Created in PyCharm
Created as nTof_x17/Mx17StripMap.py

@author: Dylan Neff, dylan
"""

import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class Mx17StripMap:
    """
    Handles mapping:
      (axis, connector, local_channel) -> (x_mm, y_mm)

    Also converts FEU channel [0..511] -> (connector, local_channel)
    """

    CHANNELS_PER_CONNECTOR = 64
    N_CONNECTORS = 8

    def __init__(self, csv_path: str | Path):
        self.map: Dict[Tuple[str, int, int], Tuple[float, float]] = {}
        self._load(csv_path)

    def _load(self, csv_path: str | Path) -> None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                axis = row["axis"]
                connector = int(row["connector"])
                local_channel = int(row["channel"])

                self.map[(axis, connector, local_channel)] = (
                    float(row["x_position_mm"]),
                    float(row["y_position_mm"]),
                )

    @classmethod
    def feu_channel_to_connector(
        cls, feu_channel: int
    ) -> Tuple[int, int]:
        """
        Convert FEU channel [0..511] -> (connector [1..8], local_channel [0..63])
        """
        if not (0 <= feu_channel < cls.N_CONNECTORS * cls.CHANNELS_PER_CONNECTOR):
            raise ValueError(f"Invalid FEU channel: {feu_channel}")

        connector = feu_channel // cls.CHANNELS_PER_CONNECTOR + 1
        local_channel = feu_channel % cls.CHANNELS_PER_CONNECTOR
        return connector, local_channel

    def lookup(
        self,
        axis: str,
        connector: int,
        local_channel: int,
    ) -> Optional[Tuple[float, float]]:
        return self.map.get((axis, connector, local_channel))



class Detector:
    """
    One detector (e.g. mx17_1)

    Maps (feu_id, feu_channel) -> (x_mm, y_mm)
    """

    def __init__(
        self,
        name: str,
        det_cfg: dict,
        strip_map: Mx17StripMap,
    ):
        self.name = name
        self.strip_map = strip_map

        self.dream_feus = det_cfg["dream_feus"]
        self.feu_orientation = det_cfg.get("dream_feu_orientation", {})

        # feu_id -> list of (axis, connector)
        self.feu_map: Dict[int, list[Tuple[str, int]]] = {}
        self._build_feu_map()

    def _build_feu_map(self) -> None:
        """
        dream_feus example:
          'x_1': (4, 1)
        """
        for key, (feu_id, connector) in self.dream_feus.items():
            axis = key[0]  # 'x' or 'y'
            self.feu_map.setdefault(feu_id, []).append(
                (axis, connector)
            )

    @staticmethod
    def apply_orientation(
            local_channel: int,
            orientation: str,
            n_channels: int = 64,
    ) -> int:
        if orientation == "normal":
            return local_channel
        elif orientation == "inverted":
            return (n_channels - 1) - local_channel
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")

    def map_hit(
            self,
            feu_id: int,
            feu_channel: int,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:

        if feu_id not in self.feu_map:
            return None

        hit_connector, local_channel = \
            self.strip_map.feu_channel_to_connector(feu_channel)

        x = y = None

        for axis, det_connector in self.feu_map[feu_id]:

            if det_connector != hit_connector:
                continue

            # Look up orientation using key like 'x_1'
            key = f"{axis}_{det_connector}"
            orientation = self.feu_orientation.get(key, "normal")

            oriented_channel = self.apply_orientation(
                local_channel,
                orientation,
                n_channels=self.strip_map.CHANNELS_PER_CONNECTOR,
            )

            pos = self.strip_map.lookup(
                axis,
                det_connector,
                oriented_channel,
            )
            if pos is None:
                continue

            px, py = pos
            if axis == "x":
                x = px
            else:
                y = py

        if x is None and y is None:
            return None

        return x, y


class RunConfig:
    """
    Loads run_config.json and builds Detector objects
    """

    def __init__(
        self,
        run_config_path: str | Path,
        strip_map_csv: str | Path,
    ):
        self.strip_map = Mx17StripMap(strip_map_csv)
        self.detectors: Dict[str, Detector] = {}

        self._load(run_config_path)

    def _load(self, run_config_path: str | Path) -> None:
        with open(run_config_path) as f:
            cfg = json.load(f)

        included = set(cfg.get("included_detectors", []))

        for det_cfg in cfg["detectors"]:
            name = det_cfg["name"]

            if included and name not in included:
                continue

            self.detectors[name] = Detector(
                name=name,
                det_cfg=det_cfg,
                strip_map=self.strip_map,
            )

    def get_detector(self, name: str) -> Detector:
        return self.detectors[name]


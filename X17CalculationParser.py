#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 02 18:03 2025
Created in PyCharm
Created as nTof_x17/X17Calculations

@author: Dylan Neff, dn277127
"""

import re
import pandas as pd

class X17CalculationParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = {}
        self.df = None
        self._parse()

    def _parse(self):
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        metadata = {}
        tables = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # --------------------
            # Parse metadata lines
            # --------------------
            if line.startswith("#") and ":" in line and not line.startswith("# elow"):
                key, val = line[1:].split(":", 1)
                metadata[key.strip()] = val.strip()
                i += 1
                continue

            # --------------------
            # Parse a data table
            # --------------------
            if line.startswith("# elow"):
                # Header line
                col_line = line[1:].strip().split()
                unit_line = lines[i + 1][1:].strip().split()

                # Merge column name + unit
                columns = [f"{c} [{u}]" for c, u in zip(col_line, unit_line)]

                # Deduplicate repeated names
                seen = {}
                final_cols = []
                for col in columns:
                    if col not in seen:
                        seen[col] = 0
                        final_cols.append(col)
                    else:
                        seen[col] += 1
                        base, unit = col.rsplit("[", 1)
                        new_name = f"{base.strip()}_{seen[col]} [{unit}"
                        final_cols.append(new_name)

                data_lines = []
                i += 2

                # Read until separator or next header
                while i < len(lines):
                    l = lines[i].strip()
                    if not l or l.startswith("#---") or l.startswith("# elow"):
                        break
                    if not l.startswith("#"):
                        data_lines.append(l)
                    i += 1

                # Build DataFrame for this block
                df = pd.DataFrame([l.split() for l in data_lines], columns=final_cols)

                # Convert all columns to numeric when possible
                for c in df.columns:
                    try:
                        df[c] = pd.to_numeric(df[c])
                    except Exception:
                        pass

                tables.append(df)
                continue

            i += 1

        # If multiple tables exist, concatenate them
        self.df = pd.concat(tables, ignore_index=True)
        self.metadata = metadata

    def get_metadata(self):
        return self.metadata

    def get_dataframe(self):
        return self.df


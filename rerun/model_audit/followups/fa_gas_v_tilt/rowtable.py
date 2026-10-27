#!/usr/bin/env python3
"""rowtable.py <key> --table T --alignment A --out O  -- per-event table of one bench row (FA items 2, 3).

Inside rerun/stage_job.py: joins the row's wft reconstruction (events.parquet) with its M3 rays (evttime, chi2, NClus,
M3-frame tangents) and its waveform alignment (reference mesh positions, reference tangents rotated into the strip
frame) exactly as ws_v_w0/build_table.py does for the golden rows.  Geometry comes only from the wft fit (p0, w).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'ws_v_w0')))
import build_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--table', required=True); ap.add_argument('--alignment', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from qa_config import get_config
    cfg = get_config(a.key)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    build_table.build('row', key=a.key, table=a.table, alignment=a.alignment, out=a.out, m3dir=cfg.m3_tracking_dir)


if __name__ == '__main__':
    main()

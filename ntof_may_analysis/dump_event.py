#!/usr/bin/env python3
"""
dump_event.py

Print every row in combined_hits for a given eventId with zero filtering.
Edit EVENT_ID, run, and subrun below, then run:
    python dump_event.py
"""

from pathlib import Path
import uproot
import pandas as pd

# ---------------------------------------------------------------------------
EVENT_ID = 5

run    = 'run_51'
subrun = 'hv_scan_drift_600_resist_510'
# ---------------------------------------------------------------------------

SUBRUN_DIR   = Path(f'/media/dylan/data/x17/may_beam/runs/{run}/{subrun}')
COMBINED_DIR = SUBRUN_DIR / 'combined_hits_root'

files = sorted(
    f for f in COMBINED_DIR.iterdir()
    if f.suffix == '.root' and '_datrun_' in f.name and 'feu-combined' in f.name
)
print(f'Loading {len(files)} combined_hits files ...')
df = uproot.concatenate([f'{f}:hits' for f in files], library='pd')
print(f'Total hits: {len(df):,}   Total events: {df["eventId"].nunique():,}')
print(f'Columns: {list(df.columns)}\n')

ev = df[df['eventId'] == EVENT_ID]
print(f'Event {EVENT_ID}: {len(ev)} hits')
if ev.empty:
    print('(no hits found for this eventId)')
else:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(ev.sort_values(['feu', 'time']).to_string(index=False))

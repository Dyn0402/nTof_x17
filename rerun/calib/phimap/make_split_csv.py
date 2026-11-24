"""make_split_csv.py -- diagnostic: mountings.csv with chamber E split into its two days (one map per day, as r3), for the
comparison with the joint E map (output: $PHIMAP_JOB/phimap_diag/mountings_Esplit.csv; run driver.py with
PHIMAP_OUT=$PHIMAP_JOB/phimap_diag PHIMAP_CSV=<that file> E_0623 E_0624)."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rows as R  # noqa: E402

df = pd.read_csv(os.path.join(HERE, 'mountings.csv'))
m = df.mounting == 'E'
day = df.run.str.contains('6-23')
df.loc[m & day, 'mounting'] = 'E_0623'
df.loc[m & ~day, 'mounting'] = 'E_0624'
df.loc[m & day, 'unit'] = 'E_0623'
df.loc[m & ~day, 'unit'] = 'E_0624'
out = os.path.join(os.path.dirname(R.OUTDIR), 'phimap_diag')
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, 'mountings_Esplit.csv'), index=False)
df.to_csv(os.path.join(out, 'mountings.csv'), index=False)
print(out)

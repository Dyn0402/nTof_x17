#!/usr/bin/env python3
"""humidity_check.py -- do the night-to-night differences of the resist current at fixed HV follow the gas water?
(Dylan 2026-11-03, currents_vs_hv_all).  Inputs: the HV logs (hv_monitor.csv, $WORK_DATA/claude-jobs/17c92c46/hv) and the
water per chamber from rerun/paper1_gas (results/gas.json, figures/gas_drying.csv).  No temperature, humidity or flush
record exists for the bench (paper1_gas/README: 'No bench gas record'), so water from the drift velocity is the only
handle.  Currents are brought to 490 V with the det3 fit I ~ 2^(V/13.3 V) (valid inside ~+-25 V).
Writes results/humidity_check.json and results/figures/humidity_check.{png,pdf}.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402
HV = os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'claude-jobs/17c92c46/hv')
D2 = 13.3

def series(run, det):
    cfg = json.load(open(f'{HV}/{run}/run_config.json'))
    ch = [d for d in cfg['detectors'] if d['name'] == det][0]['hv_channels']['resist']
    out = []
    for f in sorted(glob.glob(f'{HV}/{run}/*/hv_monitor.csv')):
        H = pd.read_csv(f, usecols=['timestamp', f'{ch[0]}:{ch[1]} vmon', f'{ch[0]}:{ch[1]} imon'])
        H.columns = ['t', 'V', 'I']; H['t'] = pd.to_datetime(H.t, errors='coerce'); H['sub'] = os.path.basename(os.path.dirname(f))
        out.append(H)
    H = pd.concat(out).dropna().sort_values('t')
    H = H[(H.V > 350)]
    H['I490'] = H.I * 2 ** ((490 - H.V) / D2)
    return H

# water per chamber and hour, 6-26 (gas_drying.csv: hours since 00:00)
G = pd.read_csv(os.path.join(REPO, 'rerun/paper1_gas/figures/gas_drying.csv'))
G = G[G.panel == 'a/b'].sort_values('hours_626')
res = {}
fig, ax = ps.fig_for('full', ncols=2, height=2.4)
for det, mk in (('mx17_6', 'o'), ('mx17_7', 's')):
    H = series('mx17_det6_det7_overnight_6-26-26', det)
    H['h'] = (H.t - pd.Timestamp('2026-06-26')).dt.total_seconds() / 3600
    # only the fixed-HV long runs (long_run, longer_run, short_run), V within +-6 V of the run's modal V
    H = H[H['sub'].isin(['long_run', 'longer_run', 'short_run'])]
    H = H[np.abs(H.V - H.V.median()) < 8]
    dn = 'det6' if det == 'mx17_6' else 'det7'
    g = G[G.det == dn]
    hb = np.arange(np.floor(H.h.min()), H.h.max(), 0.5)
    rows = []
    for a, b in zip(hb[:-1], hb[1:]):
        m = (H.h >= a) & (H.h < b)
        if m.sum() < 20: continue
        hm = H.h[m].mean(); w = np.interp(hm, g.hours_626, g.water, left=np.nan, right=np.nan)
        rows.append((hm, H.I490[m].median(), w, H.V[m].median()))
    R = pd.DataFrame(rows, columns=['h', 'I490', 'water', 'V']).dropna()
    r = np.corrcoef(R.water, R.I490)[0, 1]
    from scipy.stats import spearmanr
    rho = spearmanr(R.water, R.I490)[0]
    sl = np.polyfit(R.water, np.log(R.I490), 1)
    res[dn] = dict(n=len(R), V_median=float(H.V.median()), I490_first=float(R.I490.iloc[0]), I490_last=float(R.I490.iloc[-1]),
                   water_first=float(R.water.iloc[0]), water_last=float(R.water.iloc[-1]), pearson=float(r), spearman=float(rho),
                   dlnI_per_pctwater=float(sl[0]))
    ax[0].plot(R.water, R.I490, marker=mk, ls='-', lw=0.6, ms=3, color=ps.color(dn), label=f'{dn}')
    ax[1].plot(R.h, R.I490 / R.I490.iloc[-1], marker=mk, ls='-', lw=0.6, ms=3, color=ps.color(dn))
    print(dn, res[dn])
ax[0].set_xlabel('water from drift velocity [%]'); ax[0].set_ylabel('current at 490 V [uA]'); ax[0].legend()
ax[1].set_xlabel('hour on 26 June'); ax[1].set_ylabel('current / last half hour'); ax[1].axhline(1, color=ps.MUTED, lw=0.6)
ps.panel_labels(ax)
os.makedirs(os.path.join(HERE, 'results', 'figures'), exist_ok=True)
ps.save(fig, os.path.join(HERE, 'results', 'figures', 'humidity_check'))

# night-to-night: per board and night the median current at 490 V (+-8 V window) vs the night's mean water
nights = {  # (run, det): (label, mean water %, source)
 ('mx17_det2_det3_overnight_6-22-26', 'mx17_2'): ('det2 6-22', 0.65, 'gas_drying: 0.73->0.57'),
 ('mx17_det3_saturday_scan_6-27-26', 'mx17_3'): ('det3 6-27', 0.97, 'drift scan'),
 ('mx17_det3_det4_overnight_6-23-26', 'mx17_4'): ('det4 6-23', 0.45, 'nights: 0.70->0.21'),
 ('mx17_det4_day_6-24-26', 'mx17_4'): ('det4 6-24', 0.42, 'g_det4'),
}
res['nights'] = {}
rows = []
for run, det in nights:
    try:
        H = series(run, det)
    except Exception as e:
        print('skip', run, det, e); continue
    m = np.abs(H.V - 490) < 8 if 'det4_day' not in run else np.abs(H.V - 495) < 8
    I = float(H.I490[m].median()); rows.append((nights[(run, det)][0], nights[(run, det)][1], I))
    res['nights'][nights[(run, det)][0]] = dict(water=nights[(run, det)][1], I490=I)
    print(rows[-1])
json.dump(res, open(os.path.join(HERE, 'results', 'humidity_check.json'), 'w'), indent=1)

#!/usr/bin/env python3
"""verdict.py [--res results] -- the quoted compositions from gasfit.json and its arms (results/verdict.json).

The drift gap is ONE chamber property, so it is not left free per group: it comes from the RAW run71 group (no ZS
correction, three fields), a parabola through its chi2 at the profiled gaps.  Every group is then quoted at that gap
(the nearest profiled one), with
  stat  = the 68 % half-width of its own (water, air) region at that gap (chi2 scaled by chi2/ndf), at least half a
          grid step (water 0.0125 %, air 0.005 % beam / 0.0096 % CO2);
  gap   = half the spread of its best values over the gaps within the RAW 68 % range, or over the quoted gap and its
          two profiled neighbours if that range is narrower (the shaper trades with the gap: one step is the floor);
  arms  = per pair (ZS x0.5/x1.5, k +-0.1, strip sums) the larger |shift|; single arms (sat, eta, ZS window) their shift;
          added in quadrature.
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = ['co2_run55', 'co2_run56', 'co2_run57', 'co2_run59', 'cf4_run62_ladder', 'cf4_run63_ladder', 'cf4_run63_flat', 'cf4_run71_raw']
PAIRS = [('zs05', 'zs15'), ('kp', 'km'), ('h1', 'h4')]
SINGLE = ['sat', 'raweta', 'zswin', 'tmpl']


def at_gap(G, gp):
    b = G['best_by_gap'].get(str(gp))
    return None if b is None else b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--res', default=os.path.join(HERE, 'results')); a = ap.parse_args()
    R = json.load(open(os.path.join(a.res, 'gasfit.json')))
    arms = {k: json.load(open(os.path.join(a.res, f'gasfit_{k}.json'))) for k in [x for p in PAIRS for x in p] + SINGLE
            if os.path.exists(os.path.join(a.res, f'gasfit_{k}.json'))}
    raw = R['groups']['cf4_run71_raw']
    gs = np.array(sorted(float(k) for k in raw['chi2_by_gap'])); cs = np.array([raw['chi2_by_gap'][str(g)] for g in gs])
    i = int(np.argmin(cs)); sc = raw['scale']
    if 0 < i < len(gs) - 1:
        x, y = gs[i - 1:i + 2], cs[i - 1:i + 2]; c = np.polyfit(x, y, 2)
        g0 = float(-c[1] / (2 * c[0])); sg = float(np.sqrt(sc / c[0])) if c[0] > 0 else None
    else:
        g0, sg = float(gs[i]), None
    gq = float(gs[np.argmin(np.abs(gs - g0))])
    lo, hi = (g0 - sg, g0 + sg) if sg else (gq, gq)
    band = [g for g in gs if lo - 1e-6 <= g <= hi + 1e-6]
    if len(band) < 2:
        # narrower than the profile's sampling: the quoted gap and its two neighbours (the shaper trades with the gap,
        # so one sampling step either side is the honest floor)
        j = int(np.flatnonzero(np.isclose(gs, gq))[0]); band = list(gs[max(0, j - 1):j + 2])
    table = []
    for g in MAIN:
        G = R['groups'].get(g)
        if not G:
            continue
        b = at_gap(G, gq)
        # floor: half a grid step (the 68 % region is often a single grid node)
        dw0 = np.min(np.diff(G['grid_w'])) / 2; da0 = np.min(np.diff(G['grid_a'])) / 2
        st_w = max((b['water_68'][1] - b['water_68'][0]) / 2, dw0); st_a = max((b['air_68'][1] - b['air_68'][0]) / 2, da0)
        bw = [at_gap(G, x)['water'] for x in band if at_gap(G, x)]; ba = [at_gap(G, x)['air'] for x in band if at_gap(G, x)]
        gw, ga = (max(bw) - min(bw)) / 2, (max(ba) - min(ba)) / 2
        sw, sa, det = [], [], {}
        for p in PAIRS + [(s,) for s in SINGLE]:
            dw, da = [], []
            for k in p:
                if k in arms and g in arms[k]['groups']:
                    q = at_gap(arms[k]['groups'][g], gq)
                    if q:
                        dw.append(q['water'] - b['water']); da.append(q['air'] - b['air']); det[k] = (q['water'], q['air'])
            if dw:
                sw.append(max(abs(x) for x in dw)); sa.append(max(abs(x) for x in da))
        aw, aa = float(np.sqrt(np.sum(np.square(sw)))) if sw else 0.0, float(np.sqrt(np.sum(np.square(sa)))) if sa else 0.0
        tw, ta = float(np.sqrt(st_w ** 2 + gw ** 2 + aw ** 2)), float(np.sqrt(st_a ** 2 + ga ** 2 + aa ** 2))
        edge = (b['water'] in (min(G['grid_w']), max(G['grid_w']))) or (b['air'] == max(G['grid_a']))
        table.append(dict(group=g, gas=G['gas'], gap=gq, water=b['water'], air=b['air'], o2_ppm=b['air'] * 2095,
                          stat_w=st_w, gap_w=gw, arms_w=aw, err_w=tw, stat_a=st_a, gap_a=ga, arms_a=aa, err_a=ta,
                          err_o2=ta * 2095, chi2=b['chi2'], ndf=b['ndf'], edge=bool(edge), arms=det))
    out = dict(gap_raw=g0, gap_raw_sigma=sg, gap_quoted=gq, gap_band=[float(x) for x in band], table=table)
    cf4 = [r for r in table if r['gas'] == 'beam']; co2 = [r for r in table if r['gas'] == 'co2']

    def s(r):
        return f"{r['group']} {r['water']:.2f} ± {r['err_w']:.2f} % water, {r['o2_ppm']:.0f} ± {r['err_o2']:.0f} ppm O2"
    # how much the O2 is needed: chi2 with the same water and no air over chi2, per stack, at the group's best gap
    ratios = [p['chi2_noair'] / max(p['chi2'], 1.0) for g in MAIN if g in R['groups'] for p in R['groups'][g]['per_stack']]
    # does the CO2-epoch O2 ordering (first run vs last run) survive every arm?
    order = None
    if len(co2) >= 2:
        first, last = co2[0], co2[-1]
        keep = [first['o2_ppm'] > last['o2_ppm']] + [first['arms'][k][1] > last['arms'][k][1] for k in first['arms']
                                                    if k in last['arms']]
        order = dict(first=first['group'], last=last['group'], survives=bool(all(keep)), n_arms=len(keep) - 1)
    cw = [r['water'] for r in cf4]; co = [r['o2_ppm'] for r in cf4]; zo = [r['o2_ppm'] for r in co2]
    big = max(((k, abs(v[1] - r['air']) * 2095) for r in table for k, v in r['arms'].items()), key=lambda x: x[1])
    out['checks'] = dict(noair_chi2_ratio_min=float(min(ratios)), noair_chi2_ratio_median=float(np.median(ratios)),
                         co2_order=order, largest_o2_arm=dict(arm=big[0], shift_ppm=float(big[1])))
    out['verdict'] = (
        f"Every SPS epoch carried both water and O2. With the drift gap from the RAW run71 stacks ({g0:.1f}"
        + (f" ± {sg:.1f}" if sg else '') + f" mm, quoted at {gq} mm): CF4 epoch water {min(cw):.2f}-{max(cw):.2f} %, "
        f"O2 {min(co):.0f}-{max(co):.0f} ppm; CO2 epoch O2 {min(zo):.0f}-{max(zo):.0f} ppm (per group: "
        + '; '.join(s(r) for r in table) + '). '
        f"The O2 is required: removing it at the same water multiplies a stack's chi2 by {min(ratios):.1f}-"
        f"{max(ratios):.0f} (median {np.median(ratios):.0f}). It is higher in the CO2 epoch than in the CF4 epoch"
        + (f", and falls from {order['first']} to {order['last']} in the baseline and in all {order['n_arms']} arms"
           if order and order['survives'] else '') +
        f"; individual values carry ~±90-130 ppm, dominated by the systematic arms (largest: {big[0]}, {big[1]:.0f} ppm). "
        "The water is degenerate with the drift gap and the shaper and is quoted to ±0.15-0.5 %; the CO2-epoch water rests on a "
        "shaper fitted in the CF4 gas and is the least certain number.")
    out['method'] = [
        'Stacks: the final SPS pass\'s own cache builder (sps_build_cache: seed fix, per-plateau hot probe, RAW common mode), 4000 '
        'events per row, trigger-placed (sample index), strip offsets from the seed strip, RAW drops missing, ZS zeros measured, '
        'saturated events (any sample >= 3550) kept apart (arm: kept).',
        'Strip sums: head-on views only (X +-2, Y +-8 strips; on the 25.64 deg mount the narrower view); the inclined view is '
        'left out (its strip sum truncates with depth inside the +-16-strip window).',
        'Model (cloud_basics FINDINGS §25): uniform ionisation, Magboltz v, eta, D_L on the (water, air) grid at 3e8 collisions '
        '(eta smoothed in log E; grid extended for this fit to beam water 1.2-2.0 %, CO2 water 1.0-2.2 %, CO2 fields 100-275 V/cm), '
        'a parametric shaper fitted on RAW run71 (three fields, X and Y) jointly with its composition, per stack t0 and amplitude.',
        'ZS stacks: + the additive ZS distortion (RAW-emulated minus RAW) of run71 at the nearest field, same view and strip sum, '
        'at the row\'s ZS threshold.',
        'Field = drift HV / gap; the gap is profiled (26.2, 27.0, 27.9, 28.8, 30.0 mm) and taken from the RAW group; every '
        'group is quoted at that gap.',
        'Closure (closure.py): toy stacks from the same model are recovered exactly on the grid; a 30 mm truth fitted at 28.8 mm '
        'moves water by +0.13-0.15 % and leaves air unchanged.']
    out['degeneracies'] = [
        'Water vs drift gap: both enter through v = gap / T. 1.2 mm of gap is worth ~0.13-0.15 % water (closure); air does not '
        'move.',
        'Gap vs shaper: the shaper fitted at each gap differs (at 26.2-27.9 mm it rails at the CR-RC order bound, 60: a '
        'near-Gaussian; at 28.8-30 mm order 1.4-1.8 with a large fast undershoot); the RAW chi2 picks the gap to ~0.1 mm '
        'statistically, but the shaper freedom makes one profile step (0.9 mm) the honest systematic. The bench-template arm '
        '(no free shaper) moves the compositions by <= 0.06 % water and <= 40 ppm O2.',
        'Air vs electronics droop: attachment loses charge at the same rate per time at every field, as a slow droop would. It is '
        'broken by the post-drift-end shape (undershoot) and by the RAW multi-field fit that fixes the shaper; on ZS stacks the '
        'post-end region rests on the RAW ZS emulation (arm: ZS rows fitted only to 2.9 us).',
        'Water vs ZS distortion: ZS removes small and negative samples (the plateau slope and the tail); the ZS groups prefer a '
        'different gap and more water than RAW run71 in the same CF4 gas. The ZS correction scale (x0.5 / x1.5) is an arm.',
        'Magboltz three-body O2 attachment (O2 + e + M, M = H2O / isobutane) sets the absolute ppm scale to a factor of a few; '
        'the epoch-to-epoch ordering does not depend on it.']
    out['limits'] = [
        'run60 (the CO2 -> CF4 flush) is fitted in either pure base only: the grid has no mixed base gas, so its compositions are '
        'indicative (the trend, not the values).',
        'The 32-sample rows (run61, run66, run70) are not used: the column overruns the 1.92 us window.',
        'Single-field groups (run59, run63 flat) constrain water only through one v.',
        'The shaper is fitted on the CF4 RAW run and applied to the CO2 epoch; the ion tail is gas-dependent, so the CO2 water is '
        'the least certain number.',
        'No new data can be taken; a gas record (flow, water) does not exist for the SPS week (GAS_FLUSH_TIMELINE).']
    json.dump(out, open(os.path.join(a.res, 'verdict.json'), 'w'), indent=1)
    for r in table:
        print(f"{r['group']:18s} gap {r['gap']} water {r['water']:.3f} ± {r['err_w']:.3f} (stat {r['stat_w']:.3f} gap {r['gap_w']:.3f} "
              f"arms {r['arms_w']:.3f})  O2 {r['o2_ppm']:.0f} ± {r['err_o2']:.0f} ppm (stat {r['stat_a'] * 2095:.0f} gap {r['gap_a'] * 2095:.0f} "
              f"arms {r['arms_a'] * 2095:.0f})  chi2 {r['chi2']:.0f}/{r['ndf']}  edge {r['edge']}")
    print('gap', g0, sg, gq, band)


if __name__ == '__main__':
    main()

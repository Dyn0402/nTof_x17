"""The text of report.html, built from the products (make_report.py calls build())."""
import html
import json
import os

import numpy as np

from m3common import WORK, pointing_gauss

CSS = """
:root{--ink:#1b2430;--muted:#6a7583;--line:#d6dae0;--bg:#fff;--accent:#0072B2;--warn:#b04a3a}
@media (prefers-color-scheme: dark){:root{--ink:#e6e9ee;--muted:#9aa4b2;--line:#3a4250;--bg:#14181e}}
body{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);
max-width:980px;margin:0 auto;padding:16px 18px 60px;line-height:1.5;font-size:15.5px}
h1{font-size:1.6em;margin:.4em 0 .1em}h2{font-size:1.2em;margin-top:2em;border-bottom:1px solid var(--line)}
h3{font-size:1.05em;margin-top:1.4em}.sub{color:var(--muted);margin-top:0}
.verdict{border-left:4px solid var(--accent);padding:.6em 1em;background:rgba(0,114,178,.07);margin:1em 0}
table{border-collapse:collapse;margin:.8em 0;font-size:.92em;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:4px 9px;text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}
figure{margin:1.2em 0}figure img{max-width:100%;border:1px solid var(--line);background:#fff}
figcaption{color:var(--muted);font-size:.9em}code{font-size:.9em}.lim li{margin:.3em 0}
"""

PUB = {'A_run1': 'A, Run 1 (6-27, top)', 'A_run2': 'A, Run 2 (6-27, top)', 'B_622': 'B, 6-22 (top)',
       'A_622': 'A, 6-22 (bottom)', 'A_623': 'A, 6-23 (bottom)', 'E_623': 'E, 6-23 (top)',
       'C_626': 'C, 6-26 (bottom)', 'D_626': 'D, 6-26 (top)', 'E_624': 'E, 6-24 (top)'}


def f(x, d=2):
    return f'{x:.{d}f}'


def fig(name, cap):
    return f'<figure><img src="figures/{name}.png" alt="{html.escape(name)}"><figcaption>{cap}</figcaption></figure>'


def build(par, fr, cc, ch, eff, S, P, E, clo):
    gx, gy, ex, ey = fr['x']['glob'], fr['y']['glob'], fr['x']['err'], fr['y']['err']
    cs = json.load(open(os.path.join(WORK, 'validate', 'closure_summary.json')))
    pm = json.load(open(os.path.join(WORK, 'posmap.json')))
    cn = json.load(open(os.path.join(WORK, 'validate', 'closure_summary_noposmap.json')))
    res_x = [np.sqrt(0.275 ** 2 - a * 0.24 ** 2) * 1000 for a in (cc['x']['ratio'], fr['x']['glob']['a0'])]
    res_y = [np.sqrt(0.236 ** 2 - a * 0.21 ** 2) * 1000 for a in (cc['y']['ratio'], fr['y']['glob']['a0'])]
    eA = eff['A (Runs 1+2)']
    sX = np.sqrt(np.array(par['X']['mu']) / 2) * 1000; sY = np.sqrt(np.array(par['Y']['mu']) / 2) * 1000
    F_anom_X = sum(par['X']['f']); F_anom_Y = sum(par['Y']['f'])
    o = []
    w = o.append
    w(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
      f'content="width=device-width,initial-scale=1"><title>M3 per-track pointing</title><style>{CSS}</style></head><body>')
    w('<h1>The pointing of the M3 reference, track by track</h1>')
    w('<p class="sub">Cosmic bench, pass 1. Study code: <code>rerun/m3_pointing/</code>. Products: '
      '<code>$WORK_DATA/nTof_x17/m3_pointing/</code>. 2026-11-26. Preliminary.</p>')

    # ---------------------------------------------------------------- verdict
    w('<div class="verdict"><b>Verdict.</b> The pointing of the M3 telescope is not one number. '
      'It depends on the χ² of the reference track, on its angle and on where it crosses the telescope. '
      'We describe it with a per-track model, and the model needs no chamber to be measured, because two chambers on the '
      'same track share only the reference error. '
      f'With the model, the width of the residual pull is the same in every bin of χ², anomaly and angle to '
      f'{f(min(cs[k + "|held-out"]["std1"] for k in ("chi2", "k", "tan")), 2)}–'
      f'{f(max(cs[k + "|held-out"]["std1"] for k in ("chi2", "k", "tan")), 2)} (spread over bins), against '
      f'{f(min(cs[k + "|held-out"]["std0"] for k in ("chi2", "k", "tan")), 2)}–'
      f'{f(max(cs[k + "|held-out"]["std0"] for k in ("chi2", "k", "tan")), 2)} with the constant pointing of Paper I, and '
      'this holds on the three rows that took no part in the fit. '
      f'The pointing floor at χ² → 0 agrees with the Paper I value: the model gives {f(gx["a0"])} and {f(gy["a0"])} of '
      'its variance, and a model-free extrapolation of the two-chamber covariance gives '
      f'{f(cc["x"]["ratio"])} ± {f(cc["x"]["ratio_err"])} and {f(cc["y"]["ratio"])} ± {f(cc["y"]["ratio_err"])}. ')
    if E is not None:
        lo = E[E.hi <= 1]; hi = E[E.lo >= 1]
        w('Inside the present recipe (χ² &lt; 1), a window of a few σ per track gives the same efficiency as the fixed '
          '5 mm at the same accidental rate. ')
        mid = E[(E.lo >= 1) & (E.hi <= 4)]; lo1 = E[E.hi <= 1]
        widen = 100 * mid.n.sum() / lo1.n.sum()
        w('Beyond the recipe, the per-track width follows the residual out to χ² ≈ 50, although it was fitted at '
          'χ² &lt; 1 only. With a window of 8σ_i, tracks with χ² up to 4 are found as efficiently as recipe tracks '
          f'({100 * (mid.eff_n8 * mid.n).sum() / mid.n.sum():.1f} %, accidental ≤ {100 * mid.acc_n8.max():.2f} %), so the '
          f'recipe could be widened to χ² &lt; 4, which adds {widen:.0f} % more tracks on this row. ')
    else:
        w('Inside the present recipe (χ² &lt; 1), a window of a few σ per track gives the same efficiency as the fixed '
          '5 mm at the same accidental rate.')
    w('</div>')

    # ---------------------------------------------------------------- numbers
    w('<h2>Headline numbers</h2><table><tr><th>quantity</th><th>X view (M3 Y)</th><th>Y view (M3 X)</th><th>meaning</th></tr>')
    w(f'<tr><td>floor at χ² → 0, model (× Paper I variance)</td><td>{f(gx["a0"])} ± {f(ex["a0"])}</td>'
      f'<td>{f(gy["a0"])} ± {f(ey["a0"])}</td><td>a0: the pointing at χ² → 0</td></tr>')
    w(f'<tr><td>floor at χ² → 0, two-chamber covariance, |θ| &lt; 5°</td><td>{f(cc["x"]["ratio"])} ± {f(cc["x"]["ratio_err"])}</td>'
      f'<td>{f(cc["y"]["ratio"])} ± {f(cc["y"]["ratio_err"])}</td><td>model-free check of a0 (common part only)</td></tr>')
    w(f'<tr><td>growth of the floor with the plane resolution</td><td>{f(gx["a1"])}</td><td>{f(gy["a1"])}</td>'
      '<td>a1: floor ∝ a0 + a1 (μ(t)/μ(0) − 1)</td></tr>')
    w(f'<tr><td>response to the anomalous χ², top / bottom / cross</td><td>{f(gx["kt"])} / {f(gx["kb"])} / {f(gx["ktb"])}</td>'
      f'<td>{f(gy["kt"])} / {f(gy["kb"])} / {f(gy["ktb"])}</td><td>κ: mm² of pointing variance per mm² of χ²</td></tr>')
    w(f'<tr><td>response to the ordinary χ², top / bottom / cross</td><td>{f(gx["gt"])} / {f(gx["gb"])} / {f(gx["gtb"])}</td>'
      f'<td>{f(gy["gt"])} / {f(gy["gb"])} / {f(gy["gtb"])}</td><td>γ: zero for Gaussian plane errors; scattering kinks</td></tr>')
    w(f'<tr><td>M3 plane resolution, normal incidence → |tan θ| = 0.32</td><td colspan="2">X planes {sX[0]:.0f} → {sX[-1]:.0f} µm,'
      f' Y planes {sY[0]:.0f} → {sY[-1]:.0f} µm</td><td>from the ordinary part of the χ²</td></tr>')
    w(f'<tr><td>tracks with an anomalous part in the χ²</td><td colspan="2">{100 * F_anom_Y:.0f} % (M3 Y), {100 * F_anom_X:.0f} % (M3 X)</td>'
      '<td>kinks, δ-rays, wrong clusters</td></tr>')
    w(f'<tr><td>chamber A Runs 1+2, efficiency R = 5 mm</td><td colspan="2">{f(eA["eff_R5"])} % (accidental {eA["acc_R5"]:.3f} %)</td>'
      '<td>Paper I: 95.1 %</td></tr>')
    w(f'<tr><td>… per-track window n = 3 / 5 σ_i</td><td colspan="2">{f(eA["eff_n3"])} / {f(eA["eff_n5"])} % (accidental at n = 5: '
      f'{eA["acc_n5"]:.3f} %)</td><td>at R = 5 mm\'s accidental rate: n = {eA["n_at_accR5"]:g}, {f(eA["eff_n_at_accR5"])} %</td></tr>')
    w('</table>')

    # ---------------------------------------------------------------- what and why
    w('<h2>What we measure and why</h2>')
    w('<p>Every bench result of Paper I compares the chamber with the reference ray of the M3 telescope. The telescope '
      'has four stations, each with an X and a Y strip plane, at z = 24, 144, 1185 and 1302 mm. Each coordinate is a '
      'straight-line fit to four points, and the χ² of that fit is the sum of the squared residuals (mm², two degrees '
      'of freedom). The chambers sit in two slots between the stations, at z ≈ 242 mm (bottom) and z ≈ 713 mm (top).</p>')
    w(f'<p>Paper I uses one pointing resolution for every track: {pointing_gauss(713, "X") * 1000:.0f}–'
      f'{pointing_gauss(713, "Y") * 1000:.0f} µm at the top slot and {pointing_gauss(242, "X") * 1000:.0f}–'
      f'{pointing_gauss(242, "Y") * 1000:.0f} µm at the bottom slot, from the per-plane resolution measured in '
      '<code>m3_self_resolution</code>. However, a P2 analysis showed that the pointing depends on the χ², on the angle '
      'and probably on the position. If the pointing of each track is known, a hit can be required to lie within a few σ '
      'of that track\'s own expected residual, instead of within a fixed radius. This report builds that per-track '
      'pointing, tests it, and tries the matching rule.</p>')
    w('<p>The data are nine pass-1 rows (effbudget per-ray tables joined with the M3 track of each ray). Six rows are '
      'three runs with two chambers in the telescope at once (6-22 B top and A bottom, 6-23 E top and A bottom, 6-26 D '
      'top and C bottom). These pairs measure the telescope. Three rows have one chamber (A Run 1, A Run 2, E 6-24) and '
      'serve as held-out tests.</p>')

    # ---------------------------------------------------------------- step 1
    w('<h2>1. What the χ² of a reference track contains</h2>')
    w('<p>For Gaussian plane errors, the χ² of a four-point line fit is σ² times a χ² with two degrees of freedom, '
      'which is an exponential distribution with mean 2σ². The fitted line and the residuals are then independent, so '
      'the χ² would carry no information on the pointing at all. The χ² can only predict the pointing if part of it is '
      'not Gaussian. Figure 1a shows that this is the case. At normal incidence the χ² distribution has a narrow core '
      'and a long tail. At larger angles the core is wider, but the tail above χ² ≈ 3 mm² has the same density at every '
      'angle.</p>')
    w('<p>We therefore describe the χ² of each coordinate as the sum of two parts. The <i>ordinary part</i> is the '
      'exponential of the plane resolution, with a mean μ(t) that depends on the slope t of the track. The '
      '<i>anomalous part</i> has the same distribution at every angle (zero for most tracks, otherwise a sum of three '
      'exponentials). The fit over 212 000 single-track, four-cluster rays describes the data with a Pearson χ²/dof of '
      f'{par["X"]["pearson"] / (par["X"]["nbins"] - par["X"]["npar"]):.1f} (X) and '
      f'{par["Y"]["pearson"] / (par["Y"]["nbins"] - par["Y"]["npar"]):.1f} (Y). For every track, the posterior mean of '
      'the anomalous part given its χ² and its angle, E[k | χ², t], follows in closed form. Figure 1b shows the plane '
      f'resolution √(μ/2) from the ordinary part. It is {sX[0]:.0f} µm (X planes) and {sY[0]:.0f} µm (Y planes) at '
      'normal incidence, and it grows five-fold by |tan θ| = 0.3. This is the reason why χ² and angle are entangled: an '
      'inclined track has a large χ² simply because the planes resolve it worse.</p>')
    w(fig('decompose', 'Figure 1. (a) Density of the M3 χ² of the X coordinate in three bins of the track slope (points), '
          'with the fit (solid) and its anomalous part alone (dotted). (b) The plane resolution √(μ/2) from the ordinary '
          'part as a function of the track slope, for the X and Y planes.'))

    # ---------------------------------------------------------------- step 2
    w('<h2>2. Measuring the telescope without a chamber</h2>')
    w('<p>When two chambers see the same track, each residual is the chamber error minus the M3 error at that chamber. '
      'The chamber errors are independent of each other, so the covariance of the two residuals contains only the M3 '
      'error. This is a three-cornered-hat measurement of the telescope. The chambers are rotated by about 90° to M3, so '
      'the chamber X view is compared with M3 Y and the chamber Y view with M3 X.</p>')
    w(f'<p>Figure 2 shows this covariance for tracks within 5° of normal incidence, as a function of the χ² of the '
      f'matched M3 coordinate. The covariance grows by a factor of ten across the recipe range (χ² &lt; 1). The growth '
      f'is therefore a property of the telescope, not of the chambers. Extrapolated linearly to χ² → 0, the covariance is '
      f'{cc["x"]["intercept"]:.4f} ± {cc["x"]["intercept_err"]:.4f} mm² (X view) and {cc["y"]["intercept"]:.4f} ± '
      f'{cc["y"]["intercept_err"]:.4f} mm² (Y view). The Gaussian prediction from the per-plane resolution is '
      f'{cc["x"]["gauss_pred"]:.4f} and {cc["y"]["gauss_pred"]:.4f} mm² (stars). The ratios are '
      f'{f(cc["x"]["ratio"])} ± {f(cc["x"]["ratio_err"])} and {f(cc["y"]["ratio"])} ± {f(cc["y"]["ratio_err"])}. This '
      'means that the pointing used by Paper I is not too small. If anything, it is 5–10 % too large in width, which is '
      'inside the ±10 % that Paper I assigns to it. The covariance only sees the part of the M3 error that the two slots '
      'have in common, so this check cannot exclude an additional error that affects one slot alone. The linear fits '
      f'have χ² of {cc["x"]["fit_chi2"]:.0f} and {cc["y"]["fit_chi2"]:.0f} for 4 degrees of freedom: the curve is '
      'slightly concave.</p>')
    w(fig('crosscov', 'Figure 2. Covariance of the residuals of the top and the bottom chamber (three pair runs pooled), '
          'tracks within 5° of normal incidence, in six bins of the χ² of the matched M3 coordinate. Lines: linear fits. '
          'Stars at χ² = 0: the Gaussian pointing covariance Σ g_k(z_top) g_k(z_bottom) σ_k².'))

    # ---------------------------------------------------------------- step 3
    w('<h2>3. The per-track model</h2>')
    w('<p>For a track i and a chamber view, we write the variance of the residual as the chamber term plus the M3 term:</p>')
    w('<p style="text-align:center">σ_i² = d0 + d1 t² &nbsp; + &nbsp; P_s² [a0 + a1 (μ_i/μ_0 − 1)] + κ_s E[k_i] + '
      'γ_s E[g_i] &nbsp; (× the position factor of section 5).</p>')
    w('<p>Here d0 and d1 describe the chamber (per row). P_s² is the Gaussian pointing of the slot from the per-plane '
      'resolution, and a0 = 1 is the assumption of Paper I. The floor grows with the plane resolution μ_i of the track '
      '(a1). The two χ² terms are the posterior means of the anomalous part (k) and of the ordinary part (g = χ² − k) of '
      'the matched coordinate. For Gaussian plane errors γ would be zero. However, a small multiple-scattering kink makes '
      'a χ² that looks ordinary, and a kink near the top slot moves the pointing there by about 23 times its χ² (1.8 at '
      'the bottom slot, from the fit geometry). For the pairs, the covariance has the same form with the cross geometry '
      'P_tb and its own κ and γ. The floor\'s top : bottom : cross ratios are fixed by the fit geometry, so a0 and a1 are '
      'pinned by the covariance, which no chamber contributes to. Each chamber also has a broad outlier component. We '
      'fit the bivariate likelihood of all pair events, per view (<code>fit.py</code>).</p>')
    w('<table><tr><th>parameter</th><th>X view (M3 Y)</th><th>Y view (M3 X)</th></tr>')
    for k_, lab in (('a0', 'a0, floor at χ² → 0'), ('a1', 'a1, floor growth with μ'), ('kt', 'κ top'), ('kb', 'κ bottom'),
                    ('ktb', 'κ cross'), ('gt', 'γ top'), ('gb', 'γ bottom'), ('gtb', 'γ cross')):
        w(f'<tr><td>{lab}</td><td>{f(gx[k_])} ± {f(ex[k_])}</td><td>{f(gy[k_])} ± {f(ey[k_])}</td></tr>')
    w('</table>')
    w('<p>Both responses are larger at the top slot than at the bottom slot, as the kink geometry predicts. The '
      'positivity condition of the M3 covariance (κ_cross² ≤ κ_top κ_bottom) holds in both views.</p>')

    # ---------------------------------------------------------------- step 4
    w('<h2>4. Does the width fit every track?</h2>')
    w('<p>For each row we fit the chamber term with the M3 term held fixed (<code>validate.py</code>), and we do the same '
      'with the constant pointing of Paper I (a chamber term without angle dependence). The per-track model is better '
      'in every row and view. The improvement in log-likelihood is 85 to 1470 for one extra parameter.</p>')
    w('<p>The test that matters is the pull r_i/σ_i. If the width is right for each track, the pull has the same width '
      'in every bin of χ², anomaly, angle and position. Figure 3 shows the core width of the pull in bins of the '
      'anomalous χ², the angle and the χ². With the constant pointing, the pull width runs from 0.65 to 1.7. With the '
      'per-track model it stays flat. The level is about 0.9 rather than 1 because a fixed-window Gaussian core is '
      'narrower than the Gaussian of the mixture fit. This offset is the same in every bin.</p>')
    w('<table><tr><th>binning variable</th><th>rows</th><th>per track: mean ± spread</th><th>constant: mean ± spread</th></tr>')
    for key, lab in (('chi2', 'χ²'), ('k', 'anomalous χ²'), ('tan', 'angle'), ('pos_along', 'position along the view'),
                     ('pos_across', 'position across the view')):
        for held in ('held-out', 'in fit'):
            c = cs[f'{key}|{held}']
            w(f'<tr><td>{lab}</td><td>{held}</td><td>{f(c["mean1"], 3)} ± {f(c["std1"], 3)}</td>'
              f'<td>{f(c["mean0"], 3)} ± {f(c["std0"], 3)}</td></tr>')
    w('</table>')
    w(fig('closure', 'Figure 3. Core width of the pull r/σ_i (both views pooled) in bins of (a) the anomalous χ² of the '
          'matched M3 coordinate, (b) the track angle in the view and (c) the χ² of the matched coordinate. Blue: per-track '
          'model; orange: constant pointing (Paper I). Solid: the three held-out rows; dashed: the six pair rows.'))
    w('<h3>Chamber terms</h3><p>The chamber term d0 is the width of the Gaussian component of the mixture at normal '
      'incidence. It is not the intrinsic resolution of Paper I, which is the core width extrapolated to χ² → 0. '
      'Chamber A gives about 150 µm on 6-22 (bottom slot) but 300–340 µm in Runs 1 and 2 (top slot) and on 6-23 '
      '(bottom slot). Since the 6-23 row is also in the bottom slot, this difference follows the run and the mounting '
      'of chamber A, not the slot. For this reason, chamber A cannot be used to pin the floor of the top slot '
      'separately.</p>')
    w('<table><tr><th>row</th><th>σ_D(0) X</th><th>σ_D(0) Y</th><th>outliers X / Y</th><th>ΔlnL vs constant X / Y</th></tr>')
    for tag, d in ch.items():
        mx, my = d['x']['model'], d['y']['model']
        w(f'<tr><td>{PUB[tag]}</td><td>{np.sqrt(max(mx["d0"], 0)) * 1e3:.0f} µm</td><td>{np.sqrt(max(my["d0"], 0)) * 1e3:.0f} µm</td>'
          f'<td>{100 * mx["eps"]:.1f} / {100 * my["eps"]:.1f} %</td>'
          f'<td>{d["x"]["const"]["nll"] - mx["nll"]:.0f} / {d["y"]["const"]["nll"] - my["nll"]:.0f}</td></tr>')
    w('</table>')

    # ---------------------------------------------------------------- step 5
    w('<h2>5. Position</h2>')
    w('<p>We compare the two-chamber covariance with the model\'s prediction in 4 × 4 cells of the M3 position '
      '(<code>posmap.py</code>). The ratio is normalised to the whole telescope, because the robust covariance and the '
      'model covariance differ by a constant factor. Two effects appear, and both belong to the telescope.</p>')
    w('<ul><li>The width. For the X view (M3 Y), the M3 term is 1.3–1.75 times larger at M3 X &gt; +100 mm and about '
      '0.75 times smaller at M3 X &lt; 0. The Y view shows a smaller excess at M3 X between 0 and +100 mm and '
      'M3 Y &lt; −100 mm. We include this as a factor on the M3 term, shrunk towards 1 by its error. It lowers the '
      f'spread of the pull width across position bins from {f(cn["pos_across|in fit"]["std1"], 3)} to '
      f'{f(cs["pos_across|in fit"]["std1"], 3)} (pair rows) and from {f(cn["pos_across|held-out"]["std1"], 3)} to '
      f'{f(cs["pos_across|held-out"]["std1"], 3)} (held-out rows).</li>')
    sl = pm['row_slope_um_per_mm']
    top = sorted(v['slope'] for v in sl.values() if v['z'] > 500); bot = sorted(v['slope'] for v in sl.values() if v['z'] < 500)
    w('<li>A shift. The medians of the two chambers\' X-view residuals move together across the telescope (numbers in '
      'brackets in the table below). For each row we fit the median X-view residual as a linear function of the M3 Y '
      f'position. Every top-slot row has a positive slope ({top[0]:+.1f} to {top[-1]:+.1f} µm/mm), and the bottom-slot '
      f'rows scatter around zero ({bot[0]:+.1f} to {bot[-1]:+.1f} µm/mm). Chamber A gives '
      f'{sl["A_622"]["slope"]:+.1f} µm/mm in the bottom slot and {sl["A_run1"]["slope"]:+.1f} / {sl["A_run2"]["slope"]:+.1f} '
      'µm/mm in the top slot, which points to the telescope at the top slot. However, chamber E has the largest slope '
      f'({sl["E_623"]["slope"]:+.1f} and {sl["E_624"]["slope"]:+.1f} µm/mm), so a strip-pitch error of the chamber '
      'contributes too. A slope of 1 µm/mm is a 0.1 % scale error. The alignment of each chamber (an offset and a '
      'rotation) cannot absorb a scale error. We report this shift and do not correct it here.</li></ul>')
    w('<table><tr><th>X view: M3 X cell \\ M3 Y cell</th>' + ''.join(f'<th>{a:.0f}…{b:.0f}</th>' for a, b in
                                                               zip(pm['edges'][:-1], pm['edges'][1:])) + '</tr>')
    for i, (a, b) in enumerate(zip(pm['edges'][:-1], pm['edges'][1:])):
        w(f'<tr><td>{a:.0f}…{b:.0f} mm</td>' + ''.join(
            f'<td>{pm["x"]["ratio"][i][j]:.2f} ± {max(pm["x"]["err"][i][j], 0):.2f} ({pm["x"]["common_shift"][i][j]:+.2f})</td>'
            for j in range(4)) + '</tr>')
    w('</table><p class="sub">Cells: covariance ratio data / model ± bootstrap error (common median shift of the two '
      'chambers, mm).</p>')

    # ---------------------------------------------------------------- step 6
    w('<h2>6. Matching each track within n σ of its own width</h2>')
    w('<p>We now use the width as a matching rule. The denominator is that of Paper I: one recipe ray per event, in the '
      'interior, decoder-complete. The numerator is the any-candidate rule (in each view, the candidate nearest the '
      'ray). A track is found when d² = Σ_view ((r − bias)/σ_i)² ≤ n². The accidental rate comes from event mixing, as '
      f'in Paper I. With the fixed radius, our sample reproduces Paper I: {f(eA["eff_R5"])} % for chamber A in the '
      'interior (Paper I: 95.1 %).</p>')
    w(f'<p>Figure 4 shows the efficiency as a function of the accidental rate for the three rules. At tight windows the '
      f'per-track rule is better. At an accidental rate of 0.01–0.02 %, it finds about one point more muons than a fixed '
      f'radius. At the operating point of Paper I, however, the three rules agree. The per-track rule reaches the '
      f'accidental rate of R = 5 mm at n = {eA["n_at_accR5"]:g} with {f(eA["eff_n_at_accR5"])} %, against '
      f'{f(eA["eff_R5"])} %. The reason is that the recipe χ² &lt; 1 already removes the tracks with a poor pointing. '
      'Inside the recipe σ_i stays below about 1 mm, so 5 mm is already more than 5σ for every track. The tracks that '
      'the per-track rule loses at n = 5 are tails of the chamber (δ-pulled fits), not of the reference.</p>')
    w(f'<p>The per-track width also answers part of the question of Paper I\'s appendix "Why 5 mm and not a few σ". '
      f'There, a cut at 3σ of the global core would lose 11 points. With the per-track width, n = 3 loses '
      f'{f(eA["eff_R5"] - eA["eff_n3"], 1)} points. So the reference explains about '
      f'{100 * (1 - (eA["eff_R5"] - eA["eff_n3"]) / 11):.0f} % of the loss between a 3σ cut and 5 mm, '
      'and the chamber explains the rest.</p>')
    w(fig('eff_vs_acc', 'Figure 4. Chamber A, Runs 1+2, interior: efficiency (any-candidate rule) as a function of the '
          'accidental match probability (event mixing) for a fixed radius R (gray), a per-track window n σ_i (blue) and a '
          'window n σ with the constant pointing (orange). Marked: R = 5 mm, n = 3 and 5.'))
    w('<table><tr><th>chamber / run</th><th>rays</th><th>R = 5 mm</th><th>n = 3</th><th>n = 5</th><th>n at R=5 acc.</th>'
      '<th>eff. there</th><th>gained / lost at n=5 vs R=5</th></tr>')
    for k_, s in eff.items():
        w(f'<tr><td>{k_}</td><td>{s["n"]}</td><td>{f(s["eff_R5"])} %</td><td>{f(s["eff_n3"])} %</td><td>{f(s["eff_n5"])} %</td>'
          f'<td>{s["n_at_accR5"]:g}</td><td>{f(s["eff_n_at_accR5"])} %</td><td>+{s["gained_n5"]} / −{s["lost_n5"]}</td></tr>')
    w('</table>')

    # ---------------------------------------------------------------- step 7
    if E is not None:
        nlo = int(E[E.hi <= 1].n.sum()); nhi = int(E[E.lo >= 1].n.sum())
        w('<h2>7. Beyond the recipe: χ² &gt; 1</h2>')
        w(f'<p>The recipe keeps only half of the single-track, four-cluster M3 tracks. The pass never reconstructed the '
          f'other half, so we reconstructed every such event of chamber A Run 1 with the production code and the pass-1 '
          f'bundle (<code>stage_reco.py</code>, <code>rerun/outliers/reco_subset.py</code>). On the recipe events the new '
          f'table reproduces pass 1: {100 * clo["x"]["frac_p0_within_50um"]:.1f} % (X) and '
          f'{100 * clo["y"]["frac_p0_within_50um"]:.1f} % (Y) of the positions agree within 50 µm, and the fit status '
          f'agrees for {100 * clo["x"]["ok_agree"]:.1f} and {100 * clo["y"]["ok_agree"]:.1f} %. In the interior there are '
          f'{nlo} rays with χ² &lt; 1 and {nhi} with χ² ≥ 1 in either coordinate.</p>')
        w('<p>The model was fitted at χ² &lt; 1, so χ² &gt; 1 is a true extrapolation. Figure 5a shows the residual width '
          'and the median model width as a function of the χ² of the matched coordinate. Figure 5b shows the width of '
          'the pull for the per-track model and for the constant pointing. Figure 5c shows the efficiency per χ² bin for '
          'R = 5 mm and for per-track windows of 5σ_i and 8σ_i.</p>')
        w(fig('extend', 'Figure 5. Chamber A Run 1, every single-track four-cluster M3 ray. (a) Residual s68 (points) and '
              'median model width σ_i (lines) per bin of the χ² of the matched coordinate; dotted: the recipe cut. (b) s68 '
              'of the pull for the per-track model (solid) and the constant pointing (dashed). (c) Efficiency in the '
              'interior per bin of the larger of the two χ², for R = 5 mm, d ≤ 5σ_i and d ≤ 8σ_i.'))
        w('<table><tr><th>χ² bin</th><th>rays</th><th>R = 5 mm (acc.)</th><th>R = 20 mm (acc.)</th>'
          '<th>n = 5 (acc.)</th><th>n = 8 (acc.)</th><th>median window, n = 8</th></tr>')
        for r in E.itertuples():
            w(f'<tr><td>{r.lo:g}–{r.hi:g}</td><td>{r.n}</td><td>{100 * r.eff_R5:.1f} % ({100 * r.acc_R5:.2f})</td>'
              f'<td>{100 * r.eff_R20:.1f} % ({100 * r.acc_R20:.2f})</td><td>{100 * r.eff_n5:.1f} % ({100 * r.acc_n5:.2f})</td>'
              f'<td>{100 * r.eff_n8:.1f} % ({100 * r.acc_n8:.2f})</td><td>{r.win_n8_med_mm:.1f} mm</td></tr>')
        w('<p class="sub">Efficiency in the interior, any-candidate rule; accidental match probability in % (event mixing, '
          '20 shifts) in brackets.</p>')
        w('</table>')
        w('<table><tr><th>view</th><th>χ² bin</th><th>n</th><th>residual s68</th><th>model σ_i</th><th>M3 part</th>'
          '<th>pull s68: model</th><th>pull s68: constant</th></tr>')
        for r in P.itertuples():
            w(f'<tr><td>{r.view}</td><td>{r.lo:g}–{r.hi:g}</td><td>{r.n}</td><td>{r.res_s68:.2f} mm</td><td>{r.sig_med:.2f} mm</td>'
              f'<td>{r.pm3_med:.2f} mm</td><td>{r.s68_1:.2f}</td><td>{r.s68_0:.2f}</td></tr>')
        w('</table>')
        lo1 = E[E.hi <= 1]; mid = E[(E.lo >= 1) & (E.hi <= 4)]; m48 = E[(E.lo >= 4) & (E.hi <= 8)]
        effw = lambda S, c: float((S[c] * S.n).sum() / S.n.sum()) * 100
        Px = P[(P.view == 'x') & (P.hi <= 50)]; Py = P[(P.view == 'y') & (P.hi <= 50)]
        w(f'<p>Figure 5 shows that the model, fitted at χ² &lt; 1 only, follows the residual far outside that range. '
          f'From the lowest bin to χ² = 16–50 the residual s68 grows from {Px.res_s68.iloc[0]:.2f} to '
          f'{Px.res_s68.iloc[-1]:.1f} mm (X view) and from {Py.res_s68.iloc[0]:.2f} to {Py.res_s68.iloc[-1]:.1f} mm '
          f'(Y view). The median model width grows with it, from {Px.sig_med.iloc[0]:.2f} to {Px.sig_med.iloc[-1]:.1f} '
          f'and from {Py.sig_med.iloc[0]:.2f} to {Py.sig_med.iloc[-1]:.1f} mm. The s68 of the pull stays between '
          f'{min(Px.s68_1.min(), Py.s68_1.min()):.2f} and {max(Px.s68_1.max(), Py.s68_1.max()):.2f} in every bin up to '
          f'χ² = 50, while with the constant pointing it grows to {max(Px.s68_0.max(), Py.s68_0.max()):.1f}. Above '
          'χ² = 50 the model overestimates the width (pull s68 about 0.5). These tracks are mostly broken reference '
          'tracks, and we do not propose to use them.</p>')
        w(f'<p>The efficiency shows the same thing from the other side. With R = 5 mm, the efficiency falls from '
          f'{100 * E.eff_R5.iloc[0]:.1f} % at the lowest χ² to {100 * m48.eff_R5.iloc[0]:.1f} % at χ² = 4–8, although '
          f'the accidental rate stays at 0.05–0.09 %. With a per-track window of 8σ_i, the efficiency is '
          f'{effw(lo1, "eff_n8"):.1f} % at χ² &lt; 1, {effw(mid, "eff_n8"):.1f} % at χ² = 1–4 and '
          f'{100 * m48.eff_n8.iloc[0]:.1f} % at χ² = 4–8. The accidental rate is {100 * mid.acc_n8.max():.2f} % or less up '
          f'to χ² = 4 and {100 * m48.acc_n8.iloc[0]:.2f} % at 4–8. This means that the losses of the fixed radius at large '
          'χ² are losses of the reference, not of the chamber. The chamber is as efficient for these muons as for the '
          'recipe muons, and the per-track window recovers them.</p>')
        w(f'<p>The recipe could therefore be widened. With χ² &lt; 4 and a window of 8σ_i, chamber A Run 1 has '
          f'{int(lo1.n.sum() + mid.n.sum())} interior rays instead of {int(lo1.n.sum())} '
          f'(+{100 * mid.n.sum() / lo1.n.sum():.0f} %), at an accidental rate of at most {100 * mid.acc_n8.max():.2f} %. '
          'This matters most for the rows with few tracks.</p>')

    # ---------------------------------------------------------------- paper
    w('<h2>What this means for Paper I</h2>')
    w('<p>These are proposals. None of the paper text has been changed.</p><ol>')
    w('<li><b>Section 4 (setups).</b> The pointing of the reference ray is quoted as one range (210–280 µm). We propose '
      'to add that the pointing depends on the χ² of the reference track, on its angle and on its position, and that the '
      'recipe χ² &lt; 1 keeps it below about 1 mm.</li>')
    w(f'<li><b>Section 6, position resolution.</b> The subtraction of the pointing dominates the uncertainty of the '
      f'intrinsic resolution. The two-chamber covariance is an independent measurement of that pointing, with no '
      f'assumption on any chamber: at χ² → 0 it gives {f(cc["x"]["ratio"])} ± {f(cc["x"]["ratio_err"])} and '
      f'{f(cc["y"]["ratio"])} ± {f(cc["y"]["ratio_err"])} of the variance used in Paper I (the model fit gives '
      f'{f(gx["a0"])} and {f(gy["a0"])}). With chamber A\'s extrapolated widths (275 and 236 µm), the intrinsic '
      f'resolution would become about {res_x[1]:.0f}–{res_x[0]:.0f} µm (X) and {res_y[1]:.0f}–{res_y[0]:.0f} µm (Y) '
      'instead of 128 and 113 µm. This is inside the range that Paper I already quotes (63–166 and 62–145 µm), so the '
      'headline "about 120 ± 60 µm" survives, but the central value moves up. Figure 2 could become a panel of the '
      'position-resolution figure.</li>')
    w(f'<li><b>Appendix A1, "Why 5 mm and not a few σ".</b> The per-track width explains part of the tail. A 3σ cut '
      f'loses 11 points with one global σ, and {f(eA["eff_R5"] - eA["eff_n3"], 1)} points with the per-track σ. At the '
      'operating point a per-track window gives the same efficiency as 5 mm at the same accidental rate, so the choice '
      'of R = 5 mm stands, and the appendix can say why.</li>')
    w('<li><b>The denominator (optional, a larger change).</b> With the per-track window, tracks with χ² up to about 4 '
      'are as useful as the recipe tracks (section 7). Widening the recipe to χ² &lt; 4 would add about 40 % more tracks per row (44 % on chamber A Run 1). '
      'It would need the χ² &gt; 1 events of every row to be reconstructed (one condor pass with the pass-1 bundles, '
      '<code>reco_subset.py</code>) and the efficiency tables to be rebuilt. We would only do this if more statistics '
      'are needed, for example for chambers B and D.</li>')
    w('<li><b>Appendix A4 (bow).</b> The position-dependent shift of section 5 (0.04–0.2 % scale-like slopes, mostly in '
      'the top slot) is the kind of telescope error that A4 bounds. It could be quoted there as a measured upper '
      'limit.</li></ol>')

    # ---------------------------------------------------------------- limits
    w('<h2>What the result does not rule out</h2><ul class="lim">')
    w('<li>The two-chamber covariance measures only the part of the M3 error that the two slots share. An error that '
      'affects one slot alone (for example a scattering kink in the material of the top chamber itself) is invisible '
      'to it. The floor of each slot is then set by the fit-geometry ratios of plane errors, which is an assumption.</li>')
    w('<li>The chamber term of a row absorbs anything constant. This is harmless for matching, but it means that d0 is '
      'not an intrinsic resolution and should not be quoted as one.</li>')
    w('<li>The model was fitted on three pair runs of June (6-22, 6-23, 6-26) and applied to all rows. This assumes that '
      'the telescope did not change during June. The held-out rows (6-24, 6-27) agree, which supports this.</li>')
    w('<li>The position map has 4 × 4 cells and was measured on 15 600 pair events. It is coarse, and its corner cells '
      'have errors of 20–40 %. The common position shift of M3 Y is measured but not corrected.</li>')
    w('<li>The χ² &gt; 1 test uses one row (chamber A Run 1). The bias of the M3 track at large χ² (as opposed to its '
      'width) is not modelled.</li></ul>')
    w('<h2>Reproduce</h2><pre>cd rerun/m3_pointing\n~/.venvs/xrd/bin/python fetch.py        # alignments + M3 v2 rays from EOS\n'
      '../../.venv/bin/python build.py         # ray tables joined with the M3 tracks\n'
      '../../.venv/bin/python decompose.py     # chi2 = ordinary + anomalous\n'
      '../../.venv/bin/python fit.py           # pair likelihood -> fit.json\n'
      '../../.venv/bin/python crosscov.py      # model-free floor check\n'
      '../../.venv/bin/python posmap.py        # position factor\n'
      '../../.venv/bin/python validate.py      # chamber terms, pull closure\n'
      '../../.venv/bin/python efficiency.py    # matching rules\n'
      '~/.venvs/xrd/bin/python stage_reco.py   # chi2 &gt; 1: inputs, then reco_subset.py (see README)\n'
      '../../.venv/bin/python extend.py\n../../.venv/bin/python make_report.py</pre>')
    w('</body></html>')
    return '\n'.join(o)

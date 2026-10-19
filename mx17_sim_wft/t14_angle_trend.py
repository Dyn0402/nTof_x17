#!/usr/bin/env python3
"""
T14-ANG — the angled ladder: how sim-vs-data agreement moves with track angle.

DIAGNOSIS, NOT THE VERDICT. These points carry the CORRECTED noise spec
(noise_det3_satscan_feu07.json) and were produced after the frozen default was
closed. The frozen verdict lives in `t14_compare/` and is untouched by this.

The frozen (vertical) point answered "does the simulation reproduce the data at
one angle" — no in absolute amplitude (x0.53-0.64), roughly yes in shape. This
script answers the question Dylan actually needs for the forward-reco
parameterization: *does the shape agreement hold as the track inclines?* An
inclined track shortens the per-strip drift ladder, so rise time and FWHM are
predicted to fall with angle; whether the simulation falls the same way is the
test that matters more than the absolute gain.

It also does the one thing the per-point reports cannot: check the reconstructed
angle against the KNOWN gun angle. Stage A's gun is a pencil beam in angle
(per-event cluster fits reproduce tan(theta) to 5 decimals), so each point has an
exact truth value -- see TRUTH_LOCAL_DEG below for why they are negative.

    python3 mx17_sim_wft/t14_angle_trend.py \
        --ang-root ~/x17/response_sim/stageB_w2 \
        --out-dir ~/x17/response_sim/stageB_w2/t14_ang_trend
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os

import numpy as np
import pandas as pd

# Truth gun angle per point, in the DETECTOR-LOCAL frame that the strip maps
# and therefore wft live in.
#
# WHY NEGATIVE. Stage A's gun is set in the GEANT frame as
# SetParticleMomentumDirection(tan(theta_x), tan(theta_y), 1), so thx10 is
# +10 deg there, and a per-event fit to the drift-region ClusterTree hits
# reproduces exactly that (dx/dz = +0.17633 = tan 10 deg). But every Stage A
# file's `Meta` tree carries the detector-local mapping
#     origin = (0, 0, 17.64405) mm,  sign = (+1, +1, -1)
# and sign_z = -1 mirrors the drift-depth axis. An angle is dx/dz, so mirroring
# z flips the sign of BOTH view angles alike -- sign_y is +1 and the y axis is
# NOT flipped. Re-fitting the same clusters in the local frame gives the values
# below to 3 decimals.
#
# This is a PREDICTION for what wft should return, not a measurement of what it
# does return: it holds if wft's theta is dx/d(depth) with depth measured from
# the mesh into the drift volume. If wft instead defines depth toward the mesh,
# the reconstructed sign flips back and the difference is a wft-internal axis
# convention with no bearing on the physics. That is exactly what the
# sign-agreement column below measures.
TRUTH_LOCAL_DEG = {
    "th00":  (0.0, 0.0),
    "thx10": (-10.0, 0.0),
    "thx20": (-20.0, 0.0),
    "thy10": (0.0, -10.0),
    "thy20": (0.0, -20.0),
}
ORDER = ["thx20", "thx10", "th00", "thy10", "thy20"]
FAST_RISE_NS = 200.0   # the data's fast-rise population sits near 190 ns


def load_points(ang_root, prefix="t14_angW_"):
    """Every <prefix><point>/ that has finished, in ladder order.

    Default is the FIXED-WINDOW set (`t14_angW_`), not the p5/p95 set: p5/p95
    gives each point a different window width and breaks down entirely at
    thx20 X, where the reco tails span 61 deg and the cut stops being an angle
    match at all. Pass --prefix t14_ang_ to read the p5/p95 set instead.
    """
    pts = {}
    for d in sorted(glob.glob(os.path.join(ang_root, prefix + "*"))):
        name = os.path.basename(d)[len(prefix):]
        sp = os.path.join(d, "t14_summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            s = json.load(f)
        wf = {}
        for leg in ("sim", "data"):
            for v in ("x", "y"):
                p = os.path.join(d, f"wf_{leg}_{v}.parquet")
                if os.path.exists(p):
                    wf[(leg, v)] = pd.read_parquet(p)
        ev = None
        cand = glob.glob(os.path.join(
            ang_root, "w2_rho2M_ang", name, f"events_ang_{name}.parquet"))
        if cand:
            ev = pd.read_parquet(cand[0])
        pts[name] = dict(dir=d, summary=s, wf=wf, ev=ev)
    return {k: pts[k] for k in ORDER if k in pts}


def angle_closure(pts):
    """Reconstructed angle vs the known gun angle, per point and view."""
    rows = []
    for name, p in pts.items():
        ev = p["ev"]
        if ev is None:
            continue
        for v in ("x", "y"):
            truth = TRUTH_LOCAL_DEG.get(name, (np.nan, np.nan))[0 if v == "x" else 1]
            g = ev[ev[f"{v}_ok"] & ev[f"{v}_quality_ok"] & ~ev["spark"]]
            th = g[f"{v}_theta_deg"].dropna()
            if not len(th):
                continue
            med = float(th.median())
            rows.append(dict(
                point=name, view=v, truth_deg=truth, reco_med_deg=med,
                reco_iqr_deg=float(th.quantile(0.75) - th.quantile(0.25)),
                n=int(len(th)),
                # is the reconstruction consistent with the truth as given, or
                # with its mirror? only meaningful away from 0.
                d_same=abs(med - truth), d_flip=abs(med + truth),
                sign_verdict=("n/a (vertical)" if abs(truth) < 1e-6 else
                              "MATCHES truth sign" if abs(med - truth) < abs(med + truth)
                              else "FLIPPED vs truth"),
            ))
    return pd.DataFrame(rows)


def trend_table(pts):
    rows = []
    for name, p in pts.items():
        s = p["summary"]
        for v in ("x", "y"):
            d = s["views"][v]
            truth = TRUTH_LOCAL_DEG.get(name, (np.nan, np.nan))[0 if v == "x" else 1]
            r = dict(
                point=name, view=v, truth_deg=truth,
                theta_win=f"{s['theta_windows_deg'][v][0]:+.1f}..{s['theta_windows_deg'][v][1]:+.1f}",
                n_sim=d["n_sim"], n_data=d["n_data"],
                peak_ratio=d["peak_amp_med"]["ratio"],
                q_ratio=d["q_event_med"]["ratio"],
                qsum_ratio=d["q_sum_reco_tight3deg"]["ratio"],
                rise_sim=d["rise_ns_med"]["sim"], rise_data=d["rise_ns_med"]["data"],
                fwhm_sim=d["fwhm_ns_med"]["sim"], fwhm_data=d["fwhm_ns_med"]["data"],
                nover_sim=d["n_over_med"]["sim"], nover_data=d["n_over_med"]["data"],
                shape_rms=d["shape_rms_frac_of_peak"],
            )
            r["rise_ratio"] = r["rise_sim"] / r["rise_data"]
            r["fwhm_ratio"] = r["fwhm_sim"] / r["fwhm_data"]
            # THE RISE FLOOR, not the median. The vertical point showed the sim
            # has a hard lower limit on rise time (~250 ns) that the data
            # undercuts; a floor is a LOW-QUANTILE feature and the median hides
            # it. This is the discriminating test the ladder was built for:
            # inclining a track shortens the per-strip drift ladder, so if the
            # floor is track geometry the sim's p05 must fall with angle. If
            # the data keeps undercutting the sim at EVERY angle, the floor is
            # in the electronics/impulse response instead, and no amount of
            # geometry fixes it.
            for leg in ("sim", "data"):
                t = p["wf"].get((leg, v))
                rs = t.rise_ns.dropna() if t is not None else pd.Series(dtype=float)
                r[f"fastrise_{leg}"] = (float((rs < FAST_RISE_NS).mean())
                                        if len(rs) else np.nan)
                r[f"rise_p05_{leg}"] = float(rs.quantile(0.05)) if len(rs) else np.nan
                r[f"rise_p25_{leg}"] = float(rs.quantile(0.25)) if len(rs) else np.nan
            r["rise_p05_gap"] = r["rise_p05_sim"] - r["rise_p05_data"]
            rows.append(r)
    return pd.DataFrame(rows)


def make_figures(fig_dir, tr, cl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    figs = []

    def _save(fig, name, cap):
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name), dpi=130)
        plt.close(fig)
        figs.append((name, cap))

    # the ladder is per-view: an x-tilt moves the X view and leaves Y vertical,
    # so plot each view against ITS OWN truth angle and mark which points
    # actually tilt that view.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, v in enumerate(("x", "y")):
        t = tr[tr.view == v].copy()
        t["a"] = t.truth_deg.abs()
        t = t.sort_values("a")
        tilt = t[t.a > 0]
        axes[0, j].plot(t.a, t.peak_ratio, "o-", label="peak amp")
        axes[0, j].plot(t.a, t.q_ratio, "s-", label="event charge")
        axes[0, j].plot(t.a, t.qsum_ratio, "^-", label="reco q_sum")
        axes[0, j].axhline(1.0, color="k", lw=0.8)
        axes[0, j].set_ylim(0, 1.2)
        axes[0, j].set_title(f"{v.upper()} view — sim/data amplitude vs |θ|")
        axes[1, j].plot(t.a, t.rise_sim, "o-", color="tab:red", label="rise sim")
        axes[1, j].plot(t.a, t.rise_data, "o--", color="tab:blue", label="rise data")
        axes[1, j].plot(t.a, t.fwhm_sim, "s-", color="darkred", label="FWHM sim")
        axes[1, j].plot(t.a, t.fwhm_data, "s--", color="navy", label="FWHM data")
        axes[1, j].set_title(f"{v.upper()} view — pulse width vs |θ|")
        axes[1, j].set_ylabel("ns")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xlabel(f"|gun θ_{v}| [deg]")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            if len(tilt) == 0:
                ax.text(0.5, 0.5, "no point tilts this view",
                        transform=ax.transAxes, ha="center", color="#888")
    _save(fig, "trend.png",
          "Top: sim/data amplitude ratios vs the gun's inclination in that "
          "view (1.0 = agreement). Bottom: absolute rise and FWHM, both legs. "
          "An inclined track shortens the per-strip drift ladder, so both "
          "legs should fall with angle — the test is whether they fall "
          "TOGETHER, which is the shape-fidelity question.")

    # the rise FLOOR vs angle — the discriminating figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        t = tr[tr.view == v].copy()
        t["a"] = t.truth_deg.abs()
        t = t.sort_values("a")
        axes[j].plot(t.a, t.rise_p05_sim, "o-", color="tab:red", label="sim p05")
        axes[j].plot(t.a, t.rise_p05_data, "o--", color="tab:blue", label="data p05")
        axes[j].plot(t.a, t.rise_p25_sim, "s-", color="darkred", label="sim p25")
        axes[j].plot(t.a, t.rise_p25_data, "s--", color="navy", label="data p25")
        axes[j].set_title(f"{v.upper()} view — rise-time FLOOR vs |θ|")
        axes[j].set_xlabel(f"|gun θ_{v}| [deg]")
        axes[j].set_ylabel("rise 10–90 % [ns]")
        axes[j].legend(fontsize=8)
        axes[j].grid(alpha=0.3)
    _save(fig, "rise_floor.png",
          "The fastest pulses on each leg (5th and 25th percentile of rise "
          "time) vs inclination. Inclining a track shortens the per-strip "
          "drift ladder, so a GEOMETRIC floor must fall with angle. If the "
          "sim's floor stays put while the data undercuts it at every angle, "
          "the floor lives in the electronics/impulse response and no "
          "geometry change will move it.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, v in enumerate(("x", "y")):
        t = tr[tr.view == v].copy()
        t["a"] = t.truth_deg.abs()
        t = t.sort_values("a")
        axes[j].plot(t.a, t.rise_ratio, "o-", label="rise sim/data")
        axes[j].plot(t.a, t.fwhm_ratio, "s-", label="FWHM sim/data")
        axes[j].plot(t.a, t.shape_rms, "^-", label="avg-waveform shape RMS")
        axes[j].axhline(1.0, color="k", lw=0.8)
        axes[j].set_title(f"{v.upper()} view — shape fidelity vs |θ|")
        axes[j].set_xlabel(f"|gun θ_{v}| [deg]")
        axes[j].legend(fontsize=8)
        axes[j].grid(alpha=0.3)
    _save(fig, "shape_fidelity.png",
          "Shape agreement vs angle. A flat line means the mismatch is an "
          "angle-independent offset the forward model can absorb into a "
          "constant; a sloping line means the angular response itself is "
          "wrong, which a constant cannot fix.")

    if len(cl):
        fig, ax = plt.subplots(figsize=(6, 5))
        for v, mk in (("x", "o"), ("y", "s")):
            c = cl[cl.view == v]
            ax.plot(c.truth_deg, c.reco_med_deg, mk, ms=9, label=f"{v.upper()} view")
        lim = 25
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8, label="reco = truth")
        ax.plot([-lim, lim], [lim, -lim], "r--", lw=0.8, label="reco = −truth (flip)")
        ax.set_xlabel("gun θ, detector-local frame [deg]")
        ax.set_ylabel("reconstructed θ, median [deg]")
        ax.set_title("Angle closure: wft reco vs known gun")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, "angle_closure.png",
              "Reconstructed angle against the gun's angle in the "
              "detector-local frame. Points on the solid line mean wft's θ "
              "convention agrees with the Meta frame; points on the dashed "
              "line mean wft's depth axis runs the other way — a convention "
              "difference, not a physics error.")
    return figs


def write_report(out_dir, figs, tr, cl, pts):
    def tbl(df, fmts):
        h = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
        body = ""
        for _, r in df.iterrows():
            cells = ""
            for c in df.columns:
                f = fmts.get(c, "{}")
                try:
                    cells += f"<td>{f.format(r[c])}</td>"
                except (ValueError, TypeError):
                    cells += f"<td>{html.escape(str(r[c]))}</td>"
            body += f"<tr>{cells}</tr>"
        return f"<table><tr>{h}</tr>{body}</table>"

    missing = [p for p in ORDER if p not in pts]
    banner = ("" if not missing else
              f"<p class='warn'><b>Partial ladder.</b> Points not yet "
              f"included: {', '.join(missing)}. Trends below are drawn from "
              f"the points that have landed.</p>")

    # verdict text on shape-vs-angle, computed rather than asserted
    lines = []
    for v in ("x", "y"):
        t = tr[(tr.view == v)].copy()
        t["a"] = t.truth_deg.abs()
        t = t.sort_values("a")
        tilt = t[t.a > 0]
        if len(tilt) < 1 or len(t) < 2:
            lines.append(f"{v.upper()}: not enough tilted points yet.")
            continue
        d_amp = t.peak_ratio.max() - t.peak_ratio.min()
        d_rs = t.rise_ratio.max() - t.rise_ratio.min()
        lines.append(
            f"{v.upper()} view over |θ| = {t.a.min():.0f}–{t.a.max():.0f}°: "
            f"amplitude ratio spans {t.peak_ratio.min():.2f}–{t.peak_ratio.max():.2f} "
            f"(Δ {d_amp:.2f}); rise ratio spans {t.rise_ratio.min():.2f}–"
            f"{t.rise_ratio.max():.2f} (Δ {d_rs:.2f}).")
    sign_txt = ""
    if len(cl):
        c = cl[cl.truth_deg.abs() > 1e-6]
        if len(c):
            flipped = (c.sign_verdict == "FLIPPED vs truth").sum()
            sign_txt = (
                f"<p><b>Angle closure:</b> {len(c)} tilted view-points; "
                f"{flipped} reconstruct with the OPPOSITE sign to the "
                f"detector-local gun angle. "
                + ("wft's θ therefore agrees with the Meta frame convention."
                   if flipped == 0 else
                   "wft's depth axis runs opposite to Meta's sign_z, so its θ "
                   "is the mirror of the local-frame gun angle. This is a "
                   "convention mapping, not a physics error — but any signed "
                   "angle quoted from wft must state which it is.") + "</p>")

    fig_html = "\n".join(
        f'<figure><img src="figures/{n}" style="max-width:100%">'
        f"<figcaption>{html.escape(c)}</figcaption></figure>"
        for n, c in figs)

    fmt_tr = {c: "{:.2f}" for c in ("peak_ratio", "q_ratio", "qsum_ratio",
                                    "rise_ratio", "fwhm_ratio", "nover_sim",
                                    "nover_data")}
    fmt_tr.update({c: "{:.0f}" for c in ("rise_sim", "rise_data", "fwhm_sim",
                                         "fwhm_data", "truth_deg",
                                         "rise_p05_sim", "rise_p05_data",
                                         "rise_p25_sim", "rise_p25_data",
                                         "rise_p05_gap")})
    fmt_tr.update({c: "{:.3f}" for c in ("shape_rms", "fastrise_sim",
                                         "fastrise_data")})
    fmt_cl = {c: "{:.2f}" for c in ("truth_deg", "reco_med_deg", "reco_iqr_deg",
                                    "d_same", "d_flip")}

    body = f"""<meta charset="utf-8">
<title>T14-ANG — sim vs data across the angled ladder (DIAGNOSIS)</title>
<style>
 body {{ font-family: sans-serif; max-width: 1050px; margin: 2em auto; }}
 table {{ border-collapse: collapse; margin: 1em 0; font-size: .92em; }}
 td, th {{ border: 1px solid #999; padding: 3px 8px; text-align: right; }}
 th {{ text-align: left; background: #f0f0f0; }}
 .warn {{ background: #fff6e0; border-left: 5px solid #c60; padding: .6em 1em; }}
 .diag {{ background: #eef3ff; border-left: 6px solid #36c; padding: .6em 1em;
   font-size: 1.05em; }}
 figure {{ margin: 1.5em 0; }} figcaption {{ color: #444; font-size: .92em; }}
</style>
<h1>T14-ANG — does sim/data agreement survive track inclination?</h1>
<p class="diag"><b>DIAGNOSIS, not the verdict.</b> These points carry the
CORRECTED noise spec and were produced after the frozen default closed. The
one-shot verdict — sim ×0.53–0.64 low in amplitude, shapes close — stands
unchanged in <code>t14_compare/</code> and nothing here revises it. What this
adds is the angular dependence, which is what the forward-reco
parameterization actually needs.</p>
{banner}
<h2>What the ladder shows</h2>
<ul>{''.join(f'<li>{html.escape(l)}</li>' for l in lines)}</ul>
{sign_txt}
<h2>Angle closure — reco vs the known gun</h2>
<p>Stage A's gun is a pencil beam in angle: per-event fits to the drift-region
clusters reproduce tan θ to 5 decimals with zero-width IQR. The truth column is
that gun angle expressed in the detector-local frame, where
<code>Meta.sign_z = −1</code> mirrors the drift-depth axis and therefore flips
the sign of both views' angles (<code>sign_y</code> is +1 — the y axis is not
flipped).</p>
{tbl(cl, fmt_cl) if len(cl) else '<p>No reco tables yet.</p>'}
<h2>Per-point comparison</h2>
{tbl(tr, fmt_tr)}
<h2>What could make the amplitude deficit grow with angle?</h2>
<p>A pure gain error — wrong avalanche gain, wrong kernel normalisation, or gas
contamination suppressing gain — is <b>angle-independent</b>: it multiplies
every event by the same factor whatever the track does. So the observed trend
is a constraint that a gain-only explanation does not satisfy on its own, and
it is worth being explicit about what survives.</p>
<p><b>Data saturation is excluded, and it excludes itself.</b> Data clips at
3550 ADC and the simulation barely saturates, so one could worry the vertical
ratio is flattered by clipped data peaks. The saturation fractions rule it out:
in the X view data saturation is essentially flat with angle (0.263 → 0.241 →
0.241), and in the Y view it <i>rises</i> (0.040 → 0.156 → 0.147). More
clipping means a more under-estimated data peak, which pushes the ratio UP —
so in Y saturation works against the observed fall, and the true degradation is
if anything stronger than measured. Both views nonetheless fall together.</p>
<p><b>Transverse over-spreading argues the wrong way too.</b> Peak amplitude
goes roughly as charge / width, and the simulation runs a near-constant 4–5
strips wider than data at every angle. As the track inclines, the data's own
width grows, so the width <i>ratio</i> w_data/w_sim rises (6/11 → 10/14 →
15/19) and would push the peak ratio UP with angle. It falls instead, so the
collected-charge term must fall faster still — and indeed the event-charge
ratio falls independently (0.612 → 0.516 → 0.461 in X).</p>
<p><b>Time structure was the surviving candidate, and it was TESTED rather than
left as an argument.</b> An inclined track spreads each strip's arrival over a
wider range of depths, so the induced current per strip is broader. A response
that is already slow converts that extra spread into peak loss where a faster
one retains it — and the simulation's response is ion-dominated (~90 % of
induced charge in the ion term, reaching half only at ~172 ns), the same excess
that sets its rise floor. The prediction was that removing the ion term should
flatten the amplitude-vs-angle trend.</p>
<p><b>Result: confirmed at low angle, and NOT the whole story.</b>
<code>--no-ions</code> was run at 10° and 20° on the same angled cluster sets
(condor 3919436) and compared against the with-ions ladder at the same angles.
X-view peak ratio:</p>
<table>
<tr><th></th><th>0°</th><th>10°</th><th>20°</th>
<th>slope 0→10</th><th>slope 10→20</th></tr>
<tr><th>with ions</th><td>0.553</td><td>0.447</td><td>0.378</td>
<td>−0.106</td><td>−0.069</td></tr>
<tr><th>ions removed</th><td>0.632</td><td>0.601</td><td>0.535</td>
<td><b>−0.031</b></td><td><b>−0.066</b></td></tr>
</table>
<p>The <b>0→10° trend is ~70 % ion-driven</b> — the slope collapses from
−0.106 to −0.031 once the ion term is gone, so for low inclination the
mechanism above is confirmed. The <b>10→20° step is untouched</b> (−0.069 vs
−0.066, a 4 % change): whatever drives the amplitude loss beyond 10° is
<b>not</b> the ion term and remains <b>unexplained</b>. It is not saturation
and not transverse over-spreading (both argue the wrong way, above), and it is
not gain alone (gain is angle-independent by construction). This is an open
question, and the 20° end is also the statistically weakest — 814 angle-matched
data events in X, 435 in Y, with reco IQR 2.4–5.2° — so part of the step could
be sampling. Confirming it needs either more 20° statistics or an intermediate
15° point.</p>
<h2>The rise floor — is it geometry or electronics?</h2>
<p>The vertical point showed the simulation has a hard lower limit on rise
time that the data undercuts. A floor is a low-quantile feature, so
<code>rise_p05_*</code> (the fastest 5 % of pulses on each leg) is the number
to watch, not the median; <code>rise_p05_gap</code> is sim − data in ns, and
<code>fastrise_*</code> is the fraction of pulses rising in under
{FAST_RISE_NS:.0f} ns.</p>
<p>This is the one thing the ladder can decide that a single vertical point
cannot. Inclining a track shortens the per-strip drift ladder, so a floor that
is <b>track geometry</b> must fall as |θ| grows. A floor that <b>stays put
while the data keeps undercutting it at every angle</b> is in the electronics
or impulse response, and no geometry change in the forward model will move it
— it would have to be fixed in the shaper.</p>
{fig_html}
<hr><p><small>generated by mx17_sim_wft/t14_angle_trend.py from
{len(pts)} point(s): {', '.join(pts)}</small></p>
"""
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ang-root", default="~/x17/response_sim/stageB_w2")
    ap.add_argument("--out-dir", default="~/x17/response_sim/stageB_w2/t14_ang_trend")
    ap.add_argument("--prefix", default="t14_angW_",
                    help="per-point output dir prefix: t14_angW_ (fixed +-3 "
                         "deg window, the defensible ladder) or t14_ang_ "
                         "(p5/p95 window, width varies per point)")
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.ang_root))
    out = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(out, exist_ok=True)

    pts = load_points(root, a.prefix)
    if not pts:
        print(f"no finished t14_ang_* points under {root}")
        return 1
    print("points:", ", ".join(pts))
    tr = trend_table(pts)
    cl = angle_closure(pts)
    tr.to_csv(os.path.join(out, "trend.csv"), index=False)
    cl.to_csv(os.path.join(out, "angle_closure.csv"), index=False)
    figs = make_figures(os.path.join(out, "figures"), tr, cl)
    write_report(out, figs, tr, cl, pts)
    print(tr.to_string(index=False))
    print()
    print(cl.to_string(index=False))
    print(f"\nreport: {os.path.join(out, 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

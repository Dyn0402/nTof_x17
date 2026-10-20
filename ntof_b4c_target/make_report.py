#!/usr/bin/env python3
"""make_report.py -- report.html for the B4C-target expectation.

    python -m ntof_b4c_target.make_report     # needs OUT/expect.json (expect.py)

Writes OUT/report.html and OUT/figures/.  Every number in the text comes from
expect.json or from the constants below (the data-side census of 2026-10-20,
`act_b4c.py` on lxplus, quoted in README.md).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .expect import OUT

# data side (README.md, "What the data hold"): run_39 + run_40, hits census
DATA = dict(triggers=93314, multi_chamber=2300, pulses=2635, protons=2.175e16,
            run79_multi_per_hour=2570)
COL = {"neutrons": "#2a78d6", "casc": "#eb6834", "ipc": "#1baf7a"}   # dataviz slots 1-3
LABEL = {"neutrons": "beam neutrons (n,α), escapes",
         "casc": "¹⁰B(n,γ) cascade γs (EGAF)",
         "ipc": "¹⁰B(n,γ) internal pairs (Born)"}


def fig_scan(res, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    n = res["constants"]["EXPOSURE_PULSES"]
    for src in ("casc", "ipc"):
        for r, ls in ((50, "-"), (25, "--")):
            s = res[f"scan_r{r}"]
            y = [v * n if v > 0 else float("nan") for v in s[src]]   # no MC events: no point
            ax.plot(s["b"], y, ls, color=COL[src], lw=2,
                    label=f"{LABEL[src]}, Ø{2 * r} mm" if r == 50 else f"  (Ø{2 * r} mm)")
    ax.axvline(0.9, color="#52514e", lw=1, ls=":")
    ax.text(0.88, 2e-3, "production\nplastic 0.9 MIP", fontsize=8, color="#52514e", va="bottom", ha="right")
    ax.axhline(1, color="#9a9890", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3); ax.set_xlim(0.1, 1.5)
    ax.set_xlabel("plastic threshold [MIP]  (SiPM wall fixed at 0.5 MIP)")
    ax.set_ylabel("two-arm (Doubles) events\nin the whole B4C exposure")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(1.0, 1.15, "1 event", fontsize=8, color="#52514e", ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fmt(x):
    return f"{x:.2g}" if x else "0"


def main():
    res = json.loads((OUT / "expect.json").read_text())
    c = res["constants"]; n = c["EXPOSURE_PULSES"]
    (OUT / "figures").mkdir(exist_ok=True)
    fig_scan(res, OUT / "figures" / "doubles_vs_plastic_threshold.png")
    rows = [("sipm_any", "any SiPM bar ≥ 0.5 MIP (wall hits)"),
            ("leg1", "≥ 1 trigger leg (wall ≥ 0.5 & plastic ≥ 0.9 MIP, same arm)"),
            ("leg2", "≥ 2 legs in different arms (Doubles)"),
            ("mm2", "≥ 2 chambers with ≥ 3 keV charged deposit"),
            ("leg1_mm2", "≥ 1 leg and ≥ 2 chambers lit"),
            ("leg2_mm2legs", "2 legs, both leg chambers lit"),
            ("pair_split_leg2mm", "true e⁺e⁻ split over two arms, 2 legs, both chambers lit")]
    tab = []
    for key, desc in rows:
        cells = []
        for r in (50, 25):
            for src in ("neutrons", "casc", "ipc"):
                v = res[f"r{r}"][src][key]
                cells.append(f"<td>{fmt(v[0])}<span class=mc> ({v[2]})</span></td>")
        tot50 = sum(res["r50"][s][key][0] for s in ("neutrons", "casc", "ipc"))
        cells.append(f"<td><b>{fmt(tot50 * n)}</b></td>")
        tab.append(f"<tr><th>{desc}</th>{''.join(cells)}</tr>")
    pair = res["r50"]["ipc"]["pair_split_leg2mm"][0] * n
    dbl = sum(res["r50"][s]["leg2"][0] for s in ("neutrons", "casc", "ipc")) * n
    legs = sum(res["r50"][s]["leg1"][0] for s in ("neutrons", "casc", "ipc"))
    ipc_made = c["CAPT_PER_PULSE"] * sum(i / 100 * a for i, a in c["IPC_LINES"].values())
    loose = res["scan_r50"]; i01 = loose["b"].index(0.1)
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>B4C Target Expectation</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--rule:#e4e3df;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--rule:#3a3936;--acc:#3987e5}}}}
:root[data-theme=dark]{{--bg:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--rule:#3a3936;--acc:#3987e5}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px}}
h1{{font-size:1.5em;margin:.2em 0}} h2{{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid var(--rule)}}
.verdict{{border-left:4px solid var(--acc);padding:.5em 1em;background:color-mix(in srgb,var(--acc) 8%,transparent)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
td,th{{padding:4px 8px;border-bottom:1px solid var(--rule);text-align:right}} th{{text-align:left;font-weight:500}}
.mc{{color:var(--mut);font-size:11px}} img{{max-width:100%}} .cap{{color:var(--mut);font-size:13px}}
.tiles{{display:flex;flex-wrap:wrap;gap:12px}} .tile{{border:1px solid var(--rule);border-radius:8px;padding:8px 14px;min-width:170px}}
.tile b{{font-size:1.6em;display:block}} code{{font-size:12.5px}}
</style></head><body>
<h1>The July B4C target: what Geant4 says we should have seen</h1>
<p class=cap>n_TOF EAR2, run_39 + run_40 (15 July 2026, 05:06–09:59), 2.5 mm B4C disc in place of the ³He capsule.
Generated by <code>ntof_b4c_target/make_report.py</code>.</p>

<div class=verdict><b>Verdict: the B4C night cannot give pairs.</b> Over the whole exposure
({DATA['pulses']} pulses, {DATA['protons']:.2e} protons) Geant4 expects <b>{pair:.2f}</b>
fully triggered, two-chamber internal-pair events, and about <b>{dbl:.1f}</b> two-arm (Doubles) coincidences
of any origin at the production thresholds. Even with the plastic threshold dropped to 0.1 MIP the
internal pairs give <b>{loose['ipc'][i01] * n:.1f}</b> Doubles. The target <em>makes</em> plenty of pairs —
{ipc_made:.1f} internal pairs per nominal pulse, ~10⁴ times the ³He capsule in the thermal gate — but at
≤ 11.45 MeV their leptons do not reach the plastics, which sit behind the liquid-scintillator vessels. The
~{DATA['multi_chamber']} multi-chamber events in the data are therefore not pair physics.</div>

<div class=tiles>
<div class=tile><b>{pair:.2f}</b>expected pair events (exposure)</div>
<div class=tile><b>{dbl:.1f}</b>expected Doubles (exposure)</div>
<div class=tile><b>{legs:.1f}</b>physics trigger legs / pulse</div>
<div class=tile><b>{DATA['triggers'] / DATA['pulses']:.0f}</b>recorded triggers / pulse (data)</div>
</div>

<h2>What was simulated</h2>
<ul>
<li><b>Geometry.</b> <code>MX17_Full_Geant</code> branch <code>b4c-slab</code> (July as-built arms), the capsule replaced by a
G4_BORON_CARBIDE disc (2.52 g/cm³), 2.5 mm thick, normal to the beam at the target position. Its diameter is
<b>not recorded anywhere</b>, so it is run at Ø100 mm and Ø50 mm; the EAR2 beam ends at r ≈ 22 mm, so both
intercept all of it ({c['F_ABS'] * 100:.1f} % absorbed, median depth 0.12 mm).</li>
<li><b>Beam.</b> Phase-4 evaluated EAR2 flux and the 2D profile at 19.5 m, 2.2 meV–2 eV (time of flight 1–30 ms,
the scintillator-trigger gate): {c['N_PULSE']:.3g} neutrons per nominal 7×10¹² pulse. The exposure is
{n:.0f} nominal-pulse equivalents (8.25×10¹² protons per pulse on the day).</li>
<li><b>¹⁰B(n,γ).</b> σ<sub>γ</sub> = 0.29 b against 3837 b of (n,α): {c['CAPT_PER_PULSE']:.0f} captures per
pulse. Geant4's ParticleHP samples the ¹¹B lines independently (cascades summing to 20–50 MeV), and its
photon-evaporation mode uses a statistical branching unlike the measured one, so the cascade is
<b>generated from the EGAF thermal-capture scheme</b> (7.007+4.444 54 %, 4.711+6.740 17 %, 4.711+2.297+4.444 8 %,
2.533+8.917 13 %, 11.448 4.6 %) at the slab's capture vertices. Those events are removed from the neutron run.</li>
<li><b>Internal pairs.</b> Geant4 has none. Born IPC (<code>ipc_born</code>) for the seven strongest lines,
coefficients 1.3–3.7×10⁻³, i.e. 4.1×10⁻³ pairs per capture; the 6.74 MeV E2 line is generated with M1
kinematics.</li>
<li><b>Trigger.</b> As production: SiPM bar ≥ 0.5 MIP (0.475 MeV) and plastic ≥ 0.9 MIP (3.349 MeV) in one arm
make a leg; Doubles = legs in two arms. A chamber is "lit" by ≥ 3 keV of charged deposit in its drift gas,
within 200 ns of the event's first hit.</li>
</ul>

<h2>Rates per nominal pulse</h2>
<p class=cap>Weighted rate per 7×10¹² pulse; MC events in brackets. Last column: all sources, Ø100 mm, summed over the exposure.</p>
<div class=wrap><table><tr><th></th><th colspan=3>Ø100 mm</th><th colspan=3>Ø50 mm</th><th></th></tr>
<tr><th>class</th><th>neutrons</th><th>cascade γ</th><th>IPC</th><th>neutrons</th><th>cascade γ</th><th>IPC</th><th>exposure</th></tr>
{''.join(tab)}</table></div>

<h2>Figure</h2>
<img src="figures/doubles_vs_plastic_threshold.png" alt="Doubles over the exposure vs plastic threshold">
<p class=cap>Two-arm events expected in the whole B4C exposure against the plastic threshold, SiPM wall at 0.5 MIP.
Solid Ø100 mm, dashed Ø50 mm. The beam-neutron source gives none at any threshold (no MC events).</p>

<h2>What the data hold, for comparison</h2>
<p>{DATA['triggers']} scintillator triggers (Singles OR Doubles, 32 × 60 ns), {DATA['triggers'] / DATA['pulses']:.0f}
per pulse recorded against {legs:.1f} physics legs per pulse from the simulation: almost all triggers are not
B4C capture physics (accidental wall × plastic coincidences at a very high wall rate, room background, cosmics).
The hits census finds ~{DATA['multi_chamber']} events with x+y activity in two or more chambers, about one
production hour's worth (run_79: ~{DATA['run79_multi_per_hour']} per hour).</p>

<h2>What this does not rule out</h2>
<ul>
<li><b>The chamber occupancy does not match.</b> The simulation has each chamber lit ~1900 times per pulse,
mostly by Compton electrons from the 478 keV ⁷Li line (and Geant4 emits that line in only ~51 % of (n,α)
instead of 94 %), which would put activity in two or more chambers in nearly every 1.92 µs readout. In the data
2.5 % of triggers have two or more chambers active, the same as production run_79 (2.4 %). The data criterion (≥ 2
strips above 500 ADC in both views) is much harder than a 3 keV deposit and was not matched, so this is open. It may
also mean the disc was small, thin or off the beam: the May test found the B4C off axis.</li>
<li><b>Unknowns of the target itself:</b> diameter, density (sintered 2.52 g/cm³ assumed; a polymer-bound sheet
would be lighter), position along the beam, and the "2.5 mm, I think?" thickness. None changes the verdict: the
pairs are lost on the plastics, not at the target.</li>
<li><b>Not simulated:</b> room-return and epithermal neutrons, the flash and 0–1 ms, accidental coincidences
between captures, ¹¹B(n,γ) (3.37 MeV) and the ¹²B β decay, the run_39 threshold changes (wall thresholds were
re-tuned through the night and the plastics only rebalanced at 09:04).</li>
</ul>
</body></html>"""
    (OUT / "report.html").write_text(html)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()

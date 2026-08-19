"""How the acceptance comb was squeezed, 20–27 July.

DREAM cannot be triggered while it is reading out, so the accepted triggers in
the post-flash gate are not a smooth sampling of the neutron arrival time — they
are a *comb*: a burst of events, then a dead gap while the SCA drains, then the
next burst.  Two weeks of DAQ work went into making that comb both **denser**
(more events per beam pulse) and **flatter** (a more even sampling of the
neutron energy spectrum, which is what the physics actually wants).

This figure shows five configuration epochs on the house `ipc_vs_runs` format —
the simulated in-gate IPC production spectrum as the light-blue field behind,
the *measured* recorded triggers in front, both against time since the gamma
flash.  Every panel is real recorded DREAM data, extracted identically.

The epochs, and the one lever that changed at each:

| epoch  | date      | what changed                                        |
|--------|-----------|-----------------------------------------------------|
| run_61 | 20–21 Jul | the starting point: 1 GbE, IPD 90, 64 samples        |
| run_67 | 22–23 Jul | 10 GbE card + switch; IPD 90 → 5; 32 samples; window |
|        |           | opening moved from ~5 ms to 1 ms after the flash     |
| run_77 | 26 Jul    | read clock 16.7 → 25 MHz (measured 23 Jul)           |
| run_79 | 26–27 Jul | 20 samples / latency 27 from the run_78 latency scan;|
|        |           | trigger-FIFO watermark forced to Hwm 2 / Lwm 1       |
| run_86 | 27 Jul →  | watermark to Hwm 1 / Lwm 0 — the production point     |

**Provenance.**  The per-epoch histograms are flash-anchored time-since-flash
distributions of FEU 01, in 0.05 ms bins, produced by
`data/comb/extract_comb.py` against `/eos/experiment/ntof/data/x17/july_beam`.
The algorithm is copied verbatim from the DAQ repo's
`projections/ipc_yield.py:extract_subrun`, and the run_79 / run_86 caches *are*
that script's own output, so all five epochs are anchored the same way: on the
gamma flash itself (tagged by ADC saturation), never on "the first event we
happened to record".  Spills with no captured flash are dropped.  The `.npz`
files are committed because the extraction needs EOS.

The IPC curve is `ipc_ingate_spectrum.npz` from `MX17_Full_Geant`, taken
verbatim — the same input every other flash-comb figure in the campaign plots,
so this one cannot drift away from them.

**Uniformity metric.**  Coefficient of variation of the recorded trigger yield
across 1–10 ms in 0.1 ms bins, exactly as `projections/run82_comb.py` defines
it.  The bin width is not a free choice: coarser bins are wider than the comb's
gaps and hide the very structure being measured.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA = Path(__file__).resolve().parent / "data/comb"

# --- what the figure is measured over ---------------------------------------
TPLOT = (0.0, 30.0)   # the 30 ms thermal window; the IPC spectrum ends at 31.6
GATE_MS = 1.0         # below this the N93B veto kills everything (final config)
BIN_MS = 0.1          # also the CV bin — see the module docstring
COMB_BAND = (1.0, 10.0)
STARVED_FRAC = 0.25   # a bin below this fraction of the band mean is "starved"
YCAP = 16.0           # shared trigger scale; taller bins get an off-scale caret

# --- house palette, from nTof_x17_DAQ/projections/run82_comb.py --------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
IPC_LINE, IPC_FILL = "#5598e7", "#9ec5f4"
DATA_COL, GOOD = "#c0392b", "#0ca30c"

EPOCHS = [
    dict(
        file="run_61_tsf.npz", run="run_61", when="20–21 Jul",
        head="the starting point",
        change="1 GbE  ·  IPD 90  ·  64 samples × 60 ns  ·  30 ms N93B window",
    ),
    dict(
        file="run_67_tsf.npz", run="run_67", when="22–23 Jul",
        head="10 GbE + inter-packet delay",
        change="new PCIe card and area switch (22 Jul)  ·  IPD 90 → 5  ·  "
               "32 samples  ·  window opening 5 ms → 1 ms",
    ),
    dict(
        file="run_77_tsf.npz", run="run_77", when="26 Jul",
        head="read clock 25 MHz",
        change="read clock 16.7 → 25 MHz (measured 23 Jul; DREAM is rated 20)",
    ),
    dict(
        file="run_79_tsf.npz", run="run_79", when="26–27 Jul",
        head="measured read-out window",
        change="20 samples, latency 27 from the run_78 scan  ·  "
               "FIFO watermark Hwm 2 / Lwm 1",
    ),
    dict(
        file="run_86_tsf.npz", run="run_86", when="27 Jul →",
        head="watermark — the production point",
        change="FEU watermark forced to Hwm 1 / Lwm 0",
        production=True,
    ),
]


# ---------------------------------------------------------------- inputs
def _load_ipc():
    Z = np.load(DATA / "ipc_ingate_spectrum.npz", allow_pickle=True)
    return dict(
        t=Z["t_ms"], dndt=Z["dNdt_ipc_per_pulse_per_ms"],
        bt=0.5 * (Z["bin_t_lo"] + Z["bin_t_hi"]),
        bd=Z["bin_ipc_per_pulse"] / (Z["bin_t_hi"] - Z["bin_t_lo"]),
        per_pulse=float(Z["ipc_per_pulse_ingate"]),
        per_day=float(Z["ipc_per_day_ingate"]),
    )


def _rebin(counts, edges, n_pulse, bin_ms=BIN_MS):
    """Fine histogram -> (centres, triggers/pulse/bin) at ``bin_ms``."""
    g = int(round(bin_ms / (edges[1] - edges[0])))
    n = (counts.size // g) * g
    y = counts[:n].astype(float).reshape(-1, g).sum(axis=1) / max(n_pulse, 1)
    c = edges[0] + bin_ms * (np.arange(y.size) + 0.5)
    return c, y


def _metrics(counts, edges, n_pulse):
    """Yield and uniformity, on run82_comb.py's definitions."""
    c, y = _rebin(counts, edges, n_pulse)
    band = y[(c >= COMB_BAND[0]) & (c < COMB_BAND[1])]
    fine_c = 0.5 * (edges[:-1] + edges[1:])
    per = counts.astype(float) / max(n_pulse, 1)

    def window(lo, hi):
        return float(per[(fine_c >= lo) & (fine_c < hi)].sum())

    # where the read-out actually starts letting triggers through
    live = np.where(y > 0)[0]
    first = float(c[live[0]]) if live.size else float("nan")
    return dict(
        cv=float(band.std() / band.mean()) if band.mean() else float("nan"),
        starved=float((band < STARVED_FRAC * band.mean()).mean()),
        per_pulse_1_10=window(*COMB_BAND),
        per_pulse_1_30=window(1.0, 30.0),
        per_pulse_total=float(per.sum()),
        first_trigger_ms=first,
    )


def _load_epoch(ep):
    Z = np.load(DATA / ep["file"], allow_pickle=True)
    n_pulse = int(Z["n_spill_flash"])
    counts, edges = Z["counts"], Z["edges"]
    d = dict(ep)
    d.update(_metrics(counts, edges, n_pulse))
    d["pulses"] = n_pulse
    d["events"] = int(Z["n_events"])
    d["subruns"] = [str(s) for s in Z["subruns"]]
    d["curve"] = _rebin(counts, edges, n_pulse)
    return d


# ---------------------------------------------------------------- drawing
def _panel(ax, ep, ipc, ipc_max):
    axr = ax.twinx()
    for a in (ax, axr):
        a.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    # the flash-blind / gate-closed region, and the band the CV is measured over
    ax.axvspan(TPLOT[0], GATE_MS, color=MUTED, alpha=0.28, hatch="//",
               linewidth=0, zorder=1)
    ax.axvspan(*COMB_BAND, color=MUTED, alpha=0.07, linewidth=0, zorder=0)

    # the simulated spectrum, behind
    ax.fill_between(ipc["t"], 0, ipc["dndt"] * 1e6, color=IPC_FILL, alpha=0.75,
                    linewidth=0, zorder=2)
    ax.plot(ipc["t"], ipc["dndt"] * 1e6, color=IPC_LINE, linewidth=1.6, zorder=3)
    ax.plot(ipc["bt"], ipc["bd"] * 1e6, "o", ms=2.6, color="#d62728", zorder=4)
    ax.set_ylim(0, ipc_max * 1.72)
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(axis="y", colors=IPC_LINE, labelsize=8, length=3)
    ax.set_ylabel("IPC / pulse / ms\n[$\\times10^{-6}$]", color=IPC_LINE,
                  fontsize=8)

    # the thermal peak, which is the whole reason uniformity here matters
    m = ipc["t"] > 3.5
    tpk = float(ipc["t"][m][np.argmax(ipc["dndt"][m])])
    ax.axvline(tpk, color="#b1301a", ls=":", lw=1.1, zorder=6)

    # what was actually recorded, in front
    c, y = ep["curve"]
    sel = (c >= TPLOT[0]) & (c <= TPLOT[1])
    axr.step(c[sel], np.minimum(y[sel] / BIN_MS, YCAP), where="mid",
             color=DATA_COL, linewidth=0.9, zorder=5)
    axr.set_ylim(0, YCAP * 1.72)
    axr.set_yticks([0, 5, 10, 15])
    axr.tick_params(axis="y", colors=DATA_COL, labelsize=8, length=3)
    axr.set_ylabel("triggers / pulse / ms", color=DATA_COL, fontsize=8)

    over = sel & (y / BIN_MS > YCAP)
    if over.any():
        i = int(np.argmax(np.where(over, y, 0)))
        axr.annotate(f"{y[i] / BIN_MS:.0f} / ms", xy=(c[i], YCAP),
                     xytext=(13, -13), textcoords="offset points", fontsize=7.5,
                     color=DATA_COL, ha="left", va="top", zorder=9,
                     arrowprops=dict(arrowstyle="-|>", color=DATA_COL, lw=0.8))

    win = ("production" if ep.get("production") else None)
    label = (f"{ep['run']}   ·   {ep['when']}   ·   {ep['head']}\n"
             f"{ep['change']}\n"
             f"{ep['per_pulse_1_30']:.0f} triggers / pulse in 1–30 ms   ·   "
             f"CV(1–10 ms) = {ep['cv']:.2f}   ·   "
             f"{ep['starved'] * 100:.0f} % of the band starved")
    ax.text(0.008, 0.97, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.6, color=INK2, linespacing=1.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=SURFACE,
                      edgecolor=GOOD if win else AXIS,
                      linewidth=1.7 if win else 0.8, alpha=0.95))
    if win:
        ax.text(0.995, 0.97, "the production point", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, color=GOOD,
                fontweight="bold", zorder=9)

    ax.set_xlim(*TPLOT)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    for s in ("top",):
        ax.spines[s].set_visible(False)
        axr.spines[s].set_visible(False)
    ax.spines["left"].set_color(IPC_LINE)
    axr.spines["right"].set_color(DATA_COL)
    axr.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=3)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK2)
    return tpk


def comb_evolution(out: Path) -> dict:
    """Five configuration epochs of the acceptance comb, on the IPC spectrum.

    Writes the PNG to ``out`` and returns the numbers the prose quotes.
    """
    ipc = _load_ipc()
    eps = [_load_epoch(e) for e in EPOCHS]
    ipc_max = float((ipc["dndt"] * 1e6).max())

    n = len(eps)
    fig, axes = plt.subplots(n, 1, figsize=(12.4, 2.05 * n + 1.3), sharex=True,
                             gridspec_kw={"hspace": 0.14})
    fig.patch.set_facecolor(SURFACE)

    tpk = None
    for ax, ep in zip(axes, eps):
        tpk = _panel(ax, ep, ipc, ipc_max)

    axes[-1].annotate(f"IPC thermal peak\n{tpk:.1f} ms  (E ≈ 71 meV)",
                      xy=(tpk, ipc_max * 0.99), xytext=(11.0, ipc_max * 0.98),
                      fontsize=8.5, color="#b1301a", fontweight="bold",
                      zorder=10, linespacing=1.4, va="center",
                      arrowprops=dict(arrowstyle="->", color="#b1301a", lw=0.9))
    axes[-1].set_xlabel(
        "neutron arrival time  t  [ms]      (t = 0 is the gamma flash)",
        color=INK2, fontsize=10.5)

    first, last = eps[0], eps[-1]
    axes[0].set_title(
        "Squeezing the acceptance comb — five configuration epochs, "
        "20 to 27 July",
        color=INK, fontsize=14, fontweight="bold", loc="left", pad=44)
    axes[0].text(
        0, 1.085,
        f"light blue = simulated in-gate IPC production (reweighted Geant4)  ·  "
        f"dark red = recorded DREAM triggers, {BIN_MS * 1000:.0f} µs bins, "
        f"flash-anchored  ·  shared axes across panels\n"
        f"{first['per_pulse_1_30']:.0f} → {last['per_pulse_1_30']:.0f} triggers "
        f"per pulse in 1–30 ms (×{last['per_pulse_1_30'] / first['per_pulse_1_30']:.0f}) "
        f"while the 1–10 ms uniformity went from CV {first['cv']:.1f} to "
        f"{last['cv']:.2f}",
        transform=axes[0].transAxes, color=MUTED, fontsize=9.3, va="bottom",
        linespacing=1.6)

    fig.text(0.008, -0.012,
             f"Hatched = gate closed / flash-blind (t < {GATE_MS:g} ms).  Grey band = "
             f"the 1–10 ms region the CV is measured over, in {BIN_MS * 1000:.0f} µs "
             f"bins — coarser bins are wider\nthan the comb's gaps and hide it.  "
             f"Starved = bins below {STARVED_FRAC:.0%} of the band mean.  Both axes "
             f"are shared across panels; the trigger axis is clipped at "
             f"{YCAP:g}/ms,\nwith taller bins labelled.  Every panel is recorded "
             f"DREAM data, FEU 01, flash-anchored on the saturated flash event.",
             color=MUTED, fontsize=8, linespacing=1.6, va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    return {
        "epochs": [
            {k: ep[k] for k in
             ("run", "when", "head", "change", "cv", "starved",
              "per_pulse_1_10", "per_pulse_1_30", "per_pulse_total",
              "first_trigger_ms", "pulses", "events")}
            for ep in eps
        ],
        "first": first["run"],
        "last": last["run"],
        "rate_gain": last["per_pulse_1_30"] / first["per_pulse_1_30"],
        "cv_gain": first["cv"] / last["cv"],
        "cv_first": first["cv"],
        "cv_last": last["cv"],
        "per_pulse_first": first["per_pulse_1_30"],
        "per_pulse_last": last["per_pulse_1_30"],
        "peak_per_pulse": max(e["per_pulse_1_30"] for e in eps),
        "peak_run": max(eps, key=lambda e: e["per_pulse_1_30"])["run"],
        "ipc_per_pulse": ipc["per_pulse"],
        "ipc_per_day": ipc["per_day"],
        "thermal_peak_ms": tpk,
    }


if __name__ == "__main__":
    import json

    here = Path(__file__).resolve().parent
    s = comb_evolution(here / "figures/comb_evolution.png")
    print(json.dumps(s, indent=2, default=float))

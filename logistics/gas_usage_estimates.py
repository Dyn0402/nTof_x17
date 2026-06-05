#!/usr/bin/env python3
"""
Gas usage estimates and bottle replacement schedule simulation for nTof_x17.
"""

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory


PALETTE = {
    'flow':         '#1A6FA0',
    'gas':          '#D4570A',
    'warning_thr':  '#B07D10',
    'switch_thr':   '#8B1A1A',
    'band':         '#F9D8A0',
    'change_line':  '#2C3E50',
    'stock':        '#2C6E49',
    'bg':           '#F7F7F7',
    'axes_bg':      '#FFFFFF',
    'grid':         '#CCCCCC',
}


def main():
    start_date        = datetime.datetime(2026, 6, 29, 8, 0)
    bottle_capacity_l = 2150.0
    warning_pct       = 0.10
    switch_pct        = 0.05

    phases = [
        {'duration_hours': 8,       'rate': 20.0},
        {'duration_hours': 48,      'rate': 5.0},
        {'duration_hours': 40 * 24, 'rate': 3.0},
    ]

    # Bottle delivery schedule (when each full bottle arrives on site)
    deliveries = [
        datetime.datetime(2026, 6, 26),
        datetime.datetime(2026, 7, 20),
        # datetime.datetime(2026, 8, 2),
    ]

    plt_show = True

    df, warning_bands, bottle_changes = run_simulation(
        start_date, phases, bottle_capacity_l, warning_pct, switch_pct,
    )
    plot_results(
        df, warning_bands, bottle_changes,
        bottle_capacity_l, warning_pct, switch_pct,
        deliveries=deliveries,
        save_path='gas_usage_plot_updated.png',
        plt_show=plt_show
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(start_date, phases, bottle_capacity_l, warning_pct, switch_pct):
    """
    Simulate hourly gas consumption through the given phases.

    Returns
    -------
    df : pd.DataFrame  – hourly history
    warning_bands : list of (start, end) datetimes
    bottle_changes : list of (change_time, days_this_bottle_lasted)
    """
    warning_vol = bottle_capacity_l * warning_pct
    switch_vol  = bottle_capacity_l * switch_pct

    current_time  = start_date
    bottle_id     = 1
    remaining_l   = bottle_capacity_l
    total_l       = 0.0
    bottle_start  = start_date

    history        = []
    warning_bands  = []
    bottle_changes = []
    warn_start     = None

    for phase in phases:
        rate = phase['rate']
        for _ in range(int(phase['duration_hours'])):

            # --- Bottle switch (5% threshold) ---
            if remaining_l <= switch_vol:
                if warn_start is not None:
                    warning_bands.append((warn_start, current_time))
                    warn_start = None
                days_lasted = (current_time - bottle_start).total_seconds() / 86400
                bottle_changes.append((current_time, days_lasted))
                bottle_id   += 1
                remaining_l  = bottle_capacity_l
                bottle_start = current_time

            # --- Warning flag (10% threshold) ---
            if remaining_l <= warning_vol and warn_start is None:
                warn_start = current_time

            remaining_l -= rate
            total_l     += rate

            history.append({
                'timestamp':        current_time,
                'flow_rate_lh':     rate,
                'bottle_id':        bottle_id,
                'bottle_remaining': max(0.0, remaining_l),
                'total_consumed':   total_l,
            })
            current_time += datetime.timedelta(hours=1)

    return pd.DataFrame(history), warning_bands, bottle_changes


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        10,
        'axes.titlesize':   14,
        'axes.titleweight': 'bold',
        'axes.titlepad':    12,
        'axes.labelsize':   10,
        'axes.spines.top':  False,
        'axes.grid':        True,
        'grid.linestyle':   ':',
        'grid.alpha':       0.55,
        'grid.color':       PALETTE['grid'],
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor':   PALETTE['axes_bg'],
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
    })


def _annotate_bottle_changes(ax_label, ax_top, bottle_changes):
    """
    Draw vertical dashed change lines on both panels.
    Rotated text labels (date + duration) go on ax_label, left of each line.
    """
    trans = blended_transform_factory(ax_label.transData, ax_label.transAxes)

    for change_time, days_lasted in bottle_changes:
        for ax in (ax_label, ax_top):
            ax.axvline(
                x=change_time,
                color=PALETTE['change_line'],
                linestyle='--', linewidth=1.2,
                alpha=0.60, zorder=5,
            )
        label = f"{change_time.strftime('%b %d  %H:%M')}  —  {days_lasted:.1f} days"
        ax_label.text(
            change_time, 0.96, label,
            transform=trans,
            rotation=90,
            ha='right', va='top',
            fontsize=8,
            color=PALETTE['change_line'],
            zorder=6,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.75),
        )


def _annotate_flow_rate_labels(ax, df):
    """
    Add a labelled callout for each distinct flow-rate phase on the flow panel.
    Works for any number of phases; uses position heuristics based on duration.
    """
    # Detect phase boundaries (first row of each new rate)
    mask   = df['flow_rate_lh'] != df['flow_rate_lh'].shift()
    starts = df[mask]['timestamp'].tolist()
    rates  = df[mask]['flow_rate_lh'].tolist()
    ends   = starts[1:] + [df['timestamp'].iloc[-1] + datetime.timedelta(hours=1)]
    phases = list(zip(rates, starts, ends))

    x0        = df['timestamp'].iloc[0]
    total_sec = (df['timestamp'].iloc[-1] - x0).total_seconds()
    y_max     = df['flow_rate_lh'].max() * 1.4   # matches ax ylim top

    arrow_kw = dict(arrowstyle='->', lw=1.1, color=PALETTE['flow'], mutation_scale=10)
    label_kw = dict(
        fontsize=9.5, color=PALETTE['flow'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='none', alpha=0.85),
    )

    for i, (rate, t0, t1) in enumerate(phases):
        t_mid     = t0 + (t1 - t0) / 2
        dur_frac  = (t1 - t0).total_seconds() / total_sec  # fraction of total x span

        if dur_frac < 0.05:
            # Very short phase (<5 % of x range): offset label to the right,
            # long nearly-horizontal arrow pointing back to the step.
            text_x = x0 + datetime.timedelta(seconds=total_sec * 0.09)
            ax.annotate(
                f'{rate:.0f} L/h',
                xy=(t_mid, rate),
                xytext=(text_x, rate),
                ha='left', va='center',
                arrowprops=dict(**arrow_kw, connectionstyle='arc3,rad=0.15'),
                **label_kw,
            )

        elif dur_frac < 0.12:
            # Short phase (5–12 % of x range): place label in the open space
            # between this rate and the one above, curve down to the midpoint.
            rate_above = phases[i - 1][0] if i > 0 else rate * 2
            text_x = x0 + datetime.timedelta(seconds=total_sec * 0.13)
            text_y = rate + (rate_above - rate) * 0.48
            ax.annotate(
                f'{rate:.0f} L/h',
                xy=(t_mid, rate),
                xytext=(text_x, text_y),
                ha='left', va='center',
                arrowprops=dict(**arrow_kw, connectionstyle='arc3,rad=0.28'),
                **label_kw,
            )

        else:
            # Long flat phase: label directly above the line with a short arrow.
            tx = t0 + (t1 - t0) * 0.38
            ty = rate + y_max * 0.20
            ax.annotate(
                f'{rate:.0f} L/h',
                xy=(tx, rate),
                xytext=(tx, ty),
                ha='center', va='bottom',
                arrowprops=dict(**arrow_kw, connectionstyle='arc3,rad=0'),
                **label_kw,
            )


def _plot_bottle_stock(ax_flow, df, bottle_changes, deliveries):
    """
    Overlay a bottle-inventory step function on the right y-axis of the
    flow panel.  Extends the shared x-axis left to include delivery dates
    that predate the simulation start.

    Returns the Line2D artist for inclusion in the shared figure legend.
    """
    color = PALETTE['stock']

    # Sorted event list: (time, +1 for delivery, -1 for switch)
    events = sorted(
        [(t, +1) for t in deliveries] +
        [(t, -1) for t, _ in bottle_changes],
        key=lambda e: e[0],
    )

    # Extend x range left so pre-simulation deliveries are visible
    x_left  = min(df['timestamp'].iloc[0], min(deliveries)) - datetime.timedelta(days=2.5)
    x_right = df['timestamp'].iloc[-1]
    ax_flow.set_xlim(left=x_left)   # sharex propagates to gas panel

    # Build step-function series (duplicate timestamps create the vertical edges)
    ts, ss = [x_left], [0]
    stock = 0
    for t, d in events:
        ts += [t, t]
        ss += [stock, stock + d]
        stock += d
    ts.append(x_right)
    ss.append(stock)

    # Right y-axis for stock count (twinx of the flow panel)
    ax_s = ax_flow.twinx()
    ax_s.set_facecolor('none')
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_color(color)
    ax_s.spines['right'].set_alpha(0.5)

    line_stock, = ax_s.plot(ts, ss, color=color, linewidth=2.0, alpha=0.85,
                             zorder=3, label='Bottles in stock',
                             solid_capstyle='round')
    max_stock = max(ss)
    ax_s.set_ylim(0, max_stock + 0.25)
    ax_s.set_yticks(list(range(max_stock + 1)))
    ax_s.set_ylabel('Bottles in stock', color=color, labelpad=6, linespacing=1.3)
    ax_s.tick_params(axis='y', labelcolor=color)

    # Re-draw change lines on ax_s so they appear above the stock step function
    for t, _ in bottle_changes:
        ax_s.axvline(x=t, color=PALETTE['change_line'], linestyle='--',
                     linewidth=1.2, alpha=0.60, zorder=10)

    # Markers and labels at each event
    running = 0
    delivery_count = 0
    for t, d in events:
        running += d
        if d > 0:
            delivery_count += 1
            ax_s.plot(t, running, marker='^', color=color,
                      markersize=10, zorder=7, linewidth=0)
            # Label just above the marker; nudge right slightly to avoid clipping
            label_x = t + datetime.timedelta(hours=10)
            ax_s.text(
                label_x, running + 0.10,
                f'Bottle {delivery_count} arrives',
                ha='left', va='bottom',
                fontsize=7.8, color=color, linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                          edgecolor='none', alpha=0.88),
                zorder=8,
            )
        else:
            # Bottle exhausted — small downward triangle to show stock drop
            ax_s.plot(t, running, marker='v', color=color,
                      markersize=8, zorder=7, linewidth=0, alpha=0.80)

    return line_stock


def _add_summary_box(ax, df):
    """Place the run-summary info box in the upper-right of the flow panel."""
    n_bottles = int(df['bottle_id'].iloc[-1])
    total_m3  = df['total_consumed'].iloc[-1] / 1000
    run_days  = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    info = (f"Duration:  {run_days:.0f} days\n"
            f"Bottles:   {n_bottles}\n"
            f"Total gas: {total_m3:.2f} m³")
    ax.text(
        0.987, 0.97, info,
        transform=ax.transAxes,
        fontsize=8.5, va='top', ha='right', linespacing=1.55,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#C8C8C8', alpha=0.92, linewidth=0.8),
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Main plot  (two-panel layout: flow on top, gas volume on bottom)
# ---------------------------------------------------------------------------

def plot_results(df, warning_bands, bottle_changes, bottle_capacity_l,
                 warning_pct, switch_pct, deliveries, save_path, plt_show=False):
    _apply_style()

    warning_vol = bottle_capacity_l * warning_pct
    switch_vol  = bottle_capacity_l * switch_pct

    # Two panels: flow (top, ~30 % height of gas panel), gas (bottom)
    fig, (ax_flow, ax_gas) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1.5, 3.3], 'hspace': 0.05},
    )
    fig.patch.set_facecolor(PALETTE['bg'])

    # ------------------------------------------------------------------ #
    # Top panel – Flow rate
    # ------------------------------------------------------------------ #
    line_flow, = ax_flow.plot(
        df['timestamp'], df['flow_rate_lh'],
        color=PALETTE['flow'], linewidth=2.3,
        label='Flow Rate (L/h)', zorder=4, solid_capstyle='round',
    )
    ax_flow.set_ylabel('Flow Rate (L/h)', color=PALETTE['flow'], labelpad=6)
    ax_flow.tick_params(axis='y', labelcolor=PALETTE['flow'])
    ax_flow.set_ylim(0, df['flow_rate_lh'].max() * 1.2)
    ax_flow.set_facecolor(PALETTE['axes_bg'])

    # Hide bottom spine so panels read as one connected figure
    ax_flow.spines['bottom'].set_visible(False)
    ax_flow.tick_params(axis='x', which='both', bottom=False)

    # Flow rate phase labels
    _annotate_flow_rate_labels(ax_flow, df)

    # Bottle stock step function (right y-axis) — must come after flow labels
    # so the twinx doesn't interfere with ylim already set
    line_stock = _plot_bottle_stock(ax_flow, df, bottle_changes, deliveries)

    # Summary box sits in the empty upper-right of the flow panel
    _add_summary_box(ax_flow, df)

    # ------------------------------------------------------------------ #
    # Bottom panel – Gas remaining
    # ------------------------------------------------------------------ #
    line_gas, = ax_gas.plot(
        df['timestamp'], df['bottle_remaining'],
        color=PALETTE['gas'], linewidth=2.0,
        label='Remaining Gas (L)', zorder=4, solid_capstyle='round',
    )
    ax_gas.fill_between(
        df['timestamp'], df['bottle_remaining'],
        color=PALETTE['gas'], alpha=0.055, zorder=2,
    )
    ax_gas.set_ylabel('Remaining Gas in Bottle  (L)', color=PALETTE['gas'], labelpad=8)
    ax_gas.tick_params(axis='y', labelcolor=PALETTE['gas'])
    ax_gas.set_ylim(0, bottle_capacity_l * 1.04)
    ax_gas.set_facecolor(PALETTE['axes_bg'])
    ax_gas.spines['right'].set_visible(False)

    # X-axis: no "Date" label, weekly major ticks + daily minor ticks
    ax_gas.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax_gas.xaxis.set_minor_locator(mdates.DayLocator())
    ax_gas.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_gas.tick_params(axis='x', which='minor', length=3, color='#AAAAAA')

    # Threshold lines
    line_warn = ax_gas.axhline(
        y=warning_vol,
        color=PALETTE['warning_thr'], linestyle='--', linewidth=1.5, alpha=0.9,
        label=f'Warning {warning_pct*100:.0f}%  ({warning_vol:.0f} L)', zorder=3,
    )
    line_switch = ax_gas.axhline(
        y=switch_vol,
        color=PALETTE['switch_thr'], linestyle='--', linewidth=1.5, alpha=0.9,
        label=f'Switch  {switch_pct*100:.0f}%  ({switch_vol:.0f} L)', zorder=3,
    )

    # Inline threshold labels at ~72 % of the x-range, just above each line
    label_x = df['timestamp'].quantile(0.72)
    for vol, label_str, color in [
        (warning_vol, f'Warning {warning_pct*100:.0f}%', PALETTE['warning_thr']),
        (switch_vol,  f'Switch {switch_pct*100:.0f}%',  PALETTE['switch_thr']),
    ]:
        ax_gas.text(
            label_x, vol + bottle_capacity_l * 0.012, label_str,
            ha='center', va='bottom', fontsize=7.8,
            color=color, alpha=0.95,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.7),
        )

    # Warning bands
    for start, end in warning_bands:
        ax_gas.axvspan(start, end, color=PALETTE['band'], alpha=0.55, zorder=2)

    # Bottle-change annotations (lines on both panels, text on gas panel)
    _annotate_bottle_changes(ax_gas, ax_flow, bottle_changes)

    # ------------------------------------------------------------------ #
    # Shared legend centred below the bottom panel
    # ------------------------------------------------------------------ #
    patch_band = mpatches.Patch(
        color=PALETTE['band'], alpha=0.55,
        label='Warning Period  (10% → 5%)',
    )
    # Two rows of three: [flow, gas, stock] / [warn, switch, band]
    legend_handles = [line_flow, line_gas, line_stock, line_warn, line_switch, patch_band]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        framealpha=0.95, frameon=True,
        edgecolor='#C8C8C8',
        fontsize=9,
    )

    fig.suptitle(
        'Gas Consumption Profile  &  Bottle Replacement Schedule',
        fontsize=14, fontweight='bold',
    )
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.97)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if plt_show:
        plt.show()
    print(f'Saved → {save_path}')


if __name__ == '__main__':
    main()

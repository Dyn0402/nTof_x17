#!/usr/bin/env python3
"""make_report.py [--out DIR]: report.html + figures/ for the SPS follow-ups (handoff step 4), from results/*.json,
results/v_no_v_rows.csv and results/run71_residuals.npz.  Generated, so re-running after the analysis updates numbers,
figures and verdict text together.  Figures are referenced by relative links (figures/x.png).

Default output: $WORK_DATA/sps_followup/report if WORK_DATA is set, else rerun/sps_followup/report/ (PNGs gitignored)."""
import argparse, csv, html, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
S1, S2, INK, INK2, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'


def style(ax, xl, yl):
    ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig_resid(R, Z, out):
    q = R['raw700']['views']['y']
    r = Z['raw700_y']; r = r[np.abs(r) < 3]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(r, bins=np.arange(-3, 3.01, 0.1), color=S1, edgecolor='white', lw=0.5, label=f'det4 X (wft p0), n = {q["n"]}')
    x = np.linspace(-1.2, 1.2, 200); s = q['sigma_res']
    h, e = np.histogram(r, bins=np.arange(-1.2, 1.21, 0.1)); A = h.max()
    ax.plot(x, A * np.exp(-0.5 * (x / s) ** 2), color=S2, lw=2, label=f'core Gaussian, sigma = {s*1e3:.0f} um')
    style(ax, 'telescope y at det4 - aligned det4 X  [mm]', 'events / 0.1 mm'); ax.legend(frameon=False)
    ax.set_title('run_71 raw700: det4 against the uRWELL telescope', color=INK, loc='left', fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_p0(R, out):
    H = R['pass_p0_hist']; e = np.array(H['edges']); c = 0.5 * (e[1:] + e[:-1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.step(c, H['raw700']['x'], where='mid', color=S1, lw=2, label='X view')
    ax.step(c, H['raw700']['y'], where='mid', color=S2, lw=2, label='Y view')
    style(ax, 'det4 p0 [mm]', 'pass-1 events / 2.5 mm'); ax.legend(frameon=False)
    ax.set_title('pass-1 run71 raw700: det4 positions (Y piles up at 155-160 mm)', color=INK, loc='left', fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_flush(V, T, out):
    ks = [f'run60_flush__ov{i:02d}' for i in range(15)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]; i = np.arange(15)
    ax.plot(i, [V[k]['x']['T_box'] for k in ks], 'o-', color=S1, lw=2, ms=6, label='X view')
    ax.plot(i, [V[k]['y']['T_box'] for k in ks], 's-', color=S2, lw=2, ms=6, label='Y view')
    ctl = V['run62_rot25_ladder__d700']
    ax.axhline(ctl['x']['T_box'], color=S1, ls=':', lw=1); ax.axhline(ctl['y']['T_box'], color=S2, ls=':', lw=1)
    ax.text(0, ctl['x']['T_box'] + 15, 'dotted: CF4 control (run62 d700, same mount and field)', ha='left', color=INK2, fontsize=8)
    style(ax, 'run_60 sub-run (30 min each, gas CO2 mix -> CF4 mix)', 'column duration T_box [ns]'); ax.legend(frameon=False)
    ax = axs[1]
    tv = [T[k]['tan_y_rescaled'] if T[k]['tan_y_rescaled'] is not None else np.nan for k in ks]
    raw = [T[k]['tan_y'] for k in ks]
    ax.plot(i, np.abs(raw), 'o', color=INK2, ms=6, mfc='none', label='as in pass 1 (bench v 33 um/ns)')
    ax.plot(i, np.abs(tv), 'o', color=S1, ms=7, label='rescaled by v = 28.8 mm / T')
    ax.axhline(abs(T['run61_op25__op25']['tan_y']), color=S2, ls='--', lw=1.5, label='CF4 operating rows (run61 op25)')
    ax.axhline(np.tan(np.radians(25.64)), color=INK2, ls=':', lw=1, label='mount 25.64 deg')
    style(ax, 'run_60 sub-run', '|median tan| (inclined view)'); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 0.6)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_ov07(O, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for s, col, mk in (('overnight_06', S1, 'o'), ('overnight_07', S2, 's')):
        h = np.array(O['rate_per_min'][s]['per_min']) / 60.0
        ax.plot(np.arange(len(h)) + 0.5, np.maximum(h, 0.1), mk + '-', color=col, lw=2, ms=5, label=s.replace('_', ' '))
    ax.set_yscale('log'); style(ax, 'minute of the sub-run', 'triggers / s'); ax.legend(frameon=False)
    ax.set_title('run_60: the beam is off from ov06 minute 15 to ov07 minute 28', color=INK, loc='left', fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def table(rows, cols):
    h = '<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td>{html.escape(str(r.get(c, "")))}</td>' for c in cols) + '</tr>'
    return h + '</table>'


def main():
    ap = argparse.ArgumentParser()
    wd = os.environ.get('WORK_DATA')
    ap.add_argument('--out', default=os.path.join(wd, 'sps_followup', 'report') if wd else os.path.join(HERE, 'report'))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    R = json.load(open(os.path.join(RES, 'run71_resolution.json')))
    Z = np.load(os.path.join(RES, 'run71_residuals.npz'))
    V = json.load(open(os.path.join(RES, 'vbox.json')))
    T = json.load(open(os.path.join(RES, 'tan_check.json')))
    O = json.load(open(os.path.join(RES, 'ov07.json')))
    VT = list(csv.DictReader(open(os.path.join(RES, 'v_no_v_rows.csv'))))
    F = os.path.join(a.out, 'figures')
    fig_resid(R, Z, os.path.join(F, 'run71_residual.png')); fig_p0(R, os.path.join(F, 'run71_p0.png'))
    fig_flush(V, T, os.path.join(F, 'flush_v.png')); fig_ov07(O, os.path.join(F, 'ov07_rate.png'))

    q = R['raw700']['views']['y']; oc = R['raw700']['own_centroid']['y']
    n_def = [r for r in VT if r['verdict'] == 'defined' and r['control'] == 'False']
    ok = 'beam_on'; off = 'beam_off'
    body = f"""<h1>SPS follow-ups on pass 1 (handoff step 4)</h1>
<h2>Verdict</h2><ul>
<li><b>(i) run71 RAW resolution vs the uRWELL telescope:</b> det4 X (wft p0) = <b>{q['sigma_det4']*1e3:.0f} &plusmn; {q['sigma_det4_err']*1e3:.0f} &micro;m</b>
at 700 V (residual {q['sigma_res']*1e3:.0f} &plusmn; {q['err']*1e3:.0f} &micro;m, telescope pointing {q['pointing_mix']*1e3:.0f} &micro;m from run 53),
against {oc['sigma_det4']*1e3:.0f} &micro;m for det4's own charge centroid on the same events. det4 Y is <b>not measurable</b>: it is dead over the
telescope's footprint. 450 V and 275 V have 40&ndash;70 telescope events each: not measurable.</li>
<li><b>(ii) the 34 rows without v:</b> {len(n_def)} rows get a v from the column duration (run60 ov07&ndash;ov14 except ov10, plus ov01); rescaled with it their inclined
angle matches the CF4 operating rows to 2&ndash;3 %. The other rows stay positions-only: CO2 rows and the early flush have no column end edge (the charge
fades), the low-field rows overrun the window.</li>
<li><b>(iii) run60 ov07:</b> the beam was off for the first 27 of its 30 minutes. The pass took the plateau's first 3000 triggers, so {O[off]['n']} of them are
beam-off (x quality {O[off]['x_quality_ok']:.0%}); the {O[ok]['n']} beam-on events read x quality {O[ok]['x_quality_ok']:.0%}, like the other plateaus.</li></ul>
<h2>(i) Resolution</h2>{table([dict(plateau=p, **{k: R[p]['corr'][c][k] for k in ('n', 'x', 'y')}, coord=c) for p in ('raw700', 'raw450', 'raw275') for c in 'xy' if p in R],
                       ['plateau', 'coord', 'n', 'x', 'y'])}
<p>Correlation of each telescope coordinate with det4's two views (n = selected telescope events with det4 x and y ok).</p>
<figure><img src="figures/run71_residual.png"><figcaption>Residual of det4 X against the interpolated telescope track (z = 1140 mm), raw700.</figcaption></figure>
<figure><img src="figures/run71_p0.png"><figcaption>Pass-1 raw700 positions: {R['pass_p0_hist']['raw700']['y_frac_155_160']:.0%} of the Y fits sit in one 5 mm bin at the dead-band edge.</figcaption></figure>
<h2>(ii) v for the rows without one</h2>
{table(VT, ['key', 'field_Vcm', 'n_samples', 'verdict', 'v_box', 'v_sys', 'T_box_x', 'T_flat_x', 'grad_x', 'T_box_y', 'T_flat_y', 'grad_y'])}
<figure><img src="figures/flush_v.png"><figcaption>Left: the run_60 flush column duration per sub-run, with the CF4 control. Right: the inclined-view tan, as fitted with the bench v and rescaled by the column-duration v.</figcaption></figure>
<h2>(iii) run60 ov07</h2><figure><img src="figures/ov07_rate.png"><figcaption>Trigger rate per minute from the FEU 3 timestamps.</figcaption></figure>
<h2>What this does not rule out</h2><ul>
<li>The det4 X number rests on the run-53 telescope model (back plane 0.301 &times; pitch). Here the uRWELL clusters come from RAW in-time strip maxima, not run 53's ZS hits; a different back-plane resolution moves det4 X between ~150 and ~250 &micro;m. The residual itself ({q['sigma_res']*1e3:.0f} &micro;m) is an upper bound with no model.</li>
<li>v from the column duration carries ws_spsv's unexplained &minus;8 % tan scale, as every CF4 row does.</li></ul>
<p>Details, method and limitations: <code>rerun/sps_followup/README.md</code>.</p>"""
    css = ('body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#0b0b0b;background:#fcfcfb}'
           'table{border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #e4e3df;padding:2px 8px;text-align:right}'
           'img{max-width:100%}figcaption{color:#52514e;font-size:13px}')
    open(os.path.join(a.out, 'report.html'), 'w').write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>SPS follow-ups</title>'
                                                         f'<style>{css}</style></head><body>{body}</body></html>')
    print('wrote', os.path.join(a.out, 'report.html'))


if __name__ == '__main__':
    main()

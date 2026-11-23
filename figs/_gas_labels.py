# curve labels: water fraction (%), on the thin curves of chambers B-E; exec'd by make_gas_vE_all.py
fs = ps.BASE_PT * 0.8
for det, (e_, dx, dy, ha, va) in {'det4': (95, -2, 1.5, 'right', 'bottom'), 'det7': (120, -2, 2, 'right', 'bottom'),
                                  'det2': (195, -3, 0.3, 'right', 'bottom'),
                                  'det6': (340, 4, -2.5, 'left', 'top')}.items():
    w = S[det]['water']
    ax.text(e_ + dx, GI.v_of(w, e_) + dy, f'{r2(w)} %', color=ps.DET_COLOR[det] if det != 'det4' else '#6b6b6b',
            fontsize=fs, ha=ha, va=va, zorder=7)

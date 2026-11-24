import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/c'
Z = np.load(f'{OUT}/gap_imgs.npz')
rows = [('v', 'v rel - 1'), ('Q', 'q_u90 rel - 1'), ('P', 'v x q_u90 rel - 1'), ('hpx', 'hp_x (mm), minus mean')]
fig, ax = plt.subplots(4, 2, figsize=(5.6, 9), constrained_layout=True)
for r, (q, t) in enumerate(rows):
    a_, b_ = Z[f'4__gold__{q}'], Z[f'4__r625__{q}']
    if q != 'hpx': a_, b_ = a_ - 1, b_ - 1
    else: a_, b_ = a_ - np.nanmean(a_), b_ - np.nanmean(b_)
    vm = max(np.nanmax(abs(a_)), np.nanmax(abs(b_)))
    for c, (m, nm) in enumerate(((a_, 'golden'), (b_, '6-25'))):
        im = ax[r, c].imshow(m.T, origin='lower', extent=[-1.7, 1.7, -1.51, 1.51], cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax[r, c].set_title(f'{t}: {nm}', fontsize=8); ax[r, c].set_xlabel('U (100 mm)'); ax[r, c].set_ylabel('W (100 mm)')
    fig.colorbar(im, ax=ax[r, :], shrink=0.8)
fig.savefig(f'{OUT}/fig5_gap_dish_maps.png', dpi=130)

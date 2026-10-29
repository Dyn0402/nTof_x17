"""Fleet check of the error-model recipe (rerun/calib/saturation/ERROR_MODEL.md): per chamber, clean_check on the
1320 held-out events of its rr1 cache.  DIR holds <det>_<arm>.npz (lxplus satur/fleet); CACHES holds <det>.pkl.
Usage: python -I fleet_check.py DIR CACHES  -> results/fleet_<det>.json and tables on stdout"""
import os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_check                                                        # noqa: E402


def main(d, caches, arms=('capt', 'caponly', 'striponly', 'cap_tf')):
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        if not os.path.exists(f'{d}/{det}_rail.npz'):
            continue
        tmp = tempfile.mkdtemp(prefix=f'fleet_{det}_')
        for a in ('rail',) + tuple(arms):            # clean_check reads det3big_<arm>_c*.npz
            if os.path.exists(f'{d}/{det}_{a}.npz'):
                shutil.copy(f'{d}/{det}_{a}.npz', f'{tmp}/det3big_{a}_c0.npz')
        print(f'\n## {det}')
        clean_check.main(f'{caches}/{det}.pkl', tmp, *arms, tag=f'fleet_{det}')


if __name__ == '__main__':
    main(*sys.argv[1:3])

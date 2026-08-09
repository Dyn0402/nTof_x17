"""Sizing: resistive-sheet screening of the slow ion INDUCED signal.
Per-k forced response, exact exponential integrator (stable at all lambda dt).
Observable: central-channel share of the ion contribution accumulated at
t = 200 ns (rising-edge window), per unit arrived charge, vs the model's
injected-then-spreads treatment as the null.
"""
import numpy as np

c_prime = 4.985e-7
GAP, Z0 = 150e-6, 13.84e-6
T_ION, PITCH, SIG0, T_EVAL = 306e-9, 800e-6, 200e-6, 200e-9

k = np.linspace(1, 6e4, 4000)
Wch = np.sinc(k * PITCH / 2 / np.pi)
F0 = np.exp(-k**2 * SIG0**2 / 2)

def central_share(F):
    return PITCH / np.pi * np.trapz(Wch * F, k)

def run(rho):
    D = 1.0 / (rho * c_prime)
    lam = D * k**2
    dt = 0.5e-9
    ts = np.arange(0, T_EVAL, dt)
    decay = np.exp(-lam * dt)
    # TRUE induction: counter-charge c chases the moving image exactly per step
    c = np.zeros_like(k)
    for t in ts:
        z = Z0 + (GAP - Z0) * t / T_ION
        img = np.exp(-k * z) * F0
        c = img + (c - img) * decay
    z_end = Z0 + (GAP - Z0) * ts[-1] / T_ION
    img_end = np.exp(-k * z_end) * F0
    share_true = central_share(img_end - c)
    # MODEL null: charge injected at rate 1/T_ION, each slice spreads after
    acc = np.zeros_like(k)
    for t in ts:
        acc += (dt / T_ION) * F0 * np.exp(-lam * (T_EVAL - t))
    share_model = central_share(acc)
    q_frac = T_EVAL / T_ION
    return central_share(F0), share_model / q_frac, share_true / q_frac

print(f"{'rho_s':>6} | prompt  model(perQ)  true(perQ)  screening true/model")
for rho in (0.5e6, 1e6, 2e6, 5e6):
    p, m, tr = run(rho)
    print(f"{rho/1e6:5.1f}M | {p:.3f}   {m:.3f}        {tr:.3f}        {tr/m:.3f}")

# ===============================================
# Wykres funkcji ρ_pk(z) wg definicji piecewise
# ===============================================
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'CMU Serif', 'Libertinus Serif', 'Nimbus Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'figure.dpi': 140, 'savefig.dpi': 300,
    'font.size': 16, 'axes.labelsize': 18
})

# --- stałe ---
H = 1.3   # m
L = 1.000 # m

# przesunięcie osi X
SHIFT_MM = (H/2) * 1000.0  # 650 mm

# limit rysowania
MAX_MM = 1150

# -----------------------------------------------
# Piecewise: ρ_pk
# -----------------------------------------------
def rho_pk_from0_vec(z, H, L):
    z = np.asarray(z, dtype=float)
    y = np.zeros_like(z)
    pi_over_H = np.pi / H

    m2 = (z > -H/2) & (z <= -H/2 + L)
    m3 = (z > (-H/2 + L)) & (z <= H/2)
    m4 = (z > H/2) & (z <= H/2 + L)
    m5 = (z > H/2 + L)

    y[m2] = (H/np.pi) * (np.sin(pi_over_H * (z[m2])) + 1.0)
    y[m3] = (H/np.pi) * (np.sin(pi_over_H * (z[m3]))
                         - np.sin(pi_over_H * (z[m3] - L)))
    y[m4] = (H/np.pi) * (1.0 - np.sin(pi_over_H * (z[m4] - L)))
    y[m5] = 0.0
    return y

# -----------------------------------------------
# φ(s) liczony od góry
# -----------------------------------------------
def phi_from_top(s, H):
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)

    m11 = (s >= -H/2) & (s <= -H/2 + L)
    m12 = (s > (-H/2 + L)) & (s <= H/2)

    out[m11] = np.cos((np.pi / H) * (s[m11]))
    out[m12] = np.cos((np.pi / H) * (s[m12]))
    return out

# --- siatka i wartości ---
z_mm  = np.linspace(-1000, 1150, 2001)
z_mm2 = np.linspace(-1000, 1150, 2001)

z_m  = z_mm / 1000.0
rpk  = rho_pk_from0_vec(z_m, H, L)
rpk2 = phi_from_top(z_m, H)

# --- przesunięcie osi x ---
x_mm  = z_mm  + SHIFT_MM
x_mm2 = z_mm2 + SHIFT_MM

# maska dla φ – tylko w rdzeniu
mask_phi_core = (x_mm2 >= 0.0) & (x_mm2 <= H * 1000.0)

# + maska obcinająca wszystko powyżej MAX_MM
mask_rpk  = x_mm  <= MAX_MM
mask_phi2 = (x_mm2 <= MAX_MM) & mask_phi_core

# -----------------------------------------------
# Rysowanie
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(x_mm[mask_rpk], rpk[mask_rpk],
        lw=2, label=r"$\varrho_{pk}(s)$")

ax.plot(x_mm2[mask_phi2], rpk2[mask_phi2],
        lw=1, label=r"$\phi(s)$")

# linie pomocnicze
center_mm = SHIFT_MM
drop_mm   = MAX_MM   # spadek pręta

ax.axvline(center_mm, color='pink', lw=2, ls=':')
ax.annotate("Środek rdzenia", (center_mm, 0),
            xytext=(0, -50), textcoords='offset points',
            ha='center', va='top')

ax.axvline(drop_mm, color='red', lw=2, ls=':')
ax.annotate("Spadek pręta", (drop_mm, 0),
            xytext=(80, 50), textcoords='offset points',
            ha='center', va='top')

ax.set_xlabel("s, mm")
ax.set_ylabel(r"$\varrho_{pk}(s)$, j.w.")
ax.grid(True, alpha=0.3)
ax.legend()

ax.set_xlim(0, MAX_MM)

plt.show()

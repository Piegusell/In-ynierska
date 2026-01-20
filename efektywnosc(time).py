import numpy as np
from scipy.integrate import solve_ivp

beta_i   = np.array([2.43e-4, 1.363e-3, 1.203e-3,
                     2.605e-3, 8.19e-4, 1.67e-4])
lambda_i = np.array([1.27e-2, 3.17e-2, 1.15e-1,
                     3.11e-1, 1.40,   3.87   ])

beta          = beta_i.sum()          # 0.006552 (1 $)
Lambda_prompt = 150e-6               # 150 µs
j             = len(beta_i)          # liczba grup opóźnionych neutronów
n0            = 1.0                   # początkowa moc reaktora

rho_pom = 1.0
H = 1.286
L = 1.0
a = 1.69
g = 9.81
c = -0.019
rho_pom_emax = 2*H/np.pi * np.sin(np.pi * L / (2*H))

def z(t):
    return (1/a) * np.log(np.cosh(np.sqrt(g*a) * t + c))

def rho(t):
    zt = z(t)

    if 0 <= zt < L/2:
        return 0
    elif L/2 <= zt < H/2:
        return rho_pom - rho_pom / rho_pom_emax * (H/np.pi) * (
            np.sin(np.pi * zt / H) - np.sin(np.pi * (zt - L) / H)
        )

    elif H/2 <= zt < H/2 + L:
        return rho_pom - rho_pom / rho_pom_emax * (H/np.pi) * (
            1 - np.sin(np.pi * (zt - L) / H)
        )

    else:
        return rho_pom

def kinetics_matrix(j, t, rho):
    r = rho(t)
    A = np.zeros((j+1,j+1))
    A[0, 0]  = (r*beta- beta) / Lambda_prompt   
    A[0, 1:] =  lambda_i
    A[1:, 0] =  beta_i / Lambda_prompt
    A[1:, 1:] = -np.diag(lambda_i)
    return A

def initial_conditions(j, n0):
    f = np.zeros(j+1)
    f[0] = Lambda_prompt
    f[1:] = beta_i / lambda_i
    f *= n0 / Lambda_prompt
    return f

def rhs(j, t, f, rho):
    return kinetics_matrix(j, t, rho) @ f     # kluczowy krok

sol = solve_ivp(
    lambda t, f: rhs(j, t, f, rho),
    t_span=(0, 1),
    y0=initial_conditions(j, n0),
    method='RK45',      # dla sztywnych → użyj 'Radau' lub 'BDF'
    max_step=0.01
)

t = sol.t
f = sol.y

import matplotlib.pyplot as plt

plt.plot(t, f[0], label='Moc reaktora n(t)')
plt.show()
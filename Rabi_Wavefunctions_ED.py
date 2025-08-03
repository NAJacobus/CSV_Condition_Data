import math
import RabiHamiltonian
import scipy
import numpy as np
import matplotlib.pyplot as plt

hbar = 1
vib = 10
delta = 1
omega = 1
lam_ratio = 2
mass = 1
sym_breaking = 0.5

gc = math.sqrt(1 + math.sqrt(1 + delta**2/16))
lam = lam_ratio*gc


def QHO_eigenstate(n, x, mass, omega):
    return 1 / math.sqrt(2 ** n * math.factorial(n)) * (mass * omega / math.pi / hbar) ** (1 / 4) * math.exp(
        -1 * mass * omega * x ** 2 / 2 / hbar) * scipy.special.eval_hermite(n, math.sqrt(mass * omega / hbar) * x)

H = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega, sym_breaking)
vals, vecs = scipy.linalg.eigh(H)

x_grid = np.linspace(-8, 8, 101)

wavefxn_vals = {}
for n in range(vib):
    wavefxn_vals[n] = []
    for x in x_grid:
        wavefxn_vals[n].append(QHO_eigenstate(n, x, mass, omega))


offset = 1
eigenstates_to_plot = range(4)
fig, ax = plt.subplots()
for j in eigenstates_to_plot:
    spin_up_fxn = np.zeros(np.size(x_grid))
    spin_down_fxn = np.zeros(np.size(x_grid))
    spin_down_indices = vecs[:,j].reshape(2*vib)[:vib]
    spin_up_indices = vecs[:, j].reshape(2*vib)[vib:]
    for n in range(vib):
        spin_up_fxn += spin_up_indices[n]*np.array(wavefxn_vals[n])
        spin_down_fxn += spin_down_indices[n]*np.array(wavefxn_vals[n])
    plt.plot(x_grid, spin_up_fxn + offset*j*np.ones(np.size(x_grid)), c = "red")
    plt.plot(x_grid, spin_down_fxn + offset*j * np.ones(np.size(x_grid)), c="blue")
plt.plot([],[], c = "red", label = "Spin Up")
plt.plot([],[], c = "blue", label = "Spin Down")
plt.legend()
plt.title(r"Rabi Numerical Diagonalization; $\Delta$=" + str(delta) + ", $g/g_c$ =" + str(lam_ratio) + ", Sym. Break = " + str(sym_breaking))
plt.show()


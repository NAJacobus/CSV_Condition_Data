import numpy as np
import scipy
import qutip
from qutip import Qobj
import p3_ppt
import math
import matplotlib.pyplot as plt
def psi_n(x):
    return [np.sin(x/2),0,0,0,np.cos(x/2),0]

def psi_n1(y):
    return [0,np.sin(y/2),0,0,0,np.cos(y/2)]

def get_degen_neg(x,y):
    psi_1 = np.array(psi_n(x)).reshape([6,1])
    psi_2 = np.array(psi_n1(y)).reshape([6,1])
    rho_1 = psi_1@np.transpose(psi_1)
    rho_2 = psi_2@np.transpose(psi_2)

    rho = (1/2)*(rho_1 + rho_2)

    rho_pt = qutip.partial_transpose(Qobj(rho, dims = [[2,3], [2,3]]), [0,1]).full()
    return p3_ppt.num_negativity(rho_pt)


num_x = 200
num_y = 200
x_list = np.linspace(0,2*math.pi, num_x)
y_list = np.linspace(0, 2*math.pi, num_y)

neg_vals = np.zeros((num_y, num_x))
tf1_vals = np.zeros((num_y, num_x))

def test_fxn_1(x,y):
    return (1/4)*(math.sqrt(math.sin(y/2)**4 + math.sin(x)**2) + math.sqrt(math.cos(x/2)**4 + math.sin(y)**2) - math.sin(y/2)**2 - math.cos(x/2)**2)

for i in range(num_x):
    for j in range(num_y):
        neg_vals[j,i] = get_degen_neg(x_list[i], y_list[j])
        tf1_vals[j,i] = test_fxn_1(x_list[i], y_list[j])

fig, ax = plt.subplots()
f = ax.pcolormesh(y_list, x_list, neg_vals)
fig.colorbar(f)
# plt.ylabel(r"$\lambda$")
# plt.xlabel(r"$\Delta$")
# plt.title(model + r" Bottom Spin Population, Sym Break = " + str(sym_breaking) + r", $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}")
# plt.savefig(f"{file_name}_bottom_site_prop".replace(".",""), bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
f = ax.pcolormesh(y_list, x_list, tf1_vals)
fig.colorbar(f)
# plt.ylabel(r"$\lambda$")
# plt.xlabel(r"$\Delta$")
# plt.title(model + r" Bottom Spin Population, Sym Break = " + str(sym_breaking) + r", $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}")
# plt.savefig(f"{file_name}_bottom_site_prop".replace(".",""), bbox_inches='tight')
plt.show()
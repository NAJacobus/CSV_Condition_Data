import Full_Redfield
import numpy as np
import scipy
import RabiHamiltonian
import entanglement_entropy
import supervector, HolsteinHamiltonian
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
from qutip import Qobj
import p3_ppt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import Formatter
import matplotlib as mpl
import qutip
import matplotlib.gridspec as gridspec
import check_detailed_balance, get_schwarz_inequality_violations
from truncdec import truncdec
from matplotlib import ticker


plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})


#Run Settings`
sec_mode = "sec"
sparse_mode = "sparse"
JC = False #Note that for JC we still use the sigma x bath coupling
num_k = 4
sym_breaking = False #Still have to add symmetry breaking to JC; for Rabi set the kind of symmetry  breaking below
#System Properties
omega = 1
vib = 25
accum_max_csv_graph = True
csv_cutoff = 1e-14

beta_spin = 90
beta_boson = 90
spin_coupling = 1e-5 #Spin and boson coupling for normal bath
boson_coupling = 1e-3

delta = 2
lam = 2.3

# n = 1
# lam = math.sqrt(2*omega**2*(1/2 + n + math.sqrt((1/2 + n)**2 + (delta - omega)**2/(2*omega)**2)))


num_energy_pops = 4
subblock_size = 10

if sym_breaking:
    sym_break = 0.1*delta
else:
    sym_break = 0
if sparse_mode == "dense":
    if JC:
        Ham = RabiHamiltonian.JCHamiltonian(vib, delta, lam, omega)
    else:
        Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega, sym_break)
    evals, evecs = scipy.linalg.eigh(Ham)

if sparse_mode == "sparse":
    if JC:
        Ham = RabiHamiltonian.JCHamiltonian_sparse(vib, delta, lam, omega)
    else:
        Ham = RabiHamiltonian.RabiHamiltonian_sparse(vib, delta, lam, omega, sym_break)
    evals, evecs = scipy.sparse.linalg.eigs(Ham, k=num_k, which="SR")

spin_bath_op = np.kron(RabiHamiltonian.sig_x(), np.eye(vib))
boson_bath_op = np.kron(np.eye(2), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib)) #Watch out for factors of \sqrt{2} in this
spin_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@spin_bath_op@evecs
boson_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@boson_bath_op@evecs

bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis]

spin_cutoff = delta
boson_cutoff = omega

def J_spin(x): #Define these to be antisymmetric
    return (spin_coupling)**2*x*math.exp(-1*abs(x)/spin_cutoff)
def J_boson(x): #Define these to be antisymmetric
    return (boson_coupling)**2*x*math.exp(-1*abs(x)/boson_cutoff)

spec_density_list = [J_spin, J_boson]

if sec_mode == "non-sec":
    D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])
    if sparse_mode == "dense":
        rho_ss = supervector.unsup((scipy.linalg.null_space(D_super)).reshape((2*vib)**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
    else:
        rho_ss = supervector.unsup((scipy.linalg.null_space(D_super)).reshape(num_k**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
    rho_ss = rho_ss/np.trace(rho_ss)

if sec_mode == "sec":
    pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])
    if sparse_mode == "dense":
        rho_ss = scipy.linalg.null_space(pop_derivative).reshape(2*vib)
    else:
        rho_ss = scipy.linalg.null_space(pop_derivative).reshape(num_k)
    rho_ss = np.diag(rho_ss)
    rho_ss = rho_ss/np.trace(rho_ss)
    if np.trace(rho_ss > 1.5):
        print("here")

rho_ss_site = evecs@rho_ss@np.transpose(evecs.conj())
# spin_down_pops[d_count, lam_count] = Qobj(rho_ss_site, dims = [[2,vib], [2,vib]]).ptrace(0).full()[0,0]
# purity_data[d_count, lam_count] = np.trace(rho_ss_site@rho_ss_site)
numerical_negativity = p3_ppt.negativity_all_subdivisions(rho_ss_site, [2,vib], csv_cutoff, return_negativity_sum = True)[1]
# stored_ee_ss_rho[d_count, lam_count] = rho_ss
# stored_site_ss_rho[d_count, lam_count] = rho_ss_site
pt = qutip.partial_transpose(qutip.Qobj(rho_ss_site, dims = [[2,vib], [2,vib]]), [1,0]).full()
# stored_site_ss_pt[d_count, lam_count] = pt
tot_csv, max_csv = get_schwarz_inequality_violations.check_schwarz_inequality_violations(pt, prep_plot = False, return_abs_val_sum=True, return_max_csv=True)

if JC:
    model = "JC"
else:
    model = "Rabi"
default_backend = mpl.get_backend()

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_{model}_{sparse_mode}")
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_{model}_{sparse_mode}_{extend}")

np.save(file_name, np.empty(1), allow_pickle=True)

# def get_gc(delta):
#     return math.sqrt(1 + math.sqrt(1 + delta**2/16))
# gc_list = []
# for delta in deltas:
#     gc_list.append(get_gc(delta))


def truncdec(real_num, sig):
    ##Truncates real numbers to sig non-zero decimal places
    if real_num == 0:
        return 0
    elif real_num >= 1:
        return round(real_num, sig)
    pv = math.floor(math.log10(abs(real_num)))
    return round(real_num, -1*pv + sig - 1)


def convert_index_to_Rabi_ket(x):
    if x // vib == 0:
        spin = r"\downarrow"
    else:
        spin = r"\uparrow"
    boson = str(int(x % vib))
    return "$$|" + spin + "," + boson + r"\rangle$$"

def convert_index_to_Rabi_bra(x):
    if x // vib == 0:
        spin = r"\downarrow"
    else:
        spin = r"\uparrow"
    boson = str(int(x % vib))
    return r"$$\langle" + spin + "," + boson + r"|$$"

class Rabi_x_CustomFormatter(Formatter):
    def __init__(self, ax):
        super().__init__()
        self.set_axis(ax)

    def __call__(self, x, pos=None):
        return convert_index_to_Rabi_bra(x)

class Rabi_y_CustomFormatter(Formatter):
    def __init__(self, ax):
        super().__init__()
        self.set_axis(ax)

    def __call__(self, x, pos=None):
        return convert_index_to_Rabi_ket(x)


class Rabi_x_ShiftedFormatter(Formatter):
    def __init__(self, ax):
        super().__init__()
        self.set_axis(ax)

    def __call__(self, x, pos=None):
        return convert_index_to_Rabi_bra(x + vib)

class Rabi_y_ShiftedFormatter(Formatter):
    def __init__(self, ax):
        super().__init__()
        self.set_axis(ax)

    def __call__(self, x, pos=None):
        return convert_index_to_Rabi_ket(x + vib)


fig = plt.figure()
gs = gridspec.GridSpec(40, 40)
ax = fig.add_subplot(gs[:])
ax.axis("off")
ax.set_title("Real Parts " + model + " $\Delta$ = " + str(delta) + r" $\lambda$ = " + str(lam) + r" $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}; Negativity = " + str(truncdec(numerical_negativity, 2)))
gs.update(wspace=0.5)
ax1 = plt.subplot(gs[5:20, 5:20])
ax2 = plt.subplot(gs[5:20, 25:40])
ax3 = plt.subplot(gs[25:40, 13:28])
fig_1 = ax1.imshow(np.real(rho_ss), interpolation='nearest')
ax1.title.set_text("Real Part; Energy Eigenbasis")
fig_2 = ax2.imshow(np.real(rho_ss_site), interpolation='nearest')
ax2.title.set_text("Real Part; Product Basis")
fig_3 = ax3.imshow(np.real(pt), interpolation='nearest')
ax3.title.set_text("Real Part; Partial Transpose")
plt.colorbar(fig_1, ax=ax1)
plt.colorbar(fig_2, ax = ax2)
plt.colorbar(fig_3, ax = ax3)
y_formatter = Rabi_y_CustomFormatter(ax1)
x_formatter = Rabi_x_CustomFormatter(ax1)
ax1.yaxis.set_major_formatter(y_formatter)
ax1.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_CustomFormatter(ax2)
x_formatter = Rabi_x_CustomFormatter(ax2)
ax2.yaxis.set_major_formatter(y_formatter)
ax2.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_CustomFormatter(ax3)
x_formatter = Rabi_x_CustomFormatter(ax3)
ax3.yaxis.set_major_formatter(y_formatter)
ax3.xaxis.set_major_formatter(x_formatter)
# #Set number of ticks (see https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks)
# ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
# ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
plt.show()

pops = np.diag(rho_ss)[:num_energy_pops].reshape(num_energy_pops)
dens_op_top_left = rho_ss_site[:subblock_size, : subblock_size]
dens_op_bottom_right = rho_ss_site[vib: vib + subblock_size, vib: vib + subblock_size]
dens_op_bottom_left = rho_ss_site[:subblock_size, vib: vib + subblock_size]


fig = plt.figure()
gs = gridspec.GridSpec(40, 40)
ax = fig.add_subplot(gs[:])
ax.axis("off")
# ax.set_title("Real Parts " + model + " $\Delta$ = " + str(delta) + r" $\lambda$ = " + str(lam) + r" $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}; Negativity = " + str(truncdec(numerical_negativity, 2)))
gs.update(wspace=0.5)
ax1 = plt.subplot(gs[0:15, 0:17])
ax2 = plt.subplot(gs[0:17, 23:40])
ax4 = plt.subplot(gs[23:40, 0:17])
ax3 = plt.subplot(gs[23:40, 23:40])
fig_1 = ax1.scatter(np.arange(0,num_energy_pops), pops)
# ax1.title.set_text("Energy Eigenbasis Pops")
fig_2 = ax2.imshow(np.real(dens_op_top_left), interpolation='nearest')
# ax2.title.set_text("Density Op Top Left")
fig_3 = ax3.imshow(np.abs(np.real(dens_op_bottom_left)), interpolation='nearest')
# ax3.title.set_text("Density Op; Bottom Left Absolute Value of Coherences")
fig_4 = ax4.imshow(np.real(dens_op_bottom_right), interpolation='nearest')
# ax4.title.set_text("Density Op; Bottom Right Pops")
ax1.set_ylabel("Population")
ax1.set_xlabel("Energy Eigenstate")
ax1.set_title("(a)", loc = "left")
ax2.set_title("(b)", loc = "left")
ax3.set_title("(d)", loc = "left")
ax4.set_title("(c)", loc = "left")
plt.colorbar(fig_4, ax=ax4)
plt.colorbar(fig_2, ax = ax2)
plt.colorbar(fig_3, ax = ax3)
y_formatter = Rabi_y_CustomFormatter(ax2)
x_formatter = Rabi_x_CustomFormatter(ax2)
ax2.yaxis.set_major_formatter(y_formatter)
ax2.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_ShiftedFormatter(ax3)
x_formatter = Rabi_x_CustomFormatter(ax3)
ax3.yaxis.set_major_formatter(y_formatter)
ax3.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_ShiftedFormatter(ax4)
x_formatter = Rabi_x_ShiftedFormatter(ax4)
ax4.yaxis.set_major_formatter(y_formatter)
ax4.xaxis.set_major_formatter(x_formatter)
# #Set number of ticks (see https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks)
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
ax4.xaxis.set_major_locator(plt.MaxNLocator(5))
ax4.yaxis.set_major_locator(plt.MaxNLocator(5))
minor_ticks = np.arange(0.5, subblock_size + 0.5, 1)
minor_ticks_y = np.arange(0.5, subblock_size + 0.5, 1)
ax2.set_yticks(minor_ticks, minor = True)
ax2.set_xticks(minor_ticks, minor = True)
ax2.grid(which = "minor")
ax2.tick_params(axis="both", which='major',length=0, labelsize = 7)
ax3.set_yticks(minor_ticks, minor = True)
ax3.set_xticks(minor_ticks, minor = True)
ax3.grid(which = "minor")
ax3.tick_params(axis="both", which='major',length=0, labelsize = 7)
ax4.set_yticks(minor_ticks, minor = True)
ax4.set_yticks(minor_ticks_y, minor = True)
ax4.set_xticks(minor_ticks, minor = True)
ax4.grid(which = "minor")
ax4.tick_params(axis="both", which='major',length=0, labelsize = 7)
# plt.savefig(f"{file_name}_dens_op_visualization".replace(".", ""), bbox_inches='tight', dpi=300)
# plt.savefig("fig3", bbox_inches = "tight", dpi = 300)
plt.show()

print("S = " + str(tot_csv))
print("N = " + str(numerical_negativity))

fig = plt.figure()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig_1 = ax1.scatter(np.arange(0,num_energy_pops), pops, s = 300)
ax1.set_xticks([0,1,2,3])
ax1.set_yticks([0,0.25,0.5])
# ax1.set_yticks([0,0.5,1.0])
ax1.tick_params(axis='both', which='major', labelsize=21)



fig_2 = ax2.imshow(np.real(dens_op_top_left), interpolation='nearest')
# ax2.title.set_text("Density Op Top Left")
fig_3 = ax3.imshow(np.abs(np.real(dens_op_bottom_left)), interpolation='nearest')
# ax3.title.set_text("Density Op; Bottom Left Absolute Value of Coherences")
fig_4 = ax4.imshow(np.real(dens_op_bottom_right), interpolation='nearest')
# ax4.title.set_text("Density Op; Bottom Right Pops")

ax1.set_ylabel("Population", fontsize = 21)
ax1.set_xlabel("Energy Eigenstate", fontsize =21)
#Set number of ticks in colorbar https://stackoverflow.com/questions/22012096/how-to-set-number-of-ticks-in-plt-colorbar
tick_locator = ticker.MaxNLocator(nbins=5)
cb4 = plt.colorbar(fig_4, ax=ax4)
cb2 = plt.colorbar(fig_2, ax = ax2)
cb3 = plt.colorbar(fig_3, ax = ax3)
font = 21
cb4.ax.tick_params(labelsize=font)
cb2.ax.tick_params(labelsize=font)
cb3.ax.tick_params(labelsize=font)
cb4.locator = tick_locator
cb4.update_ticks()
tick_locator = ticker.MaxNLocator(nbins=5)
cb3.locator = tick_locator
cb3.update_ticks()
tick_locator = ticker.MaxNLocator(nbins=5)
cb2.locator = tick_locator
cb2.update_ticks()
y_formatter = Rabi_y_CustomFormatter(ax2)
x_formatter = Rabi_x_CustomFormatter(ax2)
ax2.yaxis.set_major_formatter(y_formatter)
ax2.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_ShiftedFormatter(ax3)
x_formatter = Rabi_x_CustomFormatter(ax3)
ax3.yaxis.set_major_formatter(y_formatter)
ax3.xaxis.set_major_formatter(x_formatter)
y_formatter = Rabi_y_ShiftedFormatter(ax4)
x_formatter = Rabi_x_ShiftedFormatter(ax4)
ax4.yaxis.set_major_formatter(y_formatter)
ax4.xaxis.set_major_formatter(x_formatter)
# #Set number of ticks (see https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks)
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
ax4.xaxis.set_major_locator(plt.MaxNLocator(5))
ax4.yaxis.set_major_locator(plt.MaxNLocator(5))


minor_ticks = np.arange(0.5, subblock_size + 0.5, 1)
minor_ticks_y = np.arange(0.5, subblock_size + 0.5, 1)
ax2.set_yticks(minor_ticks, minor = True)
ax2.set_xticks(minor_ticks, minor = True)
ax2.grid(which = "minor")
ax2.tick_params(axis="both", which='major',length=0, labelsize = 21)
ax3.set_yticks(minor_ticks, minor = True)
ax3.set_xticks(minor_ticks, minor = True)
ax3.grid(which = "minor")
ax3.tick_params(axis="both", which='major',length=0, labelsize = 21)
ax4.set_yticks(minor_ticks, minor = True)
ax4.set_yticks(minor_ticks_y, minor = True)
ax4.set_xticks(minor_ticks, minor = True)
ax4.grid(which = "minor")
ax4.tick_params(axis="both", which='major',length=0, labelsize = 21)
# plt.savefig(f"{file_name}_dens_op_visualization".replace(".", ""), bbox_inches='tight', dpi=300)
# plt.savefig("fig3", bbox_inches = "tight", dpi = 300)
fig1.savefig("fig5a", bbox_inches = "tight", dpi = 300)
fig2.savefig("fig5b", bbox_inches = "tight", dpi = 300)
fig3.savefig("fig5d", bbox_inches = "tight", dpi = 300)
fig4.savefig("fig5c", bbox_inches = "tight", dpi = 300)

plt.show()



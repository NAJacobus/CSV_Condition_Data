import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import Formatter
import matplotlib.gridspec as gridspec
import get_dens_op

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})


#Run Settings`
sec_mode = "sec"
sparse_mode = "sparse"
JC = True #Note that for JC we still use the sigma x bath coupling
num_k = 4
sym_breaking = False #Still have to add symmetry breaking to JC; for Rabi set the kind of symmetry  breaking below
#System Properties
omega = 1
vib = 40
accum_max_csv_graph = True
csv_cutoff = 1e-14

beta_spin = 100
beta_boson = 100
spin_coupling = 1e-5 #Spin and boson coupling for normal bath
boson_coupling = 1e-3

delta = 1.5
lam = 2

# n = 1
# lam = math.sqrt(2*omega**2*(1/2 + n + math.sqrt((1/2 + n)**2 + (delta - omega)**2/(2*omega)**2)))

def theta_n(n):
    return math.atan(2*lam*math.sqrt(n)/(delta - omega))

def eps_n_minus(n):
    vec = np.zeros((2*vib))
    vec[n] = math.cos(theta_n(n)/2)
    vec[vib + n - 1] = math.sin(theta_n(n)/2)
    return get_dens_op.pure_dens_op(vec)

def eps_n_plus(n):
    vec = np.zeros((2 * vib))
    vec[n] = -math.sin(theta_n(n) / 2)
    vec[vib + n - 1] = math.cos(theta_n(n) / 2)
    return get_dens_op.pure_dens_op(vec)

num_energy_pops = 4
subblock_size = 4

state = eps_n_minus(1)
#state = (eps_n_minus(1) + eps_n_minus(2))/2

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

dens_op_top_left = state[0:subblock_size, 0:subblock_size]
dens_op_bottom_left = state[vib: vib + subblock_size, 0:subblock_size]
dens_op_bottom_right = state[vib: vib + subblock_size, vib: vib + subblock_size]

fig = plt.figure()
gs = gridspec.GridSpec(40, 40)
ax = fig.add_subplot(gs[:])
ax.axis("off")
# ax.set_title("Real Parts " + model + " $\Delta$ = " + str(delta) + r" $\lambda$ = " + str(lam) + r" $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}; Negativity = " + str(truncdec(numerical_negativity, 2)))
gs.update(wspace=0.5)
ax2 = plt.subplot(gs[0:17, 0:17])
ax3 = plt.subplot(gs[23:40, 11:29])
ax4 = plt.subplot(gs[0:17, 23:40])
fig_2 = ax2.imshow(np.real(dens_op_top_left), interpolation='nearest')
# ax2.title.set_text("Density Op Top Left")
fig_3 = ax3.imshow(np.abs(np.real(dens_op_bottom_left)), interpolation='nearest')
# ax3.title.set_text("Density Op; Bottom Left Absolute Value of Coherences")
fig_4 = ax4.imshow(np.real(dens_op_bottom_right), interpolation='nearest')
# ax4.title.set_text("Density Op; Bottom Right Pops")
ax2.set_title("(a)", loc = "left")
ax3.set_title("(c)", loc = "left")
ax4.set_title("(b)", loc = "left")
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
num_ticks = min(subblock_size, 5)
ax2.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
ax2.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
ax3.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
ax3.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
ax4.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
ax4.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
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
plt.savefig(f"jc_visualized", bbox_inches='tight', dpi=300)
plt.show()



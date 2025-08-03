import numpy as np
import scipy
import RabiHamiltonian
import get_dens_op
import entanglement_entropy
import p3_ppt
from qutip import Qobj
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
from matplotlib.collections import LineCollection
from Holstein_Summer_2023 import Lindblad, supervector

mode = "dense"

vib = 15 #For lam = 1 the EE seems converged by vib = 20
delta = 2
omega = 1
lam = 0.1
rb = 1e-3
rs = 1e-3

#prop settings
t_range = (0, 10)
num_times = 50
t_type = "log"
include_zero = True
closed_prop = False
open_prop = True
plot_pop_data = True

#Initial state preparation
psi_0 = np.zeros((2*vib))
psi_0[vib] = 1
psi_0 = np.reshape(psi_0, [2*vib, 1])

qutip_dims = [2, vib]

if t_type == "lin":
    t_list = np.linspace(t_range[0], t_range[1], num_times)
if t_type == "log":
    t_list = np.logspace(t_range[0], t_range[1], num_times)

if include_zero:
    if 0 not in t_list:
        t_list = np.sort(np.concatenate(([0], t_list)))

closed_sys_spin_pops = np.empty((2, len(t_list)), dtype = float)
closed_sys_boson_pops =  np.empty((vib, len(t_list)), dtype = float)
open_sys_spin_pops = np.empty((2, len(t_list)), dtype = float)
open_sys_boson_pops =  np.empty((vib, len(t_list)), dtype = float)

if mode == "dense":
    if closed_prop:
        H = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)

        closed_ee_list = []
        closed_negativity_list = []

        t_index = 0
        for t in t_list:
            print(t)
            propagator = scipy.linalg.expm(-1j*H*t)

            psi_t = propagator@psi_0
            rho_t = get_dens_op.pure_dens_op(psi_t)
            rho_red = Qobj(rho_t, dims = [qutip_dims, qutip_dims]).ptrace(0).full()

            if plot_pop_data:
                closed_sys_spin_pops[:, t_index] = np.diag(rho_red)
                rho_red_boson = Qobj(rho_t, dims = [qutip_dims, qutip_dims]).ptrace(1).full()
                closed_sys_boson_pops[:, t_index] = np.diag(rho_red_boson)

            closed_ee_list.append(entanglement_entropy.get_ee(rho_red))
            closed_negativity_list.append(p3_ppt.negativity_all_subdivisions(rho_t, qutip_dims, 1e-10))

            t_index += 1


    if open_prop:
        H = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
        rho_0 = get_dens_op.pure_dens_op(psi_0)
        rho_0_super = supervector.supervector(rho_0)

        L = Lindblad.Rabi_Liouvillian(H, vib, rb, rs)

        open_ee_list = []
        open_negativity_list = []
        open_vne_list = []

        t_index = 0
        for t in t_list:
            print(t)
            propagator = scipy.linalg.expm(L*t)
            rho_t = supervector.unsup(propagator@rho_0_super)

            if plot_pop_data:
                rho_red_spin = Qobj(rho_t, dims = [qutip_dims, qutip_dims]).ptrace(0).full()
                open_sys_spin_pops[:, t_index] = np.diag(rho_red_spin)
                rho_red_boson = Qobj(rho_t, dims = [qutip_dims, qutip_dims]).ptrace(1).full()
                open_sys_boson_pops[:, t_index] = np.diag(rho_red_boson)

            t_index += 1

            open_ee_list.append(entanglement_entropy.get_average_mixed_bipartite_ee(rho_t, qutip_dims, 0))
            #TODO: Find why the entanglement of formation is so low
            open_negativity_list.append(p3_ppt.negativity_all_subdivisions(rho_t, qutip_dims, 1e-10))
            open_vne_list.append(entanglement_entropy.get_ee(rho_t))

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib_=_{vib}_lam_=_{lam}_del_=_{delta}".translate((None, ".")))
file_name_original = file_name
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = f"{file_name_original}_number_=_{extend}"

np.save(file_name, np.array([0]))

if closed_prop:
    fig, ax = plt.subplots(1)
    c_ee = ["r" if a else "g" for a in closed_negativity_list]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(t_list[:-1], closed_ee_list[:-1], t_list[1:], closed_ee_list[1:])]
    colored_lines = LineCollection(lines, colors=c_ee, linewidths=(2,))
    ax.add_collection(colored_lines)
    plt.axhline(math.log2(2), label = "log2(2) = 1 (Maximum Entanglement Entropy)")
    ax.autoscale_view()
    plt.plot([], [], ' ', c = "r", label= "Red: Negativity Detects Entanglement")
    plt.plot([], [], ' ', c = "g", label= "Green: Negativity Does Not Detect Entanglement")
    plt.title(f"Closed System Propagation; $\Delta$ = {delta}, $\lambda$ = {lam}")
    plt.xlabel(f"t")
    if t_type == "log":
        plt.xscale("log")
    plt.legend()
    plt.savefig(f"{file_name}_closed_sys_prop.png", bbox_inches='tight', dpi = 300)
    plt.show()

if open_prop:
    fig, ax = plt.subplots(1)
    c_ee = ["r" if a else "g" for a in open_negativity_list]
    lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(t_list[:-1], open_ee_list[:-1], t_list[1:], open_ee_list[1:])]
    colored_lines = LineCollection(lines, colors=c_ee, linewidths=(2,))
    ax.add_collection(colored_lines)
    plt.axhline(math.log2(2), label = "log2(2) = 1 (Maximum Pure State Entanglement Entropy)")
    ax.autoscale_view()
    plt.plot([], [], ' ', c = "r", label= "Red: Negativity Detects Entanglement")
    plt.plot([], [], ' ', c = "g", label= "Green: Negativity Does Not Detect Entanglement")
    plt.title(f"Open System Propagation; $\Delta$ = {delta}, $\lambda$ = {lam}, $r_b$ = {rb}, $r_s$ = {rs}")
    plt.xlabel(f"t")
    plt.legend()
    if t_type == "log":
        plt.xscale("log")
    plt.savefig(f"{file_name}_open_sys_prop.png", bbox_inches='tight', dpi = 300)
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(t_list, open_vne_list)
    plt.title(f"System-Environment Entanglement Entropy; rb = {rb}, rs = {rs}, vib = {vib}")
    if t_type == "log":
        plt.xscale("log")
    plt.axhline(math.log2(2*vib), c = "g", label = "log2(N) (Maximum Entanglement Entropy w/ Environment)")
    plt.xlabel(f"t")
    plt.savefig(f"{file_name}_sys_environment_entanglement_entropy.png", bbox_inches='tight', dpi = 300)
    plt.legend()




if open_prop and closed_prop:
    fig, ax = plt.subplots()
    plt.plot(t_list, closed_ee_list, label = "Entanglement Entropy; Closed System")
    plt.plot(t_list, open_ee_list, label = "Averaged Entanglement Entropy; Open System")
    plt.title(f"Open vs Closed System Propagation; $\Delta$ = {delta}, $\lambda$ = {lam}, $r_b$ = {rb}, $r_s$ = {rs}")
    plt.xlabel(f"t")
    plt.legend()
    if t_type == "log":
        plt.xscale("log")
    plt.savefig(f"{file_name}_open_vs_closed_prop.png", bbox_inches='tight', dpi = 300)
    plt.show()

if plot_pop_data:
    if closed_prop:
        fig, ax = plt.subplots()
        plt.plot(t_list, closed_sys_spin_pops[0,:], label = "Spin Down; Closed Sys")
        plt.plot(t_list, closed_sys_spin_pops[1,:], label = "Spin Up; Closed Sys")
        plt.title("Closed System Spin Population Evolution")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_closed_prop_spin_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()


        fig, ax = plt.subplots()
        for j in range(vib):
            plt.plot(t_list, closed_sys_boson_pops[j,:], label = f"Boson Level {j}; Closed Sys")
        plt.title("Closed System Boson Population Evolution")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_closed_pops_boson_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()

if open_prop:
        fig, ax = plt.subplots()
        plt.plot(t_list, open_sys_spin_pops[0,:], label = "Spin Down; Open Sys")
        plt.plot(t_list, open_sys_spin_pops[1,:], label = "Spin Up; Open Sys")
        plt.title("Open System Spin Population Evolution")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_open_sys_spin_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()

        fig, ax = plt.subplots()
        for j in range(vib):
            plt.plot(t_list, open_sys_boson_pops[j,:], label = f"Boson Level {j}; Open Sys")
        plt.title("Closed System Boson Population Evolution")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_open_sys_boson_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()

if open_prop and closed_prop:
        fig, ax = plt.subplots()
        plt.plot(t_list, open_sys_spin_pops[0,:] - closed_sys_spin_pops[0, :], label = "Spin Down; Open Sys - Closed Sys")
        plt.title(f"Difference in Open and Closed Spin Propagation Populations; rb = {rb}, rs = {rs}")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_open_vs_closed_spin_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()

        fig, ax = plt.subplots()
        for j in range(vib):
            plt.plot(t_list, open_sys_boson_pops[j, :] - closed_sys_boson_pops[j,:], label = f"Boson Level {j}, Open Sys - Closed Sys")
        plt.legend()
        if t_type == "log":
            plt.xscale("log")
        plt.savefig(f"{file_name}_open_vs_closed_boson_pops.png", bbox_inches='tight', dpi = 300)
        plt.show()

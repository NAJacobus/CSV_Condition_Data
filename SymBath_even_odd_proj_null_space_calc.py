#Alternative way to calculate the steady state via a null space instead of through either time propagation or directly exponentiating the matrices, which both run into some numerical errors
#Need to also make a non-secular version of this


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
import get_schwarz_inequality_violations
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib as mpl
import qutip
import matplotlib.gridspec as gridspec

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

#Run Settings
sec_mode = "sec"
sparse_mode = "dense"
num_k = 10
sym_breaking = False #If true, adds some kind of symmetry breaking (set below)

#System Properties
omega = 1
vib = 7

beta_e = 100
#beta_o = 0.4
beta_o = 0.4
even_coupling = 1e-5
odd_coupling = 1e-5
#del_lam_scan
num_tested_deltas = 10
tested_delta = 2
tested_delta_only = True
add_int_deltas = False
num_tested_lam = 100
delta_range = (0.15, 5*omega)
lam_range = (0.1, 4*omega)

#Even/Odd Bath Populations
pe = 0
po = 1

deltas = np.linspace(delta_range[0], delta_range[1], num_tested_deltas + 1)[1:]
if add_int_deltas:
    for j in range(1, 2):
        #Find numerators
        counter = 0
        while counter/j < delta_range[0]:
            counter += 1
        first_num = counter
        while counter/j <= delta_range[1]:
            counter += 1
        last_num = counter #Note that this is one more than the last one so that range gives all desired numerators
        for i in range(first_num, last_num):
            if i/j not in deltas:
                deltas = np.append(deltas, i/j)
            if i/j - 0.005 not in deltas and i/j - 0.005 >= delta_range[0]:
                deltas = np.append(deltas, i/j - 0.005)
            if i/j + 0.005 not in deltas and i/j + 0.005 <= delta_range[1]:
                deltas = np.append(deltas, i/j + 0.005)
if tested_delta_only:
    deltas = [tested_delta]

deltas = np.sort(deltas)
num_tested_deltas = np.size(deltas)

lam_list = np.linspace(lam_range[0], lam_range[1], num_tested_lam)

spin_down_pops = np.empty((num_tested_deltas, num_tested_lam))
purity_data = np.empty((num_tested_deltas, num_tested_lam))

average_bi_ee = np.empty((num_tested_deltas, num_tested_lam))
single_del_average_bi_ee = np.empty((num_tested_lam))
single_del_negativity = np.empty((num_tested_lam))
numerical_negativity_data = np.empty((num_tested_deltas, num_tested_lam))
stored_ee_ss_rho = np.empty((num_tested_deltas, num_tested_lam), dtype = np.ndarray)
stored_site_ss_rho = np.empty((num_tested_deltas, num_tested_lam), dtype = np.ndarray)
stored_site_ss_pt = np.empty((num_tested_deltas, num_tested_lam), dtype = np.ndarray)

d_count = -1
for d in range(num_tested_deltas):
    print(d)
    delta = deltas[d]
    d_count += 1
    lam_count = -1
    for lam in lam_list:
        lam_count += 1
        print(d_count, lam_count)
        if sym_breaking:
            sym_break = 0.1*delta
        else:
            sym_break = 0
        if sparse_mode == "dense":
            Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega, sym_break)
            evals, evecs = scipy.linalg.eigh(Ham)

        if sparse_mode == "sparse":
            Ham = RabiHamiltonian.RabiHamiltonian_sparse(vib, delta, lam, omega, sym_break)
            evals, evecs = scipy.sparse.linalg.eigs(Ham, k=num_k, which="SR")

        ee_indices = [] #indices of even eigenvectors
        oe_indices = [] #indices of odd eigenvectors
        for k in range(len(evals)):
            abs_evec = np.abs(evecs[k])
            e_sum = np.sum(abs_evec[0:vib:2]) + np.sum(abs_evec[vib + 1:2*vib:2])
            o_sum = np.sum(abs_evec[1:vib:2]) + np.sum(abs_evec[vib:2*vib:2])
            if e_sum > o_sum:
                ee_indices.append(k)
            else:
                oe_indices.append(k)




        parity_op_boson = np.zeros(vib)
        parity_op_boson[::2] = 1
        parity_op_boson[1::2] = -1
        sig_z = np.zeros((2, 2))
        sig_z[0, 0] = 1
        sig_z[1, 1] = -1
        par_op = np.kron(sig_z, np.diag(parity_op_boson))

        #Can change the interaction term to other parity-preserving things, e.g. a^2 (but might want to make them Hermitian? Not sure what parts assume a Hermitian system operator)
        interaction = np.kron(RabiHamiltonian.sig_minus(), HolsteinHamiltonian.a_dag(vib)) + np.kron(RabiHamiltonian.sig_plus(), HolsteinHamiltonian.a(vib))
        even_op = (np.eye(2 * vib) + par_op) / 2 @ interaction
        odd_op = (np.eye(2*vib) - par_op) / 2 @ interaction
        even_op_ee = np.conj(np.transpose(evecs)) @ even_op @ evecs
        odd_op_ee = np.conj(np.transpose(evecs)) @ odd_op @ evecs



        bath_coupling_list = [even_op_ee, odd_op_ee]

        #Might want to test different bath cutoffs; not clear what the physical energy scale of the even vs odd subspaces would be
        bath_cutoff = delta
        #bath_cutoff = omega

        def J_e(x): #Define these to be antisymmetric
            return (even_coupling)**2*x*math.exp(-1*abs(x)/bath_cutoff)

        def J_o(x): #Define these to be antisymmetric
            return (odd_coupling)**2*x*math.exp(-1*abs(x)/bath_cutoff)

        spec_density_list = [J_e, J_o]

        #if sec_mode == "non-sec":
        #     D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list,
        #                                                       [beta_e, beta_o])
        #     t_ind = -1
        #     if sparse_mode == "dense":
        #         rho_t = supervector.unsup((scipy.linalg.expm(D_super * prop_time) @ rho_0_super).reshape((2 * vib) ** 2,
        #                                                                                          1))  # I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
        #     else:
        #         rho_t = supervector.unsup((scipy.linalg.expm(D_super * prop_time) @ rho_0_super).reshape(num_k ** 2,
        #                                                                                          1))  # I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
        if sec_mode == "sec":
            pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list, evals,
                                                                              spec_density_list,
                                                                              [beta_e, beta_o])
            if sym_breaking:
                null = np.diag(scipy.linalg.null_space(pop_derivative).reshape(2*vib))
                rho_t = null
            else:
                even_pop_deriv = pop_derivative[ee_indices][:, ee_indices]
                odd_pop_deriv = pop_derivative[oe_indices][:, oe_indices]

                even_evals, even_evecs = scipy.linalg.eig(even_pop_deriv)
                odd_evals, odd_evecs =  scipy.linalg.eig(odd_pop_deriv)

                e_null_index = np.argmin(np.abs(even_evals))
                o_null_index = np.argmin(np.abs(odd_evals))

                aoe = np.abs(odd_evals)
                aoe.sort()
                print(aoe[0]/(aoe[1]))
                if aoe[0]/aoe[1] > 0.1:
                    print(aoe[1]/aoe[2])
                    print("here")

                even_kernel = even_evecs[:, e_null_index]
                odd_kernel = odd_evecs[:, o_null_index]



                even_kernel = pe*even_kernel/np.sum(even_kernel)
                odd_kernel = po*odd_kernel/np.sum(odd_kernel)

                rho_ss = np.empty(2*vib)
                even_count = 0
                odd_count = 0
                for k in range(2*vib):
                    if k in ee_indices:
                        rho_ss[k] = even_kernel[even_count]
                        even_count += 1
                    else:
                        rho_ss[k] = odd_kernel[odd_count]
                        odd_count += 1

        rho_ss = np.diag(rho_ss)
        rho_ss_site = evecs@rho_ss@np.transpose(evecs.conj())
        pt = qutip.partial_transpose(qutip.Qobj(rho_ss_site, dims=[[2, vib], [2, vib]]), [1, 0]).full()
        index_tuple = (d_count, lam_count)
        spin_down_pops[*index_tuple] = Qobj(rho_ss_site, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
        # print(d_count, lam_count)
        purity_data[*index_tuple] = np.trace(rho_ss_site @ rho_ss_site)
        numerical_negativity_data[*index_tuple] = \
        p3_ppt.negativity_all_subdivisions(rho_ss_site, [2, vib], 1e-8, return_negativity_sum=True)[1]
        stored_site_ss_rho[*index_tuple] = rho_ss_site
        stored_site_ss_pt[*index_tuple] = pt

default_backend = mpl.get_backend()

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_Rabi_symbaths_{sparse_mode}")
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_Rabi_symbaths_{sparse_mode}_{extend}")

np.save(file_name, np.empty(1), allow_pickle=True)

def get_gc(delta):
    return math.sqrt(1 + math.sqrt(1 + delta**2/16))
gc_list = []
for delta in deltas:
    gc_list.append(get_gc(delta))

if tested_delta_only:
    data = np.empty((2, num_tested_lam))
    data[0,:] = lam_list
    data[1,:] = numerical_negativity_data[0,:]
    # data[2,:] = tot_csv_data[0,:]

    single_del_file_name = os.path.join("Single_Del_Cross_Sections", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_Rabi_symbaths_{sparse_mode}_{extend}")


    np.save(single_del_file_name + "_single_del_cross_section", data, allow_pickle=True)
    fig,ax = plt.subplots()
    plt.plot(data[0,:], data[1,:])
    plt.show()

if not tested_delta_only:

    fig, ax = plt.subplots()
    f = ax.pcolormesh(deltas, lam_list, spin_down_pops.transpose())
    fig.colorbar(f)
    plt.ylabel(r"$\lambda$")
    plt.xlabel(r"$\Delta$")
    plt.title(r"Rabi Bottom Spin Population, Sym Break = " + str(sym_breaking) + r", $\beta_e$ = " + f"{beta_e}, " + r"$\beta_o$ = " + f"{beta_o}")
    plt.savefig(f"{file_name}_bottom_site_prop".replace(".",""), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    f = ax.pcolormesh(deltas, lam_list, purity_data.transpose())
    fig.colorbar(f)
    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Rabi Purity, Sym Break = " + str(sym_breaking) + r", $\beta_e$ = " + f"{beta_e}, " + r"$\beta_o$ = " + f"{beta_o}")
    plt.savefig(f"{file_name}_purity".replace(".",""), bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots()
    # f = ax.pcolormesh(t_list, deltas, average_bi_ee)
    # ax.set_xscale(t_type)
    # fig.colorbar(f)
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$\Delta$")
    # plt.title("(Approximate) Bipartite Entanglement of Formation")
    # plt.savefig(f"{file_name}_del_t_ee", bbox_inches='tight')
    # plt.show()
    numerical_negativity_original = np.copy(numerical_negativity_data)
    for (x,y), i in np.ndenumerate(numerical_negativity_data):
        if i <= 0: #Note that we already performed a cutoff in negativity_all_subdivisions when we only added to the negativity when it was greater than the threshold
            numerical_negativity_data[x,y] = np.nan

    def truncdec(real_num, sig):
        ##Truncates real numbers to sig non-zero decimal places
        if real_num == 0:
            return 0
        elif real_num >= 1:
            return round(real_num, sig)
        pv = math.floor(math.log10(abs(real_num)))
        return round(real_num, -1*pv + sig - 1)

    def on_pick(event):
        print(r"Delta = " + str(round(event.xdata, 2)) + r" lambda = ", str(round(event.ydata,2)))
        #get indices of data
        d_index = 0
        l_index = 0
        while d_index + 1 < len(deltas) and deltas[d_index + 1] < event.xdata:
            d_index += 1
        while l_index + 1 < len(lam_list) and lam_list[l_index + 1] < event.ydata:
            l_index += 1
        mpl.use(default_backend)

        fig = plt.figure()
        gs = gridspec.GridSpec(40, 40)
        ax = fig.add_subplot(gs[:])
        ax.axis("off")
        ax.set_title(r"Rabi Real Parts $\Delta$ = " + str(truncdec(deltas[d_index],2)) + r" $\lambda$ = " + str(truncdec(lam_list[l_index],2)) + r" $\beta_e$ = " + f"{beta_e}, " + r"$\beta_o$ = " + f"{beta_o}; Negativity = " + str(truncdec(numerical_negativity_original[d_index, l_index], 2)))
        gs.update(wspace=0.5)
        ax1 = plt.subplot(gs[5:20, 5:20])
        ax2 = plt.subplot(gs[5:20, 25:40])
        ax3 = plt.subplot(gs[25:40, 13:28])
        fig_1 = ax1.imshow(np.real(stored_ee_ss_rho[d_index, l_index]), interpolation='nearest')
        ax1.title.set_text("Real Part; Energy Eigenbasis")
        fig_2 = ax2.imshow(np.real(stored_site_ss_rho[d_index, l_index]), interpolation='nearest')
        ax2.title.set_text("Real Part; Product Basis")
        fig_3 = ax3.imshow(np.real(stored_site_ss_pt[d_index, l_index]), interpolation='nearest')
        ax3.title.set_text("Real Part; Partial Transpose")
        plt.colorbar(fig_1, ax=ax1)
        plt.colorbar(fig_2, ax = ax2)
        plt.colorbar(fig_3, ax = ax3)
        plt.show()

        fig = plt.figure()
        gs = gridspec.GridSpec(40, 40)
        ax = fig.add_subplot(gs[:])
        ax.axis("off")
        ax.set_title(r"Rabi Im Parts $\Delta$ = " + str(truncdec(deltas[d_index], 2)) + r" $\lambda$ = " + str(
            truncdec(lam_list[l_index],
                  2)) + r" $\beta_e$ = " + f"{beta_e}, " + r"$\beta_o$ = " + f"{beta_o}; Negativity = " + str(
            truncdec(numerical_negativity_original[d_index, l_index], 2)))
        gs.update(wspace=0.5)
        ax1 = plt.subplot(gs[5:20, 5:20])
        ax2 = plt.subplot(gs[5:20, 25:40])
        ax3 = plt.subplot(gs[25:40, 13:28])
        fig_1 = ax1.imshow(np.imag(stored_ee_ss_rho[d_index, l_index]), interpolation='nearest')
        ax1.title.set_text("Im Part; Energy Eigenbasis")
        fig_2 = ax2.imshow(np.imag(stored_site_ss_rho[d_index, l_index]), interpolation='nearest')
        ax2.title.set_text("Im Part; Product Basis")
        fig_3 = ax3.imshow(np.imag(stored_site_ss_pt[d_index, l_index]), interpolation='nearest')
        ax3.title.set_text("Im Part; Partial Transpose")
        plt.colorbar(fig_1, ax=ax1)
        plt.colorbar(fig_2, ax=ax2)
        plt.colorbar(fig_3, ax=ax3)
        plt.show()

        get_schwarz_inequality_violations.check_schwarz_inequality_violations(stored_site_ss_pt[d_index, l_index])
        plt.show()


        mpl.use('Qt5Agg')
        fig, ax = plt.subplots()
        cm = plt.get_cmap("viridis")
        cm.set_bad("black")
        f = ax.pcolormesh(deltas, lam_list, numerical_negativity_data.transpose(), cmap=cm)
        fig.colorbar(f)
        plt.ylabel(r"$\lambda$")
        plt.xlabel(r"$\Delta$")
        plt.title(r"Rabi Negativity, $\beta_e$ = " + f"{beta_e}, " + r"$\beta_o$ = " + f"{beta_o} (Black means neg = 0)")
        plt.savefig(f"{file_name}_negativity".replace(".", ""), bbox_inches='tight')

        fig.canvas.callbacks.connect('button_press_event', on_pick)
        plt.show()




    mpl.use('Qt5Agg')
    fig, ax = plt.subplots()
    cm = plt.get_cmap("viridis")
    cm.set_bad("black")
    f = ax.pcolormesh(deltas, lam_list, numerical_negativity_data.transpose(), cmap = cm)
    fig.colorbar(f)
    plt.ylabel(r"$\lambda$")
    plt.xlabel(r"$\Delta$")
    plt.title(r"Rabi Negativity, Sym Break = " + str(sym_breaking) + r", $\beta_e$ = " + f"{beta_e}, " + r"$\beta_b$ = " + f"{beta_o} (Black means neg = 0)")

    plt.savefig(f"{file_name}_negativity".replace(".", ""), bbox_inches='tight', dpi = 300)
    fig.canvas.callbacks.connect('button_press_event', on_pick)
    plt.show()
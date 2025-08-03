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
import matplotlib as mpl
import qutip
import matplotlib.gridspec as gridspec
import check_detailed_balance, get_schwarz_inequality_violations
from truncdec import truncdec

plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})


#Run Settings`
sec_mode = "sec"
sparse_mode = "sparse"
num_k = 4
#System Properties
omega = 1
vib = 20
accum_max_csv_graph = True
csv_cutoff = 1e-14

#Bath Properties (note the cutof13fs are chosen to match Delta and Lambda within the Delta/Lambda Loop)
beta_spin = 100
beta_boson = 100
spin_coupling = 1e-5 #Spin and boson coupling fr normal bath
boson_coupling = 1e-3

#del_lam_scan
tested_deltas = [2] #note that in the current labels for the plots we don't label delta
add_int_deltas = False
num_tested_lam = 500
lam_range = (0, 4*omega)

num_tested_deltas = np.size(tested_deltas)

lam_list = np.linspace(lam_range[0], lam_range[1], num_tested_lam)

jc_negativity_data = np.empty((num_tested_deltas, num_tested_lam))
rabi_negativity_data = np.empty((num_tested_deltas, num_tested_lam))
jc_tot_csv_data = np.empty((num_tested_deltas, num_tested_lam))
rabi_tot_csv_data = np.empty((num_tested_deltas, num_tested_lam))

sym_break = 0
d_count = -1
for d in range(num_tested_deltas):
    print(d)
    delta = tested_deltas[d]
    d_count += 1
    lam_count = -1
    for lam in lam_list:
        lam_count += 1

        if sparse_mode == "dense":
            JC_Ham = RabiHamiltonian.JCHamiltonian(vib, delta, lam, omega)
            Rabi_Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega, sym_break)

        if sparse_mode == "sparse":
            JC_Ham = RabiHamiltonian.JCHamiltonian_sparse(vib, delta, lam, omega)
            Rabi_Ham = RabiHamiltonian.RabiHamiltonian_sparse(vib, delta, lam, omega, sym_break)

        #JC_Routine
        if sparse_mode == "dense":
            evals, evecs = scipy.linalg.eigh(JC_Ham)
        if sparse_mode == "sparse":
            evals, evecs = scipy.sparse.linalg.eigs(JC_Ham, k=num_k, which="SR")

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
        jc_negativity_data[d_count, lam_count] = p3_ppt.negativity_all_subdivisions(rho_ss_site, [2,vib], csv_cutoff, return_negativity_sum = True)[1]
        pt = qutip.partial_transpose(qutip.Qobj(rho_ss_site, dims = [[2,vib], [2,vib]]), [1,0]).full()
        if accum_max_csv_graph:
            abs_val, max_csv = get_schwarz_inequality_violations.check_schwarz_inequality_violations(pt, prep_plot = False, return_abs_val_sum=True, return_max_csv=True)
            jc_tot_csv_data[d_count, lam_count] = abs_val

        #Rabi_Routine
        if sparse_mode == "dense":
            evals, evecs = scipy.linalg.eigh(Rabi_Ham)
        if sparse_mode == "sparse":
            evals, evecs = scipy.sparse.linalg.eigs(Rabi_Ham, k=num_k, which="SR")

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
        rabi_negativity_data[d_count, lam_count] = p3_ppt.negativity_all_subdivisions(rho_ss_site, [2,vib], csv_cutoff, return_negativity_sum = True)[1]
        pt = qutip.partial_transpose(qutip.Qobj(rho_ss_site, dims = [[2,vib], [2,vib]]), [1,0]).full()
        if accum_max_csv_graph:
            abs_val, max_csv = get_schwarz_inequality_violations.check_schwarz_inequality_violations(pt, prep_plot = False, return_abs_val_sum=True, return_max_csv=True)
            rabi_tot_csv_data[d_count, lam_count] = abs_val


default_backend = mpl.get_backend()

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_{sparse_mode}")
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_dellam_scan_{sparse_mode}_{extend}")
np.save(file_name, np.empty(1), allow_pickle=True)


fig, ax = plt.subplots()
d_count = -1
for delta in tested_deltas:
    d_count += 1
    ax.plot(lam_list, jc_negativity_data[d_count, :], label = "$\mathcal{N}$; JCM")
    ax.plot(lam_list, rabi_negativity_data[d_count, :], label = "$\mathcal{N}$; QRM")
    ax.plot(lam_list, jc_tot_csv_data[d_count,:], label = "$\mathcal{S}$; JCM")
    ax.plot(lam_list, rabi_tot_csv_data[d_count,:], label = "$\mathcal{S}$; QRM")
    ax.set_xlabel("$\lambda$")
    ax.set_xlim(lam_range[0] - 0.01, lam_range[1] + 0.01)
    ax.set_ylim(-0.0005, 0.501)
plt.legend()
plt.savefig(file_name + "_JCM_vs_Rabi", dpi = 300)
plt.show()






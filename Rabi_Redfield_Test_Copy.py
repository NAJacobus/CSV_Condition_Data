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


#Run Settings
sec_mode = "sec"
sparse_mode = "dense"

#System Properties
omega = 1
vib = 10
lam = 0.2*omega

beta_spin = 0.4
beta_boson = 0.4
spin_coupling = 1e-5
boson_coupling = 1e-3

#del_t_scan
num_tested_deltas = 50
add_int_deltas = True
num_tested_times = 50
delta_range = (0, 3.9*omega)
t_range = (-2, 10)
t_type = "log"

#propagation
rho_0_spin = np.zeros((2,2))
rho_0_spin[1,1] = 1
rho_0_boson = np.zeros((vib, vib))
rho_0_boson[0,0] = 1
rho_0_site_basis = np.kron(rho_0_spin, rho_0_boson)

deltas = np.linspace(delta_range[0], delta_range[1], num_tested_deltas + 1)[1:]
if add_int_deltas:
    for j in range(1, 2): #Is this biasing results because we're only adding the expected fractional resonances?
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

    deltas = np.sort(deltas)
    num_tested_deltas = np.size(deltas)

if t_type == "log":
    t_list = np.logspace(t_range[0], t_range[1], num_tested_times)
    # if t_list[0] == 1:
    #     t_list[0] = 0
else:
    t_list = np.linspace(t_range[0], t_range[1], num_tested_times)

spin_down_pops = np.zeros((num_tested_deltas, num_tested_times))
purity_data = np.zeros((num_tested_deltas, num_tested_times))

d_count = -1
for d in range(num_tested_deltas):
    delta = deltas[d]
    d_count += 1
    if sparse_mode == "dense":
        Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
        evals, evecs = scipy.linalg.eigh(Ham)

        spin_bath_op = np.kron(RabiHamiltonian.sig_x(), np.eye(vib))
        boson_bath_op = np.kron(np.eye(2), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib)) #Watch out for factors of \sqrt{2} in this

        spin_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@spin_bath_op@evecs
        boson_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@boson_bath_op@evecs
        rho_0_energy_eigenbasis = np.transpose(evecs.conj())@rho_0_site_basis@evecs
        pops_0, coh_0 = Full_Redfield.convert_rho_to_p_and_coh(rho_0_energy_eigenbasis)
        rho_0_super = supervector.supervector(rho_0_energy_eigenbasis)

        bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis]

        spin_cutoff = delta
        boson_cutoff = omega

        def J_spin(x): #Define these to be antisymmetric
            return (spin_coupling)**2*x*math.exp(-1*abs(x)/spin_cutoff)

        def J_boson(x): #Define these to be antisymmetric
            return (boson_coupling)**2*x*math.exp(-1*abs(x)/boson_cutoff)

        eps_0 = 1
        hbar = 1
        c = 1
        solar_atten_fac = 0.2
        beta_sun = 1
        def G(x):
            return 1 / (eps_0 * hbar * (2 * math.pi * c) ** 3) * 4 * math.pi ** 2 / 3 * x ** 3  # Also try Chern's Version; should there be an hbar here?
        A_ijkl = np.empty((2*vib, 2*vib, 2*vib, 2*vib), dtype = complex)
        for i in range(2*vib):
            for j in range(2*vib):
                for k in range(2*vib):
                    for l in range(2*vib):
                        A_ijkl[i,j,k,l] = spin_bath_op_energy_eigenbasis[i,j]*spin_bath_op_energy_eigenbasis[k,l]
        A_ij = np.empty((2*vib, 2*vib), dtype = complex)
        for i in range(2*vib):
            for j in range(2*vib):
                A_ij[i,j] = spin_bath_op_energy_eigenbasis[i,j]*spin_bath_op_energy_eigenbasis[j,i]

        # A_ijkl = np.zeros((2*vib, 2*vib, 2*vib, 2*vib), dtype = complex)
        # A_ij = np.zeros((2*vib, 2*vib), dtype = complex)


        bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis, A_ijkl]
        bath_coupling_list_sec = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis, A_ij]

        bath_coupling_list_all_ops = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis, spin_bath_op_energy_eigenbasis]

        spec_density_list = [J_spin, J_boson, G]

        if sec_mode == "non-sec":
            D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson, beta_sun], atten_fac_list = [1,1,solar_atten_fac], type_list = ["op", "op", "A"])
            D_super_2 = Full_Redfield.get_relaxation_derivative(bath_coupling_list_all_ops, evals, spec_density_list, [beta_spin, beta_boson,beta_sun], atten_fac_list = [1,1, solar_atten_fac])
            print(f"Difference is {np.max(np.abs(D_super - D_super_2))}")
            t_count = -1
            for t in t_list:
                t_count += 1
                rho_t_ee = supervector.unsup((scipy.linalg.expm(D_super*t)@rho_0_super).reshape((2*vib)**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
                rho_t_site = evecs@rho_t_ee@np.transpose(evecs.conj())
                purity_data[d_count, t_count] = np.trace(rho_t_site@rho_t_site)
                rho_qobj = Qobj(rho_t_site, dims = [[2,vib], [2,vib]]).ptrace(0).full()
                spin_down_pops[d_count, t_count] = rho_qobj[0,0]

                # print(d,t)

        if sec_mode == "sec":
            print(d)
            pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list_sec, evals, spec_density_list, [beta_spin, beta_boson, beta_sun], atten_fac_list = [1,1, solar_atten_fac], type_list = ["op", "op", "A"])
            pop_derivative_2, coh_derivatives_2 = Full_Redfield.get_secular_t_dep(bath_coupling_list_all_ops, evals, spec_density_list, [beta_spin, beta_boson, beta_sun], atten_fac_list = [1,1, solar_atten_fac])
            print(f"Difference is {np.max(np.abs(pop_derivative - pop_derivative_2))}")


            rho_t = Full_Redfield.full_sec_prop(pop_derivative, coh_derivatives, t_list, pops_0, coh_0)
            t_count = -1
            for t in t_list:
                t_count += 1
                full_rho_t = evecs@rho_t[t_count]@np.transpose(evecs.conj())
                spin_down_pops[d_count, t_count] = Qobj(full_rho_t, dims = [[2,vib], [2,vib]]).ptrace(0).full()[0,0]
                purity_data[d_count, t_count] = np.trace(full_rho_t@full_rho_t)

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}")
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_{extend}")

np.save(file_name, np.empty(1), allow_pickle=True)

fig, ax = plt.subplots()
f = ax.pcolormesh(t_list, deltas, spin_down_pops)
ax.set_xscale("log")
fig.colorbar(f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\Delta$")
plt.title("Bottom Spin Population")
plt.savefig(f"{file_name}_bottom_site_prop", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
f = ax.pcolormesh(t_list, deltas, purity_data)
ax.set_xscale("log")
fig.colorbar(f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\Delta$")
plt.title("Purity")
plt.savefig(f"{file_name}_purity", bbox_inches='tight')
plt.show()



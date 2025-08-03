import numpy as np
import RabiHamiltonian
import matplotlib.pyplot as plt
from qutip import Qobj
import Lindblad, secular_approximation, supervector
import scipy
import entanglement_entropy
import os
from datetime import datetime
import math

run_mode = "run"
prop_mode = "sec"
sparse_mode = "dense" #Haven't written sparse code yet

omega = 1
vib = 40
rs = 1e-6 #used when sys_environ_ee_rate_scan = False
rb = 1e-12 #used when sys_environ_ee_rate_scan = False
num_k = min(2*vib - 3, 200)

#Dellam scan
num_tested_deltas = 100
num_tested_lambdas = 100
tested_delta_only = True
tested_lambda_only = False
tested_delta = 1*omega
tested_lambda = 0.1*omega
delta_range = (0, 10*omega)
lambda_range = (0, 2.5*omega)

#Rate Scan Sys-Environment Entanglement Entropy Calculation (uses tested lambda and tested delta
sys_environ_ee_rate_scan = True #If false, uses fixed rb and rs from above
rs_range = (-10, -1)
rb_range = (-10, -1)
num_tested_rs = 50
num_tested_rb = 50

#Ratio scanning (Requires sys_environ_ee_rate_scan = True)
ratio_scan_only = True
fixed_r = "rb" #rs or rb
fixed_r_val = 1e-6
ratio_range = (-5, 3)
num_ratios = 100
ratio_range_type = "log" #lin or log

#Delta vs r ratio scan (Scans above r ratios as a function of delta for tested_lambda; uses above range for lambda)
del_vs_r_ratio_scan = False

#Lambda vs r ratio scan (Scans above r ratios as a function of lambda for tested_delta; uses above range for lambda)
lam_vs_r_ratio_scan = True


#TODO: Make sure there are no errors; verify that the sparse and dense versions give the same results; check if the rs >> rb version is correct; check if Ls is correct (originally had it reversed)

deltas = np.linspace(delta_range[0], delta_range[1], num_tested_deltas + 1)[1:]
lambdas = np.linspace(lambda_range[0], lambda_range[1], num_tested_lambdas + 1)[1:]

if sys_environ_ee_rate_scan:
    rs_list = np.logspace(rs_range[0], rs_range[1], num_tested_rs)
    rb_list = np.logspace(rb_range[0], rb_range[1], num_tested_rb)
    if ratio_scan_only:
        if ratio_range_type == "log":
            ratio_list = np.logspace(ratio_range[0], ratio_range[1], num_ratios)
        if ratio_range_type == "lin":
            ratio_list = np.linspace(ratio_range[0], ratio_range[1], num_ratios)

        if fixed_r == "rs":
            rs_list = [fixed_r_val]
            rb_list = fixed_r_val*ratio_list
        else:
            rb_list = [fixed_r_val]
            rs_list = fixed_r_val*ratio_list
else:
    rs_list = [rs]
    rb_list = [rb]

ee_data = np.empty((len(rb_list), len(rs_list)), dtype = float)

if tested_delta_only:
    deltas = [tested_delta]
    num_tested_deltas = 1
if tested_lambda_only:
    lambdas = [tested_lambda]
    num_tested_lambdas = 1

spin_pops = np.empty((2, num_tested_lambdas, num_tested_deltas))
single_lam_spin_pops = np.empty((2, num_tested_deltas))
boson_pops = np.empty((vib, num_tested_lambdas, num_tested_deltas))
single_lam_boson_pops = np.empty((vib, num_tested_deltas))
spin_flux = np.empty((num_tested_lambdas, num_tested_deltas))
single_lam_spin_flux = np.empty(num_tested_deltas)
fixed_r_dellam_scan_ee_data = np.empty((num_tested_lambdas, num_tested_deltas), dtype = float)

del_vs_r_ratio_ee_data = np.empty((num_tested_deltas, num_ratios), dtype = float)
lam_vs_r_ratio_ee_data = np.empty((num_tested_lambdas, num_ratios), dtype = float)

for rb_index in range(len(rb_list)):
    rb = rb_list[rb_index]
    for rs_index in range(len(rs_list)):
        rs = rs_list[rs_index]
        print(rb, rs)

        if ratio_scan_only and fixed_r == "rb":
            increasing_r_index = rs_index
        elif ratio_scan_only:
            increasing_r_index = rb_index

        for d in range(num_tested_deltas):
            delta = deltas[d]
            for l in range(num_tested_lambdas):
                lam = lambdas[l]
                if sparse_mode == "dense":
                    if prop_mode == "non-sec":
                        Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
                        L = Lindblad.Rabi_Liouvillian(Ham, vib, rb, rs)
                        null = scipy.linalg.null_space(L)
                        null = supervector.unsup(null)

                    if prop_mode == "sec":
                        Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
                        L_list = Lindblad.Rabi_L_list(vib, rb, rs)
                        null = secular_approximation.secular_approx(L_list, Ham)

                    null = null/np.trace(null)
                    print(delta, lam)

                    null_spin = Qobj(null, dims = [[2, vib], [2,vib]]).ptrace(0).full()
                    null_boson = Qobj(null, dims = [[2, vib], [2,vib]]).ptrace(1).full()

                    spin_pops[:, l, d] = np.diag(null_spin)
                    boson_pops[:, l, d] = np.diag(null_boson)
                    if lam == tested_lambda or tested_lambda_only:
                        single_lam_spin_pops[:,d] = np.diag(null_spin)
                        single_lam_boson_pops[:,d] = np.diag(null_boson)

                    #Get spin flux
                    L_flux = Lindblad.Rabi_Ls(vib, rs)
                    L_flux_adj = np.conj(np.transpose(L_flux))
                    prod = L_flux_adj@L_flux

                    spin_flux[l,d] = np.trace((Qobj(L_flux@null@L_flux_adj - (1/2)*(prod@null + null@prod), dims = [[2, vib], [2,vib]]).ptrace(0).full()) @ RabiHamiltonian.sig_z()) / 2
                    # print(spin_flux[l,d])
                    if lam == tested_lambda or tested_lambda_only:
                        single_lam_spin_flux[d] = spin_flux[l,d]

                    if sys_environ_ee_rate_scan and (lam == tested_lambda or tested_lambda_only) and (delta == tested_delta or tested_delta_only):
                        ee_data[rb_index, rs_index] = entanglement_entropy.get_ee(null)

                    if not sys_environ_ee_rate_scan: #Will collect ee data for delta lambda scan instead of for rs rb scan
                        fixed_r_dellam_scan_ee_data[l, d] = entanglement_entropy.get_ee(null)

                    if sys_environ_ee_rate_scan and ratio_scan_only and del_vs_r_ratio_scan and (lam == tested_lambda or tested_lambda_only):
                        del_vs_r_ratio_ee_data[d, increasing_r_index] = entanglement_entropy.get_ee(null)
                    if sys_environ_ee_rate_scan and ratio_scan_only and lam_vs_r_ratio_scan and (delta == tested_delta or tested_delta_only):
                        lam_vs_r_ratio_ee_data[l, increasing_r_index] = entanglement_entropy.get_ee(null)

file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_steady_state_scan".translate((None, ".")))
file_name_original = file_name
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = f"{file_name_original}_number_=_{extend}"

if not tested_lambda_only and not tested_delta_only:

    fig, ax0 = plt.subplots()
    im = ax0.pcolormesh(deltas, lambdas, spin_pops[0,:,:])
    plt.title(f"Spin Down Population; rs = {rs} r_b = {rb}")
    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$\lambda$")
    # ax0.invert_xaxis()
    fig.colorbar(im)
    # if run_mode == "run":
    #     plt.savefig(f"{file_name}_dellam_site_{i+1}.png", bbox_inches='tight')
    plt.show()

    fig, ax0 = plt.subplots()
    im = ax0.pcolormesh(deltas, lambdas, spin_flux)
    plt.title(f"NESS Spin Flux; rs = {rs} r_b = {rb}")
    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$\lambda$")
    # ax0.invert_xaxis()
    fig.colorbar(im)
    # if run_mode == "run":
    #     plt.savefig(f"{file_name}_dellam_site_{i+1}.png", bbox_inches='tight')
    plt.show()

    for j in range(min(3,vib)):
        fig, ax0 = plt.subplots()
        im = ax0.pcolormesh(deltas, lambdas, boson_pops[j,:,:])
        plt.title(f"Boson Level {j}; rs = {rs} r_b = {rb}")
        plt.xlabel(r"$\Delta$")
        plt.ylabel(r"$\lambda$")
        # ax0.invert_xaxis()
        fig.colorbar(im)
        # if run_mode == "run":
        #     plt.savefig(f"{file_name}_dellam_site_{i+1}.png", bbox_inches='tight')
        plt.show()

    if not sys_environ_ee_rate_scan:
        fig, ax0 = plt.subplots()
        im = ax0.pcolormesh(deltas, lambdas, fixed_r_dellam_scan_ee_data)
        plt.title(f"Steady State VN Entropy; $r_s/r_b$ = {rs/rb}")
        plt.xlabel(r"$\Delta$")
        plt.ylabel(r"$\lambda$")
        fig.colorbar(im)
        plt.savefig(f"{file_name}_dellam_vn_entropy_scan.png", dpi = 300, bbox_inches = "tight")
        plt.show()



if sys_environ_ee_rate_scan:
    if ratio_scan_only:
        fig, ax = plt.subplots()
        if fixed_r_val == "rs":
            x_label = r"$\frac{r_b}{r_s}$"
        else:
            x_label = r"$\frac{r_s}{r_b}$"
        plt.plot(ratio_list, ee_data.flatten())
        if ratio_range_type == "log":
            plt.xscale("log")
        plt.xlabel(x_label)
        plt.axhline(math.log2(2*vib), c = "g", label = "log2(N) (Maximum Entanglement Entropy w/ Environment)")
        plt.axhline(0, c = "r", label = "0")
        plt.title(f"Entanglement Entropy; $\Delta$ = {tested_delta}, $\lambda$ = {tested_lambda}, {fixed_r} = {fixed_r_val}")
        plt.savefig(f"{file_name}_sys_environ_ee_rate_ratio.png", dpi = 300, bbox_inches = 'tight')
        plt.legend()
        plt.show()
    if del_vs_r_ratio_scan:
        fig, ax = plt.subplots()
        if fixed_r_val == "rs":
            x_label = r"$\frac{r_b}{r_s}$"
        else:
            x_label = r"$\frac{r_s}{r_b}$"
        plt.ylabel(r"$\Delta$")
        im = ax.pcolormesh(ratio_list, deltas, del_vs_r_ratio_ee_data)
        if ratio_range_type == "log":
            plt.xscale("log")
        plt.xlabel(x_label)
        plt.title(f"Entanglement Entropy; $\lambda$ = {tested_lambda}, {fixed_r} = {fixed_r_val}")
        plt.savefig(f"{file_name}_sys_environ_ee_delta_vs_rate_ratio.png", dpi = 300, bbox_inches = 'tight')
        fig.colorbar(im)
        plt.show()

    if lam_vs_r_ratio_scan:
        fig, ax = plt.subplots()
        if fixed_r_val == "rs":
            x_label = r"$\frac{r_b}{r_s}$"
        else:
            x_label = r"$\frac{r_s}{r_b}$"
        plt.ylabel(r"$\lambda$")
        im = ax.pcolormesh(ratio_list, lambdas, lam_vs_r_ratio_ee_data)
        if ratio_range_type == "log":
            plt.xscale("log")
        plt.xlabel(x_label)
        plt.title(f"Entanglement Entropy; $\Delta$ = {tested_delta}, {fixed_r} = {fixed_r_val}")
        plt.savefig(f"{file_name}_sys_environ_ee_lambda_vs_rate_ratio.png", dpi = 300, bbox_inches = 'tight')
        fig.colorbar(im)
        plt.show()

    else:
        fig, ax = plt.subplots()
        im = ax.pcolormesh(rs_list, rb_list, ee_data)
        plt.title(f"System-Environment Steady State VN Entropy; $\Delta$ = {tested_delta}, $\lambda$ = {tested_lambda}")
        plt.xlabel(r"$r_s$")
        plt.ylabel(r"$r_b$")
        plt.xscale("log")
        plt.yscale("log")
        fig.colorbar(im)
        # plt.legend()
        plt.savefig(f"{file_name}_sys_environ_ee_rate_scan.png", dpi = 300, bbox_inches = 'tight')
        plt.show()


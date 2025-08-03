# import numpy as np
# import math
# from Holstein_Summer_2023 import supervector
#
# #Redfield code based on Pollard Freisner algorithm (see https://pubs.aip.org/aip/jcp/article/100/7/5054/482671/Solution-of-the-Redfield-equation-for-the and https://pubs.rsc.org/en/content/articlelanding/2015/cp/c5cp01388g)
#
# def get_B_plus(energies, spec_density_fxn, T):
#     """Note we assume the spectral density function is defined to be antisymmetric over \mathbb{R}"""
#     B = np.zeros((len(energies), len(energies)))
#     for i in range(len(energies)):
#         for k in range(len(energies)):
#             omega_ik = energies[i] - energies[k]
#             if T != 0 and omega_ik != 0:
#                 B[i,k] = math.pi*spec_density_fxn(omega_ik)/(math.exp(-omega_ik/T) - 1) #Why is there a -omega in Chern's version?
#             else:
#                 B[i,k] = 0
#     return B
#
# #Photon B operators for retinal
# def detailed_balance_factor(omega, T, C, degen_error):
#     if abs(omega) < degen_error:
#         return 0
#     if omega > 0:
#         return C/(math.exp(omega/T) - 1)
#     if omega < 0:
#         return -1 - C/(math.exp(-omega/T) - 1)
#
#
# degen_error = 1e-17
# def get_B_0_plus_photon_bath(energies, spec_density_fxn, T, mu0):
#     B = np.zeros((len(energies), len(energies)))
#     for i in range(len(energies)):
#         for k in range(len(energies)):
#             omega_ik = energies[i] - energies[k]
#             if T != 0 and omega_ik != 0:
#                 B[i,k] = math.pi*spec_density_fxn(mu0, omega_ik)*detailed_balance_factor(omega_ik, T, 0, degen_error) #Why is there a -omega in Chern's version?
#             else:
#                 B[i,k] = 0
#     return B
#
# def get_B_c_plus_photon_bath(energies, spec_density_fxn, T, mu0):
#     B = np.zeros((len(energies), len(energies)))
#     for i in range(len(energies)):
#         for k in range(len(energies)):
#             omega_ik = energies[i] - energies[k]
#             if T != 0 and omega_ik != 0:
#                 B[i,k] = math.pi*spec_density_fxn(mu0, omega_ik)*(detailed_balance_factor(omega_ik, T, 1, degen_error) - detailed_balance_factor(omega_ik, T, 0, degen_error)) #Why is there a -omega in Chern's version?
#             else:
#                 B[i,k] = 0
#     return B
#
#
# def get_B_plus_list(energies, spec_density_list, T_list, photon_bath_index, mu_list):
#     B_plus_list = []
#     index = 0
#     for i in range(len(spec_density_list)):
#         if i not in photon_bath_index:
#             B_plus_list.append(get_B_plus(energies, spec_density_list[i], T_list[i]))
#         else:
#             mu = mu_list[index]
#             B_plus_list.append(get_B_0_plus_photon_bath(energies, spec_density_list[i], T_list[i], mu))
#             B_plus_list.append(get_B_c_plus_photon_bath(energies, spec_density_list[i], T_list[i], mu))
#             index += 1
#     return B_plus_list
#
# def get_P(A,B_plus):
#     P_plus = A * B_plus/(2*math.pi)
#     P_minus = np.transpose(P_plus)
#     return P_plus, P_minus
#
# def get_P_lists(A_list, B_plus_list):
#     P_plus_list = []
#     P_minus_list = []
#     for i in range(len(A_list)):
#         P_plus_i, P_minus_i = get_P(A_list[i], B_plus_list[i])
#         P_plus_list.append(P_plus_i), P_minus_list.append(P_minus_i)
#     return P_plus_list, P_minus_list
#
# def get_M_ops(A_list, P_plus_list, P_minus_list):
#     M_plus = A_list[0]@P_plus_list[0]
#     M_minus = P_minus_list[0]@A_list[0]
#     for i in range(1, len(A_list)):
#         M_plus += A_list[i]@P_plus_list[i]
#         M_minus += P_minus_list[i]@A_list[i]
#     return M_plus, M_minus
#
# def get_secular_Redfield(A_eigenbasis_list, energies, spec_density_list, T_list, photon_bath_index, mu_list):
#
#     B_plus_list = get_B_plus_list(energies, spec_density_list, T_list, photon_bath_index, mu_list)
#     P_plus_list, P_minus_list = get_P_lists(A_eigenbasis_list, B_plus_list)
#     M_plus, M_minus = get_M_ops(A_eigenbasis_list, P_plus_list, P_minus_list)
#     return M_plus, M_minus, P_plus_list, P_minus_list
#
# def get_secular_Redfield_superoperator(A_eigenbasis_list, energies, spec_density_list, T_list, photon_bath_index, mu_list):
#     M_plus, M_minus, P_plus_list, P_minus_list = get_secular_Redfield(A_eigenbasis_list, energies, spec_density_list, T_list, photon_bath_index, mu_list)
#     M_plus_super = supervector.left_multiply(M_plus)
#     M_minus_super = supervector.right_multiply(M_minus)
#
#     summation = supervector.left_multiply(P_plus_list[0])@supervector.right_multiply(A_eigenbasis_list[0]) + supervector.left_multiply(A_eigenbasis_list[0])@supervector.right_multiply(P_minus_list[0])
#     for i in range(1, len(P_plus_list)):
#         summation += supervector.left_multiply(P_plus_list[i])@supervector.right_multiply(A_eigenbasis_list[i]) + supervector.left_multiply(A_eigenbasis_list[i])@supervector.right_multiply(P_minus_list[i])
#     return -1*(M_plus_super + M_minus_super) + summation #Should we also include the omega_ij term?
#
#
#
#
#
#
#































# def Redfield_tensor(gamma):
#     dim = gamma.shape[0]
#     r = np.empty((dim,dim,dim,dim), dtype = complex)
#     for m in range(dim):
#         for n in range(dim):
#             for o in range(dim):
#                 for p in range(dim):
#                     r[m,n,o,p] += (gamma[p,n,m,o] + np.conj(gamma[o,m,n,p]))
#                     if n == p:
#                         r[m,n,o,p] += -1*np.trace(gamma[m,:,:, o])
#                     if m == o:
#                         r[m,n,o,p] += -1*np.conj(np.trace(gamma[p,:,:,n]))
#     return r
#
# def supervector_derivative_redfield(redfield_tensors: list, evals): #Should we be passing in the mean field eigenvalues instead (see point 2 on page 383 of Chemical Dynamics in Condensed Phases by Nitzan)
#     """Takes in list of redfield tensors and closed-system energy eigenvalues as input and returns the derivative matrix for the rho supervector (using our supervector conventions from the Holstein code)"""
#     dim = redfield_tensors[0].shape[0]
#     D = np.empty((dim*dim, dim*dim), dtype = complex)
#     for i in range(dim*dim):
#         col_index = i // dim
#         row_index = i % dim
#         gap = evals[row_index] - evals[col_index]
#         D[i,i] = -1j*gap
#         for j in range(dim*dim):
#             for k in redfield_tensors:
#                 col_index_2 = j // dim
#                 row_index_2 = j % dim
#                 D[i,j] += k[row_index, col_index, row_index_2, col_index_2]
#     return D
#
# def harmonic_gamma_tensor_index(m,n,o,p, T, gap, sys_op_energy_eigenbasis, spec_density): #Need to ask Chern about what to do for T = 0 or omega = 0; for now I am replacing n_BE with n_BE + 1 based on equation 2.10a in https://pubs.aip.org/aip/jcp/article/130/23/234110/924762/On-the-adequacy-of-the-Redfield-equation-and
#     if gap == 0: #For now I am setting this to zero since in an Ohmic bath there are no oscillators at 0 T
#         return 0
#     if T == 0:
#         return sys_op_energy_eigenbasis[m,n]*sys_op_energy_eigenbasis[o,p]*spec_density(gap) #Assuming we set n_BE to 0 and use n_BE + 1
#     else:
#         return sys_op_energy_eigenbasis[m,n]*sys_op_energy_eigenbasis[o,p]*spec_density(gap)*(1/(math.exp(np.abs(gap)/T) - 1) + 1)
#
# def get_harmonic_gamma_tensor(T, evals, sys_op_energy_eigenbasis, spec_density):
#     dim = len(evals)
#     gamma = np.empty((dim, dim, dim, dim), dtype = complex) #Is there a more efficient way to do this than a nested for loop
#     for m in range(dim):
#         for n in range(dim):
#             for o in range(dim):
#                 for p in range(dim):
#                     gamma[m,n,o,p] = harmonic_gamma_tensor_index(m,n,o,p, T, evals[o] - evals[p], sys_op_energy_eigenbasis, spec_density) #Should we use the absolute value of the gap, or should we extend J(w) to negative frequencies as an antisymmetric function? I feel like we should use the absolute value of the gap, at least in the BE distribution
#     return gamma

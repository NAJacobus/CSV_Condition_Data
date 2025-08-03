import numpy as np
import matplotlib.pyplot as plt
import truncdec
def check_schwarz_inequality_violations(rho, title = "", tol = 0, prep_plot = True, return_mat_S = False, return_abs_vals = False, return_abs_val_sum = False, return_max_csv = False):
    """Checks the input matrix rho for Schwarz inequality violations; assumes rho is a Hermitian matrix"""
    dim = np.shape(rho)[0]
    S = 0
    blckwhite = np.zeros((dim,dim))
    absvals = np.zeros((dim,dim))
    absval_sum = 0
    max_csv = 0
    for i in range(dim):
        for j in range(i):
            S += np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j]
            if np.square(np.abs(rho[i,j])) > rho[i,i]*rho[j,j] + tol:
                blckwhite[i,j] = 1
                blckwhite[j,i] = 1
            if np.square(np.abs(rho[i, j])) > rho[i, i] * rho[j, j]:
                absvals[i,j] = np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j]
                absvals[j,i] = np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j]
                absval_sum += np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j]
            if np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j] > max_csv:
                max_csv = np.square(np.abs(rho[i,j])) - rho[i,i]*rho[j,j]
    S = truncdec.truncdec(np.real(S), 3)
    if prep_plot:
        fig1, ax1 = plt.subplots()
        plt.imshow(blckwhite, cmap = "Greys", interpolation = "nearest")
        plt.title(title + r" $\mathcal{S}$ = " + str(S))
    to_return = []
    if return_mat_S:
        to_return.append(blckwhite)
        to_return.append(S)
    if return_abs_vals:
        to_return.append(absvals)
    if return_abs_val_sum:
        to_return.append(np.real(absval_sum))
    if return_max_csv:
        to_return.append(np.real(max_csv))
    return to_return
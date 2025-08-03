import numpy as np

def pure_dens_op(psi):
    """Input must be n by 1 array; if not it changes it into one and gives a warning"""
    dims = np.shape(psi)
    if len(dims) == 1 or dims[1] != 1:
        print("Warning: Vector entered as 1xn instead of nx1; reshaping within file for density operator")
        psi_copy = np.copy(psi)
        psi = np.reshape(psi_copy, (dims[0], 1)) #I think this avoids aliasing issues by reassiging the variable psi to a different array than the input
    return psi@np.conj(np.transpose(psi))

if __name__ == "__main__":
    psi = np.reshape(np.array([0.6, 0.8]), (2,1))
    print(pure_dens_op(psi))

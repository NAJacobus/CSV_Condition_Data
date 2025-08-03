import scipy
import numpy

def custom_sparse_kron(A,B):
    """O(n^2) time complexity; still causes memory issues"""
    A = A.tocoo()
    B = B.tocoo()

    arows = A.row
    brows = B.row
    browlen = B.shape[0]
    #Construct rows of A kron B
    abrows = []
    for i in arows:
        for j in brows:
            abrows.append(i*browlen + j)

    acol = A.col
    bcol = B.col
    bcollen = B.shape[1]

    #Construct cols of A kron B
    abcols = []
    for i in acol:
        for j in bcol:
            abcols.append(i*bcollen + j)

    adata = A.data
    bdata = B.data

    #Construct data of A kron B
    abdata = []
    for i in adata:
        for j in bdata:
            abdata.append(i*j)

    abrow_dims = A.shape[0]*B.shape[0]
    abcol_dims = A.shape[1]*B.shape[1]
    return scipy.sparse.coo_matrix((abdata, (abrows, abcols)), shape =(abrow_dims, abcol_dims)).tocsr()


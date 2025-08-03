class RateConstantError(Exception):
    "var_x == var_y, or invalid expressions for var_x or var_y"

class Rb_Version_Error(Exception):
    "rb_version should be 'down' or 'up'"

class CSR_Error_1(Exception):
    """Sparse matrix format should be csr before taking kron"""

class CSR_Error_2(Exception):
    """Sparse matrix format should be csr after taking kron"""

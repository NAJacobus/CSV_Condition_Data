import numpy as np
import math

def db_ratio(pop_deriv, evals, beta):
    ratios = np.empty(np.shape(pop_deriv))
    for i in range(np.shape(pop_deriv)[0]):
        for j in range(np.shape(pop_deriv)[0]):
            log_k_ratio = math.log(pop_deriv[i,j]/pop_deriv[j,i])
            db = -beta*(evals[i] - evals[j])
            log_db_ratio = log_k_ratio - db
            ratios[i,j] = math.exp(log_db_ratio)
    return ratios
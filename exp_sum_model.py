import analysis as an
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def biexponential_ccdf(n, rate_1, rate_2):
    ccdf = .5*(np.exp(-rate_1*n) + np.exp(-rate_2*n))
    return ccdf

def estimate_parameters(ndata, ccdfdata, initial_guess, bounds):

    params, cov = curve_fit(biexponential_ccdf, ndata, ccdfdata,
                                p0=initial_guess,
                                bounds=bounds,
                                max_nfev=10_000,
                                xtol=1e-6,
                                gtol=1e-6,
                                method='dogbox',
                                verbose=2)
    
    return params, cov

if __name__ == '__main__':
    datalocation = 'Data/20220222_idx.xlsm'

    # Initial guess for parameters.
    guesses = [[3/400., 9., 8.],
               [3/400., 9., 5.],
               [3/400., 9., 6.],
               [3/400., 9., 5.],
               [3/400., 9., 30.],
               [3/400., 9., 17.],
               [3/400., 9., 9.],
               [3/400., 9., 54.]]

    tcodes = [1, 2, 3, 4, 5, 6, 7, 8]
import analysis as an
import celldynamics as cdn
import numpy as np
import birdepy as bd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def estimate_parameters(t, ndata, ccdfdata, initial_guess):

    params, cov = curve_fit(lambda n, birth_rate, death_rate, delta, n_crit: np.array([cdn.ccdfunc(n_element, [birth_rate, death_rate, delta, n_crit], t) for n_element in n]), ndata, ccdfdata, p0=initial_guess)

    return params, cov

if __name__ == '__main__':

    datalocation = 'Data/20220222_idx.xlsm'
    tcode = 8
    
    t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]

    data = an.read_excel_data(datalocation)
    ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)

    initial_guess = [1/100, 1/400, 1, 10]

    params, cov = estimate_parameters(t, ndata, ccdfdata, initial_guess)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ndata, ccdfdata, 'o', label='Data')
    ax.plot(ndata, [cdn.ccdfunc(n, params, t) for n in ndata], label='Fit')
    ax.set_yscale('log')
    ax.set_xlabel('Clone size')
    ax.set_ylabel('CCDF')
    ax.legend()
    fig.show()



import analysis as an
import celldynamics as cdn
import numpy as np
import birdepy as bd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def estimate_parameters(t, ndata, ccdfdata, initial_guess, bounds):

    params, cov = curve_fit(lambda n, birth_rate, death_rate, delta, n_crit: np.log(cdn.ccdfunc(n, [birth_rate, death_rate, delta, n_crit], t)), ndata, np.log(ccdfdata),
                            p0=initial_guess,
                            bounds=bounds,
                            max_nfev=100,
                            xtol=1e-8,
                            method='dogbox',
                            verbose=2) # Lmao doesnt change parameters

    return params, cov


if __name__ == '__main__':

    # Profiling boilerplate:

    import cProfile, pstats, io
    profiler = cProfile.Profile()
    profiler.enable()

    # Start code here

    datalocation = 'Data/20220222_idx.xlsm'
    tcode = 7
    initial_guess = [1/100., 1/1000., 9., 100.]

    
    t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]

    data = an.read_excel_data(datalocation)
    ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
    bounds = ([0, 0, 0, 0], [10, 10, 10, 1000])  # Tuneable.

    params, cov = estimate_parameters(t, ndata, ccdfdata, initial_guess, bounds)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ndata, ccdfdata, 'o', label='Data')
    ax.plot(ndata, cdn.ccdfunc(ndata, params, t), label='Fit')
    ax.set_yscale('log')
    ax.set_xlabel('Clone size')
    ax.set_ylabel('CCDF')
    ax.legend()
    fig.show()

    # End code here

    # Profiling boilerplate:

    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats()
    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())

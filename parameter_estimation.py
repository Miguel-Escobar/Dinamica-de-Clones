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
                            max_nfev=10_000,
                            xtol=1e-6,
                            gtol=1e-7,
                            method='dogbox',
                            verbose=2,
                            x_scale=[1, 1, 1, 10]) 

    return params, cov


if __name__ == '__main__':

    # Profiling boilerplate:

    import cProfile, pstats, io
    profiler = cProfile.Profile()
    profiler.enable()

    # Start code here

    datalocation = 'Data/20220222_idx.xlsm'
    # initial_guess = [1/100., 1/1000., 9., 1.]
    guesses = [[1/100., 1/1000., 9., 10.],
               [1/100., 1/1000., 9., 10.],
               [1/100., 1/1000., 9., 5.],
               [1/100., 1/1000., 9., 4.],
               [1/100., 1/1000., 9., 30.],
               [1/100., 1/1000., 9., 17.],
               [1/100., 1/1000., 9., 20.],
               [1/100., 1/1000., 9., 50.]]
    
    tcodes = [1, 2, 3, 4, 5, 6, 7, 8]


    with open('params.txt', 'w') as file:
        for tcode in tcodes:
            
            t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]
            initial_guess = guesses[tcode-1]

            data = an.read_excel_data(datalocation)
            ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
            bounds = ([0, 0, 0, 0], [10, 10, 100, 5000])  # Tuneable.

            params, cov = estimate_parameters(t, ndata, ccdfdata, initial_guess, bounds)

            print(f"tcode: {tcode}, params: {params}")

            file.write(f"tcode: {tcode}, params: {params} (birth_rate, death_rate, delta, n_crit)\n")

            fig = plt.figure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(ndata, ccdfdata, 'o', label='Data')
            ax.plot(ndata, cdn.ccdfunc(ndata, params, t), label='Fit')
            ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
            ax.set_yscale('log')
            ax.set_xlabel('Clone size')
            ax.set_ylabel('CCDF')
            ax.legend()
            fig.savefig(f"images/tcodefit_{tcode}.png", dpi=600)


    # End code here

    # Profiling boilerplate:

    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats()
    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())

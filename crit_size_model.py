import analysis as an
import celldynamics as cdn
import numpy as np
import birdepy as bd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def estimate_parameters(t, ndata, ccdfdata, n_crit, initial_guess, bounds):

    params, cov = curve_fit(lambda n, birth_rate, delta: np.log(cdn.ccdfunc(n, [birth_rate, delta, n_crit], t)), ndata, np.log(ccdfdata),
                            p0=initial_guess,
                            bounds=bounds,
                            max_nfev=10_000,
                            xtol=1e-6,
                            gtol=1e-6,
                            method='dogbox',
                            verbose=2,
                            x_scale=[1, 1])  # Not sure if this does anything.

    return params, cov


def func_to_optimize(birth_rate, delta, n_crit, t, ndata, ccdfdata):

    return np.sum((np.log(cdn.ccdfunc(ndata, [birth_rate, delta, n_crit], t)) - np.log(ccdfdata))**2)



if __name__ == '__main__':

    # # Profiling boilerplate:

    # import cProfile
    # import pstats
    # import io
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Start code here

    datalocation = 'Data/20220222_idx.xlsm'

    # Initial guesses for parameters. The critical sizes were obtained via a sweep of ndata:

    initial_guess = [3/400., 9.]
    n_critical_values = [5., 2., 5., 5., 28., 15., 2., 54.]

    tcodes = [1, 2, 3, 4, 5, 6, 7, 8]

    with open('critsize_params.csv', 'w') as file:

        file.write(f"tcode, birth_rate, delta, n_crit, r_score\n")

        for tcode in tcodes:

            t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]
            data = an.read_excel_data(datalocation)
            ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
            bounds = ([0, 0], [10, 100])  # Tuneable.

            n_crit = n_critical_values[tcode-1]

            print(f"\n Processing tcode {tcode}, with initial params {initial_guess[0]}, {initial_guess[1]}, {n_crit}\n")

            params, cov = estimate_parameters(t, ndata, ccdfdata, n_crit, initial_guess, bounds)
            r2 = an.r_score(lambda n, birth_rate, delta: np.log(cdn.ccdfunc(n, [birth_rate, delta, n_crit], t)), params, ndata, np.log(ccdfdata))

            print(f"\n Params at tcode {tcode} and n crit {n_crit}: {params}, with score {r2}\n")
            file.write(f"{tcode}, {params[0]}, {params[1]}, {n_crit}, {r2}\n")

            fig = plt.figure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(ndata, ccdfdata, 'o', label='Data')
            ax.plot(ndata, cdn.ccdfunc(ndata, [params[0], params[1], n_crit], t), label='Fit')
            ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
            ax.set_yscale('log')
            ax.set_xlabel('Clone size')
            ax.set_ylabel('CCDF')
            ax.legend()
            fig.savefig(f"images/critsize model/tcodefit_{tcode}.png", dpi=600)

    # End code here

    # # Profiling boilerplate:

    # profiler.disable()
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    # stats.print_stats()
    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())

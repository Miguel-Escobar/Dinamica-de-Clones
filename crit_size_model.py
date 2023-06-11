import analysis as an
import celldynamics as cdn
import numpy as np
import birdepy as bd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def estimate_parameters(t, ndata, ccdfdata, initial_guess, bounds):

    params, cov = curve_fit(lambda n, birth_rate, delta, n_crit: np.log(cdn.ccdfunc(n, [birth_rate, delta, n_crit], t)), ndata, np.log(ccdfdata),
                            p0=initial_guess,
                            bounds=bounds,
                            max_nfev=10_000,
                            xtol=1e-6,
                            gtol=1e-6,
                            method='dogbox',
                            verbose=2,
                            x_scale=[1, 1, 1e6])  # Not sure if this does anything.

    return params, cov


def func_to_optimize(birth_rate, delta, n_crit, t, ndata, ccdfdata):

    return np.sum((np.log(cdn.ccdfunc(ndata, [birth_rate, delta, n_crit], t)) - np.log(ccdfdata))**2)



if __name__ == '__main__':

    # Profiling boilerplate:

    import cProfile
    import pstats
    import io
    profiler = cProfile.Profile()
    profiler.enable()

    # Start code here

    datalocation = 'Data/20220222_idx.xlsm'

    # Initial guesses for parameters. These are tuned by hand. Trampa trampita jeje.
    guesses = [[3/400., 9., 8.],
               [3/400., 9., 5.],
               [3/400., 9., 6.],
               [3/400., 9., 5.],
               [3/400., 9., 30.],
               [3/400., 9., 17.],
               [3/400., 9., 9.],
               [3/400., 9., 54.]]

    tcodes = [8]#[1, 2, 3, 4, 5, 6, 7, 8]

    with open('params.txt', 'w') as file:

        file.write(f"tcode, birth_rate, delta, n_crit\n")

        for tcode in tcodes:

            t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]
            initial_guess = guesses[tcode-1]

            data = an.read_excel_data(datalocation)
            ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
            bounds = ([0, 0, 0], [10, 100, 300])  # Tuneable.

            print(f"\n Processing tcode {tcode}, with initial params {initial_guess}\n")

            params, cov = estimate_parameters(t, ndata, ccdfdata, initial_guess, bounds)

            print(f"\n Params at tcode {tcode}: {params}\n")

            # TESTING

            n_crit_sweep = np.arange(1, 101).astype(float)

            curve = [func_to_optimize(params[0], params[1], n_crit, t, ndata, ccdfdata) for n_crit in n_crit_sweep]

            argcurvemin = n_crit_sweep[np.argmin(curve)]

            test_fig = plt.figure(figsize=(8, 6))
            test_fig.clf()
            test_ax = test_fig.add_subplot(111)
            test_ax.plot(n_crit_sweep, curve)
            test_ax.plot(argcurvemin, np.min(curve), 'o', label=f'Minimum at n = {argcurvemin}')
            test_ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
            test_ax.set_xlabel(r'$n_{crit}$')
            test_ax.set_ylabel('Sum of squared residuals')
            test_ax.legend()
            test_fig.savefig(f"images/critsize model/tcode_{tcode}_nsweep.png", dpi=600)

            # END TESTING

            file.write(f"{tcode}, {params[0]}, {params[1]}, {params[2]}\n")

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
            fig.savefig(f"images/critsize model/tcodefit_{tcode}.png", dpi=600)

    # End code here

    # Profiling boilerplate:

    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats()
    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())

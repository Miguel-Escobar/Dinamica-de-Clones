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

    # Initial guesses for parameters. These are tuned by hand. Trampa trampita jeje.
    initial_guess = [3/400., 9.]

    tcodes = [1, 2, 3, 4, 5, 6, 7, 8]

    with open('critsize_params.csv', 'w') as file:

        file.write(f"tcode, birth_rate, delta, n_crit, r_score\n")

        for tcode in tcodes:

            t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]
            # initial_guess = guesses[tcode-1]
            # init_n0 = guesses[tcode-1][2]
            data = an.read_excel_data(datalocation)
            ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
            bounds = ([0, 0], [10, 100])  # Tuneable.

            for n_crit in ndata:

                print(f"\n Processing tcode {tcode}, with initial params {initial_guess}, {n_crit}\n")

                n_crit = float(n_crit)
                params, cov = estimate_parameters(t, ndata, ccdfdata, n_crit, initial_guess, bounds)
                r2 = an.r_score(lambda n, birth_rate, delta: np.log(cdn.ccdfunc(n, [birth_rate, delta, n_crit], t)), params, ndata, np.log(ccdfdata))

                print(f"\n Params at tcode {tcode} and n crit {n_crit}: {params}, with score {r2}\n")
                file.write(f"{tcode}, {params[0]}, {params[1]}, {n_crit}, {r2}\n")


                # TESTING

                # n_crit_sweep = np.linspace(3, 70, 135)

                # curve = [func_to_optimize(params[0], params[1], n_crit, t, ndata, ccdfdata) for n_crit in n_crit_sweep]

                # argcurvemin = n_crit_sweep[np.argmin(curve)]

                # test_fig = plt.figure(figsize=(8, 6))
                # test_fig.clf()
                # test_ax = test_fig.add_subplot(111)
                # test_ax.plot(n_crit_sweep, curve)
                # test_ax.plot(argcurvemin, np.min(curve), 'o', label=f'Minimum at n = {argcurvemin}')
                # test_ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
                # test_ax.set_xlabel(r'$n_{crit}$')
                # test_ax.set_ylabel('Sum of squared residuals')
                # test_ax.legend()
                # test_fig.savefig(f"images/critsize model/tcode_{tcode}_nsweep.png", dpi=600)

                # END TESTING

                # fig = plt.figure(figsize=(8, 6))
                # fig.clf()
                # ax = fig.add_subplot(111)
                # ax.plot(ndata, ccdfdata, 'o', label='Data')
                # ax.plot(ndata, cdn.ccdfunc(ndata, params, t), label='Fit')
                # ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
                # ax.set_yscale('log')
                # ax.set_xlabel('Clone size')
                # ax.set_ylabel('CCDF')
                # ax.legend()
                # fig.savefig(f"images/critsize model/tcodefit_{tcode}.png", dpi=600)

    # End code here

    # # Profiling boilerplate:

    # profiler.disable()
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    # stats.print_stats()
    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())

import analysis as an
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Note: this needs a more careful choice of initial guess than the current one.


def biexponential_ccdf(n, rate_1, rate_2):
    ccdf = .5*(np.exp(-rate_1*n) + np.exp(-rate_2*n))
    return ccdf

def log_biexponential_ccdf(n, rate_1, rate_2):
    logccdf = np.log(.5*(np.exp(-rate_1*n) + np.exp(-rate_2*n)))
    return logccdf

def estimate_parameters(ndata, ccdfdata, initial_guess, bounds):

    params, cov = curve_fit(log_biexponential_ccdf, ndata, np.log(ccdfdata),
                                p0=initial_guess,
                                bounds=bounds,
                                #max_nfev=10_000,
                                xtol=1e-6,
                                gtol=1e-6,
                                method='trf')
                                #verbose=2)
    
    return params, cov

if __name__ == '__main__':
    datalocation = 'Data/20220222_idx.xlsm'

    # Initial guess for parameters.

    initial_guess = [0.002, 0.001]
    tcodes = [1, 2, 3, 4, 5, 6, 7, 8]

    with open('expsum_params.csv', 'w') as file:

        file.write(f"tcode, rate_1, rate_2, r_score\n")

        for tcode in tcodes:
            data = an.read_excel_data(datalocation)
            ndata, ccdfdata = an.ccdf_at_tcode(tcode, data)
            bounds = ([0, 0], [10, 10])  # Tuneable.

            print(f"\n Processing tcode {tcode}, with initial params {initial_guess}\n")

            params, cov = estimate_parameters(ndata, ccdfdata, initial_guess, bounds)

            print(f"\n Params at tcode {tcode}: {params}\n")

            r2 = an.r_score(log_biexponential_ccdf, params, ndata, np.log(ccdfdata))
            file.write(f"{tcode}, {params[0]}, {params[1]}, {r2}\n")

            t = [24, 24, 3*24, 3*24, 6*24, 6*24, 13*24, 13*24][tcode-1]

            fig = plt.figure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(ndata, ccdfdata, 'o', label='Data')
            ax.plot(ndata, biexponential_ccdf(ndata, params[0], params[1]), label='Fit')
            ax.set_title(f"t: {t} [Hrs], EGF = {-1 + 2*(tcode % 2)}")
            ax.set_yscale('log')
            ax.set_xlabel('Clone size')
            ax.set_ylabel('CCDF')
            ax.legend()
            fig.savefig(f"images/biexp model/tcodefit_{tcode}.png", dpi=600)


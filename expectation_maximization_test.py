import numpy as np
import matplotlib.pyplot as plt
import birdepy as bd
import celldynamics
import analysis
from scipy.stats import nbinom
from scipy.integrate import quad

def estimate_parameters_critsize(measurement_times, measured_data, initial_guess, parameter_bounds, **kwargs): 
    estimation_output = bd.estimate([measurement_times for i in range(len(measured_data[0]))],
                                     measured_data.T,
                                     initial_guess,
                                     parameter_bounds,
                                     model='custom',
                                     b_rate=celldynamics.custom_birth_rate,
                                     d_rate=celldynamics.custom_death_rate,
                                     **kwargs)
    return estimation_output

def birth_death_pmf(nbirths, t, birth_rate, n0):
    return nbinom.pmf(nbirths, n0, np.exp(-birth_rate*t))

def probability(n, t, birth_rate_1, birth_rate_2, ncrit):
    """
    Probability of a clone of size n at time t.

    Parameters:
    n (int): The size of the clone.
    t (float): The time to compute the probability at.
    ncrit (int): The critical size of the clone.

    Returns:
    float: The probability of a clone of size n at time t.
    """
    if n <= ncrit:
        return np.exp(-birth_rate_1*t)*(1 - np.exp(-birth_rate_1*t))**(n - 1)
    else:
        integrand = lambda x: birth_death_pmf(ncrit, x, birth_rate_1, 1)*birth_death_pmf(n-ncrit, t-x, birth_rate_2, ncrit)
        return quad(integrand, 0, t)[0]



if __name__ == "__main__":

    # Generate fictituous data

    times = [1*24,3*24,6*24,13*24]
    data = celldynamics.simulate_critsize_model(20, times, 3/(4*82), 2., 30., 1).T

    # Plot some CCDFs

    measurement = 3
    sizes, distribution = analysis.clone_size_distribution(data[measurement])

    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(sizes, 1 - np.cumsum(distribution))
    ax.set_yscale("log")
    fig.show()

    # Estimate parameters

    initial_guess = [1/(4*82), 1., 30.]
    parameter_bounds = [(0, 1), (0, 10), (0, 100)]
    estimation = estimate_parameters_critsize(times, data, initial_guess, parameter_bounds, display=True)#, framework="em")
    print(estimation.p)
    print([ 3/(4*82), 2., 30.])



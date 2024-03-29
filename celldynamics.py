import birdepy as bd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import nbinom
from numba import njit


@njit
def heaviside(x): return .5*(np.sign(x) + 1)


@njit
def custom_birth_rate(z, p): return p[0]*(1 + p[1]*heaviside(z - round(p[2])))*z


@njit
def custom_death_rate(z, p): return 0


def P(i, j, t, params, method='ilt'):
    """
    Transition probability matrix.
    """
    p_ij = bd.probability(i, j, t, params,
                          model='custom',
                          b_rate=custom_birth_rate,
                          d_rate=custom_death_rate,
                          method=method)
    return p_ij


def prob_distribution(t, params, i0=1, imax=1000, method='expm', **kwargs):
    """
    Probability distribution at time t.
    """
    n = np.arange(i0, imax+1)
    p = bd.probability(i0, n, t, params,
                       model='custom',
                       b_rate=custom_birth_rate,
                       d_rate=custom_death_rate,
                       method=method,
                       **kwargs)
    return p.T


def ccdfunc(n, params, t, i0=1, method='expm'):
    """
    Complementary cumulative distribution function.
    """

    p = prob_distribution(t, params, i0=i0, imax=np.max(n), method=method, z_trunc=[i0, np.max(n) + 100])
    compdist = 1 - np.cumsum(p)
    
    return compdist[(n-1).astype(int)]


def plot_ccdf(t, params, i0=1, imax=500, method='expm'):
    """
    Plot complimentary cumulative probability distribution at time t.
    """
    n = np.arange(i0, imax+1)
    p = prob_distribution(t, params, i0=i0, imax=imax, method=method)
    p = 1 - np.cumsum(p)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(n, p)
    ax.set_xlabel('n')
    ax.set_ylabel('P(n)')
    ax.set_yscale("log")
    ax.set_title('t = {} [Hrs]'.format(t))

    fig.show()

    return

def simulate_critsize_model(n_realizations, measure_times, birth_rate, delta, crit_size, init_size):
    data = bd.simulate.discrete([birth_rate, delta, crit_size], 'custom', init_size,
                         b_rate = custom_birth_rate,
                         d_rate = custom_death_rate,
                         times=measure_times,
                         k=n_realizations
                        #  display=True
                         )
    return data

def birth_death_pmf(nbirths, t, birth_rate, n0):
    return nbinom.pmf(nbirths, n0, np.exp(-birth_rate*t))


def probability_density(n, t, birth_rate, delta, ncrit):
    """
    Probability of a clone of size n at time t.

    Parameters:
    n (int): The size of the clone.
    t (float): The time to compute the probability at.
    ncrit (int): The critical size of the clone.

    Returns:
    float: The probability of a clone of size n at time t.
    """

    print("DETERMINISTIC PROBABILITY DENSITY ALERT: THIS FUNCTION IS WRONG, OR AT LEAST NEEDS THINKING. HOLD OFF ON USING FOR NOW.")
    if n <= ncrit:
        return np.exp(-birth_rate*t)*(1 - np.exp(-birth_rate*t))**(n - 1)
    else:
        integrand = lambda x: birth_death_pmf(ncrit, x, birth_rate, 1)*birth_death_pmf(n-ncrit, t-x, birth_rate*(1 + delta), ncrit)
        return quad(integrand, 0, t)[0]

def deterministic_critsize_ccdf(n, birth_rate, delta, ncrit, t):
    ccdf = 1-np.cumsum([probability_density(i, t, birth_rate, delta, ncrit) for i in np.arange(1,np.max(n)+1)])
    return ccdf[(n-1).astype(int)]


if __name__ == '__main__':

    # import cProfile, pstats, io
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Start code here

    times = [1*24,3*24,6*24,13*24]
    data = simulate_critsize_model(100000, times, 3/(4*82), 2., 30., 1)
    

    # End code here

    # profiler.disable()
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    # stats.print_stats()
    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())

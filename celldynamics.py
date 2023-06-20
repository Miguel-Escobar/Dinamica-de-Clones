import birdepy as bd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    # import cProfile, pstats, io
    # profiler = cProfile.Profile()
    # profiler.enable()

    # Start code here

    params = [3/(4*82), 2., 30.]
    plot_ccdf(13*24, params, method='expm')

    # End code here

    # profiler.disable()
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    # stats.print_stats()
    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())

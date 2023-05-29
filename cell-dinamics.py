import birdepy as bd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def heaviside(x): return .5*(np.sign(x) + 1)

@njit
def custom_b_rate(z, p): return p[0]*(1 + p[2]*heaviside(z - p[3]))*z

@njit
def custom_d_rate(z, p): return p[1]*z


def P(i, j, t, params):
    """
    Transition probability matrix.
    """
    p_ij = bd.probability(i, j, t, params,
                          model='custom',
                          b_rate=custom_b_rate,
                          d_rate=custom_d_rate,
                          method='ilt')
    return p_ij

def prob_distribution(t, params, i0=1, imax=1000, method='expm'):
    """
    Probability distribution at time t.
    """
    n = np.arange(i0, imax+1)
    p = bd.probability(i0, n, t, params,
                       model='custom',
                       b_rate=custom_b_rate,
                       d_rate=custom_d_rate,
                       method=method)
    return p

def ccdf(p):
    """
    Complementary cumulative distribution function.
    """
    return 1 - np.cumsum(p)

def plot_ccdf(t, params, i0=1, imax=1000, method='expm'):
    """
    Plot probability distribution at time t.
    """
    n = np.arange(i0, imax+1)
    p = prob_distribution(t, params, i0=i0, imax=imax, method=method)
    p = ccdf(p)
    
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

    params = [1/82, 1/(4*82), 2., 30.]

    plot_ccdf(13*24, params, i0=1, imax=1000, method='expm')


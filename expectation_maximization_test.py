import numpy as np
import matplotlib.pyplot as plt
import celldynamics
import analysis



if __name__ == "__main__":

    # Generate fictituous data

    times = [1*24,3*24,6*24,13*24]
    data = celldynamics.simulate_critsize_model(100000, times, 3/(4*82), 2., 30., 1).T

    # Plot some CCDFs

    measurement = 3
    sizes, distribution = analysis.clone_size_distribution(data[measurement])

    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(sizes, 1 - np.cumsum(distribution))
    ax.set_yscale("log")
    fig.show()




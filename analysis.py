import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16  # Set the font size to 16
from collections import Counter
from scipy.optimize import curve_fit

def read_crit_size_params(file_name):
    """
    Reads critical size parameters from a CSV file with the given name.

    Parameters:
    file_name (str): The name of the CSV file to read.

    Returns:
    numpy.ndarray: A 2D array containing the critical size parameters.
    """
    df = pd.read_csv(file_name)
    params = df.values[:, 1:4]
    return params



def read_excel_data(file_name):
    """
    Reads excel data from the file with the given name, extracting the
    columns "Tcode" and "idx" from the "raw" sheet.

    Parameters:
    file_name (str): The name of the excel file to read.

    Returns:
    pandas.DataFrame: A dataframe containing the "Tcode" and "idx" columns
    from the "raw" sheet.
    """

    df = pd.read_excel(file_name, sheet_name='raw', usecols=['Tcode', 'idx'])

    return df


def clone_sizes(tcode, df):
    """
    Returns the clone sizes for the given Tcode in the provided dataframe.
    
    Parameters:
    tcode (str): The Tcode to retrieve the clone sizes for.
    df (pandas.DataFrame): The dataframe containing the "Tcode" and "idx" columns.
    
    Returns:
    numpy.ndarray: An array containing the sizes of the clones for the given Tcode.
    """

    df_tcode = df.query('Tcode == @tcode')
    idx = df_tcode['idx'].values
    clones_and_sizes = Counter(idx)  # This is a Dict.
    sizes = np.array(list(clones_and_sizes.values()))

    return sizes


def clone_size_distribution(sizes):
    """
    Computes the clone size distribution for a given list of clone sizes.

    Parameters:
    sizes (list): A list of integers representing the sizes of the clones.

    Returns:
    tuple: A tuple containing two arrays. The first array contains the clone
    sizes, and the second array contains the corresponding probabilities of
    each size in the input list.
    """

    dist = np.bincount(sizes)[1:]

    return np.arange(len(dist)) + 1, dist/sum(dist)
 

def ccdf_at_tcode(tcode, df):
    """
    Computes the complementary cumulative distribution function (CCDF) for a
    given Tcode in the provided dataframe.

    Parameters:
    tcode (str): The Tcode to compute the CCDF for.
    df (pandas.DataFrame): The dataframe containing the "Tcode" and "idx" columns.

    Returns:
    tuple: A tuple containing two arrays. The first array contains the clone
    sizes, and the second array contains the corresponding CCDF values.
    """

    measured_sizes = clone_sizes(tcode, df)
    sizes, dist = clone_size_distribution(measured_sizes)
    ccdf = 1 - np.cumsum(dist[1:])

    return sizes[1:], ccdf

def r_score(model, fit_params, ndata, ccdfdata):
    """
    Computes the R^2 score for a given model and its parameters, given the
    data to fit to.

    Parameters:
    model (function): The model function to fit to the data.
    fit_params (list): The parameters to fit to the data.
    ndata (numpy.ndarray): The x-values of the data to fit to.
    ccdfdata (numpy.ndarray): The y-values of the data to fit to.

    Returns:
    float: The R^2 score for the given model and parameters.
    """

    ccdfmodel = model(ndata, *fit_params)
    ss_res = np.sum((ccdfdata - ccdfmodel)**2)
    ss_tot = np.sum((ccdfdata - np.mean(ccdfdata))**2)
    r2 = 1 - ss_res/ss_tot

    return r2

def plot_critsize_params():
    """
    Plots the critical size parameters against time for EGF positive and negative
    conditions.

    Reads the critical size parameters from the "critsize_params.csv" file and
    plots the birth rate, delta * birth rate, and critical size against time
    for both EGF positive and negative conditions.

    Saves the resulting plot as "critsize_params_vs_t.png" if the user chooses
    to do so.

    Returns:
    None
    """
    t = [24, 3*24, 6*24, 13*24]
    params = read_crit_size_params("critsize_params.csv")
    egfnegative = params[np.arange(0, 4)*2 + 1]
    egfpositive = params[np.arange(0, 4)*2]

    fig = plt.figure("critsize params vs t", figsize=(15, 6))
    fig.clf()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(t, egfnegative[:, 0], 'o-', label="EGF negative")
    ax2.plot(t, egfnegative[:, 1]*egfnegative[:, 0], 'o-', label=r"EGF negative")
    ax3.plot(t, egfnegative[:, 2], 'o-', label=r"EGF negative")

    ax1.plot(t, egfpositive[:, 0], 'o-', label="EGF positive")
    ax2.plot(t, egfpositive[:, 1]*egfpositive[:, 0], 'o-', label=r"EGF positive")
    ax3.plot(t, egfpositive[:, 2], 'o-', label=r"EGF positive")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel("Time [Hrs]")
    ax2.set_xlabel("Time [Hrs]")
    ax3.set_xlabel("Time [Hrs]")

    ax1.set_ylabel("Birth rate [1/Hrs]")
    ax2.set_ylabel(r"$\delta * $ Birth rate [1/Hrs]")
    ax3.set_ylabel("Critical size")
    fig.tight_layout()
    fig.show()

    if input("Guardar el gráfico? [y/n]: ") == "y":
        fig.savefig("critsize_params_vs_t.png", dpi=600)
    plt.close()
    return

def plot_biexp_params():
    t = [24, 3*24, 6*24, 13*24][1:]
    params = read_crit_size_params("expsum_params.csv")
    egfnegative = params[np.arange(0, 4)*2 + 1][1:]
    egfpositive = params[np.arange(0, 4)*2][1:]

    fig = plt.figure("expsum params vs t", figsize=(15, 6))
    fig.clf()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(t, egfnegative[:, 0], 'o-', label="EGF negative")
    ax2.plot(t, egfnegative[:, 1], 'o-', label=r"EGF negative")
    ax3.plot(t, egfnegative[:, 2], 'o-', label=r"EGF negative")

    ax1.plot(t, egfpositive[:, 0], 'o-', label="EGF positive")
    ax2.plot(t, egfpositive[:, 1], 'o-', label=r"EGF positive")
    ax3.plot(t, egfpositive[:, 2], 'o-', label=r"EGF positive")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlabel("Time [Hrs]")
    ax2.set_xlabel("Time [Hrs]")
    ax3.set_xlabel("Time [Hrs]")

    ax1.set_ylabel("Population 1 exponential parameter")
    ax2.set_ylabel("Population 2 exponential parameter")
    ax3.set_ylabel("Initial population 1 ratio")
    fig.tight_layout()
    fig.show()

    if input("Guardar el gráfico? [y/n]: ") == "y":
        fig.savefig("expsum_params_vs_t.png", dpi=600)
    plt.close()
    return



    
if __name__ == "__main__":
    plot_biexp_params()
    plt.close()

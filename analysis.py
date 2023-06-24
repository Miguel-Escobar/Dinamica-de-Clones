import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit

def read_crit_size_params(file_name):

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


if __name__ == "__main__":
    t = [24, 3*24, 6*24, 13*24]
    params = read_crit_size_params("critsize_params.csv")
    egfnegative = params[np.arange(0, 4)*2 + 1]
    egfpositive = params[np.arange(0, 4)*2]

    fig = plt.figure("critsize params vs t")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(t, egfnegative[:, 0], label="Birth rate")
    ax1.plot(t, egfnegative[:, 1]*egfnegative[:, 0], label=r"$\delta*$ Birth rate")
    # ax1.plot(t, egfnegative[:, 2], label=r"$n_{crit}$")

    ax2.plot(t, egfpositive[:, 0], label="Birth rate")
    ax2.plot(t, egfpositive[:, 1]*egfpositive[:, 0], label=r"$\delta*$ Birth rate")
    # ax2.plot(t, egfpositive[:, 2], label=r"$n_{crit}$")

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel("Time (hours)")
    ax2.set_xlabel("Time (hours)")
    ax1.set_ylabel("Parameter value")
    # ax2.set_ylabel("Parameter value")

    ax1.set_title("EGF negative")
    ax2.set_title("EGF positive")
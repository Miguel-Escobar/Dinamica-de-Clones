import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import curve_fit


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

if __name__ == "__main__":

    datalocation = 'Data/20220222_idx.xlsm'
    tcode = 7

    df = read_excel_data(datalocation)
    n, ccdf = ccdf_at_tcode(tcode, df)

    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(n[:-1], ccdf[:-1], 'o', label='Tcode {}'.format(tcode))
    ax.set_xlabel('Clone size')
    ax.set_ylabel('CCDF')
    ax.legend()
    fig.show()

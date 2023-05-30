import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def read_excel_data(file_name):
    """
    Reads excel data, extracting from datasheet "raw" the columns "Tcode"
    and "idx". Returns a pandas dataframe with Tcode and idx.
    """
    df = pd.read_excel(file_name, sheet_name='raw', usecols=['Tcode', 'idx'])

    return df


def clone_sizes(tcode, df):
    """
    Returns the clone sizes at tcode.
    """
    df_tcode = df.query('Tcode == @tcode')
    idx = df_tcode['idx'].values
    clones_and_sizes = Counter(idx)  # This is a Dict.
    sizes = np.array(list(clones_and_sizes.values()))
    return sizes


def clone_size_distribution(sizes):
    dist = np.zeros(max(sizes))
    for s in sizes:
        dist[s-1] += 1
    return dist/sum(dist)

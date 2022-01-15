import numpy as np

def standardise_data(datamtx):
    """
    Standardise a data-matrix

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)

    returns
    -------
    std_datamtx - data matrix where columns have mean 0 and variance 1
    """
    means = np.mean(inputs, axis=0)
    stds = np.std(inputs, axis=0)
    return (inputs-means)/stds

def subsample_datapoints(datamtx, K):
    """
    Subsamples K data-points from a matrix

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    K - integer number of points to sample

    returns
    -------
    subsamples - a random sample of K data-points
    """
    ids = np.arange(datamtx.shape[0])
    np.random.shuffle(ids)
    sample_ids = ids[:K]
    subsamples = datamtx[sample_ids,:]
    return subsamples


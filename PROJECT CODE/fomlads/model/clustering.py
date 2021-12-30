import numpy as np

from fomlads.data.wrangle import subsample_datapoints

def squared_euclidean_distance(datamtx1, datamtx2):

    """
    parameters
    ----------
    datamtx1 - NxD array where each row is a vector
    datamtx2 - MxD array wehere each row is a vector

    returns
    -------
    distances - an NxM matrix whose (i,j)th element is the Euclidean distance
        from row i of datamtx1 to jth row of datamtx2
    """
    N, D1 = datamtx1.shape
    M, D2 = datamtx2.shape
    if D1 != D2:
        raise ValueError("Incompatible data matrices")
    datamtx1 = datamtx1.reshape(N, D1, 1)
    datamtx2 = datamtx2.T.reshape(1, D1, M)
    return np.sum((datamtx1-datamtx2)**2, axis=1)

def kmeans(datamtx, K, initial_centres=None, threshold=0.01, iterations=None):
    """
    Finds K clusters in data using the K-means algorithm

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    K - integer number of clusters to find
    initial_centres (optional) - KxD matrix of initial cluster centres
    threshold (optional) - the threshold magnitude for termination
    iterations (optional) - the maximum number of iterations

    returns
    -------
    centres - (KxD) matrix of cluster centres
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster
    """
    if initial_centres is None:
        centres = subsample_datapoints(datamtx, K)
    else:
        centres = initial_centres
    # initially change must be larger than threshold for algorithm to run 
    # at least one iteration
    change = 2*threshold
    iteration = 1
    while change > threshold:
        cluster_assignments = kmeans_e_step(datamtx, centres)
        centres, change = kmeans_m_step(datamtx, centres, cluster_assignments)
        if not iterations is None and iteration >= iterations:
            break
        iteration += 1
    #
    return centres, cluster_assignments

def kmeans_e_step(datamtx, centres):
    """
    The E step for the K-means algorithm

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    centres - (KxD) matrix of cluster centres

    returns
    -------
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster
    """
    N,D = datamtx.shape
    K = centres.shape[0]
    #datamtx = datamtx.reshape(N,D,1)
    #centres = centres.T.reshape(1,D,K)
    #distances = np.sum((datamtx - centres)**2, axis=1)
    distances = squared_euclidean_distance(datamtx, centres)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

def kmeans_m_step(datamtx, centres, cluster_assignments):
    """
    The M step for the K-means algorithm

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    centres - (KxD) matrix of cluster centres
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster

    returns
    -------
    centres - (KxD) matrix of updated cluster centres
    change - the magnitude of parameter change (the largest Euclidean distance
        between old centres and new centres)
    """
    change = 0.
    for k in range(centres.shape[0]):
        cluster_points = datamtx[cluster_assignments == k,:]
        new_centre = np.mean(cluster_points, axis=0)
        old_centre = centres[k,:]
        change = max(change, np.sum((new_centre-old_centre)**2))
        centres[k,:] = new_centre
    return centres, change


def calculate_kmeans_loss(datamtx, centres, cluster_assignments):
    """
    Evaluates the loss function J for kmeans

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    centres - (KxD) matrix of cluster centres
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster

    returns
    -------
    loss - numeric value of the loss function J
    """
    N, D = datamtx.shape
    loss = 0.
    for cluster_id in np.unique(cluster_assignments):
        # identify which points belong to this cluster
        assigned_to_this_cluster = (cluster_assignments == cluster_id)
        # get the centre as a 1xD matrix
        centre = centres[cluster_id,:].reshape((1,D))
        clustermtx = datamtx[assigned_to_this_cluster, :]
        #
        individual_distances = squared_euclidean_distance(clustermtx, centre)
        loss += np.sum(individual_distances)
    return loss


def kmedoids(proxmtx, K, initial_medoids=None, threshold=1, iterations=100):
    """
    Finds K clusters in data using the K-medoids algorithm

    parameters
    ----------
    proxmtx - (NxN) proximity matrix (array-like)
    K - integer number of clusters to find
    initial_medoids (optional) - K vector of initial medoids, , each is a
        data-point id 0..(N-1)
    threshold (optional) - the threshold magnitude for termination
    iterations (optional) - the maximum number of iterations

    returns
    -------
    medoids - (K) vector of medoids, each is a data-point id 0..(N-1)
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster
    """
    N = proxmtx.shape[0]
    if initial_medoids is None:
        medoids = np.random.choice(N, K, replace=False)
    else:
        medoids = initial_medoids
    # initially change must be larger than threshold for algorithm to run 
    # at least one iteration
    change = 2*threshold
    total_loss = np.inf
    iteration = 1
    while change > threshold:
        cluster_assignments = kmedoids_e_step(proxmtx, medoids)
        medoids, new_total_loss = kmedoids_m_step(
            proxmtx, medoids, cluster_assignments)
        change = total_loss - new_total_loss
        total_loss = new_total_loss
        if not iterations is None and iteration >= iterations:
            break
        iteration += 1
    #
    return medoids, cluster_assignments

def kmedoids_e_step(proxmtx, medoids):
    """
    The E step for the K-means algorithm

    parameters
    ----------
    proxmtx - (NxN) proximity matrix (array-like)
    medoids - K vector of medoids, each is a data-point id 0..(N-1)

    returns
    -------
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster
    """
    # each row is that data-points distance to each of the medoids
    distances_to_medoids = proxmtx[:,medoids]
    cluster_assignments = np.argmin(distances_to_medoids, axis=1)
    return cluster_assignments

def kmedoids_m_step(proxmtx, medoids, cluster_assignments):
    """
    The M step for the K-means algorithm

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    medoids - (K) vector of cluster centroid ids
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster

    returns
    -------
    centres - (KxD) matrix of updated cluster centres
    total_loss - the new total loss of the clustering
    """
    total_loss = 0.
    for k, medoid in enumerate(medoids):
        # points assigned to this cluster
        cluster_points = (cluster_assignments == k)
        # the indices where cluster_points is true are the indices of this
        # cluster's points
        cluster_ids = np.where(cluster_points)[0]
        # first filter rows then filter columns to keep only proximities within
        # this cluster
        cluster_proxmtx = proxmtx[cluster_points, :]
        cluster_proxmtx = cluster_proxmtx[:, cluster_points]
        # the cluster loss  for each cluster point
        cluster_losses = np.sum(cluster_proxmtx, axis=1)
        # the new medoid is the id of the point in this cluster with the
        # smallest cluster loss
        index_of_min_loss = np.argmin(cluster_losses)
        cluster_loss = cluster_losses[index_of_min_loss]
        new_medoid = cluster_ids[index_of_min_loss]
        # add cluster loss to the total
        total_loss += cluster_loss
        medoids[k] = new_medoid
    return medoids, total_loss

def calculate_kmedoids_loss(proxmtx, medoids, cluster_assignments):
    """
    Evaluates the loss function J for kmeans

    parameters
    ----------
    proxmtx - (NxD) data matrix (array-like)
    medoids - (KxD) matrix of cluster centres
    cluster_assignments - a vector of N integers 0..(K-1), one per data-point,
        assigning that data-point to a cluster

    returns
    -------
    loss - numeric value of the loss function J
    """
    N = proxmtx.shape[0]
    loss = 0.
    for cluster_id in np.unique(cluster_assignments):
        medoid = medoids[cluster_id]
        # identify which points belong to this cluster
        cluster_members = (cluster_assignments == cluster_id)
        # get the cluster point proximities to this medoid
        all_medoid_proximities = proxmtx[:,medoid]
        medoid_proximities = all_medoid_proximities[cluster_members]
        #
        loss += np.sum(medoid_proximities)
    return loss




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from fomlads.plot.exploratory import plot_scatter_array_classes

def plot_K_versus_loss(Ks, losses, datamtx=None, title=None, ylabel=None):
    """
    Plots the loss as a function of the number of clusters. 

    parameters
    ----------
    Ks - sequence of K values (1d array) where K is the number of clusters to 
        find for K-means or similar algorithm
    losses - associated losses, one per K value
    datamtx (optional) - Nx2 array of data points. If provided the funciton will
      also plot datapoints in accompanying plot
    """
    if datamtx is None:
        fig = plt.figure()
        ax2 = fig.add_subplot(1,1,1)
    else:
        w, h = plt.figaspect(0.4)
        fig = plt.figure(figsize=(w,h))
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(datamtx[:,0], datamtx[:,1], 'ko')
        ax2 = fig.add_subplot(1,2,2)
    # plot K versus the loss 
    ax2.plot(Ks, losses, 'b-', linewidth=3)
    ax2.set_xticks(Ks)
    ax2.set_xlabel('K')
    # label the y-axis
    if not ylabel is None:
        ax2.set_ylabel(ylabel)
    # give the plots a title
    if not title is None:
        fig.suptitle(title)

def scatter_plot_responsibilities(
        datamtx, log_resps, means=None, cmap_name='rainbow',
        markersize=5):
    """
    Plots a single scatter plots of input data, coloured according to 
    responsibilities

    parameters
    ----------
    datamtx - (Nx2) data matrix of input values (array-like)
    log_resps - (NxK) matrix of logs of component responsibilities one row per
        datapoint, one column per component
    means (optional) - a (Kx2) array of the component means
    cmap_name (optional) - the name of the color map to use
    """
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    N, K = log_resps.shape
    cmap = plt.cm.get_cmap(cmap_name, K)
    component_colours = np.array([cmap(k) for k in range(K)])
    data_colours = np.empty((N, component_colours.shape[1]))
    for n in range(N):
        r_n = np.exp(log_resps[n,:]).reshape((K,1))
        data_colours[n,:] = np.sum(component_colours*r_n, axis=0)
    for n,(x, y) in enumerate(datamtx):
        # each point is coloured as a mixture of component colours with mixing
        # proportions according to the responsibilitiy r_nk
        c = data_colours[n,:]
        # precision errors can cause numbers outside of [0,1] so we correct
        c = np.minimum(c, np.ones(c.size))
        c = np.maximum(c, np.zeros(c.size))
        ax.plot(x,y, 'o', c=c, markersize=markersize)
    if not means is None:
        for k, mean_k in enumerate(means):
            colour_k = component_colours[k,:]
            ax.plot(
                mean_k[0], mean_k[1], 'x', color=colour_k,
                markersize=3*markersize, markeredgewidth=2)
    return fig, ax

def scatter_plot_clusters(
        datamtx, cluster_assignments, centres=None, cmap_name='rainbow',
        marker_types=None, centre_color=None, **kwargs):
    """
    Plots a single scatter plots of input data, coloured according to cluster assignments

    parameters
    ----------
    datamtx - (Nx2) data matrix of input values (array-like)
    cluster_assignments - 1d vector of class values as integers (array-like)
    centres (optional) - a (Kx2) array of the cluster means
    cmap_name (optional) - the name of the color map to use
    """
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    cluster_ids = np.unique(cluster_assignments)
    num_clusters = cluster_ids.size
    if cmap_name is None:
        cmap = lambda x: 'k'
    else:
        cmap = plt.cm.get_cmap(cmap_name, num_clusters)
    if marker_types is None:
        marker_types = ['o']*num_clusters

    for cluster_id, marker_type in zip(cluster_ids, marker_types):
        cluster_color = cmap(cluster_id)
        cluster_points = datamtx[cluster_assignments==cluster_id,:]
        x = cluster_points[:,0]
        y = cluster_points[:,1]
        ax.plot(x,y, marker_type, color=cluster_color, **kwargs)
    if not centres is None:
        for cluster_id, cluster_mean in enumerate(centres):
            if centre_color is None:
                this_color = cmap(cluster_id)
            ax.plot(
                cluster_mean[0], cluster_mean[1], 'x', color=this_color,
                markersize=15, markeredgewidth=2)
    return fig, ax




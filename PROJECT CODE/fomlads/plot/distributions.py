import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 


def plot_simple_gaussian(mu, sigma2, xlim):
    """
    This function plots a simple gaussian curve, with mean mu, variance sigma2
    in the x range xlim.
    """
    # get the x values we wish to plot at
    xs = np.linspace(xlim[0],xlim[1],101)
    # calculate the associated y values
    ys = prob_dens_gaussian(xs, mu, sigma2)
    # plot likelihood function
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xs,ys)
    ax.set_xlabel("x")
    ax.set_ylabel("$p(x)$")
    fig.tight_layout()
    return fig, ax

def overlay_2d_gaussian_contour(ax, mu, Sigma, num_grid_points=60, levels=10):
    """
    Overlays the contours of a 2d-gaussian with mean, mu, and covariance matrix
    Sigma onto an existing set of axes.

    parameters
    ----------
    ax -- a matplotlib.axes.Axes object on which to plot the contours
    mu -- a 2-vector mean of the distribution
    Sigma -- the (2x2)-covariance matrix of the distribution.
    num_grid_points (optional) -- the number of grid_points along each dimension
      at which to evaluate the pdf
    levels (optional) -- the number of contours (or the function values at which 
      to draw contours)
    """
    # generate num_grid_points grid-points in each dimension
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xpoints = np.linspace(xmin, xmax, num_grid_points)
    ypoints = np.linspace(ymin, ymax, num_grid_points)
    # meshgrid produces two 2d arrays of the x and y coordinates
    xgrid, ygrid = np.meshgrid(xpoints, ypoints)
    # Pack xgrid and ygrid into a single 3-dimensional array
    pos = np.empty(xgrid.shape + (2,))
    pos[:, :, 0] = xgrid
    pos[:, :, 1] = ygrid
    # create a distribution over the random variable 
    rv = stats.multivariate_normal(mu, Sigma)
    # evaluate the rv probability density at every point on the grid
    prob_density = rv.pdf(pos)
    # plot the contours
    ax.contour(xgrid, ygrid, prob_density, levels=levels)


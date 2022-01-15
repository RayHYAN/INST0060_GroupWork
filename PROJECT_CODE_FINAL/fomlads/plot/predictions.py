import numpy as np
import matplotlib.pyplot as plt

def plot_function(true_func, linewidth=3, xlim=None):
    """
    Plot a function in a given range

    parameters
    ----------
    true_func - the function to plot
    xlim (optional) - the range of values to plot for. A pair of values lower,
        upper. If not specified, the default will be (0,1)
    linewidth (optional) - the width of the plotted line

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the (one) line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(xlim[0], xlim[1], 101)
    true_ys = true_func(xs)
    line, = ax.plot(xs, true_ys, 'g-', linewidth=linewidth)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.5, 1.5)
    return fig, ax, [line]

def plot_function_and_data(inputs, targets, true_func, markersize=5, **kwargs):
    """
    Plot a function and some associated regression data in a given range

    parameters
    ----------
    inputs - the input data
    targets - the targets
    true_func - the function to plot
    markersize (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    fig, ax, lines = plot_function(true_func)
    line, = ax.plot(inputs, targets, 'bo', markersize=markersize)
    lines.append(line)
    return fig, ax, lines

def plot_function_data_and_approximation(
        predict_func, inputs, targets, true_func, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    true_func - the true function
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig, ax, lines = plot_function_and_data(
        inputs, targets, true_func, linewidth=linewidth, xlim=xlim, **kwargs)
    xs = np.linspace(0, 1, 101)
    ys = predict_func(xs)
    line, = ax.plot(xs, ys, 'r-', linewidth=linewidth)
    lines.append(line)
    return fig, ax, lines

def plot_2d_decision_boundary(pred_func, X, y, flush=True, fig_ax=None):
    N, D = X.shape
    if D != 2:
        raise ValueError("Only works with 2d input data")
    fig, ax = plt.subplots()
    y = (y>0).astype(int)
    # min and max values (with padding)
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # generate grid of points with distance h
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    # predict function value for whole grid
    Z = pred_func(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    # plot contour and training examples
    ax.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c=y, cmap=plt.cm.Spectral)


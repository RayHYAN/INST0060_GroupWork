import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors,
        train_stes=None, test_stes=None):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')

    if not train_stes is None:
        train_upper = train_errors + train_stes
        train_lower = train_errors - train_stes
        ax.fill_between(
            experiment_sequence,train_lower,train_upper, color='b', alpha=0.2)
    if not test_stes is None:
        test_upper = test_errors + test_stes
        test_lower = test_errors - test_stes
        ax.fill_between(
            experiment_sequence,test_lower,test_upper, color='r', alpha=0.2)

    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line], ["train", "test"])
    return fig, ax

def plot_roc(
        false_positive_rates, true_positive_rates, linewidth=3, fig_ax=None,
        colour=None, ofname=None, **kwargs):
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    else:
        fig, ax = fig_ax
    if colour is None:
        colour='r'
    ax.plot(
        false_positive_rates, true_positive_rates, '-', linewidth=linewidth,
        color=colour, **kwargs)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    plt.tight_layout()
    if not ofname is None:
        print("Saving ROC curve as: %s" % ofname)
        fig.savefig(ofname)
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt

def eval_accuracy(targets, predicts):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predicts - the predictions, a 1d array  of 1s and 0s respectively
        predicting targets of class 1 and 0 

    Returns
    -------
    error - The minimum-misclassification error between true and predicted target
    """
    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    N = targets.size
    return np.sum(targets == predicts)/N

def misclassification_error(targets, predicts):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predicts - the predictions, a 1d array  of 1s and 0s respectively
        predicting targets of class 1 and 0 

    Returns
    -------
    error - The minimum-misclassification error between true and predicted target
    """
    return 1 - eval_accuracy(targets, predicts)

def expected_loss(targets, predicts, lossmtx):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predicts - the predictions, a 1d array  of 1s and 0s respectively
        predicting targets of class 1 and 0 
    lossmtx - a matrix (2x2) of loss values for misclassification

    Returns
    -------
    error - An estimate of the expected loss between true and predicted target
    """
    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    class0 = (targets == 0)
    class1 = np.invert(class0)
    predicts0 = (predicts == 0)
    predicts1 = np.invert(predicts0)
    class0loss = lossmtx[0,0]*np.sum(class0 & predicts0) \
        + lossmtx[0,1]*np.sum(class1 & predicts1)
    class1loss = lossmtx[1,0]*np.sum(class0 & predicts0) \
        + lossmtx[1,1]*np.sum(class1 & predicts1)
    N = targets.size
    error = (class0loss + class1loss)/N
    return error

def cross_entropy_error(targets, predict_probs):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predict_probs - the prediction probabilities, a 1d array of probabilities 
        each predicting the probability of class 1 for the corresponding target

    Returns
    -------
    error - The cross-entropy error between true and predicted target
    """
    # flatted
    targets = np.array(targets).flatten()
    predict_probs = np.array(predict_probs).flatten()
    N = targets.size
    error = - np.sum(
        targets*np.log(predict_probs) + (1-targets)*np.log(1-predict_probs))/N
    return error


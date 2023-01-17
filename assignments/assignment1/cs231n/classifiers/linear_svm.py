from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
              continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                ################################
                dW[:,j] += X[i]             ##add a fraction of every_class W
                dW[:,y[i]] -= X[i]          ##add a fraction of corrrect_class W
                ################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dW /= num_train
    
    dW_regFrac = np.zeros(W.shape)
    for c in range(num_classes):
      dW_regFrac[:,c] += 2 * W[:,c]
    
    dW += reg * dW_regFrac
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 0. num_train = N, num_dimension = D, num_classes = C 
    num_train = X.shape[0]  
    num_classes = W.shape[1]

    # 1. create XW (N * D)
    XW = X @ W

    # 2. create correct_mask (N * D) indicating the position of correct class
    correct_mask = np.ones_like(XW, dtype=bool)
    correct_mask[np.arange(num_train), y] = False
    
    # 3. divide XW into two with correct_mask 
    #    XW_noCorrectClass (N * D-1)
    #    XW_CorrectClass   (N * 1)
    XW_noCorrectClass = XW[correct_mask].reshape((num_train, num_classes-1))
    XW_CorrectClass = XW[~correct_mask].reshape((num_train, 1))

    # 4. calculate scores (N * D-1) from above
    #    remember loss is not calculated in the position of correct class
    scores = XW_noCorrectClass - XW_CorrectClass + 1  # broadcasting, 1 is delta
    scores = np.where(scores < 0, 0, scores)          # max(0,___)
    
    # 5. calculate loss from scores
    loss = np.sum(scores) / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. create score_mask (N * D-1) to indicate the position where score is 0
    score_mask = np.where(scores == 0, 0, 1)
    
    # 2. expand it to calc_table (N * D) which has the value 1 where score is positive, and else 0 
    calc_table = np.zeros_like(XW)                        #value == 0       i == j
    calc_table[correct_mask] = score_mask.ravel()         #value == 1       positive score
                                                          #value == 0       negative score

    # 3. modify calc_table to make sure in gives dW when multiplied with X.T (X.T @ calc_table)
    #    Which means calc_table has coordinate of each Wj respect to X.T
    #    leave the cell with value 1
    #    add -sum value of 1 in rows to position of correct class
    num_subtraction_list = calc_table.sum(axis=1)
    calc_table[~correct_mask] = -1 * num_subtraction_list
    
    # 4. calculate X.T @ calc_table, and normalize to get dW
    dW = X.T @ calc_table / num_train
    
    # 5. add the regulation fragment
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

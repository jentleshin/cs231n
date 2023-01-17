from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_score = scores[y[i]]
        
        # normalize scores by subtracting with correct_score
        n_scores = scores - correct_score #broadcasting
        
        # get exp(normalized correct score) (which is e^0), and sum of exp(normalized scores)
        en_scores = np.exp(n_scores)
        en_correct_score = np.exp(0.0)
        sum_en_scores = np.sum(en_scores)
        
        # dW fraction for each X[i]
        dW_fraction = np.zeros_like(W)
        
        for j in range(num_classes):
          # when correct class
          if j == y[i]:
            dW_fraction[:,j] += X[i]
          # when not correct class
          dW_fraction[:,j] -= (en_scores[j] / sum_en_scores) * X[i]

        # update loss and dW
        loss += np.log(en_correct_score) - np.log(sum_en_scores)
        dW += dW_fraction

    
    dW_regFrac = np.zeros(W.shape)
    for c in range(num_classes):
      dW_regFrac[:,c] += 2 * W[:,c]

    # -1/N * (loss, dW)
    loss = -1 / num_train * loss + reg * np.sum(W * W)
    dW = -1 / num_train * dW + reg * dW_regFrac

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X @ W

    # 1. normalize, exp scores
    n_scores = scores - scores[np.arange(num_train), y].reshape(num_train,1)
    en_scores = np.exp(n_scores)

    # 2. Sum of en_scores in columns (N)
    column_sums = np.sum(en_scores, axis =1).reshape(num_train,1)

    ###########################################################

    # 3. optimization + regularization
    loss = 1 / num_train * np.sum(np.log(column_sums))  #Broadcasting
    loss += reg * np.sum(W * W)
    
    ###########################################################

    # 1. calculate ( e^cell / column_sum(e^cell) ) for each cell
    coordinate = -1 * en_scores / column_sums #Broadcasting

    # 2. create correct_mask (N * D) of scores,
    #    indicating the position of correct class as True
    correct_mask = np.full_like(scores, False, dtype=bool)
    correct_mask[np.arange(num_train), y] = True
    coordinate[correct_mask] += 1

    # 3. optimization + regularization
    dW = -1 / num_train * (X.T @ coordinate) 
    dW += reg * 2 * W
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

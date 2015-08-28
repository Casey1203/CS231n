import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  for i in xrange(num_train):
    for j in xrange(num_classes):
      scores = W.dot(X[:, i])
      scores = math.exp(1) ** scores
      dW[j, :] += (scores[j] / np.sum(scores)) * X[:, i].transpose()/num_train + reg * W[j, :]
      if j == y[i]:        
        loss += -math.log(scores[j] / np.sum(scores))/num_train + reg * 0.5 * np.sum(W ** 2)
        dW[j, :] -= X[:, i].transpose() / num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  dim, num_train = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = W.dot(X) # C x X
  scores = np.exp(scores)
  sumScores = sum(scores, 0)
  scores = scores / sumScores
  ground_truth = np.zeros_like(scores) # C x X
  # the value is 1.0 at the y[i] element in each column, otherwise 0.0
  ground_truth[tuple([y, range(len(y))])] = 1.0 
  loss = -np.sum(ground_truth * np.log(scores)) / num_train + reg * 0.5 * np.sum(W ** 2)
  dW = -((ground_truth - scores).dot(X.transpose())) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

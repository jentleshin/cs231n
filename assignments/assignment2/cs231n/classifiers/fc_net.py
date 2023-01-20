from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        rng = np.random.default_rng()
        
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(num_classes)

        for i in range(self.num_layers-1) :
          self.params['W{}'.format(i+1)] = rng.normal(loc=0.0, scale=weight_scale, size=(dims[i], dims[i+1]))
          self.params['b{}'.format(i+1)] = np.zeros(hidden_dims[i])
        if self.normalization == "batchnorm":
          for i in range(self.num_layers-2) :
            self.params['gamma{}'.format(i+1)] = np.ones(hidden_dims[i])
            self.params['beta{}'.format(i+1)] = np.zeros(hidden_dims[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        caches = {}

        if self.use_dropout and self.normalization != "batchnorm":
          Func = {
            "n":(affine_relu_dropout_forward, "params/W", "params/b", "dropout_param"),
            "max":(affine_forward, "params/W", "params/b")
          }

        elif ~self.use_dropout and self.normalization == "batchnorm":
          Func = {
            "n":(affine_batchnorm_relu_forward, "params/W", "params/b", "params/gamma", "params/beta", "bn_params"),
            "max":(affine_forward, "params/W", "params/b")
          }

        elif self.use_dropout and self.normalization == "batchnorm":
          Func = {
            "n":(affine_batchnorm_relu_dropout_forward, "params/W", "params/b", "params/gamma", "params/beta", "bn_params", "dropout_param"),
            "max":(affine_forward, "params/W", "params/b")
          }

        else:
          Func = {
            "n":(affine_relu_forward, "params/W", "params/b"),
            "max":(affine_forward, "params/W", "params/b")
          }

        ###########################################################################

        def parse(string, count):
          if "/" not in string:
            if "bn_params" == string:
              return getattr(self, string)[count-1]
            return getattr(self, string)
          [where, name] = string.split("/")
          return getattr(self, where)[name + str(count)]
        
        ###########################################################################
        
        def repeat(Func, caches, count, count_max, initial_input):
            
          if count == 1 :
            fn ,*fn_args_st = Func["n"]
            fn_args = [ parse(st,count) for st in fn_args_st ]
            H, cache = fn(initial_input, *fn_args)
            caches["c{}".format(count)] = cache
            
            return H, caches

          X = repeat(Func, caches, count-1, count_max, initial_input)[0]

          if count == count_max:
            fmax ,*fmax_args_st = Func["max"]
            fmax_args = [ parse(st,count) for st in fmax_args_st ]
            H, cache = fmax(X, *fmax_args) 
          
          else:
            fn ,*fn_args_st = Func["n"]
            fn_args = [ parse(st,count) for st in fn_args_st ]
            H, cache = fn(X, *fn_args)
          
          caches["c{}".format(count)] = cache
          
          return H, caches
        
        ##########################################################################
        
        count_max = self.num_layers-1
        count = count_max 
        scores, caches = repeat(Func, caches, count, count_max, X)      

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss, dL = softmax_loss(scores, y)
        
        ###########################################################################
        # reccursive function for calculating backpropagation                     #
        # count flows from 1 to max, corresponding to hidden layer number.        #
        ###########################################################################

        if self.use_dropout and self.normalization != "batchnorm":
          Func = {
            "n":(affine_relu_dropout_backward, "W", "b"),
            "max":(affine_backward, "W", "b")
          }
        elif ~self.use_dropout and self.normalization == "batchnorm":
          Func = {
            "n":(affine_batchnorm_relu_backward, "W", "b", "gamma", "beta"),
            "max":(affine_backward, "W", "b")
          }
        elif self.use_dropout and self.normalization == "batchnorm":
          Func = {
            "n":(affine_batchnorm_relu_dropout_backward, "W", "b", "gamma", "beta"),
            "max":(affine_backward, "W", "b")
          }
        else:
          Func = {
            "n":(affine_relu_backward, "W", "b"),
            "max":(affine_backward, "W", "b")
          }

        ###########################################################################

        def reverse(Func, caches, grads, count, count_max, initial_input):
          cache = caches["c{}".format(count)]
          
          # initial condition
          if count == count_max :
            fmax ,*fmax_args_st = Func["max"]
            fmax_args_name = [ st + str(count) for st in fmax_args_st ]
            dX, *D = fmax(initial_input, cache)
            for name, d in zip(fmax_args_name, D):
              grads[name] = d
            return dX, grads

          fn ,*fn_args_st = Func["n"]
          fn_args_name = [ st + str(count) for st in fn_args_st ]
          dout = reverse(Func, caches, grads, count+1, count_max, initial_input)[0]  
          dX, *D = fn(dout, cache)
          for name, d in zip(fn_args_name, D):
              grads[name] = d
          return dX, grads
        
        ###########################################################################
        
        count = 1     
        count_max = self.num_layers-1
        dX, grads = reverse(Func, caches, grads, count, count_max, dL)

        ###########################################################################
        
        # add regularization
        # not when using batch/layer normalization
        if self.normalization != "batchnorm" and self.normalization != "layernorm":

          for i in range(self.num_layers-1):
            W = self.params["W{}".format(count)]
            dW = grads["W{}".format(count)]
            
            loss += 0.5 * self.reg * np.sum(W * W)
            grads["W{}".format(count)] += self.reg * W 
          
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

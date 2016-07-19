__author__ = 'dhl'

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from collections import OrderedDict

def relu(x):
    return T.maximum(0.0, x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def as_floatx(variable):
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = numpy.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatx(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatx(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie
    """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None,
                 use_bias=True):
        self.input = input
        self.activation = activation

        if W is None:
            if activation.func_name == 'relu':
                W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)),
                                         dtype=theano.config.floatX)
            else:
                W_bound = numpy.sqrt(6. / (n_in + n_out))
                W_values = numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                    dtype=theano.config.floatX
                )
            W = theano.shared(value=W_values, name='W')
            # if broadcast:
            #     W = W.dimshuffle('x', 0, 1)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
            # if broadcast:
            #     b = b.dimshuffle('x', 0)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W)
        if use_bias:
            lin_output += self.b

        self.output = (lin_output if activation is None else activation(lin_output))

        self.params = [self.W, self.b] if use_bias else [self.W]


# CNN pooling
class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear='tanh',
                 W=None, b=None):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        if W is None:
            if self.non_linear == 'none' or self.non_linear == 'relu':
                self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01, high=0.01,
                                                                 size=filter_shape),
                                                     dtype=theano.config.floatX),
                                       borrow=True,
                                       name='W_conv')
            else:
                W_bound = numpy.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                     dtype=theano.config.floatX),
                                       borrow=True,
                                       name="W_conv")
        else:
            self.W = W

        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        else:
            self.b = b

        # ftmp = theano.function([], self.W.shape)
        # print ftmp()
        conv_out = conv.conv2d(input=input,
                               filters=self.W,
                               filter_shape=self.filter_shape,
                               image_shape=self.image_shape)

        if self.non_linear == 'tanh':
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == 'relu':
            conv_out_tanh = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W,
                               filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear == 'tanh':
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == 'relu':
            conv_out_relu = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = downsample.max_pool_2d(input=conv_out_relu, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


def index_select(params, indices):
    """Indexes through the last dimension of params.

    Arguments:
        params, indices (tensor): indices.shape == params.shape[:-1]

    Returns:
        output (tensor of same shape as indices): output[...] = params[..., indices[...]]
    """
    assert params.shape[:-1] == indices.shape, "shape mismatch"
    return tf.reduce_sum(tf.one_hot(indices, int(params.shape[-1])) * params, axis=-1)


def broadcast_to(inputs, shape):
    """Broadcasts inputs to the given shape. However it starts with the leading dimensions.

    Arguments:
        inputs (tensor): the tensor to broadcast
        shape (list of ints): the output shape

    Returns:
        output (tensor): inputs broadcast to shape
    """
    l = len(inputs.shape)
    assert inputs.shape == shape[:l], "shape mismatch"
    inputs = tf.reshape(inputs, inputs.shape.as_list() + [1] * (len(shape) - l))
    return inputs + tf.zeros(shape, dtype=inputs.dtype)
    

"""
Base class
"""
class Distribution(tfd.Distribution):
    """Wrapper class for distributions, mainly used to allow moving parameters from constructors of distributions 
    to method functions (for uniformity).

    Members:
        param_dims (dict of string: int): the map from names of parameters to their sizes
        init_params (dict of string: tensor): the map from names of parameters to their values given at initialization
        num_ext_params (int): the number of external parameters for the current module (not including its submodules),
            for example, for a mixture of n distributions, num_ext_params = n for the n logits of the mixture, not 
            including parameters for each sub-distribution in the mixture
        tot_num_params (int): the total number of external parameters (including submodules)
    """
    def __init__(self, **init_params):
        # check that all parameters given agree with data in self.param_dims
        for key, value in init_params.items():
            assert key in self.param_dims, "unknown parameter name: " + key
            assert value.shape.as_list() == [self.param_dims[key]], "shape mismatch for parameter: " + key
        self.init_params = init_params
        # the number of external parameters is the number of parameters not specified in init_params
        self.num_ext_params = sum(value for key, value in self.param_dims.items()) \
                            - sum(int(value.shape[-1]) for key, value in init_params.items())

    def get_params(self, inputs):
        """Returns the dictionary from names of parameters to their values, either from initialization or given by inputs.

        Arguments:
            inputs (tensor of shape (..., tot_num_params)): external parameters of the distribution

        Returns:
            params (dict of string: tensor): The first num_ext_params parameters are converted to a map from names to values,
                and the rest are collected in the "rest" entry of params.
        """
        if inputs is None:
            assert self.tot_num_params == 0, "number of parameters mismatch"
        else:
            assert int(inputs.shape[-1]) == self.tot_num_params, "number of parameters mismatch"
        params = {}
        num = 0
        for key, value in self.param_dims.items():
            if key in self.init_params:
                # data is given at initialization
                params[key] = self.init_params[key]
            else:
                # data is given in inputs
                params[key] = inputs[..., num : num + value]
                num += value
        # the "rest" entry for unused parameters
        params["rest"] = inputs[..., num : ] if inputs is not None and num < int(inputs.shape[-1]) else None
        return params

    def sample(self, sample_shape, params=None):
        """Returns samples given the external parameters.

        Arguments:
            sample_shape (list of ints): shape of the returned tensor of samples
            params (tensor or None): external parameters of shape sample_shape + (tot_num_params,)
                None is allowed if tot_num_params == 0, to be compatible with tf.Distribution

        Returns:
            samples (tensor of shape sample_shape): the generated random samples
        """
        raise NotImplementedError

    def log_prob(self, samples, params=None):
        """Returns the log probabilities of the samples given the external parameters.

        Arguments:
            samples (tensor): the samples observed
            params (tensor): external parameters of shape samples.shape() + (tot_num_params,)
                None is allowed if tot_num_params == 0, to be compatible with tf.Distribution

        Returns:
            log_prob (tensor of same shape as samples): the log probabilities
        """
        raise NotImplementedError


"""
Useful transformations of distributions
"""
class Symmetrize(Distribution):
    """Class for a symmetrized distribution p(x) = q(|x|) / 2. No additional parameters.

    Members:
        dist (Distribution): the probability distribution q(x)
    """

    def __init__(self, dist, **init_params):
        self.dist = dist
        self.param_dims = {}
        super().__init__(**init_params)
        self.tot_num_params = dist.tot_num_params + self.num_ext_params

    def sample(self, sample_shape, params=None):
        sign = tf.cast(tf.random.uniform(sample_shape, maxval=2, dtype=tf.int32), tf.float32) * 2 - 1
        return self.dist.sample(sample_shape, params) * sign

    def log_prob(self, samples, params=None):
        return self.dist.log_prob(tf.abs(samples), params) - tf.log(2.)


class Permute(Distribution):
    """Class for permuting the event dimension. No additional parameters.

    Members:
        dist (Distribution): the original probability distribution
        perm (tfp.bijectors.Bijector): the permutation bijector
        event_size (int): size of the event dimension
    """

    def __init__(self, dist, perm, **init_params):
        self.dist = dist
        self.perm = tfp.bijectors.Permute(perm)
        self.event_size = len(perm)
        self.param_dims = {}
        super().__init__(**init_params)
        self.tot_num_params = dist.tot_num_params + self.num_ext_params

    def sample(self, sample_shape, params=None):
        assert sample_shape[-1] == self.event_size, "dimension mismatch"
        return self.perm.forward(self.dist.sample(sample_shape, params))

    def log_prob(self, samples, params=None):
        assert int(samples.shape[-1]) == self.event_size, "dimension mismatch"
        return self.dist.log_prob(self.perm.inverse(samples), params)


class Affine(Distribution):
    """Class for an affine transformed distribution p(x) = q( exp(-log_scale) (x - loc) ) exp(-log_scale). 
    There are two additional affine parameters log_scale and loc.

    Members:
        dist (Distribution): the probability distribution before transformation
    """

    def __init__(self, dist, **init_params):
        self.dist = dist
        self.param_dims = {"log_scale": 1, "loc": 1}
        super().__init__(**init_params)
        self.tot_num_params = dist.tot_num_params + self.num_ext_params

    def _get_params(self, inputs):
        params = self.get_params(inputs)
        return tf.squeeze(params["log_scale"], axis=-1), tf.squeeze(params["loc"], axis=-1), params["rest"]

    def sample(self, sample_shape, params=None):
        log_scale, loc, rest = self._get_params(params)
        log_scale = broadcast_to(log_scale, sample_shape)
        loc = broadcast_to(loc, sample_shape)
        return tf.stop_gradient(self.dist.sample(sample_shape, rest) * tf.exp(log_scale) + loc)

    def log_prob(self, samples, params=None):
        log_scale, loc, rest = self._get_params(params)
        log_scale = broadcast_to(log_scale, samples.shape)
        loc = broadcast_to(loc, samples.shape)
        return self.dist.log_prob(tf.exp(-log_scale) * (samples - loc), rest) - log_scale   


class Mixture(Distribution):
    """Class for a mixture (weighted sum) of several probability distributions

    There is one additional parameter (weight) for each distribution in the mixture, besides parameters necessary 
    for themselves. Because weights should add up to one, a softmax layer is added after the logits parameters. 
    Parameters are arranged such that logits of weights are first, then parameters for each distribution follow 
    in the order that they appear in the list.

    Members:
        dist_list (list of Distribution): a list of distributions in the mixture
        event_ndims (int): the number of event dimensions
    """
    def __init__(self, dist_list, event_ndims=0, **init_params):
        self.dist_list = dist_list
        assert event_ndims >= 0, "the number of event dimensions must be a nonnegative integer"
        self.event_ndims = event_ndims
        self.param_dims = {"logits": len(dist_list)}
        super().__init__(**init_params)
        self.tot_num_params = sum(d.tot_num_params for d in dist_list) + self.num_ext_params

    def _get_params(self, inputs):
        params = self.get_params(inputs)
        return params["logits"], params["rest"]

    def sample(self, sample_shape, params=None):
        logits, rest = self._get_params(params)
        batch_shape = sample_shape if self.event_ndims == 0 else sample_shape[:-self.event_ndims]
        logits = logits + tf.zeros(batch_shape + [1]) # broadcasting to (batch_shape, num_dists)
        # samples from each distribution
        start = 0
        samples = []
        for d in self.dist_list:
            p = None if rest is None or d.tot_num_params == 0 else rest[..., start : start + d.tot_num_params]
            samples.append(d.sample(sample_shape, p))
            start += d.tot_num_params
        samples = tf.stack(samples, axis=-1) # shape (sample_shape, num_dists)
        # choose samples from distribution i with probability softmax_i (logits)
        indices = tfd.Categorical(logits=logits).sample() # shape (batch_shape,)
        indices = broadcast_to(indices, sample_shape)
        return tf.stop_gradient(index_select(samples, indices))

    def log_prob(self, samples, params=None):
        logits, rest = self._get_params(params)
        # log probabilities from each distribution
        start = 0
        log_prob = []
        for d in self.dist_list:
            p = None if rest is None or d.tot_num_params == 0 else rest[..., start : start + d.tot_num_params]
            log_prob.append(d.log_prob(samples, p))
            start += d.tot_num_params
        log_prob = tf.stack(log_prob, axis=-1) # shape (batch_shape, num_dists)
        return tf.reduce_logsumexp(logits + log_prob, axis=-1) - tf.reduce_logsumexp(logits, axis=-1)


"""
Useful base distributions
"""
class Normal(Distribution):
    """Class for the normal distribution with zero mean and unit variance. No external parameters."""

    def __init__(self, **init_params):
        self.dist = tfd.Normal(loc=0.0, scale=1.0)
        self.param_dims = {}
        super().__init__(**init_params)
        self.tot_num_params = self.num_ext_params

    def sample(self, sample_shape, params=None):
        assert params is None or int(params.shape[-1]) == 0, "unused parameters: " + params
        return self.dist.sample(sample_shape)

    def log_prob(self, samples, params=None):
        assert params is None or int(params.shape[-1]) == 0, "unused parameters: " + params
        return self.dist.log_prob(samples)


class GeneralizedNormal(Distribution):
    """Class for the generalized normal distribution p(x) = v / Gamma(1/v) * exp(- x^v), v, x > 0."""

    def __init__(self, **init_params):
        self.param_dims = {"log_v" : 1}
        super().__init__(**init_params)
        self.tot_num_params = self.num_ext_params

    def _get_params(self, inputs):
        params = self.get_params(inputs)
        assert params["rest"] is None, "unused parameters: " + params["rest"]
        log_v = params["log_v"]
        v = tf.squeeze(tf.exp(log_v), axis=-1)
        return v

    def sample(self, sample_shape, params=None):
        v = self._get_params(params)
        v = broadcast_to(v, sample_shape)
        x = tfd.Gamma(1 / v, 1.0).sample()
        samples = tf.pow(x, 1 / v)
        return tf.stop_gradient(samples)

    def log_prob(self, samples, params=None):
        v = self._get_params(params)
        v = broadcast_to(v, samples.shape)
        return -tf.pow(samples, v) + tf.log(v) - tf.lgamma(1 / v)


class Gamma(Distribution):
    """Class for the gamma distribution p(x) = x^(a - 1) exp(- x) / Gamma(a), a >= 2, x > 0."""

    def __init__(self, **init_params):
        self.param_dims = {"log_a": 1}
        super().__init__(**init_params)
        self.tot_num_params = self.num_ext_params

    def _get_params(self, inputs):
        params = self.get_params(inputs)
        assert params["rest"] is None, "unused parameters: " + params["rest"]
        log_a = params["log_a"]
        a = 2 * tf.exp(tf.abs(log_a)) # set a >= 2 for stability near x = 0
        return tf.squeeze(a, axis=-1)

    def sample(self, sample_shape, params=None):
        a = self._get_params(params)
        a = broadcast_to(a, sample_shape)
        return tf.stop_gradient(tfd.Gamma(a, 1.0).sample())

    def log_prob(self, samples, params=None):
        a = self._get_params(params)
        a = broadcast_to(a, samples.shape)
        return tfd.Gamma(a, 1.0).log_prob(samples)


class Chi(Distribution):
    """Class for the chi distribution with one parameter df >= 2."""

    def __init__(self, **init_params):
        self.param_dims = {"log_df": 1}
        super().__init__(**init_params)
        self.tot_num_params = self.num_ext_params

    def _get_params(self, inputs):
        params = self.get_params(inputs)
        assert params["rest"] is None, "unused parameters: " + params["rest"]
        log_df = params["log_df"]
        df = 2 * tf.exp(tf.abs(log_df))
        return tf.squeeze(df, axis=-1)

    def sample(self, sample_shape, params=None):
        df = broadcast_to(self._get_params(params), sample_shape)
        return tf.stop_gradient(tfd.Chi(df=df).sample())

    def log_prob(self, samples, params=None):
        df = broadcast_to(self._get_params(params), samples.shape)
        return tfd.Chi(df=df).log_prob(samples)

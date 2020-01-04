import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from obs import normalize
from dist import Distribution


class Dense():
    """Class for a multilayer fully-connected neural network, possibly with autoregressive masks."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=tf.tanh, masked=False):
        """Constructs the network.

        Arguments:
            input_dim (int): last dimension of the input
            output_dim (int): last dimension of the output
            hidden_dim (int): number of hidden units in each layer
            num_layers (int): number of hidden layers
            activation (callable): nonlinear activation functions for hidden layers
            masked (bool): whether to use autoregressive masks
        """
        assert input_dim > 0 and output_dim > 0 and hidden_dim > 0, "dimensions must be positive"
        assert num_layers >= 0, "number of hidden layers must be nonnegative"
        self.input_dim = input_dim
        self.network = tf.keras.Sequential()
        if num_layers == 0:
            self.network.add(
                tf.keras.layers.Dense(
                    output_dim, input_dim=input_dim, 
                    **self._kernel_handler(input_dim, input_dim, output_dim, masked, True)
                ))
        else:
            self.network.add(
                tf.keras.layers.Dense(
                    hidden_dim, input_dim=input_dim, activation=activation,
                    **self._kernel_handler(input_dim, input_dim, hidden_dim, masked, True)
                ))
            for _ in range(num_layers - 1):
                self.network.add(
                    tf.keras.layers.Dense(
                        hidden_dim, activation=activation,
                        **self._kernel_handler(input_dim, hidden_dim, hidden_dim, masked, False)
                    ))
            self.network.add(
                tf.keras.layers.Dense(
                    output_dim,
                    **self._kernel_handler(input_dim, hidden_dim, output_dim, masked, False)
                ))

    def _kernel_handler(self, num_blocks, input_depth, units, masked, exclusive):
        """Returns and kernel initializer and kernel constraint for possibly masked layers."""
        if not masked:
            return {}
        else:
            mask = self._gen_mask(num_blocks, input_depth, units, exclusive).T
            def masked_initializer(shape, dtype=None, partition_info=None):
                kernel_initializer = tf.glorot_normal_initializer()
                return mask * kernel_initializer(shape, dtype, partition_info)
            return {
                    "kernel_initializer": masked_initializer, 
                    "kernel_constraint": lambda x: mask * x
                   }

    def __call__(self, inputs):
        """Passes the inputs through the network.

        Arguments:
            inputs (tensor): input tensor of shape (..., input_dim)

        Returns:
            outputs (tensor): output tensor of shape (..., output_dim)
        """
        assert int(inputs.shape[-1]) == self.input_dim, "dimension mismatch"
        return self.network(inputs)

    # the following are adapted from masked_autoregressive.py in tensorflow

    def _gen_slices(self, num_blocks, n_in, n_out, exclusive):
        """Generates the numpy slices for building an autoregressive mask."""
        slices = []
        col = 0
        d_in = n_in // num_blocks
        d_out = n_out // num_blocks
        row = d_out if exclusive else 0
        for _ in range(num_blocks):
            row_slice = slice(row, None)
            col_slice = slice(col, col + d_in)
            slices.append([row_slice, col_slice])
            col += d_in
            row += d_out
        return slices

    def _gen_mask(self, num_blocks, n_in, n_out, exclusive):
        """Generates the mask for building an autoregressive dense layer."""
        mask = np.zeros([n_out, n_in], dtype=np.float32)
        slices = self._gen_slices(num_blocks, n_in, n_out, exclusive)
        for [row_slice, col_slice] in slices:
            mask[row_slice, col_slice] = 1
        return mask


class Autoregressive(Distribution):
    """Class for an autoregressive flow. 

    Similar to tfd.Bijectors.MaskedAutoregressiveFlow but allowing for more expressive base distributions.
    """

    def __init__(self, dist, hidden_dim, num_layers, offset=None, scope="maf"):
        """Creates the masked autoregressive flow. 

        Arguments:
            dist (list of Distribution): the base distributions
            hidden_dim (int): number of units in each hidden layer
            num_layers (int): number of hidden layers
            offset (tensor): offset used to initialize the wavefunction, shape (sum(d.tot_num_params for d in dist),)
            scope (string): variable scope of the model
        """
        for d in dist:
            assert d.tot_num_params > 0, "the distribution must have external parameters"
            assert d.tot_num_params == dist[0].tot_num_params, "the distributions must have same number of parameters"
        assert offset is None or list(offset.shape) == [sum(d.tot_num_params for d in dist)], "offset dimension mismatch"
        self.param_dims = {}
        super().__init__()
        self.tot_num_params = 0
        self.dist = dist
        self.event_size = len(dist)
        self.offset = offset
        with tf.variable_scope(scope):
            self.network = Dense(self.event_size, sum(d.tot_num_params for d in dist), hidden_dim, num_layers, masked=True)

    def sample(self, sample_shape, params=None):
        assert params is None, "unused parameters"
        assert sample_shape[-1] == self.event_size, "dimension mismatch" 
        batch_shape = sample_shape[:-1] # sample_shape = batch_shape + [event_size]
        samples = tf.zeros(sample_shape)
        start = 0
        for i, d in enumerate(self.dist):
            outputs = self.network(samples)
            params = outputs[..., start : start + d.tot_num_params]
            if self.offset is not None:
                params += self.offset[start : start + d.tot_num_params]
            start += d.tot_num_params
            # update the samples
            samples_i = d.sample(batch_shape, params=params) # shape (batch_shape,)
            samples = samples + tf.expand_dims(samples_i, axis=-1) * tf.one_hot(i, self.event_size) # shape (batch_shape, event_size)
        return tf.stop_gradient(samples)

    def log_prob(self, samples, params=None):
        assert params is None, "unused parameters"
        assert samples.shape[-1] == self.event_size, "dimension mismatch"
        # only one forward pass 
        outputs = self.network(samples)
        log_prob = 0
        start = 0
        for i, d in enumerate(self.dist):
            params = outputs[..., start : start + d.tot_num_params]
            if self.offset is not None:
                params += self.offset[start : start + d.tot_num_params]
            start += d.tot_num_params
            log_prob = log_prob + d.log_prob(samples[..., i], params=params)
        return log_prob


class NormalizingFlow(Distribution):
    """Class for a multivariate normalizing flow."""

    def __init__(self, dist, num_layers, activation, offset=None, scope="nf"):
        """Initializes the model.

        Arguments:
            dist (list of Distribution): the univariate base distributions
            num_layers (int): the number of flow layers
            activation (tfp.bijectors.bijector): the nonlinearity to use
            offset (tensor): offset used to initialize the wavefunction, shape (len(dist),)
            scope (string): variable scope of the model
        """
        assert sum(d.tot_num_params for d in dist) == 0, "distributions must not have external parameters"
        assert offset is None or list(offset.shape) == [len(dist)], "offset dimension mismatch"
        self.dist = dist
        self.param_dims = {}
        super().__init__()
        self.tot_num_params = 0
        self.event_size = event_size = len(self.dist)
        self.offset = offset
        bijector = []
        with tf.variable_scope(scope):
            for i in range(num_layers + 1):
                scale = tfd.fill_triangular(tf.get_variable("weights_" + str(i), shape=[event_size*(event_size+1)//2], initializer=tf.zeros_initializer()))
                scale = tf.matrix_set_diag(scale, tf.exp(tf.matrix_diag_part(scale))) # ensure that the transformation is invertible
                scale = tf.linalg.LinearOperatorLowerTriangular(scale)
                shift = tf.get_variable("shift_" + str(i), shape=[event_size])
                bijector = [tfp.bijectors.AffineLinearOperator(shift=shift, scale=scale)] + bijector
                if i < num_layers:
                    # if not the output layer
                    bijector = [activation] + bijector
        # use tfd.Chain() to concatenate the bijectors
        self.bijector = tfp.bijectors.Chain(bijector)
        with tf.variable_scope(scope):
            # initializes the bijectors
            samples = self.sample([1, self.event_size])
            log_prob = self.log_prob(samples)

    def sample(self, sample_shape, params=None):
        assert params is None, "unused parameters"
        assert sample_shape[-1] == self.event_size, "dimension mismatch"
        batch_shape = sample_shape[:-1]
        samples = tf.stack([d.sample(batch_shape) for d in self.dist], axis=-1)
        output = self.bijector.forward(samples)
        if self.offset is not None:
            output += self.offset
        return tf.stop_gradient(output)

    def log_prob(self, samples, params=None):
        assert params is None, "unused parameters"
        assert int(samples.shape[-1]) == self.event_size, "dimension mismatch"
        if self.offset is not None:
            samples -= self.offset
        base = self.bijector.inverse(samples)
        log_prob = sum(d.log_prob(base[..., i]) for i, d in enumerate(self.dist))
        # reparametrization
        return log_prob + self.bijector.inverse_log_det_jacobian(samples, event_ndims=1)


class Vectorizer():
    """Vectorizes and de-vectorizes the bosonic matrix gauge representatives."""

    def __init__(self, algebra, bijector=None):
        """Initializes the vectorizer.

        Arguments:
            algebra (LieAlgebra): the lie algebra used to vectorize the matrices
            bijector (tfp.bijectors.bijector or None): see the encode method for more info
        """
        self.algebra = algebra
        self.bijector = bijector

    def encode(self, rep):
        """Vectorizes the bosonic matrix gauge representatives.

        Arguments:
            rep (tensor of shape (batch_size, 3, N, N)): the bosonic matrices

        Returns:
            vec_rep (tensor shape (batch_size, K)): the vectorized matrices (should be real)
                K depends on self.bijector: if bijector is None, K = 3 * algebra.dim (three matrices are simply 
                vectorized with self.algebra); otherwise K = 2 * algebra.dim, where the first matrix is diagonal 
                and its diagonal sums to zero. So only algebra.N - 1 parameters are necessary, and they
                are taken to be the difference between adjacent diagonal elements. The (i, i + 1) entries of 
                the second matrix are imaginary, so the corresponding components of real parts in the lie algebra
                are zero, and this subtracts out algebra.N - 1 parameters. The last matrix is vectorized as usual. 
                Some elements must be positive by our gauge fixing procedure, so a bijector may be necessary to 
                map general real numbers to positive ones. 
        """
        batch_size = int(rep.shape[0])
        if self.bijector is None:
            return tf.real(tf.reshape(self.algebra.matrix_to_vector(rep), [batch_size, -1]))
        else:
            e = tf.matrix_diag_part(rep[:, 0, :, :])
            # by our gauge fixing, each row of e must be nondecreasing
            diff = tf.cast(self.bijector.inverse(tf.real(e[:, 1:] - e[:, :-1])), tf.complex64) # shape (batch_size, N - 1)
            vec_rest = tf.reshape(self.algebra.matrix_to_vector(rep[:, 1:, :, :]), [batch_size, -1])
            # by our gauge fixing, first (N - 1) elements in each row of vec_rest must vanish
            # and the following (N - 1) elements must be positive
            # note this depends on our ordering of the SU(N) basis in algebra.py
            dim = self.algebra.N - 1
            vec_rep = tf.concat([diff, 
                                 tf.cast(self.bijector.inverse(tf.real(vec_rest[:, dim:2*dim])), tf.complex64), 
                                 vec_rest[:, 2*dim:]], 
                                axis=-1)
            return tf.real(vec_rep)

    def decode(self, vec_rep):
        """De-vectorizes the bosonic matrices. Reverse of the encode method.

        Arguments:
            vec_rep (tensor of shape (batch_size, K)): the vectorized gauge representatives

        Returns:
            rep (tensor of shape (batch_size, 3, N, N)): the bosonic matrix gauge representatives
        """
        batch_size = int(vec_rep.shape[0])
        if self.bijector is None:
            vec_rep = tf.reshape(vec_rep, [batch_size, -1, self.algebra.dim])
            return self.algebra.vector_to_matrix(vec_rep)
        else:
            N = self.algebra.N
            dim = N - 1
            diff = self.bijector.forward(vec_rep[:, :dim])
            # by our gauge fixing, each row of e must be nondecreasing
            e = tf.concat([tf.zeros([batch_size, 1]), tf.cumsum(diff, axis=-1)], axis=-1)
            # the sum of each row of e should also be zero
            e = e - tf.reduce_sum(e, axis=-1, keepdims=True) / N
            mat_e = tf.cast(tf.expand_dims(tf.matrix_diag(e), axis=-3), tf.complex64)
            # by our gauge fixing, first (N - 1) elements in each row of vec_rest must vanish
            # and the following (N - 1) elements must be positive
            # note this depends on our ordering of the SU(N) basis in algebra.py
            vec_rest = tf.concat([tf.zeros([batch_size, dim]), 
                                  self.bijector.forward(vec_rep[:, dim:2*dim]), 
                                  vec_rep[:, 2*dim:]], 
                                 axis=-1)
            vec_rest = tf.reshape(vec_rest, [batch_size, 2, self.algebra.dim])
            mat_rest = self.algebra.vector_to_matrix(vec_rest)
            rep = tf.concat([mat_e, mat_rest], axis=-3)
            return rep


class FermionicWavefunction():
    """Generates a multilayer perceptron as the fermionic wavefunction."""

    def __init__(self, algebra, bosonic_dim, num_matrices, num_fermions, rank, fermionic_hidden_dim=None, fermionic_num_layers=None, scope="fermions"):
        """Initializes the wavefunction.

        Arguments:
            algebra (LieAlgebra): the algebra of matrices
            bosonic_dim (int): size of the input (vectorized bosonic gauge representatives)
            num_matrices (int): the number of fermionic matrices
            num_fermions (int): number of occupied fermionic modes
            rank (int): number of free fermionic states in the superposition of the fermionic states (see obs.py for more information)
            fermionic_hidden_dim, fermionic_num_layers (int or None): if None, fermionic states do not depend on bosonic coordinates; 
                otherwise a fully-connected neural network is used with given hidden dimensions and number of hidden layers
            scope (string): variable scope for the wavefunction
        """
        assert bosonic_dim > 0 and num_matrices > 0 and num_fermions >= 0 and rank >= 0, "invalid dimensions"
        self.algebra = algebra
        self.bosonic_dim = bosonic_dim
        self.num_matrices = num_matrices
        self.num_fermions = num_fermions
        self.rank = rank
        with tf.variable_scope(scope):
            output_dim = rank * num_fermions * num_matrices * algebra.dim
            if output_dim == 0:
                self.states = None
            elif fermionic_hidden_dim is None or fermionic_num_layers is None:
                # the fermionic state does not depend on bosonic coordinates
                def complex_initializer(base_initializer):
                    f = base_initializer()
                    def initializer(*args, dtype=tf.complex64, **kwargs):
                        real = f(*args, **kwargs)
                        imag = f(*args, **kwargs)
                        return tf.complex(real, imag)
                    return initializer
                self.states = tf.get_variable("states", shape=[output_dim], dtype=tf.complex64, initializer=complex_initializer(tf.glorot_uniform_initializer))
            else:
                # the fermionic state depends on bosonic coordinates via a multilayer perceptron
                self.states_re = Dense(bosonic_dim, output_dim, fermionic_hidden_dim, fermionic_num_layers)
                self.states_im = Dense(bosonic_dim, output_dim, fermionic_hidden_dim, fermionic_num_layers)

    def __call__(self, vec_rep):
        """Returns the fermionic states at the given bosonic representatives.

        Arguments:
            vec_rep (tensor): inputs of vectorized bosonic representatives, of shape (batch_size, bosonic_dim)

        Returns:
            fermionic (tensor or None): normalized fermionic states, of shape (batch_size, rank, num_fermions, num_matrices, N, N)
        """
        assert int(vec_rep.shape[-1] == self.bosonic_dim), "dimension mismatch"
        batch_size = int(vec_rep.shape[0])
        if hasattr(self, "states"):
            if self.states is None:
                return None
            states = tf.tile(self.states, [batch_size])
        else:
            states = tf.complex(self.states_re(vec_rep), self.states_im(vec_rep))
        states = tf.reshape(states, [batch_size, self.rank, self.num_fermions, self.num_matrices, self.algebra.dim])
        states = self.algebra.vector_to_matrix(states)
        return normalize(states)


class Wavefunction():
    """Class for a wavefunction with both bosons and fermions."""

    def __init__(self, gauge, vectorizer, bosonic_wavefunc, fermionic_wavefunc):
        """Initializes the wavefunction.

        Arguments:
            gauge (Gauge): the gauge group to use
            vectorizer (Vectorizer): the piping module to (de)vectorize the matrices
            bosonic_wavefunc (Distribution): the module for generating samples of (vectorized) bosonic matrices 
                and evaluating their probabilities
            fermionic_wavefunc (FermionicWavefunction): the module for generating the normalized fermionic states
        """
        self.gauge = gauge
        self.vectorizer = vectorizer
        self.bosonic_wavefunc = bosonic_wavefunc
        self.fermionic_wavefunc = fermionic_wavefunc

    def __call__(self, bosonic):
        """Computes the fermionic state at given bosonic coordinates.

        Argument:
            bosonic (tensor of shape (batch_size, 3, N, N)): inputs of bosonic coordinates

        Returns:
            log_norm (tensor of shape (batch_size,)): log of the norm of the (possibly unnormalized) wavefunction at inputs
            state (tensor of shape (batch_size, rank, num_occupied_fermions, num_matrices, N, N) or None): 
                the normalized fermionic state
        """
        g, rep = self.gauge.gauge_fixing(bosonic)
        vec_rep = self.vectorizer.encode(rep)
        log_norm = (self.bosonic_wavefunc.log_prob(vec_rep) - self.gauge.log_orbit_measure(rep)) / 2
        fermions = self.fermionic_wavefunc(vec_rep)
        state = None if fermions is None else self.gauge.action(g, fermions)
        return log_norm, state

    def sample(self, batch_size):
        """Samples from the wavefunction according to its norm squared.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            bosonic (tensor of shape (batch_size, 3, N, N)): the sampled bosonic matrices
        """
        vec_rep = self.bosonic_wavefunc.sample([batch_size, self.bosonic_wavefunc.event_size])
        rep = self.vectorizer.decode(vec_rep)
        g = self.gauge.random_element(batch_size)
        bosonic = self.gauge.action(g, rep)
        return tf.stop_gradient(bosonic)


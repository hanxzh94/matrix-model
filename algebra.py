import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import random


class LieAlgebra():
    """Base class for a lie algebra module, including methods necessary to compute casimirs 
    (see obs.py for more info)."""

    def __init__(self, basis):
        """Initializes the algebra given a matrix representation.

        Arguments:
            basis (list of numpy arrays): matrix representation of a basis of the algebra,
                with each element in the list an N-by-N hermitian numpy matrix for a basis element
        """
        self.N = basis[0].shape[0]
        self.dim = len(basis)
        self.basis = tf.constant(np.array(basis), dtype=tf.complex64)
        metric = np.zeros((self.dim, self.dim), dtype=complex)
        for i, s in enumerate(basis):
            assert np.allclose(s, np.transpose(s.conj())), "matrices should be hermitian"
            for j, t in enumerate(basis):
                metric[i, j] = np.trace(np.matmul(s, t))
        self.metric = tf.constant(metric, dtype=tf.complex64)

    def infinitesimal_action(self, dg, x):
        """Acts the lie algebra element on the input. Note dg x = d exp (i s dg) x / ds (s = 0).

        Arguments:
            dg (tensor of shape (batch_size, dim)): the lie algebra elements (written in self.basis)
            x (tensor of shape (batch_size, ...)): the inputs

        Returns:
            dg x (tensor of shape (batch_size, ...)): the inputs after action
        """
        raise NotImplementedError

    def random_algebra_element(self, batch_size):
        """Generates a random lie algebra element with norm == sqrt(dim).

        Arguments:
            batch_size (int): the number of samples to generate

        Returns:
            dg (tensor of shape (batch_size, dim)): random elements generated
        """
        # note this implementation assumes that self.metric is proportional to identity
        dg = tf.random.normal([batch_size, self.dim])
        return tf.stop_gradient(tf.sqrt(self.dim * 1.0) * dg / tf.norm(dg, axis=-1, keepdims=True))

    def vector_to_matrix(self, vec):
        """Converts from vector representations of lie algebra elements to matrices.

        Arguments:
            vec (tensor of shape (..., dim)): coefficients on the basis

        Returns:
            mat (tensor of shape (..., N, N)): matrix representations
        """
        assert vec.shape[-1] == self.dim, "dimension mismatch"
        vec = tf.cast(vec, tf.complex64)
        mat = tf.tensordot(vec, self.basis, axes=[[-1], [0]])
        return mat

    def matrix_to_vector(self, mat):
        """Converts from matrix representations of lie algebra elements to vectors.

        Arguments:
            mat (tensor of shape (..., N, N)): matrix representations
            
        Returns:
            vec (tensor of shape (..., dim)): coefficients on the basis
        """
        assert mat.shape[-1] == mat.shape[-2] == self.N, "dimension mismatch"
        mat = tf.cast(mat, tf.complex64)
        vec = tf.tensordot(mat, self.basis, axes=[[-1, -2], [1, 2]])
        vec = tf.tensordot(vec, tf.linalg.inv(self.metric), axes=[[-1], [0]])
        return vec


class Gauge():
    """Base class for the gauge groups, including methods indispensable for the algorithm 
    (used in the Wavefunction class)."""

    def action(self, g, x):
        """Returns g x for a batch of gauge group elements g and input coordinates x.

        Arguments:
            g (tensor of shape (batch_size, ...)): a batch of group elements
            x (tensor of shape (batch_size, ...)): a batch of inputs

        Returns:
            g x (tensor of same shape as x): for each sample in the batch, 
                the result of input coordinates acted by g from left
        """
        raise NotImplementedError

    def gauge_fixing(self, x):
        """Decomposes x into g y where g is a group element and y is a representative
        of the gauge orbit.

        Argument:
            x (tensor of shape (batch_size, ...)): the input coordinates

        Returns:
            g (tensor of shape (batch_size, ...)): the group elements in the decomposition
            y (tensor of same shape as x): the representatives in the decomposition
        """
        raise NotImplementedError

    def log_orbit_measure(self, y):
        """Computes measure of the gauge orbit that has y as the representative. 

        Argument:
            y (tensor of shape (batch_size, ...)): the representatives

        Returns:
            log_measure (tensor of shape (batch_size,)): log measure of the gauge orbit
        """
        raise NotImplementedError

    def random_element(self, batch_size):
        """Generates a batch of random group elements.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (tensor of shape (batch_size, ...)): the random elements generated
        """
        raise NotImplementedError


class Trivial(Gauge):
    """Class for the trivial gauge group."""

    def action(self, g, x):
        assert g is None, "unrecognized group element"
        return x

    def gauge_fixing(self, x):
        return None, x

    def log_orbit_measure(self, x):
        batch_size = int(x.shape[0])
        return tf.zeros([batch_size])

    def random_element(self, batch_size):
        return None


class SU(Gauge, LieAlgebra):
    """Class for the SU(N) gauge group."""

    def __init__(self, N, pool_size=10000):
        basis = []
        # ordering of the basis here is important for gauge fixing
        # see the implementation of Vectorizer and gauge_fixing method for more info
        # N*(N-1)/2 nondiagonal elements
        for i in range(1, N):
            for j in range(0, N - i):
                m = np.zeros((N, N), dtype=complex)
                m[j, j + i] = 1 / np.sqrt(2)
                m[j + i, j] = 1 / np.sqrt(2)
                basis.append(m)
            for j in range(0, N - i):
                m = np.zeros((N, N), dtype=complex)
                m[j, j + i] = 1j / np.sqrt(2)
                m[j + i, j] = -1j / np.sqrt(2)
                basis.append(m)
        # Cartan subalgebra of dimension (N-1)
        for i in range(1, N):
            m = np.zeros((N, N), dtype=complex)
            for j in range(i):
                m[j, j] = 1 / np.sqrt(i * i + i)
            m[i, i] = -i / np.sqrt(i * i + i)
            basis.append(m)
        super().__init__(basis)

        self.pool_size = pool_size
        self.pool = self._random_element(pool_size) # creates a pool of random elements

    def action(self, g, x):
        """Returns g x inv(g) for a batch of unitary matrices g and input matrices x.

        Arguments:
            g (tensor of shape (batch_size, N, N)): a batch of unitary matrices
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            g x inv(g) (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices conjugated by g
        """
        assert g.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = g.shape[0]
        assert g.shape[-1] == g.shape[-2] == x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        y_flat = tf.einsum("bij,btjk,bkl->btil", g, x_flat, tf.linalg.adjoint(g))
        return tf.reshape(y_flat, x.shape)

    def infinitesimal_action(self, dg, x):
        """Returns i [dg, x] for a batch of hermitian matrices dg and input matrices x.

        Arguments:
            dg (tensor of shape (batch_size, dim)): a batch of lie algebra elements
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            i [dg, x] (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices taking commutators with dg
        """
        assert dg.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = dg.shape[0]
        assert dg.shape[-1] == self.dim and x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        dg_mat = self.vector_to_matrix(dg)
        y_flat = tf.einsum("bij,btjk->btik", dg_mat, x_flat) - tf.einsum("bjk,btij->btik", dg_mat, x_flat)
        return tf.reshape(1j * y_flat, x.shape)

    def gauge_fixing(self, x):
        """Decomposes x into g y inv(g) where g is in SU(N) and y is the unique representative
        of the gauge orbit such that the first bosonic matrix is diagonal with elements 
        arranged in a nondecreasing order and entries (i, i + 1) of the second bosonic matrix are all 
        imaginary with a positive imaginary part. See the paper for more info on the gauge choice. 

        Argument:
            x (tensor of shape (batch_size, num_matrices, N, N)): the bosonic matrices

        Returns:
            g (tensor of shape (batch_size, N, N)): the (nonunique) group elements in the decomposition
            y (tensor of the same shape as x): the unique representatives in the decomposition
        """
        assert x.shape[-1] == x.shape[-2] == self.N, "dimension mismatch"
        with tf.device("/cpu:0"):
            e, u = tf.self_adjoint_eig(x[:, 0, :, :]) # x = u diag(e) u^dagger
        local_device_protos = device_lib.list_local_devices()
        local_gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        device = "/gpu:0" if len(local_gpu_devices) > 0 else "/cpu:0"
        with tf.device(device):
            y = tf.einsum("bij,brjk,bkl->bril", tf.linalg.adjoint(u), x, u)
            # fix the residual U(1)^N gauge symemtry
            batch_size = int(x.shape[0])
            # gradient of tf.sign is problematic for complex numbers!
            sign = lambda x: x / tf.cast(tf.sqrt(tf.real(tf.conj(x) * x)), tf.complex64)
            # next_diagonal = tf.matrix_diag_part(tf.roll(y[:, 1, :, :], shift=-1, axis=-1))[:, :-1]
            next_diagonal = tf.reduce_sum(tf.matrix_band_part(y[:, 1, :, :], 0, 1), axis=-1) - tf.matrix_diag_part(y[:, 1, :, :])
            next_diagonal = next_diagonal[:, :-1]
            p = tf.concat([tf.ones([batch_size, 1], dtype=tf.complex64), sign(next_diagonal)], axis=-1)
            # gradient of tf.cumprod is also problematic for complex numbers ...
            # p = tf.exp(tf.conj(tf.cumsum(tf.log(-1j * p), axis=-1)))
            log = lambda x: tf.cast(tf.log(tf.real(x) * tf.real(x) + tf.imag(x) * tf.imag(x)) / 2, tf.complex64) \
                + 1j * tf.cast(tf.atan2(tf.imag(x), tf.real(x)), tf.complex64)
            power = lambda x, a: tf.exp(tf.cast(a, tf.complex64) * log(x))
            logp = log(-1j * p)
            p = tf.exp(tf.cast(tf.cumsum(tf.real(logp), axis=-1), tf.complex64) - 1j * tf.cast(tf.cumsum(tf.imag(logp), axis=-1), tf.complex64))
            u, y = u * tf.expand_dims(p, -2), tf.einsum("bi,brij,bj->brij", tf.conj(p), y, p)
            return u / power(tf.linalg.det(u)[:, tf.newaxis, tf.newaxis], 1 / self.N), y

    def log_orbit_measure(self, y):
        """Computes measure of the gauge orbit that has y as the representative. The gauge choice is
        stated in the gauge_fixing method, and see the paper for a derivation of the measure.

        Argument:
            y (tensor of shape (batch_size, num_matrices, N, N)): the bosonic representatives

        Returns:
            log_measure (tensor of shape (batch_size,)): log measure of the gauge orbit
        """
        assert y.shape[-1] == y.shape[-2] == self.N, "dimension mismatch"
        e  = tf.matrix_diag_part(y[:, 0, :, :]) # shape (batch_size, N)
        diff = tf.expand_dims(e, axis=-1) - tf.expand_dims(e, axis=-2)
        log_det = tf.reduce_sum(tf.log(tf.eye(self.N) + tf.abs(diff)), axis=[-1, -2])
        # next_diagonal = tf.matrix_diag_part(tf.roll(y[:, 1, :, :], shift=-1, axis=-1))[:, :-1]
        next_diagonal = tf.reduce_sum(tf.matrix_band_part(y[:, 1, :, :], 0, 1), axis=-1) - tf.matrix_diag_part(y[:, 1, :, :])
        next_diagonal = next_diagonal[:, :-1]
        return log_det + tf.reduce_sum(tf.log(tf.abs(next_diagonal)), axis=-1)

    def _random_element(self, batch_size):
        """Generates a batch of random Haar unitaries in SU(N). 

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (numpy array of shape (batch_size, N, N)): the random unitaries generated
        """
        N = self.N
        random_mat = np.random.normal(size=[batch_size, N, N]) + 1j * np.random.normal(size=[batch_size, N, N])
        q = np.array([np.linalg.qr(random_mat[i])[0] for i in range(batch_size)])
        return q / np.float_power(np.linalg.det(q)[:, np.newaxis, np.newaxis], 1 / N)

    def random_element(self, batch_size):
        """Gets a batch of random Haar unitaries in SU(N) from the pool.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (tensor of shape (batch_size, N, N)): the random unitaries generated
        """
        idx = random.randint(0, self.pool_size - batch_size)
        return tf.constant(self.pool[idx : idx + batch_size], dtype=tf.complex64)


class SO3(LieAlgebra):
    """Class for the so(3) vector or spinor representations."""

    def __init__(self, rep="vector"):
        if rep == "vector":
            Lx = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
            Ly = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
            Lz = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
            super().__init__([Lx, Ly, Lz])
        elif rep == "spinor":
            Sx = np.array([[0,   1], [ 1,  0]]) / 2
            Sy = np.array([[0, -1j], [1j,  0]]) / 2
            Sz = np.array([[1,   0], [ 0, -1]]) / 2
            super().__init__([Sx, Sy, Sz])
        else:
            raise NotImplementedError

    def infinitesimal_action(self, dg, x, axis=-1):
        """The lie algebra acts as matrix multiplications on the given axis."""
        assert int(dg.shape[-1]) == self.dim and dg.shape[0] == x.shape[0] \
                and int(x.shape[axis]) == self.N, "dimension mismatch"
        x_shape = x.shape.as_list()
        batch_size = x_shape[0]
        l = len(x_shape)
        axis = axis % l
        perm = [dim for dim in range(l) if dim != axis] + [axis]
        perm_inv = [perm.index(dim) for dim in range(l)]
        x_perm = tf.transpose(tf.cast(x, tf.complex64), perm=perm)
        x_flat = tf.reshape(x_perm, [batch_size, -1, self.N])
        y_flat = tf.einsum("bij,btj->bti", self.vector_to_matrix(dg), x_flat)
        y_perm = tf.reshape(y_flat, x_perm.shape)
        y = tf.transpose(y_perm, perm=perm_inv)
        return 1j * y


class U(LieAlgebra):
    """Class for the U(N) algebra."""

    def __init__(self, N):
        basis = []
        # Cartan subalgebra of dimension N
        for i in range(N):
            m = np.zeros((N, N), dtype=complex)
            m[i, i] = 1
            basis.append(m)
        # other N*(N-1) elements
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                m = np.zeros((N, N), dtype=complex)
                m[i, j] = 1 / np.sqrt(2)
                m[j, i] = 1 / np.sqrt(2)
                basis.append(m)
                m = np.zeros((N, N), dtype=complex)
                m[i, j] = -1j / np.sqrt(2)
                m[j, i] = 1j / np.sqrt(2)
                basis.append(m)
        super().__init__(basis)
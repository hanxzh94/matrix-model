import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp
tfd = tfp.distributions
from obs import *
from train import minimize
from wavefunc import *
from dist import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from algebra import *
from ent import *
from bent_identity import *


class TestAlgebra(unittest.TestCase):

    def test_trivial(self):
        algebra = Trivial()
        g = algebra.random_element(100)
        self.assertTrue(g is None)
        x = tf.zeros([100, 3])
        self.assertTrue(algebra.action(g, x) is x)
        g, y = algebra.gauge_fixing(x)
        self.assertTrue(g is None and y is x)
        measure = algebra.log_orbit_measure(x)
        with tf.Session() as sess:
            measure = sess.run(measure)
        self.assertTrue(np.allclose(measure, 0))

    def test_SO3vector(self):
        algebra = SO3()
        x = tf.expand_dims(tf.eye(3), 0)
        # vector representation
        Lx = -1j * algebra.infinitesimal_action(tf.constant([[1.0, 0.0, 0.0]]), x, axis=-2)
        Ly = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 1.0, 0.0]]), x, axis=-2)
        Lz = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 0.0, 1.0]]), x, axis=-2)
        Lx, Ly, Lz = tf.squeeze(Lx, axis=0), tf.squeeze(Ly, axis=0), tf.squeeze(Lz, axis=0)
        with tf.Session() as sess:
            Lx, Ly, Lz = sess.run([Lx, Ly, Lz])
        self.assertTrue(np.allclose(np.matmul(Lx, Ly) - np.matmul(Ly, Lx), 1j * Lz))
        self.assertTrue(np.allclose(np.matmul(Ly, Lz) - np.matmul(Lz, Ly), 1j * Lx))
        self.assertTrue(np.allclose(np.matmul(Lz, Lx) - np.matmul(Lx, Lz), 1j * Ly))
        # dual representation
        Lx = -1j * algebra.infinitesimal_action(tf.constant([[1.0, 0.0, 0.0]]), x)
        Ly = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 1.0, 0.0]]), x)
        Lz = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 0.0, 1.0]]), x)
        Lx, Ly, Lz = tf.squeeze(Lx, axis=0), tf.squeeze(Ly, axis=0), tf.squeeze(Lz, axis=0)
        with tf.Session() as sess:
            Lx, Ly, Lz = sess.run([Lx, Ly, Lz])
        self.assertTrue(np.allclose(np.matmul(Lx, Ly) - np.matmul(Ly, Lx), -1j * Lz))
        self.assertTrue(np.allclose(np.matmul(Ly, Lz) - np.matmul(Lz, Ly), -1j * Lx))
        self.assertTrue(np.allclose(np.matmul(Lz, Lx) - np.matmul(Lx, Lz), -1j * Ly))
        # test shapes
        dg = algebra.random_algebra_element(100)
        norm = tf.linalg.norm(dg, axis=-1)
        mat = algebra.vector_to_matrix(dg)
        dg_ = algebra.matrix_to_vector(mat)
        mat_ = algebra.vector_to_matrix(dg_)
        x1 = tf.random.normal([100, 3, 4, 5])
        x2 = tf.random.normal([100, 2, 3, 2])
        x3 = tf.random.normal([100, 3])
        y1 = algebra.infinitesimal_action(dg, x1, axis=1)
        y2 = algebra.infinitesimal_action(dg, x2, axis=-2)
        y3 = algebra.infinitesimal_action(dg, x3, axis=-1)
        self.assertEqual(y1.shape, x1.shape)
        self.assertEqual(y2.shape, x2.shape)
        self.assertEqual(y3.shape, x3.shape)
        # math tests
        ox = tf.reduce_sum(y1 * tf.cast(x1, tf.complex64), axis=-3)
        oy = tf.reduce_sum(y2 * tf.cast(x2, tf.complex64), axis=2)
        oz = tf.reduce_sum(y3 * tf.cast(x3, tf.complex64), axis=1)
        with tf.Session() as sess:
            norm, dg, dg_, mat, mat_, ox, oy, oz = sess.run([norm, dg, dg_, mat, mat_, ox, oy, oz])
        self.assertTrue(np.allclose(norm, np.sqrt(3)))
        self.assertTrue(np.allclose(dg, dg_))
        self.assertTrue(np.allclose(mat, mat_))
        # because the so(3) matrices are all antisymmetric
        self.assertTrue(np.allclose(ox, 0, atol=1e-5))
        self.assertTrue(np.allclose(oy, 0, atol=1e-5))
        self.assertTrue(np.allclose(oz, 0, atol=1e-5))

    def test_SO3spinor(self):
        algebra = SO3("spinor")
        x = tf.expand_dims(tf.eye(2), 0)
        # vector representation
        Lx = -1j * algebra.infinitesimal_action(tf.constant([[1.0, 0.0, 0.0]]), x, axis=-2)
        Ly = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 1.0, 0.0]]), x, axis=-2)
        Lz = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 0.0, 1.0]]), x, axis=-2)
        Lx, Ly, Lz = tf.squeeze(Lx, axis=0), tf.squeeze(Ly, axis=0), tf.squeeze(Lz, axis=0)
        with tf.Session() as sess:
            Lx, Ly, Lz = sess.run([Lx, Ly, Lz])
        self.assertTrue(np.allclose(np.matmul(Lx, Ly) - np.matmul(Ly, Lx), 1j * Lz))
        self.assertTrue(np.allclose(np.matmul(Ly, Lz) - np.matmul(Lz, Ly), 1j * Lx))
        self.assertTrue(np.allclose(np.matmul(Lz, Lx) - np.matmul(Lx, Lz), 1j * Ly))
        # dual representation
        Lx = -1j * algebra.infinitesimal_action(tf.constant([[1.0, 0.0, 0.0]]), x, axis=-1)
        Ly = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 1.0, 0.0]]), x, axis=-1)
        Lz = -1j * algebra.infinitesimal_action(tf.constant([[0.0, 0.0, 1.0]]), x, axis=-1)
        Lx, Ly, Lz = tf.squeeze(Lx, axis=0), tf.squeeze(Ly, axis=0), tf.squeeze(Lz, axis=0)
        with tf.Session() as sess:
            Lx, Ly, Lz = sess.run([Lx, Ly, Lz])
        self.assertTrue(np.allclose(np.matmul(Lx, Ly) - np.matmul(Ly, Lx), -1j * Lz))
        self.assertTrue(np.allclose(np.matmul(Ly, Lz) - np.matmul(Lz, Ly), -1j * Lx))
        self.assertTrue(np.allclose(np.matmul(Lz, Lx) - np.matmul(Lx, Lz), -1j * Ly))
        # test shapes
        dg = algebra.random_algebra_element(100)
        norm = tf.linalg.norm(dg, axis=-1)
        mat = algebra.vector_to_matrix(dg)
        dg_ = algebra.matrix_to_vector(mat)
        mat_ = algebra.vector_to_matrix(dg_)
        x1 = tf.random.normal([100, 2, 4, 5])
        x2 = tf.random.normal([100, 2, 2, 2])
        x3 = tf.random.normal([100, 2])
        y1 = algebra.infinitesimal_action(dg, x1, axis=1)
        y2 = algebra.infinitesimal_action(dg, x2, axis=-2)
        y3 = algebra.infinitesimal_action(dg, x3, axis=-1)
        z1 = algebra.infinitesimal_action(dg, y1, axis=1)
        z2 = algebra.infinitesimal_action(dg, y2, axis=-2)
        z3 = algebra.infinitesimal_action(dg, y3, axis=-1)
        self.assertEqual(y1.shape, x1.shape)
        self.assertEqual(y2.shape, x2.shape)
        self.assertEqual(y3.shape, x3.shape)
        self.assertEqual(z1.shape, x1.shape)
        self.assertEqual(z2.shape, x2.shape)
        self.assertEqual(z3.shape, x3.shape)
        # math tests
        o1 = tf.reduce_sum(z1 * tf.cast(x1, tf.complex64), axis=-3)
        o2 = tf.reduce_sum(z2 * tf.cast(x2, tf.complex64), axis=2)
        o3 = tf.reduce_sum(z3 * tf.cast(x3, tf.complex64), axis=1)
        p1 = tf.reduce_sum(x1 * x1, axis=-3)
        p2 = tf.reduce_sum(x2 * x2, axis=-2)
        p3 = tf.reduce_sum(x3 * x3, axis=1)
        with tf.Session() as sess:
            norm, dg, dg_, mat, mat_, o1, o2, o3, p1, p2, p3 = sess.run([norm, dg, dg_, mat, mat_, o1, o2, o3, p1, p2, p3])
        self.assertTrue(np.allclose(norm, np.sqrt(3)))
        self.assertTrue(np.allclose(dg, dg_))
        self.assertTrue(np.allclose(mat, mat_))
        # because the so(3) matrices are all antisymmetric
        self.assertTrue(np.allclose(o1, -3 * p1 / 4, atol=1e-5))
        self.assertTrue(np.allclose(o2, -3 * p2 / 4, atol=1e-5))
        self.assertTrue(np.allclose(o3, -3 * p3 / 4, atol=1e-5))

    def test_U(self):
        N = 5
        batch_size = 1000
        algebra = U(N)
        self.assertEqual(algebra.N, N)
        self.assertEqual(algebra.dim, N * N)
        # test conversions between vectors and matrices
        dg = algebra.random_algebra_element(batch_size)
        mat = algebra.vector_to_matrix(dg)
        dg_ = algebra.matrix_to_vector(mat)
        mat_ = algebra.vector_to_matrix(dg_)
        self.assertEqual(dg.shape, [batch_size, algebra.dim])
        self.assertEqual(mat.shape, [batch_size, algebra.N, algebra.N])
        with tf.Session() as sess:
            dg, dg_, mat, mat_ = sess.run([dg, dg_, mat, mat_])
        self.assertTrue(np.allclose(dg, dg_, atol=1e-5))
        self.assertTrue(np.allclose(mat, mat_, atol=1e-5))

    def test_SU(self):
        N = 5
        batch_size = 1000
        algebra = SU(N)
        # check random unitaries
        u = algebra.random_element(batch_size)
        self.assertEqual(u.shape.as_list(), [batch_size, N, N])
        det = tf.linalg.det(u)
        prod = tf.matmul(u, tf.linalg.adjoint(u))
        re = tf.reduce_mean(tf.abs(tf.real(u)))
        im = tf.reduce_mean(tf.abs(tf.imag(u)))
        with tf.Session() as sess:
            det, prod, re, im = sess.run([det, prod, re, im])
        self.assertTrue(0.9 < re / im < 1.1)
        self.assertTrue(np.allclose(det, 1))
        self.assertTrue(np.allclose(prod, np.eye(N), atol=1e-5))
        # test gauge fixing: shapes
        mat = tf.complex(tf.random.normal([batch_size, 3, N, N]), tf.random.normal([batch_size, 3, N, N]))
        mat = mat + tf.linalg.adjoint(mat) # make it hermitian
        g, rep = algebra.gauge_fixing(mat)
        self.assertEqual(g.shape, [batch_size, N, N])
        self.assertEqual(rep.shape, [batch_size, 3, N, N])
        # test gauge fixing: g should be in SU(N)
        det = tf.linalg.det(g)
        prod = tf.matmul(g, tf.linalg.adjoint(g))
        with tf.Session() as sess:
            det, prod = sess.run([det, prod])
        self.assertTrue(np.allclose(det, 1))
        self.assertTrue(np.allclose(prod, np.eye(N), atol=1e-5))
        # test gauge fixing: rep should agree with the gauge choice
        rep_adj = tf.linalg.adjoint(rep)
        diag = tf.matrix_diag_part(rep[:, 0, :, :])
        off_diag = rep[:, 0, :, :] - tf.matrix_diag(diag)
        diff = diag[:, 1:] - diag[:, :-1]
        next_diagonal = tf.matrix_diag_part(tf.roll(rep[:, 1, :, :], shift=-1, axis=-1))[:, :-1]
        with tf.Session() as sess:
            r, rep_adj, off_diag, diff, next_diagonal = sess.run([rep, rep_adj, off_diag, diff, next_diagonal])
        self.assertTrue(np.allclose(r, rep_adj, atol=1e-5))
        self.assertTrue(np.allclose(off_diag, 0, atol=1e-5))
        self.assertTrue(np.allclose(diff, np.abs(diff)))
        self.assertTrue(np.allclose(next_diagonal, -np.conj(next_diagonal), atol=1e-5))
        self.assertTrue(np.allclose(np.imag(next_diagonal), np.abs(np.imag(next_diagonal)), atol=1e-5))
        # test gauge fixing and action: g rep is mat
        mat_ = tf.einsum("bij,brjk,bkl->bril", g, rep, tf.linalg.adjoint(g))
        mat__ = algebra.action(g, rep)
        g_, rep_ = algebra.gauge_fixing(mat__)
        self.assertEqual(g_.shape, g.shape)
        self.assertEqual(rep_.shape, rep.shape)
        with tf.Session() as sess:
            g, g_, r, rep_, m, mat_, mat__ = sess.run([g, g_, rep, rep_, mat, mat_, mat__])
        self.assertTrue(np.allclose(np.abs(g), np.abs(g_), atol=1e-2)) # g and g_ may differ by an overall phase
        self.assertTrue(np.allclose(r, rep_, atol=1e-2))
        self.assertTrue(np.allclose(m, mat_, atol=1e-5))
        self.assertTrue(np.allclose(m, mat__, atol=1e-5))
        # test uniqueness of the gauge representative
        _, rep_ = algebra.gauge_fixing(algebra.action(u, mat))
        with tf.Session() as sess:
            rep1, rep2 = sess.run([rep, rep_])
        self.assertTrue(np.allclose(rep1, rep2, atol=1e-3))
        # check the shape of log_orbit_measure
        m = algebra.log_orbit_measure(rep)
        self.assertEqual(m.shape, [batch_size])
        # test infinitesimal action
        dg = algebra.random_algebra_element(batch_size)
        dmat = algebra.infinitesimal_action(dg, mat)
        dmat_adj = tf.linalg.adjoint(dmat)
        eps = 1e-4
        dmat_ = (algebra.action(tf.linalg.expm(1j * eps * algebra.vector_to_matrix(dg)), mat) - mat) / eps
        o = tf.reduce_sum(tf.conj(mat) * dmat, axis=[-2, -1])
        with tf.Session() as sess:
            dmat, dmat_adj, dmat_, o = sess.run([dmat, dmat_adj, dmat_, o])
        self.assertTrue(np.allclose(dmat, dmat_adj, atol=1e-5))
        self.assertTrue(np.allclose(dmat, dmat_, atol=1e-1))
        self.assertTrue(np.allclose(o, 0, atol=1e-4)) 
        # test conversion between vectors and matrices
        dg = algebra.random_algebra_element(batch_size)
        norm = tf.linalg.norm(dg, axis=-1)
        mat = algebra.vector_to_matrix(dg)
        dg_ = algebra.matrix_to_vector(mat)
        mat_ = algebra.vector_to_matrix(dg_)
        self.assertEqual(algebra.N, N)
        self.assertEqual(algebra.dim, N * N - 1)
        self.assertEqual(dg_.shape, [batch_size, algebra.dim])
        self.assertEqual(mat_.shape, [batch_size, algebra.N, algebra.N])
        with tf.Session() as sess:
            norm, dg, dg_, mat, mat_ = sess.run([norm, dg, dg_, mat, mat_])
        self.assertTrue(np.allclose(norm, np.sqrt(N * N - 1)))
        self.assertTrue(np.allclose(dg, dg_, atol=1e-5))
        self.assertTrue(np.allclose(mat, mat_, atol=1e-5))


class TestObservables(unittest.TestCase):

    def test_bosonic(self):
        vec = tf.constant([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
        algebra = SU(2)
        mat = algebra.vector_to_matrix(vec)
        quadratic = matrix_quadratic_potential(mat)
        commutator = matrix_commutator_square(mat)
        cubic = matrix_cubic_potential(mat)
        self.assertEqual(quadratic.shape, (1,))
        self.assertEqual(commutator.shape, (1,))
        self.assertEqual(cubic.shape, (1,))
        with tf.Session():
            self.assertTrue(np.allclose(quadratic.eval(), 3))
            self.assertTrue(np.allclose(commutator.eval(), -12))
            self.assertTrue(np.allclose(cubic.eval(), 3 * np.sqrt(2)))

    def test_overlap(self):
        batch_size = 1000
        rank = 4
        num_fermions = 5
        N = 5
        algebra = SU(N)
        state = tf.complex(
            tf.random.normal([batch_size, rank, num_fermions, 2, N, N]),
            tf.random.normal([batch_size, rank, num_fermions, 2, N, N]))
        O = overlap_matrix(state, state)
        self.assertEqual(O.shape, [batch_size, rank, rank, num_fermions, num_fermions])
        O_adj = tf.einsum("brsmn->bsrnm", tf.conj(O))
        o = overlap(state, state)
        self.assertEqual(o.shape, [batch_size])
        state_norm = normalize(state)
        g = algebra.random_element(batch_size)
        state_rand = algebra.action(g, state_norm)
        self.assertEqual(state_norm.shape, state.shape)
        self.assertEqual(state_rand.shape, state.shape)
        o_norm = overlap(state_norm, state_norm)
        o_rand = overlap(state_rand, state_rand)
        self.assertEqual(o_norm.shape, [batch_size])
        self.assertEqual(o_rand.shape, [batch_size])
        with tf.Session() as sess:
            O, O_adj, o, o_norm, o_rand = sess.run([O, O_adj, o, o_norm, o_rand])
        self.assertTrue(np.allclose(O, O_adj))
        self.assertTrue(np.allclose(o, np.abs(o)))
        self.assertTrue(np.allclose(o_norm, 1.0))
        self.assertTrue(np.allclose(o_rand, 1.0))

    # matrix free fields energies are tested in TestEntanglement
    # yukawa potentials and casimirs are tested in TestWavefunction


class TestDistribution(unittest.TestCase):

    def test_normal(self):
        dist = Normal()
        self.assertEqual(dist.param_dims, {})
        self.assertEqual(dist.init_params, {})
        self.assertEqual(dist.num_ext_params, 0)
        self.assertEqual(dist.tot_num_params, 0)
        samples = dist.sample([10000])
        log_prob = dist.log_prob(samples)
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, - samples * samples / 2 - np.log(2 * np.pi) / 2))
        self.assertTrue(np.allclose(np.mean(samples), 0.0, atol=1e-1))
        self.assertTrue(np.allclose(np.std(samples), 1.0, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(-4, 4, 50)
        plt.plot(x, np.exp(- x * x / 2) / np.sqrt(2 * np.pi))
        plt.savefig("fig/test/normal.png")
        plt.clf()

    def test_generalized_normal(self):
        dist = GeneralizedNormal()
        self.assertEqual(dist.num_ext_params, 1)
        self.assertEqual(dist.tot_num_params, 1)
        p = tf.constant([0.0])
        samples = dist.sample([10000], params=p)
        log_prob = dist.log_prob(samples, params=p)
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, -samples))
        self.assertTrue(np.allclose(np.mean(samples), 1.0, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(0, 5, 50)
        plt.plot(x, np.exp(-x))
        plt.savefig("fig/test/generalized_normal.png")
        plt.clf()

    def test_gamma(self):
        dist = Gamma()
        self.assertEqual(dist.num_ext_params, 1)
        self.assertEqual(dist.tot_num_params, 1)
        p = tf.constant([0.0])
        samples = dist.sample([10000], params=p)
        log_prob = dist.log_prob(samples, params=p)
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, -samples + np.log(samples)))
        self.assertTrue(np.allclose(np.mean(samples), 2.0, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(0, 6, 50)
        plt.plot(x, x * np.exp(-x))
        plt.savefig("fig/test/gamma.png")
        plt.clf()

    def test_symmetrize1(self):
        dist = Symmetrize(GeneralizedNormal(log_v=tf.constant([0.0])))
        self.assertEqual(dist.num_ext_params, 0)
        self.assertEqual(dist.tot_num_params, 0)
        batch_size = 10000
        samples = dist.sample([batch_size])
        log_prob = dist.log_prob(samples)
        self.assertEqual(samples.shape, [batch_size])
        self.assertEqual(log_prob.shape, [batch_size])
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, -np.abs(samples) - np.log(2.0)))
        self.assertTrue(np.allclose(np.mean(samples), 0.0, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(-4, 4, 51)
        plt.plot(x, np.exp(-np.abs(x)) / 2)
        plt.savefig("fig/test/symmetrize1.png")
        plt.clf()

    def test_symmetrize2(self):
        dist = Symmetrize(GeneralizedNormal())
        self.assertEqual(dist.num_ext_params, 0)
        self.assertEqual(dist.tot_num_params, 1)
        batch_size = 10000
        p = tf.zeros([batch_size, 1])
        samples = dist.sample([batch_size], params=p)
        log_prob = dist.log_prob(samples, params=p)
        self.assertEqual(samples.shape, [batch_size])
        self.assertEqual(log_prob.shape, [batch_size])
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, -np.abs(samples) - np.log(2.0)))
        self.assertTrue(np.allclose(np.mean(samples), 0.0, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(-4, 4, 51)
        plt.plot(x, np.exp(-np.abs(x)) / 2)
        plt.savefig("fig/test/symmetrize2.png")
        plt.clf()

    def test_affine1(self):
        dist = Affine(
            Affine(Normal(), log_scale=tf.constant([1.0]), loc=tf.constant([0.0])), 
            log_scale=tf.constant([0.0]))
        self.assertEqual(dist.num_ext_params, 1)
        self.assertEqual(dist.tot_num_params, 1)
        batch_size = 10000
        p = tf.constant([10.0])
        samples = dist.sample([batch_size], params=p)
        log_prob = dist.log_prob(samples, params=p)
        self.assertEqual(samples.shape, [batch_size])
        self.assertEqual(log_prob.shape, [batch_size])
        true_log_prob = lambda x: -(x - 10) * (x - 10) / 2 / np.exp(2) - 0.5 * np.log(2 * np.pi * np.exp(2))
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, true_log_prob(samples)))
        self.assertTrue(np.allclose(np.mean(samples), 10.0, atol=1e-1))
        self.assertTrue(np.allclose(np.std(samples), 2.72, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(4, 16, 51)
        plt.plot(x, np.exp(true_log_prob(x)))
        plt.savefig("fig/test/affine1.png")
        plt.clf()

    def test_affine2(self):
        dist = Affine(
            Affine(Normal()), 
            log_scale=tf.constant([0.0]), loc=tf.constant([10.0]))
        self.assertEqual(dist.num_ext_params, 0)
        self.assertEqual(dist.tot_num_params, 2)
        batch_size = 10000
        p = tf.constant([1.0, 0.0])
        samples = dist.sample([batch_size], params=p)
        log_prob = dist.log_prob(samples, params=p)
        self.assertEqual(samples.shape, [batch_size])
        self.assertEqual(log_prob.shape, [batch_size])
        true_log_prob = lambda x: -(x - 10) * (x - 10) / 2 / np.exp(2) - 0.5 * np.log(2 * np.pi * np.exp(2))
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, true_log_prob(samples)))
        self.assertTrue(np.allclose(np.mean(samples), 10.0, atol=1e-1))
        self.assertTrue(np.allclose(np.std(samples), 2.72, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(4, 16, 51)
        plt.plot(x, np.exp(true_log_prob(x)))
        plt.savefig("fig/test/affine2.png")
        plt.clf()

    def test_mixture(self):
        dist = Mixture([
                        Affine(Normal(), loc=tf.constant([ 2.0])),
                        Affine(Normal(), loc=tf.constant([-2.0]))
                        ])
        self.assertEqual(dist.num_ext_params, 2)
        self.assertEqual(dist.tot_num_params, 4)
        batch_size = 10000
        p = tf.constant([0.0, 1.0, 0.0, 0.0])
        samples = dist.sample([batch_size], params=p)
        log_prob = dist.log_prob(samples, params=p)
        self.assertEqual(samples.shape, [batch_size])
        self.assertEqual(log_prob.shape, [batch_size])
        true_log_prob = lambda x: np.log((np.exp(-(x-2)*(x-2)/2) + np.exp(1-(x+2)*(x+2)/2))/(1+np.exp(1))) - 0.5*np.log(2 * np.pi)
        with tf.Session() as sess:
            samples, log_prob = sess.run([samples, log_prob])
        self.assertTrue(np.allclose(log_prob, true_log_prob(samples)))
        self.assertTrue(np.allclose(np.mean(samples), -0.92, atol=1e-1))
        plt.hist(samples, bins=50, density=True)
        x = np.linspace(-5, 5, 51)
        plt.plot(x, np.exp(true_log_prob(x)))
        plt.savefig("fig/test/mix.png")
        plt.clf()

    def test_bent_identity(self):
        x = tf.constant([-1e4, 1e4])
        bij = BentIdentity()
        y = bij.forward(x)
        _x = bij.inverse(y)
        fj = bij.forward_log_det_jacobian(x, event_ndims=0)
        bj = bij.inverse_log_det_jacobian(y, event_ndims=0)
        with tf.Session() as sess:
            y, _x, fj, bj = sess.run([y, _x, fj, bj])
        self.assertTrue(np.allclose(y, [-1e4, 2e4]))
        self.assertTrue(np.allclose(_x, [-1e4, 1e4]))
        self.assertTrue(np.allclose(fj, [0.0, np.log(2.0)]))
        self.assertTrue(np.allclose(bj, [0.0, -np.log(2.0)]))


class TestWavefunction(unittest.TestCase):

    def test_vectorizer(self):
        N = 5
        batch_size = 1000
        algebra = SU(N)
        vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
        # generates random traceless hermitian matrices
        bosonic = tf.complex(tf.random.normal([batch_size, 3, N, N]), tf.random.normal([batch_size, 3, N, N]))
        bosonic = bosonic + tf.linalg.adjoint(bosonic)
        bosonic = bosonic - tf.trace(bosonic)[..., tf.newaxis, tf.newaxis] * tf.eye(N, dtype=tf.complex64) / N
        # test encode and decode
        g, rep = algebra.gauge_fixing(bosonic)
        vec_rep = vectorizer.encode(rep)
        self.assertEqual(vec_rep.shape, [batch_size, 2 * algebra.dim])
        rep_ = vectorizer.decode(vec_rep)
        self.assertEqual(rep_.shape, [batch_size, 3, N, N])
        vec_rep_ = vectorizer.encode(rep_)
        bosonic_ = algebra.action(g, rep_)
        self.assertEqual(bosonic_.shape, bosonic.shape)
        self.assertEqual(vec_rep_.shape, vec_rep.shape)
        with tf.Session() as sess:
            bosonic, bosonic_, rep, rep_, vec_rep, vec_rep_ = sess.run([bosonic, bosonic_, rep, rep_, vec_rep, vec_rep_])
        self.assertTrue(np.allclose(bosonic, bosonic_, atol=1e-5))
        self.assertTrue(np.allclose(rep, rep_, atol=1e-5))
        self.assertTrue(np.allclose(vec_rep, vec_rep_, atol=1e-5))

    def test_gauge_no_fermions(self):
        tf.reset_default_graph()
        # spec
        batch_size = 1000
        N = 2
        m = 10
        num_fermions = 0
        rank = 1
        name = "test_gauge_no_fermions"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(name):
            dim = 2 * algebra.dim
            bosonic_wavefunc = Autoregressive([Mixture([Affine(Normal())] * 2)] * dim, dim, 2)
            fermionic_wavefunc = FermionicWavefunction(algebra, dim, 2, num_fermions, rank, dim, 1)
            vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
            wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            bosonic = wavefunc.sample(batch_size)
            log_norm, _ = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        energy = fuzzy_sphere_energy(m, wavefunc, bosonic)
        # training
        print("Training ...")
        output_path = "results/" + name + "/"
        obs = minimize(name, log_norm, energy, 10000,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-3, output_path=output_path)
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-4, restore_path=output_path)
        self.assertTrue(np.abs(obs["energy"] + 12593) < 5)
        self.assertTrue(np.abs(obs["r"] - 12.99) < 1e-2)
        self.assertTrue(0 < obs["gauge"] < 1e-8)
        self.assertTrue(0 < obs["rotation"] < 1e-2)

    def test_no_gauge_single_fermion(self):
        tf.reset_default_graph()
        # spec
        batch_size = 1000
        N = 2
        m = 10
        num_fermions = 1
        rank = 1
        name = "test_no_gauge_single_fermion"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(name):
            bosonic_dim = 3 * algebra.dim
            fermionic_dim = 2 * algebra.dim
            bosonic_wavefunc = Autoregressive([Mixture([Affine(Normal())] * 2)] * bosonic_dim, bosonic_dim, 2)
            fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, num_fermions, rank, fermionic_dim, 2)
            vectorizer = Vectorizer(algebra)
            wavefunc = Wavefunction(Trivial(), vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            bosonic = wavefunc.sample(batch_size)
            log_norm, fermionic = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        kinetic = matrix_kinetic_energy(wavefunc, bosonic)
        bilinear = miniBMN_yukawa_potential(bosonic, fermionic)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        pre_energy = fuzzy_sphere_energy(m, wavefunc, bosonic)
        energy = miniBMN_energy(m, wavefunc, bosonic)
        # pretraining
        print("Pretraining ...")
        output_path = "results/" + name + "/"
        obs = minimize(name, log_norm, pre_energy, 10000,
                        {"r": radius}, 
                        1000, 5000, lr=1e-3, output_path=output_path)
        # training
        print("Training ...")
        obs = minimize(name, log_norm, energy, 10000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-3, restore_path=output_path)
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-4, restore_path=output_path)
        self.assertTrue(obs["energy"] < 15)
        self.assertTrue(obs["kinetic"] < 65)
        self.assertTrue(np.abs(obs["r"] - 8.66) < 2e-2)
        self.assertTrue(np.abs(obs["bilinear"] + 20) < 5e-2)
        # gauge and rotation should be large, and energy should be slightly larger than 5 (semiclassical value)
        # it is more difficult to learn for large mass without gauge fixing

    def test_gauge_single_fermion(self):
        tf.reset_default_graph()
        # spec
        batch_size = 1000
        N = 2
        m = 10
        num_fermions = 1
        rank = 1
        name = "test_gauge_single_fermion"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(name):
            dim = 2 * algebra.dim
            with open("data/SpinMatrices" + str(N) + ".bin", "rb") as f:
                mats = tf.constant(np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=3*N*N), [3, N, N]))
                Sx, Sy, Sz = mats[0], mats[1], mats[2]
                offset = m * tf.stack([-Sz, -Sy, -Sx])
            vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
            offset = vectorizer.encode(tf.expand_dims(offset, 0))[0]
            bosonic_wavefunc = NormalizingFlow([Normal()] * dim, 0, tfp.bijectors.Sigmoid(), offset)
            fermionic_wavefunc = FermionicWavefunction(algebra, dim, 2, num_fermions, rank, dim, 1)
            wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            bosonic = wavefunc.sample(batch_size)
            log_norm, fermionic = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        kinetic = matrix_kinetic_energy(wavefunc, bosonic)
        bilinear = miniBMN_yukawa_potential(bosonic, fermionic)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        energy = miniBMN_energy(m, wavefunc, bosonic)
        # training
        print("Training ...")
        output_path = "results/" + name + "/"
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-3, output_path=output_path)
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-4, restore_path=output_path)
        self.assertTrue(np.abs(obs["energy"] - 5) < 1)
        self.assertTrue(np.abs(obs["kinetic"] - 55) < 5)
        self.assertTrue(np.abs(obs["r"] - 8.66) < 1e-2)
        self.assertTrue(np.abs(obs["bilinear"] + 20) < 0.2)
        self.assertTrue(0 < obs["gauge"] < 1e-8)
        self.assertTrue(np.abs(obs["rotation"] - 0.75) < 4e-2)

    def test_gauge_susy(self):
        tf.reset_default_graph()
        # spec
        batch_size = 1000
        N = 2
        m = 10
        num_fermions = 2
        rank = 4
        name = "test_gauge_susy"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(name):
            dim = 2 * algebra.dim
            with open("data/SpinMatrices" + str(N) + ".bin", "rb") as f:
                mats = tf.constant(np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=3*N*N), [3, N, N]))
                Sx, Sy, Sz = mats[0], mats[1], mats[2]
                offset = m * tf.stack([-Sz, -Sy, -Sx])
            vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
            offset = vectorizer.encode(tf.expand_dims(offset, 0))[0]
            bosonic_wavefunc = NormalizingFlow([Normal()] * dim, 0, tfp.bijectors.Sigmoid(), offset)
            fermionic_wavefunc = FermionicWavefunction(algebra, dim, 2, num_fermions, rank, dim, 1)
            wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            bosonic = wavefunc.sample(batch_size)
            log_norm, fermionic = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        kinetic = matrix_kinetic_energy(wavefunc, bosonic)
        bilinear = miniBMN_yukawa_potential(bosonic, fermionic)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        energy = miniBMN_energy(m, wavefunc, bosonic)
        # training
        print("Training ...")
        output_path = "results/" + name + "/"
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-3, output_path=output_path)
        obs = minimize(name, log_norm, energy, 5000,
                        {"r": radius, "kinetic": kinetic, "bilinear": bilinear, "gauge": gauge, "rotation": rotation}, 
                        100, 5000, lr=1e-4, restore_path=output_path)
        self.assertTrue(np.abs(obs["energy"] - 0) < 1)
        self.assertTrue(np.abs(obs["kinetic"] - 55) < 5)
        self.assertTrue(np.abs(obs["r"] - 8.66) < 1e-2)
        self.assertTrue(np.abs(obs["bilinear"] + 40) < 0.4)
        self.assertTrue(0 < obs["gauge"] < 1e-8)
        self.assertTrue(np.abs(obs["rotation"] - 0) < 4e-2)


if __name__ == '__main__':
    unittest.main()

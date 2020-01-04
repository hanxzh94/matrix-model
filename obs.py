import tensorflow as tf
import numpy as np
from algebra import SU, SO3


# There are bugs in implementations of gradients of complex matrix determinant/inverse.
# Please replace linalg_grad.py into tensorflow/python/ops/.


"""
Following are some energy functions of bosonic matrices. Their definitions should be clear from formulas.
The interface is 
    Argument:
        mats (tensor of shape (batch_size, num_matrices, N, N)): values of the bosonic matrix coordinates
    Returns:
        energy (tensor of shape (batch_size,)): energy density at mats
"""
def matrix_quadratic_potential(mats):
    """sum_i tr X_i^2"""
    return tf.real(tf.einsum("birs,bisr->b", mats, mats))

def matrix_commutator_square(mats):
    """sum_ij tr [X_i, X_j]^2"""
    commutator = tf.einsum("birs,bjst->bijrt", mats, mats) - tf.einsum("bist,bjrs->bijrt", mats, mats)
    commutator_square = tf.einsum("bijrs,bijsr->b", commutator, commutator)
    return tf.real(commutator_square)

def matrix_cubic_potential(mats):
    """sum_ijk I epsilon_ijk tr X_i X_j X_k = 3 I tr [X_0, X_1] X_2"""
    assert mats.shape[-3] == 3, "the number of bosonic matrices must be three"
    mat0, mat1, mat2 = mats[:, 0, :, :], mats[:, 1, :, :], mats[:, 2, :, :]
    commutator01 = tf.einsum("bij,bjk->bik", mat0, mat1) - tf.einsum("bij,bjk->bik", mat1, mat0)
    return -3 * tf.imag(tf.einsum("bij,bji->b", commutator01, mat2))

def matrix_spherical_laplacian(mats):
    """sum_ij tr [S_i, X_j]^2 where S_i are spin matrices"""
    N = int(mats.shape[-1])
    with open("data/SpinMatrices" + str(N) + ".bin", "rb") as f:
        pauli = tf.constant(np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=3*N*N), [3, N, N]))
    commutator = tf.einsum("irs,bjst->bijrt", pauli, mats) - tf.einsum("ist,bjrs->bijrt", pauli, mats)
    commutator_square = tf.einsum("bijrs,bijsr->b", commutator, commutator)
    return tf.real(commutator_square)


"""
Following are some observables for a superposition of free fermionic states. Each fermionic state is 
represented as a tensor of shape (rank, num_fermions, num_matrices, N, N), which is a superposition of 
``rank'' free fermionic states with ``num_fermions'' occupied fermionic modes. Each occupied mode is specified 
by ``num_matrices'' complex matrices of size N by N. See the paper for more information.
"""
def overlap_matrix(state1, state2):
    """Computes the single fermion overlap matrix (O^rs)^mn = < state1^rm | state2^sn > where rs are rank indices
    and mn are fermion indices.

    Arguments:
        state1, state2 (tensor of shape (batch_size, rank, num_fermions, num_matrices, N, N)): fermionic states

    Returns:
        O (tensor of shape (batch_size, rank, rank, num_fermions, num_fermions)): the overlap matrix
    """
    assert state1.shape == state2.shape, "dimension mismatch"
    return tf.einsum("brmaij,bsnaji->brsmn", tf.linalg.adjoint(state1), state2)

def overlap(state1, state2):
    """Computes the overlap < state1 | state 2 > for fermionic states.

    Arguments:
        state1, state2 (tensor of shape (batch_size, rank, num_fermions, num_matrices, N, N)): fermionic states

    Returns:
        o (tensor of shape (batch_size,)): the overlaps
    """
    return tf.reduce_sum(tf.linalg.det(overlap_matrix(state1, state2)), axis=[-1, -2])

def normalize(state):
    """Normalizes the fermionic states.

    Arguments:
        state (tensor of shape (batch_size, rank, num_fermions, num_matrices, N, N)): fermionic states

    Returns:
        state_normalized (tensor of same shape as state)
    """
    num_fermions = int(state.shape[-4])
    norm = lambda x: tf.cast(tf.sqrt(tf.reduce_sum(tf.real(x) * tf.real(x) + tf.imag(x) * tf.imag(x), axis=-1)), tf.complex64)
    state = state / norm(tf.reshape(state, state.shape.as_list()[:-3] + [-1]))[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    norm = tf.cast(tf.pow(tf.real(overlap(state, state)), 0.5 / num_fermions), tf.complex64)
    return state / norm[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]


def miniBMN_yukawa_potential(bosonic, fermionic):
    """Computes the fermionic bilinear tr lambda^dagger sigma^i [X^i, lambda] (lambda is a fermionic matrix),
    where sigma^i are pauli matrices.

    Arguments:
        bosonic (tensor): bosonic matrices, of shape (batch_size, 3, N, N)
        fermionic (tensor or None): states of the fermions, of shape (batch_size, rank, num_fermions, 2, N, N)
            If None, there are no fermions.

    Returns:
        value (tensor of shape (batch_size,))
    """
    assert fermionic is None or bosonic.shape[0] == fermionic.shape[0], "dimension mismatch"
    batch_size = bosonic.shape[0]
    if fermionic is None:
        return tf.zeros([batch_size])

    O = overlap_matrix(fermionic, fermionic) # shape (batch_size, rank, rank, num_fermions, num_fermions)
    O_adj = tf.linalg.det(O)[..., tf.newaxis, tf.newaxis] * tf.linalg.inv(O)
    sigma = 2 * SO3("spinor").basis # pauli matrices
    value = tf.einsum("trmaij,lab,tljk,tsnbki->trsmn", tf.linalg.adjoint(fermionic), sigma, bosonic, fermionic) \
          - tf.einsum("trmaij,lab,tlki,tsnbjk->trsmn", tf.linalg.adjoint(fermionic), sigma, bosonic, fermionic)
    value = tf.reduce_sum(tf.trace(tf.matmul(O_adj, value)), axis=[-1, -2])
    return tf.real(value)


"""
Some casimirs are evaluated in the following. Recall that the state is represented as psi(X) = f(X) | M(X) > where
f(X) >= 0 and | M(X) > is a normalized matrix fermionic state. For any lie algebra action (dg psi)(X) = d / ds
f(X - s dg X) | M(X - s dg X) + s (dg M) > |_s=0. So < dg psi | dg psi > = E_{X ~ |f|^2} [d / ds ln f(X - s dg X)|_(s = 0)]^2
+ E_{X ~ |f|^2} [d^2 / ds dt < M(X - s dg X) + s (dg M) | M(X - t dg X) + t (dg M) >|_(s = t = 0)]. The casimir is 
then < dg psi | dg psi > averaged over dg with proper normalization.

For casimir functions, the interface is
    Argument:
        wavefunc (Wavefunction): the wavefunction
        bosonic (tensor of shape (batch_size, num_bosonic_matrices, N, N)): the bosonic matrix coordinates
    Returns:
        energy (tensor of shape (batch_size,)): the casimir density at bosonic
"""
def dstate(wavefunc, bosonic, dbosonic, dfermionic):
    """Creates the state with an infinitesimal change.

    Arguments:
        wavefunc (Wavefunction): the wavefunction that gives the mapping from bosonic coordinates to fermionic states
        bosonic, dbosonic (tensor of shape (batch_size, num_bosonic_matrices, N, N)): the bosonic matrix coordinates
            and its infinitesimal change
        dfermionic (tensor of shape (batch_size, rank, num_fermions, num_femionic_matrices, N, N) or None):
            infinitesimal change of the fermionic state, and None if no fermions

    Returns:
        s (tensor of shape (batch_size,)): the infinitesimal parameter
        log_norm (tensor of shape (batch_size,)): the log wavefunction norm at (bosonic - s * dbosonic)
        state (tensor of same shape as dfermionic or None): the fermionic state at (bosonic - s * d bosonic) plus 
            s * dfermionic
    """
    batch_size = int(bosonic.shape[0])
    s = 1e-8 * tf.ones([batch_size]) # tensorflow bugs in gradients if s = 0 here
    log_norm, state = wavefunc(bosonic - tf.reshape(tf.cast(s, tf.complex64), [batch_size] + [1] * 3) * dbosonic)
    if dfermionic is not None:
        state = state + tf.reshape(tf.cast(s, tf.complex64), [batch_size] + [1] * 5) * dfermionic
    return s, log_norm, state

def casimir(wavefunc, bosonic, dbosonic, dfermionic):
    """Evaluates the casimir density diven dbosonic = dg bosonic, dfermionic = dg fermionic(bosonic).

    Arguments:
        wavefunc (Wavefunction): the wavefunction that gives the mapping from bosonic coordinates to fermionic states
        bosonic, dbosonic (tensor of shape (batch_size, num_bosonic_matrices, N, N)): the bosonic matrix coordinates
            and its infinitesimal change
        dfermionic (tensor of shape (batch_size, rank, num_fermions, num_femionic_matrices, N, N) or None):
            infinitesimal change of the fermionic state, and None if no fermions

    Returns:
        casimir (tensor of shape (batch_size,)): value of the casimir at bosonic 
    """
    # bosonic contribution
    s, log_norm, state = dstate(wavefunc, bosonic, dbosonic, dfermionic)
    casimir = tf.square(tf.gradients(log_norm, s)[0])
    # fermionic contribution
    if state is not None:
        s_, _, state_ = dstate(wavefunc, bosonic, dbosonic, dfermionic)
        casimir = casimir + tf.gradients(tf.gradients(tf.real(overlap(state, state_)), s), s_)[0]
    return casimir


def matrix_kinetic_energy(wavefunc, bosonic):
    assert bosonic.shape[-1] == bosonic.shape[-2], "dimension mismatch"
    batch_size, num_matrices, N, _ = bosonic.shape.as_list()
    algebra = SU(N)
    num_matrices = int(bosonic.shape[1])
    dbosonic = tf.stack([algebra.vector_to_matrix(algebra.random_algebra_element(batch_size)) 
                            for _ in range(num_matrices)], axis=1)
    return casimir(wavefunc, bosonic, dbosonic, None)


def matrix_SUN_adjoint_casimir(wavefunc, bosonic):
    assert bosonic.shape[-1] == bosonic.shape[-2], "dimension mismatch"
    batch_size, _, N, _ = bosonic.shape.as_list()
    algebra = SU(N)
    dg = algebra.random_algebra_element(batch_size)
    dbosonic = algebra.infinitesimal_action(dg, bosonic)
    _, fermionic = wavefunc(bosonic)
    if fermionic is None:
        dfermionic = None
    else:
        dfermionic = algebra.infinitesimal_action(dg, fermionic)
    return casimir(wavefunc, bosonic, dbosonic, dfermionic)

def miniBMN_SO3_casimir(wavefunc, bosonic):
    assert bosonic.shape[1] == 3 and bosonic.shape[-1] == bosonic.shape[-2], "dimension mismatch"
    batch_size, _, N, _ = bosonic.shape.as_list()
    algebra_bosonic = SO3("vector")
    algebra_fermionic = SO3("spinor")
    dg = algebra_bosonic.random_algebra_element(batch_size)
    dbosonic = algebra_bosonic.infinitesimal_action(dg, bosonic, axis=-3)
    _, fermionic = wavefunc(bosonic)
    if fermionic is None:
        dfermionic = None
    else:
        dfermionic = algebra_fermionic.infinitesimal_action(dg, fermionic, axis=-3)
    return casimir(wavefunc, bosonic, dbosonic, dfermionic)


"""
Energy functions of the models. Note these are all energy densities; the expectation value of energy 
in the wavefunction is E_{bosonic ~ |wavefunction|^2}[ energy density at bosonic ], or tf.reduce_mean(energy) 
as a Monte Carlo estimate.
"""
def matrix_free_fields_energy(m, wavefunc, bosonic):
    """Returns the bosonic energy density of a free field on a fuzzy sphere.

    Arguments:
        m (float): the mass
        wavefunc (Wavefunction): the wavefunction object
        bosonic (tensor of shape (batch_size, 3, N, N)): vectorized bosonic matrices

    Returns:
        energy (tensors): energy density at the inputs, of shape (batch_size,)
    """
    kinetic = matrix_kinetic_energy(wavefunc, bosonic)
    potential = -matrix_spherical_laplacian(bosonic) + m * m * matrix_quadratic_potential(bosonic)
    return kinetic + potential


def fuzzy_sphere_energy(g, wavefunc, bosonic, potential_only=False):
    """Returns the bosonic energy density of the fuzzy sphere matrix Hamiltonian.

    Arguments:
        g (float): coupling of the cubic term
        wavefunc (Wavefunction): the wavefunction object
        bosonic (tensor of shape (batch_size, 3, N, N)): bosonic matrices
        potential_only (bool): whether to ignore kinetic terms, mainly for test use

    Returns:
        energy (tensors of shape (batch_size,)): energy densities
    """
    kinetic = 0.5 * matrix_kinetic_energy(wavefunc, bosonic)
    potential = -0.25 * matrix_commutator_square(bosonic) + g * matrix_cubic_potential(bosonic)
    return potential if potential_only else kinetic + potential


def miniBMN_bosonic_energy(m, wavefunc, bosonic, potential_only=False):
    """Returns the bosonic part of the energy density of the miniBMN matrix Hamiltonian.

    Arguments:
        m (float): mass parameter
        wavefunc (Wavefunction): the wavefunction
        bosonic (tensor of shape (batch_size, 3, N, N)): bosonic matrices
        potential_only (bool): whether to ignore kinetic terms, mainly for test use

    Returns:
        energy (tensors of shape (batch_size,)): energy densities
    """
    kinetic = 0.5 * matrix_kinetic_energy(wavefunc, bosonic)
    potential = -0.25 * matrix_commutator_square(bosonic)
    mass_deformation = 0.5 * m * m * matrix_quadratic_potential(bosonic) + m * matrix_cubic_potential(bosonic)

    return potential + mass_deformation if potential_only else kinetic + potential + mass_deformation

def miniBMN_energy(m, wavefunc, bosonic, potential_only=False):
    """Returns the energy density of the miniBMN matrix Hamiltonian.

    Arguments:
        m (float): mass parameter
        wavefunc (Wavefunction): the wavefunction
        bosonic (tensor of shape (batch_size, 3, N, N)): bosonic matrices
        potential_only (bool): whether to ignore kinetic terms, mainly for test use

    Returns:
        energy (tensors of shape (batch_size,)): energy densities
    """
    N = int(bosonic.shape[-1])
    _, fermionic = wavefunc(bosonic)
    num_fermions = 0 if fermionic is None else int(fermionic.shape[-4])

    kinetic = 0.5 * matrix_kinetic_energy(wavefunc, bosonic)
    potential = -0.25 * matrix_commutator_square(bosonic) + miniBMN_yukawa_potential(bosonic, fermionic)
    mass_deformation = 0.5 * m * m * matrix_quadratic_potential(bosonic) + m * matrix_cubic_potential(bosonic) \
                        + 1.5 * m * num_fermions - 1.5 * m * (N * N - 1)

    return potential + mass_deformation if potential_only else kinetic + potential + mass_deformation


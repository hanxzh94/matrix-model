import tensorflow as tf
import numpy as np 
from obs import *
from train import *
from algebra import *
from wavefunc import *
from dist import *


def logmeanexp(x, axis=-1):
    offset = tf.reduce_max(tf.real(x), axis=axis, keepdims=True)
    offset = tf.cast(offset, tf.complex64)
    res = tf.log(tf.reduce_mean(tf.exp(x - offset), axis=axis)) + tf.squeeze(offset, axis=axis)
    return res

def gauge_fix(wavefunc, bosonic):
    """Returns the gauge representatives of the bosonic samples, where the gauge slice is the linear subspace passing through 
    the saddle and perpendicular to gauge directions at the saddle. 

    The gauge fixing is done by numerically maximizing sum_i tr S^i U X^i U^-1 (S^i is the saddle) over unitaries U, and the 
    representatives are U_* X^i U_*^-1 where U_* is the argmax. First the saddle X^i = S^i is in the gauge slice; and for 
    any X^i in the gauge slice, sum_i tr S^i [Y, X^i] = 0 for any Hermitian matrix Y, i.e., sum_i tr [Y, S^i]^dagger (X^i - S^i) = 0 
    for any Y, so X^i - S^i is perpendicular to any gauge direction at S^i. This gauge may be not well-defined globally but this 
    will not be an issue because the results are meaningful only when the fluctuations around the saddle are infinitesimal anyway.

    Arguments:
        wavefunc (Wavefunction): the wavefunction object
        bosonic (tensors of shape (batch_size, 3, N, N)): bosonic matrix coordinates

    Returns:
        rep (tensors of shape (batch_size, 3, N, N)): representatives in the perpendicular gauge
    """
    batch_size, _, N, _ = bosonic.shape.as_list()
    algebra = SU(N)
    saddle = wavefunc.offset
    name = "ent"
    path = "results/" + name + "/"
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        mats = tf.get_variable("mats", shape=[batch_size, algebra.dim])
        unitaries = tf.tile(tf.expand_dims(tf.linalg.expm(1j * algebra.vector_to_matrix(mats)), axis=1), [1, 3, 1, 1])
        rep = tf.matmul(tf.matmul(unitaries, bosonic), tf.linalg.adjoint(unitaries))
        loss = -tf.real(tf.reduce_sum(tf.conj(saddle) * rep, [1, 2, 3]))
        obs = minimize(name, 0, loss, 1000, {"loss" : loss})
        saver = tf.train.Saver([mats])
        sess = tf.Session()
        assert restore(sess, saver, "results/" + name + "/model.ckpt", True), "restoration failed"
        return tf.constant(sess.run(rep))

def swap_diff(wavefunc, bosonic1, bosonic2, proj):
    """Returns the change in log wavefunction after a swap operator, useful in calculating entanglements.

    Denote each sample in bosonic1 as (x1, y1), where y are the variables in the subsystem to be traced out.
    Similarly denote bosonic2 = (x2, y2). Then this function returns log [ psi(x2, y1) / psi(x1, y1) ].
    Note fermions are always traced out.

    Arguments:
        wavefunc (Wavefunction): the wavefunction object
        bosonic1, bosonic2 (tensors of shape (batch_size, 3, N, N)): bosonic matrix coordinates
        proj (tensor of shape (3, N, N, 3, N, N)): projection onto the subsystem x

    Returns:
        diff (tensor): log psi(x2, y1) - log psi(x1, y1), of shape (batch_shape,)
    """
    project = lambda x: tf.einsum("rijskl,bskl->brij", tf.cast(proj, tf.complex64), x)

    bosonic_mixed = bosonic1 - project(bosonic1) + project(bosonic2) # (x2, y1)
    log_norm1, fermionc1 = wavefunc(bosonic1)
    log_norm_mixed, fermionic_mixed = wavefunc(bosonic_mixed)
    o = tf.log(overlap(fermionic1, fermionic_mixed)) if fermionc1 is not None else 0

    return tf.cast(log_norm_mixed - log_norm1, tf.complex64) + o

def entanglement(wavefunc, proj, samples):
    """Computes the Renyi entropy of the reduced density matrix of a subsystem given the wavefunction.

    Denote the coordinates of the subsystem as x and its complement as y, then the reduced density matrix is
    rho(x, x') = int dy psi(x, y) psi(x', y)^*, so tr rho^n = prod_{i = 0, ..., n-1} int dx_i dy_i psi(x_i, y_i) psi(x_{i+1}, y_i)^*
    where x_n = x_0. This can be estimated by Monte Carlo tr rho^n = E_{x_i, y_i}[prod_i psi(x_{i+1}, y_i)^* / psi(x_i, y_i)^*]
    and finally note H_n(rho) = log tr rho^n / (1 - n).

    Arguments:
        wavefunc (Wavefunction): the wavefunction object
        proj (tensor of shape (3, N, N, 3, N, N)): projection onto the subsystem
        samples (list of tensors of shape (batch_size, 3, N, N)): samples used to compute the entropy

    Returns:
        ent (scalar): entanglement entropy of the subsystem
    """
    n = len(samples) # the order of the Renyi entropy
    diff = sum(tf.conj(swap_diff(wavefunc, samples[i], samples[(i + 1) % n], proj)) for i in range(n))
    ent = logmeanexp(diff) / (1 - n)
    return ent


def matrix_entanglement(scope, wavefunc, restore_path, n=2, num_samples=1000):
    """Computes the mutual information of the matrix wavefunction given projections in the file.

    Information of partition is read from a file GaugeProjectionsN.bin where N is the size of the matrix. It is a binary file which 
    starts with an integer num, the number of angles to compute. Then there are num records where each record starts with a float
    which is the angle of the spherical cap region and then an array of floats of shape (3, N, N, 3, N, N), the projection matrix 
    P^rs_ijkl. It maps bosonic matrices A^r_ij to sum_skl P^rs_ijkl A^s_kl. 

    The returned result is the mutual information between bosonic regions given by the projector, and the fermions are traced out. 

    Arguments:
        scope (string): the variable scope where the model is stored
        wavefunc (Wavefunction): the model for the wavefunction
        restore_path (string): the path where the trained model is stored
        n (int): the order of the Renyi entropy to compute
        num_samples (int): the number of samples to use

    Returns:
        angles (numpy array): angles of the spherical cap
        ent (numpy array): the mutual information
    """
    # load the model
    model_path = restore_path + "model.ckpt"
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    saver = tf.train.Saver(variables)
    sess = tf.Session()
    assert restore(sess, saver, model_path, True), "failed to load the model."
    # prepare the samples beforehand
    samples = [wavefunc.sample(num_samples) for _ in range(n)]
    samples = [gauge_fix(wavefunc, tf.constant(sess.run(wavefunc.vectorizer.algebra.gauge_fixing(s)[-1]))) for s in samples]
    # read the projectors 
    N = wavefunc.vectorizer.algebra.N
    filename = "data/GaugeProjections" + str(N) + ".bin"
    with open(filename, "rb") as f:
        num = np.reshape(np.fromfile(f, dtype=np.dtype("int32"), count=1), [])
        print("Total number of angles:", num)
        angles, ent = np.zeros([num]), np.zeros([num])
        for i in range(num):
            theta = np.reshape(np.fromfile(f, dtype=np.dtype("float32"), count=1), [])
            print("Angle:", theta)
            angles[i] = theta
            proj1 = np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=9*N*N*N*N), [3, N, N, 3, N, N])
            proj2 = np.reshape(np.eye(3*N*N), [3, N, N, 3, N, N]) - proj1
            proj1, proj2 = tf.constant(proj1), tf.constant(proj2)
            ent1 = entanglement(wavefunc, proj1, samples)
            ent2 = entanglement(wavefunc, proj2, samples)
            ent1, ent2 = sess.run([ent1, ent2])
            print("Ent:", ent1, ent2)
            ent[i] = np.real(ent1 + ent2)
        ent = ent - (ent[0] + ent[-1]) / 2
    return angles, ent


if __name__ == "__main__":
    num_fermions = 0
    N = 4
    for m in [8.0]:
        tf.reset_default_graph()
        print("m =", m)
        # spec
        batch_size = 1000
        num_steps = 4000
        rank = 2 * num_fermions
        num_layers = 1
        scope = "nf" + "N" + str(N) + "f0r1demo"
        name = "nf" + "N" + str(N) + "m" + str(m) + "f0r1demo"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(scope):
            # read the spin matrices
            with open("data/SpinMatrices" + str(N) + ".bin", "rb") as f:
                mats = tf.constant(np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=3*N*N), [3, N, N]))
            Sx, Sy, Sz = mats[0], mats[1], mats[2]
            offset = m * tf.stack([-Sz, -Sy, -Sx])
            vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
            offset = vectorizer.encode(tf.expand_dims(offset, 0))[0]
            # construct wavefunction
            bosonic_dim = 2 * algebra.dim
            fermionic_dim = 2 * algebra.dim
            bosonic_wavefunc = NormalizingFlow([Normal()] * bosonic_dim, 0, tfp.bijectors.Sigmoid(), offset=offset)
            fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, num_fermions, rank, fermionic_dim, num_layers)
            wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            wavefunc.offset = m * tf.stack([-Sz, -Sy, -Sx])
            bosonic = wavefunc.sample(batch_size)
            log_norm, _ = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        energy = miniBMN_energy(m, wavefunc, bosonic)
        # training
        print("Training ...")
        output_path = "results/" + name + "/"
        obs = minimize(scope, log_norm, energy, num_steps,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        lr=1e-2, output_path=output_path)
        obs = minimize(scope, log_norm, energy, num_steps,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        lr=1e-3, restore_path=output_path, output_path=output_path)
        obs = minimize(scope, log_norm, energy, num_steps,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        lr=1e-4, restore_path=output_path, output_path=output_path)
        obs = minimize(scope, log_norm, energy, num_steps,
                        {"r": radius, "gauge": gauge, "rotation": rotation}, 
                        lr=1e-5, restore_path=output_path, output_path=output_path)
    for m in [8.0]:
        tf.reset_default_graph()
        print("m =", m)
        # spec
        batch_size = 1000
        num_steps = 4000
        rank = 2 * num_fermions
        num_layers = 1
        scope = "nf" + "N" + str(N) + "f0r1demo"
        name = "nf" + "N" + str(N) + "m" + str(m) + "f0r1demo"
        algebra = SU(N)
        # build the model
        with tf.variable_scope(scope):
            # read the spin matrices
            with open("data/SpinMatrices" + str(N) + ".bin", "rb") as f:
                mats = tf.constant(np.reshape(np.fromfile(f, dtype=np.dtype("complex64"), count=3*N*N), [3, N, N]))
            Sx, Sy, Sz = mats[0], mats[1], mats[2]
            offset = m * tf.stack([-Sz, -Sy, -Sx])
            vectorizer = Vectorizer(algebra, tfp.bijectors.Exp())
            offset = vectorizer.encode(tf.expand_dims(offset, 0))[0]
            # construct wavefunction
            bosonic_dim = 2 * algebra.dim
            fermionic_dim = 2 * algebra.dim
            bosonic_wavefunc = NormalizingFlow([Normal()] * bosonic_dim, 0, tfp.bijectors.Sigmoid(), offset=offset)
            fermionic_wavefunc = FermionicWavefunction(algebra, bosonic_dim, 2, num_fermions, rank, fermionic_dim, num_layers)
            wavefunc = Wavefunction(algebra, vectorizer, bosonic_wavefunc, fermionic_wavefunc)
            wavefunc.offset = m * tf.stack([-Sz, -Sy, -Sx])
            bosonic = wavefunc.sample(batch_size)
            log_norm, _ = wavefunc(bosonic)
        # observables
        radius = tf.sqrt(matrix_quadratic_potential(bosonic) / N)
        gauge = matrix_SUN_adjoint_casimir(wavefunc, bosonic)
        rotation = miniBMN_SO3_casimir(wavefunc, bosonic)
        energy = miniBMN_energy(m, wavefunc, bosonic)
        # evaluation
        print("Evaluating ...")
        restore_path = "results/" + name + "/"
        angles, ent = matrix_entanglement(scope, wavefunc, restore_path, num_samples=10000)
        print(list(ent))

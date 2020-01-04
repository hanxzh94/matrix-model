import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline


def restore(sess, saver, model_path, default=None):
    """Restores all the variables saved to the session.

    Arguments:
        sess (tf.Session): session to restore the variables
        saver (tf.train.Saver): saver for restoring
        model_path (string): checkpoint name of the model saved
        default (bool or None): whether to restore by default; no default if None

    Returns:
        restored (bool): whether the model is restored
    """
    restored = False
    if tf.train.checkpoint_exists(model_path):
        if default is None:
            response = ""
        else:
            response = "yes" if default else "no"
        while response not in ["y", "n", "yes", "no"]:
            response = input("Model already existed, restore variables? (y/n)")
        if response in ["y", "yes"]:
            restored = True
            saver.restore(sess, model_path)
            print("Model restored.")
    else:
        print("Model not found.")
    return restored


def zero_nan(has_nans):
    # replaces nans by zeros in the tensor
    return tf.where(tf.is_nan(has_nans), tf.zeros_like(has_nans), has_nans)


def truncate(data, lower=0.1, upper=0.9):
    # sorts and removes the exceptional (large or small) values in the list to reduce variance
    s, l = sorted(data), len(data)
    return s[int(lower * l) : int(upper * l)]


def minimize(scope, log_norm, energy, steps, check_obs={}, \
                check_every=100, save_every=1000, lr=0.001, restore_path=None, output_path=None, profiling=False):
    """Trains the wavefunction to minimize the energy.

    Arguments:
        scope (string): the variable scope where all variables to be trained reside
        log_norm (tensor): a batch of log wavefunction norm of samples, shape (batch_shape,) 
        energy (tensor): energies of the same batch of samples, shape (batch_shape,)
        steps (int): the total number of training steps
        check_obs (dict from strings to tensors): names and values of the tensors to log
        check_every (int): the number of steps between logging
        save_every (int): the number of steps between saving the model
        lr (float): learning rate for the optimizer
        restore_path (string or None): the directory (ending with "/") to restore the model
                If None, start from scratch by default.
        output_path (string or None): the directory to save the trained model
                If None, save to "results/" + scope + "/" by default.
        profiling (bool): whether to profile the code

    Returns:
        obs (dict from strings to scalars or lists of scalars): statistics of the tensors in check_obs
            For each name in check_obs, obs contains an entry of the same name for the average value of 
            the observable, and an entry of name_std for its standard deviation, with an entry of name_raw 
            for its history of values during the training.
    """
    # set up training ops
    mean_energy = tf.reduce_mean(energy)
    loss = energy + 2 * log_norm * tf.stop_gradient(energy - mean_energy)
    loss = tf.reduce_mean(loss)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    # print(variables)
    print("Total number of parameters:", sum(tf.Session().run(tf.size(v)) for v in variables))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = optimizer.compute_gradients(loss, variables)
    grads_and_vars = [(zero_nan(tf.clip_by_norm(grad, 1.0)), var) for grad, var in grads_and_vars if grad is not None]               
    train_op = optimizer.apply_gradients(grads_and_vars)
    # set up loggings
    output_path = output_path or "results/" + scope + "/"
    model_path = output_path + "model.ckpt"
    saver = tf.train.Saver(variables)
    file_writer = tf.summary.FileWriter(output_path)
    check_obs = {key: tf.reduce_mean(value) for key, value in check_obs.items()}
    check_obs["energy"] = mean_energy
    for key, value in check_obs.items():
        tf.summary.scalar(key, value)
    summary = tf.summary.merge_all()
    res = {key: [] for key in check_obs}
    # initialize variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    sess.run(tf.variables_initializer(optimizer.variables()))
    if restore_path is None or not restore(sess, saver, restore_path + "model.ckpt", True):
        sess.run(tf.variables_initializer(variables))
        print("Starting from scratch ...")
    progbar = tf.keras.utils.Progbar(steps, stateful_metrics=list(check_obs.keys()))
    if profiling:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    # training
    for step in range(1, steps + 1):        
        if profiling and step % save_every == 0:
            e, _ = sess.run([mean_energy, train_op], options=options, run_metadata=run_metadata)
        else:
            e, _ = sess.run([mean_energy, train_op])
        # log
        if step % check_every == 0:
            obs = [(key, sess.run(value)) for key, value in check_obs.items()]
            for key, value in obs:
                res[key].append(value)
            progbar.update(step, obs)
            file_writer.add_summary(sess.run(summary), step)
        else:
            progbar.update(step, [("energy", e)]) # always update energy
        # save the model
        if step % save_every == 0:
            saver.save(sess, model_path)
            print(" Model saved to", model_path)
            if profiling:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(output_path + 'timeline_%d.json' % (step // save_every), 'w') as f:
                    f.write(chrome_trace)
    # return the observables
    raw = {key + "_raw": res[key] for key in res}
    mean = {key: np.mean(truncate(res[key])) for key in res}
    std = {key + "_std": np.std(truncate(res[key])) for key in res}
    return {**raw, **mean, **std}
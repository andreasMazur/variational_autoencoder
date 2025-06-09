import tensorflow as tf


def analytical_kl_div(pred_mean, pred_var):
    return -(1 / 2) * tf.reduce_sum(1 + pred_var - pred_mean ** 2 - tf.math.exp(pred_var), axis=-1)


def mean_squared_error(y_true, reconstruction, batch_dims=1):
    # Reshape inputs to vectors
    y_true = tf.reshape(y_true, tf.concat([tf.shape(y_true)[:batch_dims], [-1]], axis=0))
    reconstruction = tf.reshape(reconstruction, tf.concat([tf.shape(reconstruction)[:batch_dims], [-1]], axis=0))

    # Compute MSE per instance in batch
    return tf.reduce_mean(tf.math.squared_difference(y_true, reconstruction), axis=-1)


def binary_cross_entropy(y_true, reconstruction):
    # Assumes log-output for reconstruction
    probabilities = tf.clip_by_value(tf.math.sigmoid(reconstruction), clip_value_min=1e-7, clip_value_max=1 - 1e-7)
    return -(y_true * tf.math.log(probabilities) + (1 - y_true) * tf.math.log(1 - probabilities))


def compute_entropy(logits):
    probabilities = tf.nn.softmax(logits, axis=-1)
    return -tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=-1)

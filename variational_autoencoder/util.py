import tensorflow as tf


def analytical_kl_div(pred_mean, pred_var):
    return -(1 / 2) * tf.reduce_sum(1 + pred_var - pred_mean ** 2 - tf.math.exp(pred_var), axis=-1)


def mean_squared_error(y_true, reconstruction):
    # Reshape inputs to vectors
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    reconstruction = tf.reshape(reconstruction, (tf.shape(reconstruction)[0], -1))
    # First average over epsilon-samples, then compute MSE for instances in batch
    return tf.reduce_mean(tf.math.squared_difference(y_true, reconstruction), axis=-1)


def binary_cross_entropy(y_true, reconstruction):
    # Assumes log-output for reconstruction
    probabilities = tf.clip_by_value(tf.math.sigmoid(reconstruction), clip_value_min=1e-7, clip_value_max=1 - 1e-7)
    return -(y_true * tf.math.log(probabilities) + (1 - y_true) * tf.math.log(1 - probabilities))


def compute_entropy(logits):
    probabilities = tf.nn.softmax(logits, axis=-1)
    return -tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=-1)

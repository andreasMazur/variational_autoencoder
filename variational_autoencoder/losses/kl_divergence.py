from variational_autoencoder.util import analytical_kl_div

import tensorflow as tf
import keras


class KLDivergence(keras.losses.Loss):
    def call(self, y_true, y_pred, reduce_sum=True):
        pred_mean, pred_var = y_pred
        if reduce_sum:
            return tf.reduce_sum(analytical_kl_div(pred_mean, pred_var))
        else:
            return analytical_kl_div(pred_mean, pred_var)

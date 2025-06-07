from variational_autoencoder.util import mean_squared_error

import tensorflow as tf
import keras


class ReconstructionLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, reduce_sum=True):
        if reduce_sum:
            return tf.reduce_sum(mean_squared_error(y_true, y_pred))
        else:
            return mean_squared_error(y_true, y_pred)

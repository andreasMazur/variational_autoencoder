from variational_autoencoder.util import mean_squared_error

import tensorflow as tf
import keras


class ReconstructionLoss(keras.losses.Loss):
    def __init__(self, batch_dims=1, *args, **kwargs):
        super(ReconstructionLoss, self).__init__(*args, **kwargs)
        self.batch_dims = batch_dims

    def call(self, y_true, y_pred, reduce_sum=True):
        if reduce_sum:
            return tf.reduce_sum(mean_squared_error(y_true, y_pred, self.batch_dims))
        else:
            return mean_squared_error(y_true, y_pred, self.batch_dims)

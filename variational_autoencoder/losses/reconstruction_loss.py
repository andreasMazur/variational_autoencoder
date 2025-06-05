from variational_autoencoder.util import mean_squared_error

import keras


class ReconstructionLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

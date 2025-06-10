from variational_autoencoder.losses.kl_divergence import KLDivergence
from variational_autoencoder.losses.reconstruction_loss import ReconstructionLoss

import tensorflow as tf
import keras


class VariationalAutoEncoder(keras.models.Model):
    """Variational Autoencoder (VAE) for unsupervised Machine Learning.

    Publication that introduces the VAE:
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." 20 Dec. 2013,

    Attributes
    ----------
    encoder : keras.models.Model
        The encoder model that encodes input features and associated labels into a latent space.
    latent_dim : int
        The dimension of the latent space.
    decoder : keras.models.Model
        The decoder model that decodes latent representations together with a label back into the original feature
        space.
    beta : float
        A regularization coefficient for the KL-divergence loss.
    """
    def __init__(self, encoder, latent_dim, decoder, beta=1., warmup_steps=300, training_steps=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # Initialize bottleneck
        self.latent_dim = latent_dim
        self.mean_predictor = keras.layers.Dense(
            self.latent_dim,
            activation="linear",
            name="mean_predictor"
        )
        self.log_var_predictor = keras.layers.Dense(
            self.latent_dim,
            activation="linear",
            name="log_std_predictor"
        )

        self.beta = beta
        self.kl_divergence = KLDivergence()
        self.reconstruction_loss = ReconstructionLoss()

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_divergence")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        # Warmup attributes
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps

    def call(self, inputs, training=False):
        reconstruction, _, _ = self.call_detailed(inputs, training=True)
        return reconstruction

    def call_detailed(self, inputs, training=False):
        pred_mean, pred_log_var = self.encode(inputs, training=training)
        reconstruction = self.decode(pred_mean, pred_log_var, training=training)
        return reconstruction, pred_mean, pred_log_var

    def encode(self, inputs, training=False):
        inputs = self.encoder(inputs, training=training)
        pred_mean = self.mean_predictor(inputs, training=training)
        pred_log_var = self.log_var_predictor(inputs, training=training)
        return pred_mean, pred_log_var

    def decode(self, pred_mean, pred_log_var, training=False):
        samples = self.sample(pred_mean, pred_log_var, training=training)
        return self.decoder(samples)

    def sample(self, pred_mean, pred_log_var, training=False):
        if training:
            epsilon = tf.random.normal(shape=tf.shape(pred_mean), mean=0.0, stddev=1.0)
        else:
            epsilon = 0.
        return pred_mean + tf.math.exp(pred_log_var / 2) * epsilon

    def train_step(self, data):
        input_features, output_features = data
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, pred_mean, pred_log_var = self.call_detailed(input_features, training=True)

            # Compute beta-weighted, negative evidence lower bound (ELBO)
            kl_loss = self.kl_divergence(y_true=(), y_pred=(pred_mean, pred_log_var))
            recon_loss = self.reconstruction_loss(y_true=output_features, y_pred=reconstruction)

            coeff = self.training_steps / self.warmup_steps if self.training_steps < self.warmup_steps else 1.0
            self.training_steps = self.training_steps + 1 if self.training_steps < self.warmup_steps else self.training_steps

            total_loss = coeff * self.beta * kl_loss + recon_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update tracker
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(recon_loss)

        return {
            "loss": total_loss,
            "kl_divergence": kl_loss,
            "reconstruction_loss": recon_loss,
        }

    def test_step(self, data):
        input_features, output_features = data

        # Forward pass
        reconstruction, pred_mean, pred_log_var = self.call_detailed(input_features, training=False)

        # Compute beta-weighted, negative evidence lower bound (ELBO)
        kl_loss = self.kl_divergence(y_true=(), y_pred=(pred_mean, pred_log_var))
        recon_loss = self.reconstruction_loss(y_true=output_features, y_pred=reconstruction)
        total_loss = self.beta * kl_loss + recon_loss

        # Update tracker
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.recon_loss_tracker.update_state(recon_loss)

        return {
            "loss": total_loss,
            "kl_divergence": kl_loss,
            "reconstruction_loss": recon_loss,
        }

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.kl_loss_tracker, self.recon_loss_tracker]

    def get_config(self):
        config = super().get_config()
        config["encoder"] = keras.utils.serialize_keras_object(self.encoder)
        config["latent_dim"] = self.latent_dim
        config["decoder"] = keras.utils.serialize_keras_object(self.decoder)
        config["beta"] = self.beta
        config["warmup_steps"] = self.warmup_steps
        config["training_steps"] = self.training_steps
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(
            encoder=keras.saving.deserialize_keras_object(config["encoder"]),
            latent_dim=config["latent_dim"],
            decoder=keras.saving.deserialize_keras_object(config["decoder"]),
            beta=config["beta"],
            warmup_steps=config["warmup_steps"],
            training_steps=config["training_steps"]
        )

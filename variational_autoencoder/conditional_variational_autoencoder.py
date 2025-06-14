from variational_autoencoder.losses.kl_divergence import KLDivergence
from variational_autoencoder.losses.reconstruction_loss import ReconstructionLoss
from variational_autoencoder.util import compute_entropy

import tensorflow as tf
import keras


class CVariationalAutoEncoder(keras.models.Model):
    """Conditional Variational Autoencoder (cVAE) with classifier for semi-supervised Machine Learning.

    Publication that introduces the cVAE:
    Kingma, Durk P., et al. "Semi-supervised learning with deep generative models."
    Advances in neural information processing systems 27 (2014).

    Attributes
    ----------
    encoder : keras.models.Model
        The encoder model that encodes input features and associated labels into a latent space.
    latent_dim : int
        The dimension of the latent space.
    decoder : keras.models.Model
        The decoder model that decodes latent representations together with a label back into the original feature
        space.
    classifier : keras.models.Model
        A classification model that is used to predict labels for unlabeled data.
    alpha : float
        A regularization coefficient for the classification models loss.
    beta : float
        A regularization coefficient for the KL-divergence loss.
    """
    def __init__(self,
                 encoder,
                 latent_dim,
                 decoder,
                 classifier,
                 alpha=0.1,
                 beta=1.,
                 kl_divergence_loss=None,
                 reconstruction_loss=None,
                 warmup_steps=300,
                 training_steps=0,
                 mean_predictor=None,
                 log_var_predictor=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        # Initialize bottleneck
        self.latent_dim = latent_dim
        if mean_predictor is None:
            self.mean_predictor = keras.layers.Dense(
                self.latent_dim,
                activation="linear",
                name="mean_predictor"
            )
        else:
            self.mean_predictor = mean_predictor
        if log_var_predictor is None:
            self.log_var_predictor = keras.layers.Dense(
                self.latent_dim,
                activation="linear",
                name="log_std_predictor"
            )
        else:
            self.log_var_predictor = log_var_predictor

        self.classification_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum_over_batch_size"
        )

        self.alpha = alpha
        self.beta = beta
        if kl_divergence_loss is None:
            self.kl_divergence = KLDivergence()
        if reconstruction_loss is None:
            self.reconstruction_loss = ReconstructionLoss()

        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_divergence")
        self.clf_loss_tracker = keras.metrics.Mean(name="classifier_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

        self.training_steps = training_steps
        self.warmup_steps = warmup_steps

    def call(self, inputs, training=False):
        reconstructions, _, _ = self.call_detailed(inputs, training=training)
        return reconstructions

    def call_detailed(self, inputs, training=False):
        # Concat labels to features
        features, labels = inputs
        c_labeled_features = self.concat_labels_to_features(features, labels)

        # Encoder of cVAE is label dependent: p(z | x, y)
        pred_mean, pred_log_var = self.encode(c_labeled_features, training=training)

        # Decoder of cVAE is label dependent: p(x | y, z)
        reconstruction = self.decode(pred_mean, pred_log_var, labels, training=training)
        return reconstruction, pred_mean, pred_log_var

    def concat_labels_to_features(self, features, labels):
        # Get number of extra dimensions
        n_dims = tf.rank(features) - 1

        # Build shape for reshaping labels
        new_shape = tf.concat([tf.shape(labels), tf.ones(n_dims, dtype=tf.int32)], axis=0)
        labels = tf.reshape(labels, new_shape)

        # Tile up to feature shape
        multiples = tf.concat([[1], tf.shape(features)[1:]], axis=0)
        labels = tf.tile(labels, multiples=multiples)

        # Concat labels to innermost dimension of features
        return tf.concat([features, tf.cast(labels, features.dtype)], axis=-1)

    def encode(self, inputs, training=False):
        inputs = self.encoder(inputs, training=training)
        pred_mean = self.mean_predictor(inputs, training=training)
        pred_log_var = self.log_var_predictor(inputs, training=training)
        return pred_mean, pred_log_var

    def decode(self, pred_mean, pred_log_var, labels, training=False):
        samples = self.sample(pred_mean, pred_log_var, training=training)

        # cVAE learns label-dependent decoder p(x | y, z) => Concat labels to features
        return self.decoder(tf.concat([samples, tf.cast(labels[:, None], samples.dtype)], axis=-1), training=training)

    def sample(self, pred_mean, pred_log_var, training=False):
        if training:
            epsilon = tf.random.normal(shape=tf.shape(pred_mean), mean=0.0, stddev=1.0)
        else:
            epsilon = 0.
        return pred_mean + tf.math.exp(pred_log_var / 2) * epsilon

    def compute_labeled_loss(self, features, labels, training=True):
        reconstruction, pred_mean, pred_log_var = self.call_detailed((features, labels), training=training)

        # Compute beta-weighted, negative evidence lower bound (ELBO)
        kl_loss = self.kl_divergence.call(y_true=(), y_pred=(pred_mean, pred_log_var), reduce_sum=False)
        recon_loss = self.reconstruction_loss.call(
            y_true=tf.cast(features, reconstruction.dtype), y_pred=reconstruction, reduce_sum=False
        )

        # Update tracker
        self.kl_loss_tracker.update_state(tf.reduce_sum(kl_loss))
        self.recon_loss_tracker.update_state(tf.reduce_sum(recon_loss))

        coeff = self.training_steps / self.warmup_steps if self.training_steps < self.warmup_steps else 1.0
        self.training_steps = (
            self.training_steps + 1 if self.training_steps < self.warmup_steps else self.training_steps
        )

        return coeff * self.beta * kl_loss + recon_loss

    def compute_unlabeled_loss(self, non_labeled_features, training=True):
        # Estimate labels using a classifier: q(y | x)
        prediction = self.classifier(non_labeled_features, training=training)
        predictions_shape = tf.shape(prediction)

        # Compute beta-weighted, negative evidence lower bound (ELBO) for unlabeled features
        # -L(x, y)
        def elbo_wrapper(label):
            return self.compute_labeled_loss(
                non_labeled_features, tf.fill((predictions_shape[0],), value=label), training=training
            )

        # Compute expected ELBO over all classes
        # neg_elbo = E_{q(y | x)}[-L(x, y)] = sum_y q(y | x)*(-L(x, y))
        neg_elbo = tf.map_fn(elbo_wrapper, tf.range(predictions_shape[1]), fn_output_signature=tf.float32)
        neg_elbo = tf.einsum("ij,ji->i", tf.nn.softmax(prediction, axis=-1), neg_elbo)

        # Compute entropy
        # H(q(y | x))
        entropy = compute_entropy(prediction)

        # Return loss for unsupervised case: E_{q(y | x)}[-L(x, y)] + H(q(y | x))
        return neg_elbo + entropy

    @tf.function
    def train_step(self, data):
        input_features, labels = data

        # Filter for given and non-given labels
        labeled_features, labels, non_labeled_features = self.divide_data(input_features, labels)

        with tf.GradientTape() as tape:
            # 1.) Compute loss for features where labels are given
            if tf.shape(labeled_features)[0] > 0:
                labeled_loss = tf.reduce_sum(self.compute_labeled_loss(labeled_features, labels, training=True))
            else:
                labeled_loss = 0.0

            # 2.) Compute loss for features with no labels and classification loss
            if tf.shape(non_labeled_features)[0] > 0:
                unlabeled_loss = tf.reduce_sum(self.compute_unlabeled_loss(non_labeled_features, training=True))
            else:
                unlabeled_loss = 0.0

            # 3.) Compute classification loss on labeled data
            predictions = self.classifier(labeled_features, training=True)
            clf_loss = self.classification_loss(labels, predictions)

            # 4.) Combine total loss
            total_loss = labeled_loss + unlabeled_loss + self.alpha * clf_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.clf_loss_tracker.update_state(clf_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_divergence": self.kl_loss_tracker.result(),
            "classifier_loss": self.clf_loss_tracker.result()
        }

    def divide_data(self, features, labels):
        # Get set of features with labels
        indices = tf.where(labels != -1)
        labeled_features = tf.gather_nd(features, indices)
        given_labels = tf.gather_nd(labels, indices)

        # Get set of features without labels
        indices = tf.where(labels == -1)
        non_labeled_features = tf.gather_nd(features, indices)

        return labeled_features, given_labels, non_labeled_features

    @tf.function
    def test_step(self, data):
        input_features, labels = data

        # Filter for given and non-given labels
        labeled_features, labels, non_labeled_features = self.divide_data(input_features, labels)

        # 1.) Compute loss for features where labels are given
        if tf.shape(labeled_features)[0] > 0:
            labeled_loss = tf.reduce_sum(self.compute_labeled_loss(labeled_features, labels, training=True))
        else:
            labeled_loss = 0.0

        # 2.) Compute loss for features with no labels and classification loss
        if tf.shape(non_labeled_features)[0] > 0:
            unlabeled_loss = tf.reduce_sum(self.compute_unlabeled_loss(non_labeled_features, training=True))
        else:
            unlabeled_loss = 0.0

        # 3.) Compute classification loss on labeled data
        predictions = self.classifier(labeled_features, training=False)
        clf_loss = self.classification_loss(labels, predictions)
        self.clf_loss_tracker.update_state(clf_loss)

        # 4.) Combine loss
        total_loss = labeled_loss + self.alpha * clf_loss + unlabeled_loss
        self.total_loss_tracker.update_state(total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_divergence": self.kl_loss_tracker.result(),
            "classifier_loss": self.clf_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.recon_loss_tracker, self.kl_loss_tracker, self.total_loss_tracker, self.clf_loss_tracker]

    def get_config(self):
        config = super().get_config()
        config["encoder"] = keras.utils.serialize_keras_object(self.encoder)
        config["latent_dim"] = self.latent_dim
        config["decoder"] = keras.utils.serialize_keras_object(self.decoder)
        config["beta"] = self.beta
        config["classifier"] = keras.utils.serialize_keras_object(self.classifier)
        config["alpha"] = self.alpha
        config["warmup_steps"] = self.warmup_steps
        config["training_steps"] = self.training_steps
        config["mean_predictor"] = keras.utils.serialize_keras_object(self.mean_predictor)
        config["log_var_predictor"] = keras.utils.serialize_keras_object(self.log_var_predictor)
        return config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(
            encoder=keras.saving.deserialize_keras_object(config["encoder"]),
            latent_dim=config["latent_dim"],
            decoder=keras.saving.deserialize_keras_object(config["decoder"]),
            beta=config["beta"],
            classifier=keras.saving.deserialize_keras_object(config["classifier"]),
            alpha=config["alpha"],
            warmup_steps=config["warmup_steps"],
            training_steps=config["training_steps"],
            mean_predictor=keras.saving.deserialize_keras_object(config["mean_predictor"]),
            log_var_predictor=keras.saving.deserialize_keras_object(config["log_var_predictor"]),
        )

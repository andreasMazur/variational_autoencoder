from variational_autoencoder import CVariationalAutoEncoder

import unittest
import keras
import os
import numpy as np
import tensorflow as tf


class TestVAE(unittest.TestCase):
    def setUp(self):
        # Load data
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        self.train_images = (train_images / 255.0).reshape(-1, 28, 28, 1)
        self.train_labels = train_labels
        self.test_images = (test_images / 255.0).reshape(-1, 28, 28, 1)
        self.test_labels = test_labels

        # Initialize encoder architecture
        latent_dim = 64
        encoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(latent_dim, activation="relu")
        ])

        # Initialize decoder architecture
        decoder = keras.Sequential([
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(28 * 28, activation="sigmoid"),
            keras.layers.Reshape((28, 28, 1))
        ])

        # Initialize classifier
        classifier = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="linear", name="classifier_output")
        ])

        # Initialize VAE
        self.vae = CVariationalAutoEncoder(
            encoder=encoder,
            latent_dim=latent_dim,
            decoder=decoder,
            classifier=classifier
        )
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    def test_model_pipeline(self):
        """Test whether VAE can be loaded after being stored"""
        # Randomly assign -1 to train-labels (simulate unlabeled data)
        remove_label = np.random.binomial(n=1, p=0.5, size=(self.train_labels.shape[0],))
        training_labels = np.array(self.train_labels).astype(np.int8)
        training_labels[remove_label.astype(np.bool)] = -1

        remove_label = np.random.binomial(n=1, p=0.5, size=(self.test_labels.shape[0],))
        test_labels = np.array(self.test_labels).astype(np.int8)
        test_labels[remove_label.astype(np.bool)] = -1

        # Train with and without labels (semi-supervised case)
        self.vae.fit(
            x=tf.constant(self.train_images),
            y=tf.constant(training_labels),
            epochs=10,
            batch_size=64,
            validation_data=(self.test_images, tf.constant(test_labels))
        )

        # Save model
        model_path = "./vae_test_model.keras"
        self.vae.save(model_path)

        # Load model
        try:
            self.vae = keras.saving.load_model(
                model_path,
                custom_objects={"CVariationalAutoEncoder": CVariationalAutoEncoder}
            )
        except:
            pass
        os.remove(model_path)

        # Check if the model can be used after loading
        test_input = self.train_images[0, None]
        test_input_label = self.train_labels[0, None]

        # Input is represented by an image and its associated label
        test_output = self.vae((test_input, test_input_label)).numpy()
        assert test_input.shape == test_output.shape, "Input shape has a different shape compared to output shape."


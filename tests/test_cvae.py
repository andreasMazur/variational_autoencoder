from variational_autoencoder import CVariationalAutoEncoder

from matplotlib import pyplot as plt

import unittest
import keras
import os
import numpy as np
import tensorflow as tf


class TestCVAE(unittest.TestCase):
    def setUp(self):
        # Load data
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        self.train_images = (train_images / 255.0).reshape(-1, 28, 28, 1)
        self.train_labels = train_labels
        self.test_images = (test_images / 255.0).reshape(-1, 28, 28, 1)
        self.test_labels = test_labels

        # Initialize encoder architecture
        latent_dim = 2
        encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 2)),  # Input features: grey-scale image and label
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
        ])

        # Initialize decoder architecture
        decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim + 1,)),  # Input features: latent vector and label
            keras.layers.Dense(units=7 * 7 * 32, activation="relu"),
            keras.layers.Reshape(target_shape=(7, 7, 32)),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
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
            epochs=5,
            batch_size=64,
            validation_data=(self.test_images, tf.constant(test_labels))
        )

        # Save model
        model_path = "./vae_test_model.keras"
        self.vae.save(model_path)

        # Load model
        self.vae = keras.saving.load_model(
            model_path,
            custom_objects={"CVariationalAutoEncoder": CVariationalAutoEncoder}
        )

        # Check if the model can be used after loading
        _, means, _ = self.vae.call_detailed((self.train_images[:1000], self.train_labels[:1000]))
        plt.scatter(means[:, 0], means[:, 1])
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.grid()
        plt.show()

        os.remove(model_path)

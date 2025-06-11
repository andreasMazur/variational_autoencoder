from variational_autoencoder import VariationalAutoEncoder

from matplotlib import pyplot as plt

import unittest
import keras
import os


class TestVAE(unittest.TestCase):
    def setUp(self):
        # Load data
        (train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0

        # Initialize encoder architecture
        latent_dim = 2
        encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
        ])

        # Initialize decoder architecture
        decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(units=7 * 7 * 32, activation="relu"),
            keras.layers.Reshape(target_shape=(7, 7, 32)),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
        ])

        # Initialize VAE
        self.vae = VariationalAutoEncoder(
            encoder=encoder,
            latent_dim=latent_dim,
            decoder=decoder
        )
        self.vae.compile(optimizer=keras.optimizers.Adam(1e-4))

    def test_model_loading(self):
        """Test whether VAE can be loaded after being stored"""
        self.vae.fit(
            x=self.train_images,
            y=self.train_images,
            epochs=10,
            batch_size=64,
            validation_data=(self.test_images, self.test_images)
        )
        # Save model
        model_path = "./vae_test_model.keras"
        self.vae.save(model_path)

        # Load model
        self.vae = keras.saving.load_model(
            model_path,
            custom_objects={"VariationalAutoEncoder": VariationalAutoEncoder}
        )

        # Check if the model can be used after loading
        _, means, _ = self.vae.call_detailed(self.train_images[:1000])
        plt.scatter(means[:, 0], means[:, 1])
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.grid()
        plt.show()

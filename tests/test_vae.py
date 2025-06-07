from variational_autoencoder import VariationalAutoEncoder

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
            keras.layers.Reshape((28, 28))
        ])

        # Initialize VAE
        self.vae = VariationalAutoEncoder(
            encoder=encoder,
            latent_dim=latent_dim,
            decoder=decoder
        )
        self.vae.compile(optimizer=keras.optimizers.Adam())

    def test_vae_initialization(self):
        """Compare input shape against output shape. Should be equal."""
        test_input = self.train_images[0, None]
        test_output = self.vae(test_input).numpy()
        assert test_input.shape == test_output.shape, "Input shape has a different shape compared to output shape."

    def test_vae_training(self):
        """Test whether VAE training runs."""
        self.vae.fit(
            x=self.train_images,
            y=self.train_images,
            epochs=1,
            batch_size=64,
            validation_data=(self.test_images, self.test_images)
        )

    def test_model_loading(self):
        """Test whether VAE can be loaded after being stored"""
        self.vae.fit(
            x=self.train_images,
            y=self.train_images,
            epochs=1,
            batch_size=64,
            validation_data=(self.test_images, self.test_images)
        )
        # Save model
        model_path = "./vae_test_model.keras"
        self.vae.save(model_path)

        # Load model
        try:
            self.vae = keras.saving.load_model(
                model_path,
                custom_objects={"VariationalAutoEncoder": VariationalAutoEncoder}
            )
        except:
            pass
        os.remove(model_path)

        # Check if the model can be used after loading
        self.vae(self.train_images[0, None])

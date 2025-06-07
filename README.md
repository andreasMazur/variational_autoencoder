# Variational Autoencoder (VAE) with TensorFlow

This repository provides a simple implementation of a Variational Autoencoder (VAE) using TensorFlow and Keras.


## Install

```bash
  pip install git+https://github.com/andreasMazur/variational_autoencoder.git/@main
```

## Example usage of a VAE

```python
from variational_autoencoder import VariationalAutoEncoder

import keras


# Initialize encoder architecture (should be serializable for saving/loading)
latent_dim = 64
encoder = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(latent_dim, activation="relu")
])

# Initialize decoder architecture (should be serializable for saving/loading)
decoder = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape((28, 28))
])

# Initialize VAE
vae = VariationalAutoEncoder(
    encoder=encoder,
    latent_dim=latent_dim,
    decoder=decoder,
    beta=1.0  # Weight for the KL-divergence loss
)

# Leave the 'loss'-parameter blank, as the VAE loss (negative ELBO) is calculated 
# internally within 'VariationalAutoEncoder.train_step'.
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Load data
(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Fit the VAE (training-parameters are mockup numbers)
vae.fit(
    x=train_images,
    y=train_images,
    epochs=10,
    batch_size=64,
    validation_data=(test_images, test_images)
)
```

## Example usage of a conditional VAE

```python
from variational_autoencoder import CVariationalAutoEncoder

import keras


# Initialize encoder architecture (should be serializable for saving/loading)
latent_dim = 64
encoder = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(latent_dim, activation="relu")
])

# Initialize decoder architecture (should be serializable for saving/loading)
decoder = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape((28, 28))
])

# Initialize classifier for semi-supervised learning
classifier = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="linear", name="classifier_output")
])

# Initialize VAE
vae = CVariationalAutoEncoder(
    encoder=encoder,
    latent_dim=latent_dim,
    decoder=decoder,
    classifier=classifier,
    alpha=0.1,  # Weight for the classifier loss
    beta=1.0  # Weight for the KL-divergence loss
)

# Leave the 'loss'-parameter blank, as the VAE loss is calculated 
# internally within 'VariationalAutoEncoder.train_step'.
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Fit the VAE (training-parameters are mockup numbers)
# Hint: Missing labels should be filled with -1.
# This indicates the VAE to use the classifier for the associated features..
vae.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    batch_size=64,
    validation_data=(test_images, test_labels)
)
```

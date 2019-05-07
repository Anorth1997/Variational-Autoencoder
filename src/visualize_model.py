import numpy as np
from src import data
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from vae import diagonal_gaussian_sampler, bernoulli_sampler, encode, decode, img_flat_size, latent_dim

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()

# Binarize the data
train_images = np.round(train_images[0:10000]).astype(np.float32)
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])

def reshape_into_image(x):
    """
    A helper function to reshape 784 array into a 28x28 image for plotting
    :param x: input data
    :return: reshaped input data
    """
    return x.reshape(-1, 28, 28).astype(np.float64)


# ------------------------- Samples from the generative model ------------------------- #
def samples_from_generative_model():
    """
    Generate samples from our trained decoder, plot an image named samples_from_generative_model.png which the first
    row is the data distribution and the second row is the x_hat under distribution.
    :return: None
    """
    # Sample 10 z from prior
    unit_mu = tf.constant(0, shape=[10, latent_dim], dtype=tf.float32)
    unit_variance = tf.constant(1, shape=[10, latent_dim], dtype=tf.float32)
    z = diagonal_gaussian_sampler(unit_mu, unit_variance, latent_dim, 10)  # 10 x 2 matrix

    # For each z, plot p(x|z)
    p = decode(z, training=False)

    # Sample x from p(x|z)
    x_hat = bernoulli_sampler(p, 10, img_flat_size)


    # Concatenate plots into a figure
    with tf.Session() as session:
        p_plot = reshape_into_image(p.eval())
        x_hat_plot = reshape_into_image(1 - x_hat.eval())

    to_plot = np.concatenate((p_plot, x_hat_plot))
    data.save_images(to_plot, 'samples_from_generative_model.png', ims_per_row=10)


# ------------------------------------ Reconstructions of data ----------------------------_#\
def reconstructions_of_data(x):
    """
    Given the input data x, process it through our vae model and output its reconstruction x_hat. Plot an image
    named reconstructions_of_data.png
    :param x: input data
    :return:
    """

    # encode the input data x
    mu, variance = encode(x, training=False)

    # For each x, sample distribution z ~ q(z|x)
    z = diagonal_gaussian_sampler(mu, variance, latent_dim, 10)  # 10 x latent_dim matrix

    # For each z, decode to distribution p(x̃|z), plot.
    p = decode(z, training=False)
    x_hat = bernoulli_sampler(p, 10, img_flat_size)

    # Concatenate all plots into a figure.
    x_plot = reshape_into_image(x)
    with tf.Session() as session:
        p_plot = reshape_into_image(p.eval())
        x_hat_plot = reshape_into_image(1 - x_hat.eval())

    to_plot = np.concatenate((x_plot, p_plot, x_hat_plot))
    data.save_images(to_plot, 'reconstructions_of_data.png', ims_per_row=10)


# ------------------------ latent space visualization ------------------------- #
def latent_embedding_of_data(x):
    """

    :param x:
    :return:
    """
    # x = train_images[:500].astype(np.float32)

    # Encode the training data
    mu, variance = encode(x, training=False)

    with tf.Session() as session:
        mu_plot = mu.eval()
    color_label = [np.where(r==1)[0][0] for r in train_labels[:500]]
    color_label = np.array([color_label]).T
    mu_plot = np.concatenate((mu_plot, color_label), axis=1)
    plt.scatter(mu_plot[:, 0], mu_plot[:, 1], c=mu_plot[:, 2])
    plt.title('mean vectors in latent space')
    plt.legend()
    plt.show()

def decoding_along_a_lattice():
    # Create a 20x20 equally spaced grid of z's
    grid_x = np.linspace(-5, 6, 20)
    grid_y = np.linspace(-3, 6, 20)
    axis_x = np.repeat(grid_x[np.newaxis, ...], len(grid_y), axis=0)
    axis_y = np.repeat(grid_y[..., np.newaxis], len(grid_x), axis=1)

    latent_coordinates = np.dstack((axis_y, axis_x))
    latent_coordinates = latent_coordinates.reshape(400, 2).astype(np.float32)
    p = decode(latent_coordinates, training=False)
    with tf.Session() as session:
        to_plot = p.eval().reshape(400, 28, 28)
    data.save_images(to_plot, 'decoding_along_lattice.png', ims_per_row=20)


if __name__ == '__main__':
    decoding_along_a_lattice()
    # latent_embedding_of_data(train_images[:500].astype(np.float32))
    # reconstructions_of_data(train_images[:10].astype(np.float32))
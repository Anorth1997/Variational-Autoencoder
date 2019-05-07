import numpy as np
from src import data
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()

# Binarize the data
train_images = np.round(train_images[0:10000]).astype(np.float32)
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])

# ----------------------hyper-parameters----------------------#

latent_dim = 2
num_hidden_units = 500
img_flat_size = 784
batch_size = 10000
# minibatch size M
M = 100
n_epoch = 200
learn_rate = 0.001

# partition training set into mini batches
num_batches = int(train_images.shape[0] / M)
mini_batches = np.reshape(train_images, (num_batches, M, -1))
test_mini_batches = np.reshape(test_images, (num_batches, M, -1))


# ------------------------ Sampler helper functions ------------------------------ #

def diagonal_gaussian_sampler(mu, variance, latent_dim, batch_size):

    gaussian_sampler = variance * np.random.randn(batch_size, latent_dim) + mu
    gaussian_sampler = tf.convert_to_tensor(gaussian_sampler, dtype=tf.float32)
    return gaussian_sampler


def bernoulli_sampler(p, batch_size, dim):
    bernoulli_sampler = tf.dtypes.cast(np.random.rand(batch_size, dim) >= p, dtype=tf.float32)
    bernoulli_sampler = tf.convert_to_tensor(bernoulli_sampler, dtype=tf.float32)
    return bernoulli_sampler


# ------------------------- log pdf computation helper functions --------------- #

def log_pdf_gaussian(z, mu, variance, z_dim):
    """
    Given sampler z, mu, covariance matrix and gaussian dimension, compute the log-pdf of the sampler x from
    Multivariate Gaussian Distribution
    :param z: batch_size x latent_dim matrix
    :param mu: batch_size x latent_dim matrix
    :param variance: batch_size x latent_dim matrix
    :param latent_dim:
    :return:
    """
    det = tf.reduce_prod(variance, axis=-1)
    variance_inverse = tf.reciprocal(variance)  # batch_size x latent_dim matrix

    return -tf.log(tf.pow(2 * math.pi, z_dim / 2) * tf.sqrt(det)) - \
           0.5 * tf.reduce_sum(tf.multiply(tf.square(z - mu), variance_inverse), axis=-1)


def log_pdf_bernoulli(y, p):
    """
    Give sampler y, p, compute the log-pdf of the sampler x from Bernoulli Distribution
    :param y: batch_size x 784
    :param p: batch_size x 784
    :return:
    """

    return tf.reduce_sum(tf.multiply(y, tf.log(p)) + tf.multiply(1 - y, tf.log(1 - p)), axis=-1)



# ------------------------- training parameters -------------------------- #


# Gaussian MLP encoder
# weights and biases initializers
init_w = tf.contrib.layers.variance_scaling_initializer()
init_b = tf.constant_initializer(value=0.)
# weights and biases for h
w_3 = tf.get_variable('w3', [img_flat_size, num_hidden_units], initializer=init_w)
b_3 = tf.get_variable('b3', [num_hidden_units], initializer=init_b)
# weights and biases for mu
w_4 = tf.get_variable('w4', [num_hidden_units, latent_dim], initializer=init_w)
b_4 = tf.get_variable('b4', [latent_dim], initializer=init_b)
# weights and biases for variance
w_5 = tf.get_variable('w5', [num_hidden_units, latent_dim], initializer=init_w)
b_5 = tf.get_variable('b5', [latent_dim], initializer=init_b)


# Bernoulli MLP as decoder
# hidden layer
w_1 = tf.get_variable('w1', [latent_dim, num_hidden_units], initializer=init_w)  # 2 x 500 matrix
b_1 = tf.get_variable('b1', [num_hidden_units], initializer=init_b)
# output layer
w_2 = tf.get_variable('w2', [num_hidden_units, img_flat_size], initializer=init_w)  # n_hidden x 784 matrix
b_2 = tf.get_variable('b2', [img_flat_size], initializer=init_b)

# --------------------------- Encoder ------------------------------- #

def encode(x, training=True):
    """
    Given the input data x, encode it to parameters of hidden representation z distribution
    :param x: input data
    :param training: define if the encoder is under training or testing. Load the trained parameters if not
    :return: mu and variance
    """
    if training:
        global w_3, w_4, w_5, b_3, b_4, b_5
    else:
        # load the trained paramters
        w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
        w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
        w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
        b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
        b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
        b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)

    # Compute mu and variance which are the parameters for q_theta(z|x)
    h = tf.matmul(x, w_3) + b_3
    h = tf.nn.tanh(h)  # 10000 x n_hidden matrix
    mu = tf.matmul(h, w_4) + b_4  # 10000 x latent_dim matrix
    log_variance = tf.matmul(h, w_5) + b_5  # 10000 x latent_dim matrix
    variance = tf.exp(log_variance)

    return mu, variance


def decode(z, training=True):
    """
    Given the input hidden representation z, decode it to the probability distribution of data
    :param z: hidden representation
    :param training: define if the decoder is under training or testing. Load the trained parameters if not
    :return: probability distribution of data
    """

    if training:
        global w_1, w_2, b_1, b_2
    else:
        # load the trained paramters
        w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
        w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
        b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
        b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)

    # hidden layer
    hidden = tf.matmul(z, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # batch_size x n_hidden matrix

    # output layer
    p = tf.matmul(hidden, w_2) + b_2  # batch_size x 784 matrix
    p = tf.nn.sigmoid(p)
    p = tf.clip_by_value(p, 1e-8, 1-1e-8)

    return p

def variational_objective(x, training=True):
    """
    This is the loss function of the variational autoencoder
    :param x: input data x
    :param training: define if the model is under training or testing. Load the trained parameters if not
    :return: loss
    """

    mu, variance = encode(x, training)

    # Samples z ~ q(z|x)
    z = diagonal_gaussian_sampler(mu, variance, latent_dim, M)  # M x latent_dim matrix

    p = decode(z, training)

    # log_q(z|x) log probability of z under approximate posterior N(μ,σ^2)
    encoder_log_prob = log_pdf_gaussian(z, mu, variance, latent_dim)  # batch_size x latent_dim

    # log_p_z(z) log probability of z under prior
    unit_mu = tf.constant(0, shape=[M, latent_dim], dtype=tf.float32)
    unit_variance = tf.constant(1, shape=[M, latent_dim], dtype=tf.float32)
    prior_log_prob = log_pdf_gaussian(z, unit_mu, unit_variance, latent_dim)  # batch_size x latent_dim

    # log_p(x|z) - conditional probability of data given latents.
    decoder_log_prob = log_pdf_bernoulli(x, p)  # batch_size x 1

    # Monte Carlo Estimator of mean ELBO with Reparameterization over M minibatch samples.
    # return tf.reduce_mean(encoder_log_prob + tf.multiply(prior_log_prob, decoder_log_prob))

    elbo = tf.reduce_mean(-encoder_log_prob + prior_log_prob + decoder_log_prob)

    return -elbo

# ------------------------------ train -------------------------------------#

def train():

    x = tf.placeholder(tf.float32, shape=(None, img_flat_size))

    loss = variational_objective(x)

    # Set up ADAM optimizer
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # ---------------------------training--------------------------#
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):

            l = session.run(variational_objective(x), feed_dict={x: mini_batches[0]})
            t_l = session.run(variational_objective(x), feed_dict={x: test_mini_batches[0]})

            print('epoch:', epoch, 'training elbo:', -l, 'test elbo', -t_l)

            for i in range(num_batches):
                session.run(train_op, feed_dict={x: mini_batches[i]})
                # if i == num_batches - 1:
                #     l = session.run(variational_objective(x), feed_dict={x: mini_batches[i]})
                #     print('epoch:', epoch, 'training loss:', l)

        # save the trained parameters
        np.save('w1', w_1.eval())
        np.save('w2', w_2.eval())
        np.save('w3', w_3.eval())
        np.save('w4', w_4.eval())
        np.save('w5', w_5.eval())
        np.save('b1', b_1.eval())
        np.save('b2', b_2.eval())
        np.save('b3', b_3.eval())
        np.save('b4', b_4.eval())
        np.save('b5', b_5.eval())

if __name__ == '__main__':
    train()


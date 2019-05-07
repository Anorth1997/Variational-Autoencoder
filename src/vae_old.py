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



# ------------------------------------------------- #
#                   Distributions                   #
# ------------------------------------------------- #
def diagonal_gaussian_sampler(mu, variance, latent_dim, batch_size):

    gaussian_sampler = variance * np.random.randn(batch_size, latent_dim) + mu
    gaussian_sampler = tf.convert_to_tensor(gaussian_sampler, dtype=tf.float32)
    return gaussian_sampler


def bernoulli_sampler(p, batch_size, dim):
    bernoulli_sampler = tf.dtypes.cast(np.random.rand(batch_size, dim) >= p, dtype=tf.float32)
    bernoulli_sampler = tf.convert_to_tensor(bernoulli_sampler, dtype=tf.float32)
    return bernoulli_sampler


def log_pdf_gaussian(z, mu, variance, latent_dim):
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

    return -tf.log(tf.pow(2 * math.pi, latent_dim / 2) * tf.sqrt(det)) - \
           0.5 * tf.reduce_sum(tf.multiply(tf.square(z - mu), variance_inverse), axis=-1)


def log_pdf_bernoulli(y, p):
    """
    Give sampler y, p, compute the log-pdf of the sampler x from Bernoulli Distribution
    :param y: batch_size x 784
    :param p: batch_size x 784
    :return:
    """

    return tf.reduce_sum(tf.multiply(y, tf.log(p)) + tf.multiply(1 - y, tf.log(1 - p)), axis=-1)


# ----------------------------------------------------------- #
#               Defining Model Architecture                   #
# ----------------------------------------------------------- #

# Tensorflow implementation of the model

# Gaussian MLP encoder
# weights and biases initializers
init_w = tf.contrib.layers.variance_scaling_initializer()

# init_w = tf.initializers.truncated_normal(stddev=0.01)
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


# ----------------------------------------------------------- #
#                    Variational Objective                    #
# ----------------------------------------------------------- #


def variational_objective(x):

    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)

    # encoding
    # Provides parameters for q(z|x)
    h = tf.matmul(x, w_3) + b_3
    h = tf.nn.tanh(h)  # 10000 x n_hidden matrix
    mu = tf.matmul(h, w_4) + b_4  # 10000 x latent_dim matrix
    log_variance = tf.matmul(h, w_5) + b_5  # 10000 x latent_dim matrix
    variance = tf.exp(log_variance)

    # Define sample from recognition model
    # Samples z ~ q(z|x)
    z = diagonal_gaussian_sampler(mu, variance, latent_dim, M)  # M x latent_dim matrix

    # decoding
    # Provides parameters for distribution p(x|z)
    # hidden layer
    hidden = tf.matmul(z, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # batch_size x n_hidden matrix

    # output layer
    p = tf.matmul(hidden, w_2) + b_2  # batch_size x 784 matrix
    p = tf.nn.sigmoid(p)
    p = tf.clip_by_value(p, 1e-8, 1-1e-8)

    # sample x_hat ~ p(x|z)
    # x_hat = bernoulli_sampler(p, M, img_flat_size)

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


# ----------------------------------------------------------------- #
#            Numerical Integration over Latent Space                #
# ----------------------------------------------------------------- #
def compute_integral(integrand_sequences, delta_z):
    """
    Given the a sequences of f(z_i) and delta_z, compute the integral
    with Riemann sum approximation
    :param integrand_sequences: array of f(z_i)s
    :param delta_z:
    :return: integral
    """
    # Implement log sum exp
    # Implement stable numerical integration
    # over a 2d grid of equally spaced (delta_z) evaluations logf(x)
    delta_area = delta_z ** 2
    return tf.reduce_logsumexp(integrand_sequences + tf.log(delta_area))


# ----------------------------------------------------------------- #
#                    Numerical Log-Likelihood                       #
# ----------------------------------------------------------------- #
def compute_true_log_p_x(x, z_x_axis_start, z_x_axis_end, z_y_axis_start, z_y_axis_end, top_half):
    """
    """
    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)

    # Define the delta_z to be the spacing of the grid  (I used delta_z = 0.1)
    delta_z = 0.1
    z_x_axis_num_seg = int((z_x_axis_end - z_x_axis_start) / delta_z)
    z_y_axis_num_seg = int((z_y_axis_end - z_y_axis_start) / delta_z)
    num_grids = z_x_axis_num_seg * z_y_axis_num_seg

    # Define a grid of delta_z spaced points [-4,4]x[-4,4]
    # This is right Riemann sum that's why start with -3.9
    grid_x = np.linspace(z_x_axis_start + delta_z, z_x_axis_end, z_x_axis_num_seg).astype(np.float32)
    grid_y = np.linspace(z_y_axis_start + delta_z, z_y_axis_end, z_y_axis_num_seg).astype(np.float32)

    xx, yy = np.meshgrid(grid_x, grid_y)  # 2 default 80 x 80 matrices

    xx = xx.reshape(z_y_axis_num_seg, z_x_axis_num_seg, 1)
    yy = yy.reshape(z_y_axis_num_seg, z_x_axis_num_seg, 1)

    z_samples = np.concatenate((xx, yy), axis=2).reshape(num_grids, 2) # 6400 x 2 matrix

    # Sample an x from the data to evaluate the likelihood
    # x = np.array([train_images[0]]) # 1 x 784 matrix

    unit_mu = tf.constant(0, shape=[num_grids, latent_dim], dtype=tf.float32)
    unit_variance = tf.constant(1, shape=[num_grids, latent_dim], dtype=tf.float32)
    # Compute log_p(x|z)+log_p(z) for every point on the grid

    hidden = tf.matmul(z_samples, w_1) + b_1  # 6400 x 500 matrix
    hidden = tf.nn.tanh(hidden)  # 6400 x 500 matrix
    p = tf.matmul(hidden, w_2) + b_2  # 6400 x 784 matrix
    p = tf.nn.sigmoid(p) # 6400 x 784 matrix
    p = tf.clip_by_value(p, 1e-8, 1-1e-8)  # 6400 x 784 matrix

    if top_half:
        p = p[:, :392]
    # 6400 x 1 matrix

    log_p_x_give_z = log_pdf_gaussian(z_samples, unit_mu, unit_variance, latent_dim)

    log_p_z = log_pdf_bernoulli(np.repeat(x, num_grids, axis=0), p)
    integrands = log_p_x_give_z + log_p_z

    # Using your numerical integration code
    # integrate log_p(x|z)+log_p(z) over z to find log_p(x)
    integrands = tf.convert_to_tensor(integrands)
    log_p_x = compute_integral(integrands, delta_z)

    # Check that your numerical integration is correct
    # by integrating log_p(x|z)+log_p(z) - log_p(x)
    # If you've successfully normalized this should integrate to 0 = log 1
    check = compute_integral(integrands, delta_z) - log_p_x
    #
    # Now compute the ELBO on x
    if not top_half:
        elbo = -variational_objective(x)
    #
    with tf.Session() as sess:
    #     print(check.eval())
    #
    #     # Now compute the ELBO on x
        print('log_p_x', log_p_x.eval())
        if not top_half:
            print('elbo', elbo.eval())

    return log_p_x, integrands, xx, yy, z_samples

# ----------------------------------------------------------------- #
#                  Data Space Visualizations                        #
# ----------------------------------------------------------------- #

# Write a function to reshape 784 array into a 28x28 image for plotting
def reshape_into_image(x):
    return x.reshape(-1, 28, 28).astype(np.float64)


# ----------------------------------------------------------------- #
#              Samples from the generative model                    #
# ----------------------------------------------------------------- #
def sampler_from_generative_model():
    # load trained parameters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)

    # Sample 10 z from prior
    unit_mu = tf.constant(0, shape=[10, latent_dim], dtype=tf.float32)
    unit_variance = tf.constant(1, shape=[10, latent_dim], dtype=tf.float32)
    z = diagonal_gaussian_sampler(unit_mu, unit_variance, latent_dim, 10)  # 10 x 2 matrix

    # For each z, plot p(x|z)
    # decoding
    # Provides parameters for distribution p(x|z)
    # hidden layer
    hidden = tf.matmul(z, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # 10 x n_hidden matrix

    # output layer
    p = tf.matmul(hidden, w_2) + b_2  # 10 x 784 matrix
    p = tf.nn.sigmoid(p)
    p = tf.clip_by_value(p, 1e-8, 1-1e-8)

    # Sample x from p(x|z)
    x_hat = bernoulli_sampler(p, 10, img_flat_size)

    # Concatenate plots into a figure
    with tf.Session() as session:
        p_plot = reshape_into_image(p.eval())
        x_hat_plot = reshape_into_image(1 - x_hat.eval())

    to_plot = np.concatenate((p_plot, x_hat_plot))
    data.save_images(to_plot, 'samples_from_generative_model.png', ims_per_row=10)


# ----------------------------------------------------------------- #
#                  Reconstructions of data                          #
# ----------------------------------------------------------------- #
def reconstructions_of_data():
    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)

    # Sample 10 xs from the data, plot.
    x = train_images[:10].astype(np.float32)  # 10 x 784 matrix

    # For each x, encode to distribution q(z|x)
    h = tf.matmul(x, w_3) + b_3
    h = tf.nn.tanh(h)  # 100 x 500 matrix
    mu = tf.matmul(h, w_4) + b_4  # 10000 x latent_dim matrix
    log_variance = tf.matmul(h, w_5) + b_5  # 10000 x latent_dim matrix
    variance = tf.exp(log_variance)

    # For each x, sample distribution z ~ q(z|x)
    z = diagonal_gaussian_sampler(mu, variance, latent_dim, 10)  # 10 x latent_dim matrix

    # For each z, decode to distribution p(x̃|z), plot.
    hidden = tf.matmul(z, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # batch_size x n_hidden matrix
    p = tf.matmul(hidden, w_2) + b_2  # batch_size x 784 matrix
    p = tf.nn.sigmoid(p)
    p = tf.clip_by_value(p, 1e-8, 1-1e-8)

    # For each x, sample from the distribution x̃ ~ p(x̃|z), plot.
    x_hat = bernoulli_sampler(p, 10, img_flat_size)

    # Concatenate all plots into a figure.
    x_plot = reshape_into_image(x)
    with tf.Session() as session:
        p_plot = reshape_into_image(p.eval())
        x_hat_plot = reshape_into_image(1 - x_hat.eval())

    to_plot = np.concatenate((x_plot, p_plot, x_hat_plot))
    data.save_images(to_plot, 'reconstructions_of_data.png', ims_per_row=10)

# ----------------------------------------------------------------- #
#                  Latent Space Visualizations                      #
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
#                  Latent embedding of data                         #
# ----------------------------------------------------------------- #
def latent_embedding_of_data():
    x = train_images[:500].astype(np.float32)

    # load the trained paramters
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)

    # Encode the training data
    h = tf.matmul(x, w_3) + b_3
    h = tf.nn.tanh(h)  # 100 x 500 matrix
    # Take the mean vector of each encoding
    mu = tf.matmul(h, w_4) + b_4  # 10000 x latent_dim matrix
    # log_variance = tf.matmul(h, w_5) + b_5  # 10000 x latent_dim matrix
    # variance = tf.exp(log_variance)
    with tf.Session() as session:
        mu_plot = mu.eval()
    color_label = [np.where(r==1)[0][0] for r in train_labels[:500]]
    color_label = np.array([color_label]).T
    mu_plot = np.concatenate((mu_plot, color_label), axis=1)
    plt.scatter(mu_plot[:, 0], mu_plot[:, 1], c=mu_plot[:, 2])
    plt.title('mean vectors in latent space')
    plt.legend()
    plt.show()

# ----------------------------------------------------------------- #
#                   Decoding along a lattice                        #
# ----------------------------------------------------------------- #
def decoding_along_a_lattice():
    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)


    # Create a 20x20 equally spaced grid of z's
    grid_x = np.linspace(-6, 2.5, 20)
    grid_y = np.linspace(4.8, -6.5, 20)

    to_plot = np.zeros((20, 20, 28, 28))
    with tf.Session() as session:
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = np.array([xi, yi]).reshape(1, latent_dim).astype(np.float32)
                # For each z on the grid plot the generative distribution over x
                hidden = tf.matmul(z_sample, w_1) + b_1
                hidden = tf.nn.tanh(hidden)  # batch_size x n_hidden matrix
                p = tf.matmul(hidden, w_2) + b_2  # batch_size x 784 matrix
                p = tf.nn.sigmoid(p)
                p = tf.clip_by_value(p, 1e-8, 1 - 1e-8)
                # concatenate these plots to a lattice of distributions

                to_plot[j, i] = reshape_into_image(p.eval())

    to_plot = to_plot.reshape(400, 28, 28)
    data.save_images(to_plot, 'decoding_along_lattice.png', ims_per_row=20)


# ----------------------------------------------------------------- #
#               Interpolate between two classes                     #
# ----------------------------------------------------------------- #
# Function which gives linear interpolation z_α between za and zb
def linear_interpolation(za, zb):
    """

    :param za: 1 x 2 matrix
    :param zb: 1 x 2 matrix
    :return: 10 x 2 matrix linear_interpolation
    """
    return np.linspace(za, zb, 10).reshape(10, latent_dim)


def interpolation_plot():
    # load the trained paramters
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)

    # Sample 3 pairs of data with different classes
    pair_1 = train_images[0:2].astype(np.float32)  # 5 and 0
    pair_2 = train_images[2:4].astype(np.float32)  # 4 and 1
    pair_3 = train_images[4:6].astype(np.float32)  # 9 and 2

    # Encode the data in each pair, and take the mean vectors
    h = tf.matmul(pair_1, w_3) + b_3
    h = tf.nn.tanh(h)  # 100 x 500 matrix
    mu_pair_1 = tf.matmul(h, w_4) + b_4  # 2 x latent_dim matrix

    h = tf.matmul(pair_2, w_3) + b_3
    h = tf.nn.tanh(h)  # 100 x 500 matrix
    mu_pair_2 = tf.matmul(h, w_4) + b_4  # 2 x latent_dim matrix

    h = tf.matmul(pair_3, w_3) + b_3
    h = tf.nn.tanh(h)  # 100 x 500 matrix
    mu_pair_3 = tf.matmul(h, w_4) + b_4  # 2 x latent_dim matrix

    with tf.Session() as session:
        mu_pair_1 = mu_pair_1.eval()
        mu_pair_2 = mu_pair_2.eval()
        mu_pair_3 = mu_pair_3.eval()

    # Linearly interpolate between these mean vectors
    pair_1_inter = linear_interpolation(mu_pair_1[0], mu_pair_1[1])  # 10 x 2 matrix
    pair_2_inter = linear_interpolation(mu_pair_2[0], mu_pair_2[1])  # 10 x 2 matrix
    pair_3_inter = linear_interpolation(mu_pair_3[0], mu_pair_3[1])  # 10 x 2 matrix

    # Along the interpolation, plot the distributions p(x|z_α)
    hidden = tf.matmul(pair_1_inter, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # 10 x n_hidden matrix
    p_1 = tf.matmul(hidden, w_2) + b_2  # 10 x 784 matrix
    p_1 = tf.nn.sigmoid(p_1)
    p_1 = tf.clip_by_value(p_1, 1e-8, 1-1e-8)

    hidden = tf.matmul(pair_2_inter, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # 10 x n_hidden matrix
    p_2 = tf.matmul(hidden, w_2) + b_2  # 10 x 784 matrix
    p_2 = tf.nn.sigmoid(p_2)
    p_2 = tf.clip_by_value(p_2, 1e-8, 1-1e-8)

    hidden = tf.matmul(pair_3_inter, w_1) + b_1
    hidden = tf.nn.tanh(hidden)  # 10 x n_hidden matrix
    p_3 = tf.matmul(hidden, w_2) + b_2  # 10 x 784 matrix
    p_3 = tf.nn.sigmoid(p_3)
    p_3 = tf.clip_by_value(p_3, 1e-8, 1-1e-8)

    with tf.Session() as session:
        p1_plot = p_1.eval()
        p2_plot = p_2.eval()
        p3_plot = p_3.eval()

    to_plot = np.concatenate((p1_plot, p2_plot, p3_plot))
    data.save_images(to_plot, 'interpolation.png', ims_per_row=10)


# ----------------------------------------------------------------- #
#         Posteriors and Stochastic Variational Inference           #
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
#                    Plotting Posteriors                            #
# ----------------------------------------------------------------- #
def plot_posteriors():
    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)

    # Sample an element x from the dataset to plot posteriors for
    x = np.array([train_images[0]]) # 1 x 784 matrix

    # Define a grid of equally spaced points in z
    # The grid needs to be fine enough that the plot is nice
    # To keep the integration tractable
    # I reccomend centering your grid at the mean of q(z|x)
    # This is my latent space, [-6, 2.5] x [-6.5, 4.8]
    # Evaluate log_p(x|z) + log_p(z) for every z on the grid
    # Numerically integrate log_p(x|z) + log_p(z) to get log_p(x)
    z_x_axis_start = -6
    z_x_axis_end = 2.5
    z_y_axis_start = -6.5
    z_y_axis_end = 4.8
    true_log_p_x, unnormalized_log_p_z_given_x, xx, yy, z_grids = compute_true_log_p_x(x, z_x_axis_start, z_x_axis_end, z_y_axis_start, z_y_axis_end, False)

    # Produce a grid of normalized log_p(z|x)
    normalized_log_p_z_given_x = unnormalized_log_p_z_given_x - true_log_p_x
    normalized_p_z_given_x = tf.exp(normalized_log_p_z_given_x)

    xx = np.squeeze(xx)
    yy = np.squeeze(yy)
    with tf.Session() as sess:
        normalized_p_z_given_x = normalized_p_z_given_x.eval()
        normalized_p_z_given_x = normalized_p_z_given_x.reshape(xx.shape)
    # Plot the contours of p(z|x) (note, not log)
    # plt.contour(xx, yy, normalized_p_z_given_x.tolist(), cmap='jet')
    # plt.colorbar()
    #
    # plt.show()

    # Evaluate log_q(z|x) recognition network for every z on grid
    # encoding x -> mu, variance
    # Provides parameters for q(z|x)
    h = tf.matmul(x, w_3) + b_3
    h = tf.nn.tanh(h)
    mu = tf.matmul(h, w_4) + b_4
    log_variance = tf.matmul(h, w_5) + b_5
    variance = tf.exp(log_variance)

    log_q_z_given_x = log_pdf_gaussian(z_grids, mu, variance, latent_dim)
    q_z_given_x = tf.exp(log_q_z_given_x)
    with tf.Session() as sess:
        q_z_given_x = q_z_given_x.eval()
        q_z_given_x = q_z_given_x.reshape(xx.shape)
    # Plot the contours of p(z|x) (note, not log)
    # Plot the contours of q(z|x) on previous plot
    plt.contour(xx, yy, normalized_p_z_given_x.tolist(), cmap='jet')
    plt.contour(xx, yy, q_z_given_x.tolist(), cmap='jet')
    plt.colorbar()
    plt.show()

# ----------------------------------------------------------------- #
#                 True posterior for top of digit                   #
# ----------------------------------------------------------------- #
# Function which returns only the top half of a 28x28 array

def true_posterior_for_top_half():
    # load the trained paramters
    w_1 = tf.convert_to_tensor(np.load('w1.npy'), dtype=tf.float32)
    w_2 = tf.convert_to_tensor(np.load('w2.npy'), dtype=tf.float32)
    w_3 = tf.convert_to_tensor(np.load('w3.npy'), dtype=tf.float32)
    w_4 = tf.convert_to_tensor(np.load('w4.npy'), dtype=tf.float32)
    w_5 = tf.convert_to_tensor(np.load('w5.npy'), dtype=tf.float32)
    b_1 = tf.convert_to_tensor(np.load('b1.npy'), dtype=tf.float32)
    b_2 = tf.convert_to_tensor(np.load('b2.npy'), dtype=tf.float32)
    b_3 = tf.convert_to_tensor(np.load('b3.npy'), dtype=tf.float32)
    b_4 = tf.convert_to_tensor(np.load('b4.npy'), dtype=tf.float32)
    b_5 = tf.convert_to_tensor(np.load('b5.npy'), dtype=tf.float32)
    # log_p(x_top | z) (hint: select top half of 28x28 bernoulli param array)


    # Sample an element from the data set and take only its top half: x_top
    x = np.array([train_images[0][:392]]) # 1 x 392 matrix
    # Define a grid of equally spaced points in z
    z_x_axis_start = -6
    z_x_axis_end = 2.5
    z_y_axis_start = -6.5
    z_y_axis_end = 4.8
    # Evaluate log_p(x_top | z) + log_p(z) for every z on grid
    true_log_p_x_top, unnormalized_log_p_z_given_x, xx, yy, z_grids = compute_true_log_p_x(x, z_x_axis_start, z_x_axis_end, z_y_axis_start, z_y_axis_end, True)

    # Numerically integrate to get log_p(x_top)

    # Normalize to produce grid of log_p(z|x_top)
    normalized_log_p_z_given_x_top = unnormalized_log_p_z_given_x - true_log_p_x_top
    normalized_p_z_given_x_top = tf.exp(normalized_log_p_z_given_x_top)

    xx = np.squeeze(xx)
    yy = np.squeeze(yy)
    with tf.Session() as sess:
        normalized_p_z_given_x_top = normalized_p_z_given_x_top.eval()
        normalized_p_z_given_x_top = normalized_p_z_given_x_top.reshape(xx.shape)
    # Plot the contours of p(z|x_top)

    plt.contour(xx, yy, normalized_p_z_given_x_top.tolist(), cmap='jet')
    plt.colorbar()
    plt.show()



def train():
    # ----------------------build graph--------------------------#

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
    # train()
    # compute_true_log_p_x(np.array([train_images[2]]), -4, 4, -4, 4, False)
    plot_posteriors()
    # true_posterior_for_top_half()
    # sampler_from_generative_model()
    # reconstructions_of_data()
    # # ----------------------build graph--------------------------#
    #
    # x = tf.placeholder(tf.float32, shape=(None, img_flat_size))
    #
    # loss = variational_objective(x)
    #
    # # Set up ADAM optimizer
    # train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    #
    # # ---------------------------training--------------------------#
    # with tf.Session() as session:
    #
    #     session.run(tf.global_variables_initializer())
    #
    #     for epoch in range(n_epoch):
    #
    #         for i in range(num_batches):
    #             session.run(train_op, feed_dict={x: mini_batches[i]})
    #             if i == num_batches - 1:
    #                 l = session.run(variational_objective(x), feed_dict={x: mini_batches[i]})
    #                 print('epoch:', epoch, 'training loss:', l)
    #
    #     # save the trained parameters
    #     np.save('w1', w_1.eval())
    #     np.save('w2', w_2.eval())
    #     np.save('w3', w_3.eval())
    #     np.save('w4', w_4.eval())
    #     np.save('w5', w_5.eval())
    #     np.save('b1', b_1.eval())
    #     np.save('b2', b_2.eval())
    #     np.save('b3', b_3.eval())
    #     np.save('b4', b_4.eval())
    #     np.save('b5', b_5.eval())






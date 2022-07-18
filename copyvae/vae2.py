#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Softmax, BatchNormalization
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.errors import InvalidArgumentError
from tqdm import tqdm


def lognormal_ll(x, x_p, eps=1e-8):

    mean = x_p[0]
    std = x_p[1]

    tf.debugging.assert_positive(std)

    pi = tf.constant(np.pi)
    inner = std**2 * pi * 2
    left = - 0.5 * tf.math.log(inner + eps)
    mid = - tf.math.log(x + eps) 
    upper = tf.math.log(x + eps) - mean
    lower = std**2 * 2
    right = -  (upper ** 2) / lower
    ll = left + mid + right

    return tf.math.reduce_sum(ll, axis=-1)


def validate_params(mu, theta):

    try:
        tf.debugging.assert_non_negative(mu)
    except InvalidArgumentError:
        print("Invalid mu")
        raise
    try:
        tf.debugging.assert_non_negative(theta)
    except InvalidArgumentError:
        print("Invalid theta")
        raise
    return True


def nb_pos(y_true, y_pred, eps=1e-8):
    """ Negative binomial reconstruction loss

    Args:
        y_true: true values
        y_pred: predicted values
        eps: numerical stability constant
    Parameters:
        x: Data
        mu: mean of the negative binomial (positive (batch x vars)
        theta: inverse dispersion parameter (positive) (batch x vars)
    Returns:
        loss (log likelihood scalar)
    """
    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]

    arg_validated = validate_params(mu, theta)
    if not arg_validated:
        print("invalid arguments for negative binomial!")
        return None

    log_theta_mu_eps = tf.math.log(theta + mu + eps)

    res = (
        theta * (tf.math.log(theta + eps) - log_theta_mu_eps)
        + x * (tf.math.log(mu + eps) - log_theta_mu_eps)
        + tf.math.lgamma(x + theta)
        - tf.math.lgamma(theta)
        - tf.math.lgamma(x + 1)
    )

    return tf.math.reduce_sum(res, axis=-1)


def zinb_pos(y_true, y_pred, eps=1e-8):
    """ Zero-inflated negative binomial reconstruction loss

    Args:
        y_true: true values
        y_pred: predicted values
        eps: numerical stability constant
    Parameters:
        x: Data
        mu: mean of the negative binomial (positive (batch x vars)
        theta: inverse dispersion parameter (positive) (batch x vars)
        pi: logit of the dropout parameter (real number) (batch x vars)
        #### π in [0,1] ####
        pi = log(π/(1-π)) = log π - log(1-π)
    Returns:
        loss (log likelihood scalar)
    """

    x = y_true
    mu = y_pred[0]
    theta = y_pred[1]
    pi = y_pred[2]

    arg_validated = validate_params(mu, theta)
    if not arg_validated:
        print("invalid arguments for zinb!")
        return None

    softplus_pi = tf.math.softplus(-pi)
    log_theta_eps = tf.math.log(theta + eps)
    log_theta_mu_eps = tf.math.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = tf.math.softplus(pi_theta_log) - softplus_pi
    mask1 = tf.cast(tf.math.less(x, eps), tf.float32)
    mul_case_zero = tf.math.multiply(mask1, case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (tf.math.log(mu + eps) - log_theta_mu_eps)
        + tf.math.lgamma(x + theta)
        - tf.math.lgamma(theta)
        - tf.math.lgamma(x + 1)
    )
    mask2 = tf.cast(tf.math.greater(x, eps), tf.float32)
    mul_case_non_zero = tf.math.multiply(mask2, case_non_zero)
    res = mul_case_zero + mul_case_non_zero

    return tf.math.reduce_sum(res, axis=-1)


def poisson_prior(batch_dim, genes_dim, max_cp=6, lam=2):
    """ poisson-like categorical distribution

    Args:
        batch_dim: number of example in minibatch
        genes_dim: number of genes
        max_cp: maximum copy number
        lam: poisson mean
    Returns:
        cat_dis: categorical distribution object
    """

    poi = tfp.distributions.Poisson(lam)
    poi_prob = poi.prob(np.arange(max_cp + 1))
    cat_prob = poi_prob / np.sum(poi_prob)
    a = tf.expand_dims(cat_prob, axis=0)
    b = tf.expand_dims(a, axis=0)
    c = tf.repeat(b, repeats=genes_dim, axis=1)
    d = tf.cast(tf.repeat(c, repeats=batch_dim, axis=0), tf.float32)
    cat_dis = tfp.distributions.Categorical(probs=d)

    return cat_dis


def dirichlet_prior(batch, genes):
    """ categorical distribution from a dirichlet prior

    Args:
        batch: number of example in minibatch
        genes: number of genes
    Returns:
        cat_dis: categorical distribution object
    """

    alpha = [1, 5, 10, 5, 1, 1, 1]
    dist = tfp.distributions.Dirichlet(alpha)
    d = dist.sample([batch, genes])
    cat_dis = tfp.distributions.Categorical(probs=d)
    return cat_dis


class FullyConnLayer(keras.layers.Layer):

    def __init__(self,
                 num_outputs,
                 STD=0.01,
                 keep_prob=None,
                 activation=None,
                 bn=False):
        super(FullyConnLayer, self).__init__()
        self.drop = keep_prob
        self.act = activation
        self.bn_on = bn
        self.fc = Dense(num_outputs,
                        kernel_initializer=TruncatedNormal(stddev=STD))
        self.bn = BatchNormalization(momentum=0.01, epsilon=0.001)
        if self.drop:
            self.dropout = Dropout(self.drop)

    def call(self, inputs):
        x = self.fc(inputs)
        if self.bn_on:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        if self.drop:
            x = self.dropout(x)
        return x


class ScaleLayer(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(ScaleLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight('weight',
                                 shape=input_shape[1:],
                                 initializer='random_normal',
                                 trainable=True)
        #self.b = self.add_weight('bias',
        #                         shape=input_shape[1:],
        #                         initializer='zeros',
        #                         trainable=True)

        self.act = Activation('sigmoid')

    def call(self, x):
        #w = self.w
        #x = tf.math.multiply(x, w) #+ self.b
        w = tf.math.exp(self.w)
        x = tf.math.multiply(self.act(x), w)
        return x
        out = self.act(x)
        return out


class BatchCorrectionLayer(keras.layers.Layer):

    def __init__(self, batch=2):
        super(ScaleLayer, self).__init__()
        self.n_batch = batch

    def build(self, input_shape):
        self.w = self.add_weight('weight',
                                 shape=(input_shape[1:], self.n_batch),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, x):
        ### self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        ### px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        ## x is batch index
        mask = tf.one_hot(x, self.n_batch)
        y = tf.matmul(mask, self.w)
        return y

class GaussianSampling(keras.layers.Layer):
    """ Gaussian sampling """

    def call(self, inputs):
        z_mean, z_var = inputs
        sample = tf.random.normal(tf.shape(z_mean),
                                  z_mean, tf.math.sqrt(z_var))
        return sample


class GumbelSoftmaxSampling(keras.layers.Layer):
    """ Reparameterize categorical distribution """

    def __init__(self,
                bin_size,
                unified_bins=False,
                temp=0.1,
                eps=1e-20) -> None:
        super().__init__()
        self.bin_size = bin_size
        self.unified_bins = unified_bins
        self.temp = temp
        self.eps = eps

    def call(self, inputs):

        # reshape the dimensions (batch x gene x copies)
        rho = tf.stack(inputs, axis=1)
        if self.unified_bins:
            gene_rho = rho
        else:
            gene_rho = tf.repeat(rho, repeats=self.bin_size, axis=-1)
        pi = tf.transpose(gene_rho, [0, 2, 1])

        # sample from Gumbel(0, 1)
        u = tf.random.uniform(tf.shape(pi), minval=0, maxval=1)
        # Gumbel-Softmax
        g = - tf.math.log(- tf.math.log(u + self.eps) + self.eps)
        z = (tf.math.log(pi + self.eps) + g) / self.temp
        y = tf.nn.softmax(z, axis=-1)

        # one-hot map using argmax, but differentiate w.r.t. soft sample y
        y_hard = tf.cast(
            tf.equal(y,
                     tf.math.reduce_max(y, axis=-1, keepdims=True)
                     ),
            tf.float32)
        y = tf.stop_gradient(y_hard - y) + y

        # constract copy number matrix
        bat = tf.shape(pi)[0]
        gene = tf.shape(pi)[1]
        copy = tf.shape(pi)[2]
        a = tf.reshape(tf.range(copy), (-1, copy))
        b = tf.tile(a, (gene, 1))
        c = tf.reshape(b, (-1, gene, copy))
        cmat = tf.cast(tf.tile(c, (bat, 1, 1)), tf.float32)

        # copy number map
        y = tf.math.multiply(y, cmat)
        sample = tf.math.reduce_sum(y, axis=-1)

        if self.unified_bins:
            pi = tf.repeat(pi, repeats=self.bin_size, axis=1)
            sample = tf.repeat(sample, repeats=self.bin_size, axis=1)

        return sample, pi


class Encoder(keras.models.Model):
    """ SCVI encoder """

    def __init__(self, latent_dim=10, intermediate_dim=128, n_layer=2,
                 name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.eps = 1e-4
        self.n_layer = n_layer

        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True,
                    keep_prob=.1))
        self.dense_mean = Dense(latent_dim)
        self.dense_var = Dense(latent_dim)
        self.sampling = GaussianSampling()

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        z_mean = self.dense_mean(x)
        z_var = tf.math.exp(self.dense_var(x)) + self.eps
        z = self.sampling((z_mean, z_var))

        return z_mean, z_var, z


class Decoder(keras.models.Model):
    """ SCVI decoder """

    def __init__(
            self,
            original_dim,
            intermediate_dim,
            n_layer=2,
            name="decoder",
            **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True))

        self.px_scale_decoder = keras.Sequential([
            Dense(original_dim),
            Softmax(axis=-1)
        ])
        self.px_r_decoder = Dense(original_dim)
        self.px_dropout_decoder = Dense(original_dim)

    def call(self, inputs):
        x = inputs[0]
        lib = inputs[1]
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)
        px = x
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = lib * px_scale
        #px_rate = tf.clip_by_value(
        #    px_rate, clip_value_min=0, clip_value_max=12)
        px_r = self.px_r_decoder(px)
        px_r = tf.math.exp(px_r)

        return [px_rate, px_r, px_dropout]

loss_metric = tf.keras.metrics.Mean()

class VAE(keras.models.Model):
    """ SCVI """

    def __init__(
        self,
        original_dim,
        intermediate_dim=128,
        latent_dim=10,
        name="VAE",
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim)
        self.loss_metric = tf.keras.metrics.Mean()

    def call(self, inputs):
        z_mean, z_var, z = self.encoder(inputs)
        l = tf.expand_dims(
                            tf.math.log(
                                        tf.math.reduce_sum(inputs, axis=1)
                                    ), 
                        axis=1)
        reconstructed = self.decoder([z,l])
        
        # Add KL divergence regularization loss.
        p_dis = tfp.distributions.Normal(
            loc=z_mean,
            scale=tf.math.sqrt(z_var)
            )
        q_dis = tfp.distributions.Normal(
            loc=tf.zeros_like(z_mean),
            scale=tf.ones_like(z_var)
            )
        kl_loss = tf.reduce_sum(
                        tfp.distributions.kl_divergence(
                            p_dis, q_dis),
                    axis=1)
        self.add_loss(kl_loss)

        return reconstructed

    def train_step(self, data):

        with tf.GradientTape() as tape:
            reconstructed = self(data)
            # Compute reconstruction loss
            recon = - zinb_pos(data, reconstructed)
            loss = recon + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_metric(loss)

        return {"loss": self.loss_metric.result()}


class CNEncoder(keras.models.Model):
    """ copy nunmber encoder """

    def __init__(self,
                 original_dim,
                 bin_size,
                 max_cp=6,
                 latent_dim=10, intermediate_dim=128, n_layer=2,
                 name="cnencoder", **kwargs):
        super(CNEncoder, self).__init__(name=name, **kwargs)
        self.eps = 1e-4
        self.n_layer = n_layer
        self.bin_size = bin_size
        self.max_cp = max_cp
        
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True,
                    keep_prob=.1))
        for i in range(self.max_cp + 1):
            setattr(self, "rho%i" % i, Dense(original_dim // bin_size))

    def call(self, inputs):
        x = inputs
        for i in range(self.n_layer):
            x = getattr(self, "dense%i" % i)(x)

        rho_list = []
        for i in range(self.max_cp + 1):
            rho = getattr(self, "rho%i" % i)(x)
            rho_list.append(rho)
        rho_list = tf.nn.softmax(rho_list, axis=0)

        exp_counts = tf.zeros_like(rho_list[0])
        for i in range(self.max_cp + 1):
                exp_counts = exp_counts + rho_list[i] * i
        copy = tf.repeat(exp_counts, repeats=self.bin_size, axis=1)
        
        pi = tf.stack(rho_list, axis=1)
        pi = tf.transpose(pi, perm=[0, 2, 1])
        cat_prob = tf.repeat(pi, repeats=self.bin_size, axis=1)

        #pi_drop = pi[:,:-1,:]
        #left_column = tf.zeros([
        #                        tf.shape(pi_drop)[0],
        #                        1,
        #                        tf.shape(pi_drop)[-1]
        #                        ])
        #pi_left = np.concatenate([left_column, pi_drop], axis=1)
        #regulariser = 1 * np.sum(np.abs(pi - pi_left))
        #print(regulariser)
        #self.add_loss(regulariser)

        return cat_prob, copy


class CNDecoder(keras.models.Model):
    """ copy number decoder """

    def __init__(
            self,
            original_dim,
            intermediate_dim,
            n_layer=2,
            name="cndecoder",
            **kwargs):
        super(CNDecoder, self).__init__(name=name, **kwargs)
        self.n_layer = n_layer
        for i in range(self.n_layer):
            setattr(
                self,
                "dense%i" %
                i,
                FullyConnLayer(
                    intermediate_dim,
                    activation=Activation('relu'),
                    bn=True))

        self.px_r_decoder = Dense(original_dim)
        self.px_dropout_decoder = Dense(original_dim)


    def call(self, inputs):
        x = inputs[0]
        px = inputs[1]
        lib = inputs[2]

        px_dropout = self.px_dropout_decoder(px)
        px_rate = x # lib * x
        px_r = self.px_r_decoder(px)
        px_r = tf.math.exp(px_r)

        return [px_rate, px_r, px_dropout], x


class CopyVAE(VAE):

    def __init__(
            self,
            original_dim,
            intermediate_dim=128,
            latent_dim=10,
            bin_size=25,
            max_cp=15,
            decoder_layers=2,
            name="CopyVAE",
            **kwargs):
        super().__init__(original_dim,
                         intermediate_dim,
                         latent_dim,
                         name)
        self.max_cp = max_cp
        self.encoder = CNEncoder(original_dim, bin_size=bin_size, max_cp=max_cp)
        self.decoder = CNDecoder(original_dim, 
                                          intermediate_dim,
                                          n_layer=decoder_layers
                                          )
        self.loss_metric = tf.keras.metrics.Mean()

    def call(self, inputs):
        inputs_en = inputs[0]
        inputs_de = inputs[1]
        rho, z = self.encoder(inputs_en)
        l = tf.expand_dims(
                            tf.math.log(
                                        tf.math.reduce_sum(inputs_en, axis=1)
                                    ), 
                        axis=1)
        reconstructed, copy = self.decoder([z,inputs_de,l])

        return reconstructed

    def train_step(self, data):

        with tf.GradientTape() as tape:
            reconstructed = self(data)
            # Compute reconstruction loss
            recon = - zinb_pos(data[0], reconstructed)
            loss = recon + self.losses

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_metric(loss)

        return {"loss": self.loss_metric.result()}


def train_vae(vae, data, batch_size=128, epochs=10):
    """ Training function

    Args:
        vae: VAE object
        data: training examples
        batch_size: number of example in minibatch
        epochs: epochs
    Returns:
        trained model
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=0.01)
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Iterate over epochs.
    tqdm_progress = tqdm(range(epochs), desc='model training')
    for epoch in tqdm_progress:

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                recon = - zinb_pos(x_batch_train, reconstructed)
                loss = recon + vae.losses

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)

            if step % 100 == 0:
                tqdm_progress.set_postfix_str(
                    s="loss={:.2e}".format(
                        loss_metric.result()), refresh=True)

    return vae

def train_cpvae(vae, data, batch_size=128, epochs=10):
    """ Training function

    Args:
        vae: VAE object
        data: training examples
        batch_size: number of example in minibatch
        epochs: epochs
    Returns:
        trained model
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,epsilon=0.01)
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # Iterate over epochs.
    tqdm_progress = tqdm(range(epochs), desc='model training')
    for epoch in tqdm_progress:

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                    reconstructed = vae(x_batch_train)
                    # Compute reconstruction loss
                    recon = - zinb_pos(x_batch_train[0], reconstructed)
                    loss = recon + vae.losses  # sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)
            if step % 100 == 0:
                tqdm_progress.set_postfix_str(
                    s="loss={:.2e}".format(
                        loss_metric.result()), refresh=True)

    return vae
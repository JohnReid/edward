import os, math, six
from PIL import Image
import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import edward as ed
from edward.models import RandomVariable
from edward.util import copy


def tf_repeat(a, times):
  """Similar to np.repeat with axis=0."""
  indices = np.repeat(np.arange(a.shape[0].value), times)
  return tf.gather(a, indices)


# a = np.arange(3)
# a_tf = tf.constant(a)
# a_rep_tf = tf_repeat(a_tf, 2)
# sess_test = tf.Session()
# a_rep = sess_test.run(a_rep_tf)
# a_rep


def create_image_array(imgs, rows=None, cols=None):
  N = len(imgs)
  if rows is None:
    rows = int(np.sqrt(N))      # Number rows in output image
  if cols is None:
    cols = math.ceil(N / rows)  # Number columns in output image
  imarray = np.zeros((rows * 28, cols * 28), dtype = imgs[0].dtype)
  for n in range(N):
    row = int(n / cols)
    col = n % cols
    imarray[row*28:(row+1)*28, col*28:(col+1)*28] = imgs[n].reshape(28, 28)
  return Image.fromarray(255 * imarray).convert('RGB')


def generative_network(y, z, K, D, M):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  yz = tf.concat([y, z], axis=1)
  net = tf.reshape(yz, [M, 1, 1, K+D])
  with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
    net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
    net = slim.conv2d_transpose(net, 32, 5, stride=2)
    net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
    net = slim.flatten(net)
  return net


def inference_network(x, K, D):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma, y_logits = neural_network(x)
  """
  net = tf.reshape(x, [-1, 28, 28, 1])
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = slim.conv2d(net, 32, 5, stride=2)
    net = slim.conv2d(net, 64, 5, stride=2)
    net = slim.conv2d(net, 128, 5, padding='VALID')
    net = slim.dropout(net, 0.9)
    net = slim.flatten(net)
    output = slim.fully_connected(net, D * 2 + K, activation_fn=None)

  mu = output[:, :D]
  sigma = tf.nn.softplus(output[:, D:-K])
  y_logits = output[:, -K:]
  return mu, sigma, y_logits


def tf_entropy(probs):
  "The entropy of a categorical distribution with the given probabilities"
  return - tf.reduce_sum(probs * tf.log(probs), axis=-1)


class SemiSuperKLqp(ed.KLqp):
  def __init__(self, K, Ml, Mu, y, y_logits, alpha, *args, **kwargs):
    super(SemiSuperKLqp, self).__init__(*args, **kwargs)
    self.K = K
    self.Ml = Ml
    self.Mu = Mu
    self.M = Ml + K * Mu
    self.y = y
    self.y_logits = y_logits
    self.alpha = alpha
    # Indices into unlabelled data structures
    self.m = self.Ml + np.repeat(np.arange(self.Mu), self.K)
    self.k = np.tile(np.arange(self.K), self.Mu)
    self.mk = self.Ml + np.arange(self.Mu * self.K)

  def build_loss_and_gradients(self, var_list):
    "Adapted from edward.inferences.klqp.build_reparam_kl_loss_and_gradients"
    # Check the data are the correct shape
    for x in six.iterkeys(self.data):
      tf.assert_equal(tf.shape(x)[0], self.M)

    # For each x and each sample, calculate log p(x|z)
    self.p_log_lik = [tf.zeros(self.M)] * self.n_samples
    for s in range(self.n_samples):
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      scope = 'inference_' + str(id(self)) + '/' + str(s)
      dict_swap = {}
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        dict_swap[z] = qz_copy.value()

      # Sum the log likelihood of each datum
      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          log_probs = self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])
          self.p_log_lik[s] += tf.reduce_mean(log_probs, axis=1)

    # Stack the list of log likelihoods for each iteration into one tensor
    # and average across samples
    self.p_log_lik = tf.reduce_mean(
        tf.stack(self.p_log_lik), axis=0, name='p_log_lik')

    # Calculate the KL divergences between q(z|x) and p(z)
    self.kl = tf.concat(
        [ self.kl_scaling.get(z, 1.0) * ed.inferences.klqp._calc_KL(qz, z)
          for z, qz in six.iteritems(self.latent_vars)],
        axis=0)
    # sum over latent z dimensions and qz distributions
    self.kl = tf.reduce_sum(self.kl, axis=1, name='kl')
    # print(self.kl)

    # Calculate L for the labelled data
    self.Ll = self.kl[:self.Ml] - self.p_log_lik[:self.Ml]

    # Entropy of q(z|x) for unlabelled data
    self.yp = tf.nn.softmax(self.y_logits, name='yp')
    # print(self.yp)
    self.H_qyu = tf_entropy(self.yp[self.Ml:])
    self.H_qyu = tf.identity(self.H_qyu, name='H_qyu')
    # print(self.H_qyu)

    # Contribution to loss from unlabelled data
    self.klu = tf.reshape(tf_repeat(self.kl[self.Ml:], self.K), (self.Mu, self.K))
    self.llu = tf.reshape(self.p_log_lik[self.Ml:], (self.Mu, self.K))
    self.Lu = self.klu - self.llu
    self.ypu = self.yp[self.Ml:]
    self.U = tf.reduce_sum(tf.reduce_sum(self.ypu * self.Lu, axis=1) - self.H_qyu)

    # Calculate J and Jalpha
    self.J = tf.reduce_sum(self.Ll) + self.U
    # alpha term has a sum over K and average over Ml
    alpha_term = - self.alpha * tf.reduce_mean(tf.reduce_sum(self.y[:self.Ml] * tf.log(self.yp[:self.Ml]), axis=1))
    self.Jalpha = self.J + alpha_term

    # Calculate gradients
    loss = self.Jalpha
    grads = tf.gradients(loss, [v._ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars

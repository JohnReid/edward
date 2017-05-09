from time import gmtime, strftime
import os, math, six
from more_itertools import chunked
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


def current_time_tag():
  "Current time as tag suitable for filename."
  return strftime("%Y-%m-%d_%H-%M-%S", gmtime())


def get_log_dir(tag):
  time_tag = current_time_tag()
  logdir = os.path.join('logs', tag, time_tag)
  print('Logging to: {}'.format(logdir))
  os.makedirs(logdir)
  return logdir


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).

  See https://www.tensorflow.org/get_started/summaries_and_tensorboard"""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def ckpt_file(model, epoch):
  "File path to store model checkpoint."
  return '{}-{:0>3}.ckpt'.format(model, epoch)


def choose_labelled(ds, tochoose, K):
  # Calculate how many of each digit to pick
  if tochoose % K:
    raise ValueError('tochoose not a multiple of K')
  perdigit = tochoose // K  # integer division
  # Choose correct number of each digit
  idxs_l = np.empty((0,), dtype=np.int32)  # Labelled indexes
  idxs_u = np.empty((0,), dtype=np.int32)  # Unlabelled indexes
  for k in range(K):  # for each digit
    idxs = np.argwhere(ds.labels[:,k])[:,0]
    np.random.shuffle(idxs)  # permute
    idxs_l = np.concatenate([idxs_l, idxs[:perdigit]])
    idxs_u = np.concatenate([idxs_u, idxs[perdigit:]])
  # Shuffle to mix up digits
  np.random.shuffle(idxs_l)  # permute
  np.random.shuffle(idxs_u)  # permute
  # Check we have correct number of indexes
  assert idxs_l.shape[0] == tochoose
  assert idxs_u.shape[0] == ds.num_examples - tochoose
  # Check we have all indexes
  assert all(np.unique(np.concatenate([idxs_l, idxs_u])) == range(ds.num_examples))
  # Check there is no overlap between labelled and unlabelled
  assert len(set(idxs_u).intersection(set(idxs_l))) == 0
  return idxs_l, idxs_u


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
    with tf.name_scope('xp'):
      self.xp = [tf.zeros(self.M)] * self.n_samples
      for s in range(self.n_samples):
        # Form dictionary in order to replace conditioning on prior or
        # observed variable with conditioning on a specific value.
        scope = 'inference_' + str(id(self)) + '/' + str(s)
        self.dict_swap = {}
        for x, qx in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            if isinstance(qx, RandomVariable):
              qx_copy = copy(qx, scope=scope)
              self.dict_swap[x] = qx_copy.value()
            else:
              self.dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          self.dict_swap[z] = qz_copy.value()

        # Sum the log likelihood of each datum
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, self.dict_swap, scope=scope)
            log_probs = self.scale.get(x, 1.0) * x_copy.log_prob(self.dict_swap[x])
            self.xp[s] += tf.reduce_sum(log_probs, axis=1)

      # Stack the list of log likelihoods for each iteration into one tensor
      # and average across samples
      self.xp = tf.reduce_mean(tf.stack(self.xp), axis=0, name='xp')
      self.xpl = self.xp[:self.Ml]
      self.xpu = tf.reshape(self.xp[self.Ml:], (self.Mu, self.K))

    # Calculate the KL divergences between q(z|x) and p(z)
    with tf.name_scope('KL'):
      self.kl = tf.concat(
          [ self.kl_scaling.get(z, 1.0) * ed.inferences.klqp._calc_KL(qz, z)
            for z, qz in six.iteritems(self.latent_vars)],
          axis=0)
      # sum over latent z dimensions and qz distributions
      self.kl = tf.reduce_sum(self.kl, axis=1, name='kl')
      self.kll = self.kl[:self.Ml]
      # repeat into shape (Mu, K) as q(z|x) is the same for each y
      self.klu = tf.reshape(tf_repeat(self.kl[self.Ml:], self.K), (self.Mu, self.K))

    # Calculate L for the labelled and unlabelled data
    self.Ll = self.kll - self.xpl
    self.Lu = self.klu - self.xpu

    # y probabilities from recognition model
    with tf.name_scope('yp'):
      self.yp = tf.nn.softmax(self.y_logits, name='yp')
      self.ypl = self.yp[:self.Ml]
      self.ypu = self.yp[self.Ml:]

    # Entropy of q(z|x) for unlabelled data
    with tf.name_scope('H_qyu'):
      self.H_qyu = tf_entropy(self.ypu)
      self.H_qyu = tf.identity(self.H_qyu, name='H_qyu')

    # Contribution to loss from unlabelled data
    self.U = tf.reduce_sum(self.ypu * self.Lu, axis=1) - self.H_qyu

    # Calculate J and Jalpha
    with tf.name_scope('J'):
      self.J = tf.reduce_sum(self.Ll) + tf.reduce_sum(self.U)
      # alpha term has a sum over K and average over Ml
      log_term = - tf.reduce_sum(self.y[:self.Ml] * tf.log(self.ypl), axis=1)
      alpha_term = self.alpha * tf.reduce_mean(log_term, axis=0)
      self.Jalpha = self.J + alpha_term

    # Calculate gradients
    loss = self.Jalpha
    grads = tf.gradients(loss, [v._ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))

    # Add summaries
    with tf.name_scope('loss'):
      tf.summary.scalar('H_qyu', tf.reduce_mean(self.H_qyu))
      tf.summary.scalar('KL', tf.reduce_mean(self.kl))
      tf.summary.scalar('U', tf.reduce_sum(self.U))
      tf.summary.scalar('J', self.J)
      tf.summary.scalar('Jalpha', self.Jalpha)

    return loss, grads_and_vars

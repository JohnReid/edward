#!/usr/bin/env python
"""Semi-supervised learning

Some ideas from:

  Semi-supervised Learning with Deep Generative Models, Kingma et al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

import semi_util, importlib
importlib.reload(semi_util)
from semi_util import *
import os, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation
import edward as ed
from edward.models import Normal, Bernoulli
import semi_util, importlib
importlib.reload(semi_util)
from semi_util import *


model_tag = 'semi-M1'
DATA_DIR = "data/mnist"
IMG_DIR = os.path.join("img", model_tag)
CKPT_DIR = os.path.join("models", model_tag)
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)
if not os.path.exists(CKPT_DIR):
  os.makedirs(CKPT_DIR)


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  with tf.name_scope('generative'):
    with slim.arg_scope([slim.conv2d_transpose],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
      net = tf.reshape(z, [M, 1, 1, d])
      net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
      net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
      net = slim.conv2d_transpose(net, 32, 5, stride=2)
      net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
      net = slim.flatten(net)
      return net


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
  """
  with tf.name_scope('inference'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
      net = tf.reshape(x, [M, 28, 28, 1])
      net = slim.conv2d(net, 32, 5, stride=2)
      net = slim.conv2d(net, 64, 5, stride=2)
      net = slim.conv2d(net, 128, 5, padding='VALID')
      net = slim.dropout(net, 0.9)
      net = slim.flatten(net)
      params = slim.fully_connected(net, d * 2, activation_fn=None)

    mu = params[:, :d]
    sigma = tf.nn.softplus(params[:, d:])
    return mu, sigma


# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)


#
# Model M1
# Latent-feature discriminative model
#


# Dimensions
d = 25    # Number of dimensions of z
M = 500  # Mini-batch size
if mnist.train.num_examples % M:
  raise ValueError('Number of examples not a multiple of mini-batch size')
n_batches = math.floor(mnist.train.num_examples / M)

#
# The generative model
#
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)
hidden_rep = tf.sigmoid(logits)

#
# Recognition model
#
# Define a subgraph of the variational model, corresponding to a
# minibatch of size M.
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
mu, sigma = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(loc=mu, scale=sigma)

#
# Inference
#
# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
logdir = get_log_dir(model_tag)
inference.initialize(optimizer=optimizer, logdir=logdir)
inference.summarize = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create session
sess = ed.get_session()
# Restore variables if we have an epoch variable
if 'epoch' in locals():
  model_path = os.path.join(CKPT_DIR, ckpt_file(model_tag, epoch))
  print('Loading model from {}'.format(model_path))
  saver.restore(sess, model_path)
  avg_losses = [130.] * (epoch + 1)
else:
  init = tf.global_variables_initializer()
  init.run()
  avg_losses = []

#
# Train
#
n_epoch = 600
# n_epoch = 1
# n_batches = 1
from tensorflow.contrib import distributions as ds
for _ in range(n_epoch):
  epoch = len(avg_losses)
  total_loss = 0.0
  total_kl = 0.0

  for t in range(n_batches):
    x_train, _ = mnist.train.next_batch(M)
    x_train = np.random.binomial(1, x_train)
    feed_dict = {x_ph: x_train}
    info_dict = inference.update(feed_dict=feed_dict)
    total_loss += info_dict['loss']
    total_kl += ds.kl(qz, z).eval(feed_dict=feed_dict).sum()

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = total_loss / n_batches / M
  avg_kl = total_kl / n_batches / M / d
  logger.info("Epoch: {:0>3}; log p(x) >= {:0.3f}; KL(qz||z) = {:0.3f}".format(len(avg_losses), -avg_loss, avg_kl))
  avg_losses += [avg_loss]

  # Posterior predictive check.
  create_image_array(hidden_rep.eval()).save(
    os.path.join(IMG_DIR, 'ppc-{:0>3}.png'.format(epoch)))

  # Save the variables to disk.
  ckpt_path = os.path.join(CKPT_DIR, ckpt_file(model_tag, epoch))
  save_path = saver.save(sess, ckpt_path)


# #
# # Train a classifier on the latent z to predict the classes
# #
# # Build the classifier
# classifier = Sequential([
#   Dense(30, input_dim=d, activation='relu'),
#   Dense(10, activation='relu'),  # Small hidden layer
#   Dense(10, activation='sigmoid')])  # 10 digits
# # Compile model - clip gradients to avoid NaNs
# classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], clipnorm=1.)
# #
# # train
# class_losses = []
# n_epoch = 40
# n_iter_per_epoch = 100
# # n_epoch = 1
# # n_iter_per_epoch = 10
# for _ in range(n_epoch):
#   epoch = len(class_losses)
#   loss = np.zeros(2)
#   for t in range(n_iter_per_epoch):
#     # Get MNIST batch
#     x_train, x_labels = mnist.train.next_batch(M)
#     x_train = np.random.binomial(1, x_train)
#     create_image_array(x_train.astype(np.float32)).save(os.path.join(IMG_DIR, 'train.png'))
#     # Sample latent variables for images
#     zhat, muhat = sess.run((qz, mu), feed_dict={x_ph: x_train})
#     # Estimate images from latent variables
#     xsampled = sess.run(hidden_rep, feed_dict={z: zhat})
#     create_image_array(xsampled).save(os.path.join(IMG_DIR, 'sampled.png'))
#     np.mean(np.abs(xsampled - x_train))
#     # Train classifier
#     loss += classifier.train_on_batch(zhat, x_labels)
#   class_losses += [loss / n_iter_per_epoch]
#   logger.info('Epoch {}: av. loss = {}'.format(epoch, class_losses[-1]))
# #
# # Check classifier accuracy
# x_train, x_labels = mnist.train.next_batch(M)
# x_train = np.random.binomial(1, x_train)
# zhat, muhat = sess.run((qz, mu), feed_dict={x_ph: x_train})
# preds = classifier.predict(zhat)
# np.argmax(preds, axis=1) != np.argmax(x_labels, axis=1)
# num_mis = sum(np.argmax(preds, axis=1) != np.argmax(x_labels, axis=1))
# logger.info('Proportion misclassifications: {}'.format(num_mis / M))


#
# Map data through inference/recognition network to latent space z
#
save_mapped_dataset(DATA_DIR, sess, x_ph, mu, sigma, mnist.train, 'train')
save_mapped_dataset(DATA_DIR, sess, x_ph, mu, sigma, mnist.validation, 'validation')
save_mapped_dataset(DATA_DIR, sess, x_ph, mu, sigma, mnist.test, 'test')

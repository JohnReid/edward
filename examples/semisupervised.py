#!/usr/bin/env python
"""Semi-supervised learning

An attempt to implement some ideas from:

  Semi-supervised Learning with Deep Generative Models, Kingma et al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from scipy.misc import imsave
from scipy.special import expit
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation
import edward as ed
from edward.models import Normal, Bernoulli


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
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


DATA_DIR = "data/mnist"
IMG_DIR = "img"

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

# The generative model
#
d = 10   # Number of dimensions of z
M = 128  # Mini-batch size
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)
hidden_rep = tf.sigmoid(logits)

# Recognition model
#
# Define a subgraph of the variational model, corresponding to a
# minibatch of size M.
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
mu, sigma = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(mu=mu, sigma=sigma)

# Inference
#
# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()

rows = int(np.sqrt(M))      # Number rows in PPC output image
cols = math.ceil(M / rows)  # Number columns in PPC output image
def create_image_array(imgs):
  imarray = np.zeros((rows * 28, cols * 28), dtype = imgs[0].dtype)
  for m in range(M):
    row = int(m / cols)
    col = m % cols
    imarray[row*28:(row+1)*28, col*28:(col+1)*28] = imgs[m].reshape(28, 28)
  return Image.fromarray(255 * imarray).convert('RGB')

#
# Train
#
avg_losses = []
n_epoch = 100
n_epoch = 20
n_iter_per_epoch = 1000
for _ in range(n_epoch):
  epoch = len(avg_losses)
  total_loss = 0.0

  for t in range(n_iter_per_epoch):
    x_train, _ = mnist.train.next_batch(M)
    x_train = np.random.binomial(1, x_train)
    info_dict = inference.update(feed_dict={x_ph: x_train})
    total_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = total_loss / n_iter_per_epoch / M
  print("Epoch: {:0>3}; log p(x) >= {:0.3f}".format(len(avg_losses), avg_loss))
  avg_losses += [avg_loss]

  # Posterior predictive check.
  create_image_array(hidden_rep.eval()).save(
      os.path.join(IMG_DIR, 'ppc-{:0>3}.png'.format(epoch)))


#
# Train a classifier on the latent z to predict the classes
#
# Build the classifier
classifier = Sequential([
  Dense(30, input_dim=d, activation='relu'),
  Dense(10, activation='relu'),  # Small hidden layer
  Dense(10, activation='sigmoid')])  # 10 digits
# Compile model - clip gradients to avoid NaNs
classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], clipnorm=1.)
#
# train
class_losses = []
n_epoch = 40
n_iter_per_epoch = 100
# n_epoch = 1
# n_iter_per_epoch = 10
for _ in range(n_epoch):
  epoch = len(class_losses)
  loss = np.zeros(2)
  for t in range(n_iter_per_epoch):
    # Get MNIST batch
    x_train, x_labels = mnist.train.next_batch(M)
    x_train = np.random.binomial(1, x_train).astype(np.float32)
    create_image_array(x_train).save(os.path.join(IMG_DIR, 'train.png'))
    # Sample latent variables for images
    zhat, muhat = sess.run((qz, mu), feed_dict={x_ph: x_train})
    # Estimate images from latent variables
    xsampled = sess.run(hidden_rep, feed_dict={z: zhat})
    create_image_array(xsampled).save(os.path.join(IMG_DIR, 'sampled.png'))
    np.mean(np.abs(xsampled - x_train))
    # Train classifier
    loss += classifier.train_on_batch(zhat, x_labels)
  class_losses += [loss / n_iter_per_epoch]
  print('Epoch {}: av. loss = {}'.format(epoch, class_losses[-1]))

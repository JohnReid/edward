#!/usr/bin/env python
"""Convolutional variational auto-encoder for binarized MNIST.

The neural networks are written with TensorFlow Slim.

References
----------
http://edwardlib.org/tutorials/decoder
http://edwardlib.org/tutorials/inference-networks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math
from PIL import Image
import numpy as np
import tensorflow as tf
import edward as ed

from edward.models import Normal, Bernoulli, Multinomial
# from edward.util import Progbar
from scipy.misc import imsave
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


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


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(z, [M, 1, 1, D])
    net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
    net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
    net = slim.conv2d_transpose(net, 32, 5, stride=2)
    net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
    net = slim.flatten(net)
    return net


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  loc, scale = neural_network(x)
  """
  net = tf.reshape(x, [M, 28, 28, 1])
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = slim.conv2d(net, 32, 5, stride=2)
    net = slim.conv2d(net, 64, 5, stride=2)
    net = slim.conv2d(net, 128, 5, padding='VALID')
    net = slim.dropout(net, 0.9)
    net = slim.flatten(net)
    params = slim.fully_connected(net, D * 2, activation_fn=None)

  mu = params[:, :D]
  sigma = tf.nn.softplus(params[:, D:])
  return mu, sigma


def inference_network_keras(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
  """
  from keras.models import Sequential
  from keras.layers.core import Dropout, Flatten, Activation, Dense
  from keras.layers.convolutional import Conv2D
  from keras.layers.normalization import BatchNormalization
  net = Sequential([
      Conv2D( 32, 5, padding='same', strides=2, input_shape=(28, 28, 1), use_bias=False),
      # BatchNormalization(),
      Activation('elu'),
      Conv2D( 64, 5, padding='same', strides=2, use_bias=False),
      # BatchNormalization(),
      Activation('elu'),
      Conv2D(128, 5, padding='valid', use_bias=False),
      # BatchNormalization(),
      Activation('elu'),
      Dropout(.1),
      Flatten(),
      Dense(D * 2),
    ])

  params = net(tf.reshape(x, [M, 28, 28, 1]))
  mu = params[:, :D]
  sigma = tf.nn.softplus(params[:, D:])
  return mu, sigma


# ed.set_seed(42)

M = 128  # batch size during training
D = 2  # latent dimension
DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# MODEL
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)

# INFERENCE
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
loc, scale = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(loc=loc, scale=scale)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

hidden_rep = tf.sigmoid(logits)

init = tf.global_variables_initializer()
init.run()

n_epoch = 25
n_iter_per_epoch = 1000
n_iter_per_epoch = 100
for epoch in range(n_epoch):
  avg_loss = 0.0

  # pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    # pbar.update(t)
    x_train, _ = mnist.train.next_batch(M)
    x_train = np.random.binomial(1, x_train)
    info_dict = inference.update(feed_dict={x_ph: x_train})
    avg_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / M
  print("log p(x) >= -{:0.3f}".format(avg_loss))

  # Visualize hidden representations.
  imgs = hidden_rep.eval()
  create_image_array(imgs).save(
      os.path.join(IMG_DIR, 'vae-conv-{:0>3}.png'.format(epoch)))

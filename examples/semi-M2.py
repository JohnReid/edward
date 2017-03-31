#!/usr/bin/env python
"""Semi-supervised learning

Generative semi-supervised model M2 from:

  Semi-supervised Learning with Deep Generative Models, Kingma et al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2DTranspose, Conv2D
import edward as ed
from edward.models import Normal, Bernoulli, Multinomial


DATA_DIR = "data/mnist"
IMG_DIR = "img"

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)


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


#
# Model M2
# Generative semi-supervised model
#


def generative_network(y, z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  yz = tf.concat([tf.cast(y, dtype = z.dtype), z], axis=1)
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


def generative_network_keras():
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(y, z)
  """
  return Sequential([
    Conv2DTranspose(128, 3, activation="elu", padding="valid", input_shape=(1, 1, D+K,)),
    Conv2DTranspose( 64, 5, activation="elu", padding="valid"),
    Conv2DTranspose( 32, 5, activation="elu", padding="same" , strides=2),
    Conv2DTranspose(  1, 5, activation=None , padding="same" , strides=2),
    Flatten()
  ])


def preprocess_inputs(y, z):
  """Preprocess the inputs y and z for the generative network."""
  # Concatenate the y and the z, making sure dtypes agree
  yz = tf.concat([tf.cast(y, dtype = z.dtype), z], axis=1)
  return tf.reshape(yz, [M, 1, 1, D+K])


# The generative model
#
D = 5    # Number of dimensions of z
Mu = 55  # Number of unknown labels in mini-batch
Ml = 65  # Number of known labels in mini-batch
M = Mu + Ml  # Mini-batch size
K = 10   # Number of digits to classify
#
# Sample latent variables, z, for both labelled and unlabelled data
z = Normal(mu=tf.zeros([M, D]), sigma=tf.ones([M, D]))
#
# Uniform prior over unlabelled digits
pi = tf.ones([Mu, K]) / K
#
# Define a y separately for the labelled and unlabelled data and also concatenate them
yu = Multinomial(n=1., p=pi)            # The unknown labels
yl_ph = tf.placeholder(dtype=yu.dtype, shape=[Ml, K], name='yl')  # The known labels
y = tf.concat([yl_ph, yu], axis=0)
#
# Define the generative network and the Bernoulli likelihood
logits = generative_network(y, z)
# gen_net = generative_network_keras()
# logits = gen_net(preprocess_inputs(y, z))
x = Bernoulli(logits=logits)
hidden_rep = tf.sigmoid(logits)


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
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
    output = slim.fully_connected(net, D * 2 + K, activation_fn=None)

  mu = output[:, :D]
  sigma = tf.nn.softplus(output[:, D:-K])
  pi_logits = output[:, -K:]
  return mu, sigma, pi_logits


def inference_network_keras():
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma, pi_logits = neural_network(x)
  """
  return Sequential([
    Conv2D( 32, 5, strides=2, padding="same" , activation="elu", input_shape=(28, 28, 1)),
    Conv2D( 64, 5, strides=2, padding="same" , activation="elu"),
    Conv2D(128, 5, strides=2, padding="valid", activation="elu"),
    # Dropout(.1),
    Flatten(),
    Dense(D*2+K, activation=None)  # First D are mu, next D are sigma, last K are pi logits
  ])


def postprocess_inference(output):
  mu = output[:, :D]
  sigma = tf.nn.softplus(output[:, D:-K])
  pi_logits = output[:, -K:]
  return mu, sigma, pi_logits


# Recognition model
#
x_ph = tf.placeholder(dtype=tf.int32, shape=[M, 28 * 28], name='x')
# inf_net = inference_network_keras()
# mu, sigma, pi_logits = postprocess_inference(inf_net(tf.cast(tf.reshape(x_ph, (M, 28, 28, 1)), tf.float32)))
mu, sigma, pi_logits = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(mu=mu, sigma=sigma)
qy = Multinomial(n=1., logits=pi_logits)
qyu = Multinomial(n=1., logits=tf.slice(pi_logits, [Ml, 0], [Mu, -1]))
qyl = Multinomial(n=1., logits=tf.slice(pi_logits, [ 0, 0], [Ml, -1]))
# qyl = tf.slice(qy, [ 0, 0], [Ml, -1])
# qyu = tf.slice(qy, [Ml, 0], [Mu, -1])


# Inference
#
# Bind p(x,y,z) and q(z,y|x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz, yu: qyu}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
debug = False
inference.initialize(optimizer=optimizer, debug=debug)
sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


#
# Train
#
avg_losses = []
n_epoch = 20
n_iter_per_epoch = 100
n_epoch = 40
n_iter_per_epoch = 1000
for _ in range(n_epoch):
  epoch = len(avg_losses)
  total_loss = 0.0

  for t in range(n_iter_per_epoch):
    # Choose whether to train on labelled data or not. We draw the labelled data
    # from the validation set and the unlabelled data from the training set. This
    # gives us a 5000 : 55000 split.
    #
    xl_train, xl_labels = mnist.validation.next_batch(Ml)
    xu_train, xu_labels = mnist.train.next_batch(Mu)
    x_train_soft = np.concatenate([xl_train, xu_train], axis=0)
    x_train = np.random.binomial(1, x_train_soft)
    feed_dict = {x_ph: x_train, yl_ph: xl_labels}
    info_dict = inference.update(feed_dict=feed_dict)
    if math.isnan(info_dict['loss']):
      raise ValueError('Loss is NaN')
    # print(np.sum(np.abs(xl_labels - qyl.eval(feed_dict=feed_dict))))
    # pi_logits.eval(feed_dict=feed_dict)[:Ml,:]
    # qyl.eval(feed_dict=feed_dict)
    # qy.log_prob(tf.cast(np.concatenate([xl_labels, xu_labels], axis=0), dtype=tf.float32)).eval(feed_dict)[:Ml]
    # Increment total loss
    total_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = total_loss / n_iter_per_epoch / M
  print("Epoch: {:0>3}; log p(x) >= {:0.3f}".format(len(avg_losses), avg_loss))
  avg_losses += [avg_loss]

  # Sample a variety of latent z and plot one digit for each
  zsampled = np.random.normal(loc=0, scale=1., size=(M, D))
  ysampled = np.zeros((M, K))
  numstyles = int(M / K)
  for style in range(numstyles):
    for digit in range(K):
      idx = style * K + digit
      ysampled[idx, digit] = 1.
  imgs = sess.run(hidden_rep, feed_dict={z: zsampled, y: ysampled})
  create_image_array(imgs).save(
      os.path.join(IMG_DIR, 'semi-M2-{:0>3}.png'.format(epoch)))


def softmax(a):
  return (np.exp(a).T / np.sum(np.exp(a), axis=1)).T

#
# Predict classes for test data
#
x_train, x_labels = mnist.test.next_batch(M)
x_train = np.random.binomial(1, x_train)
create_image_array(x_train.astype(np.float32)).save(os.path.join(IMG_DIR, 'train.png'))
# Sample latent variables for images
zhat, muhat, yhat, qpi_logits = sess.run((qz, mu, qy, pi_logits), feed_dict={x_ph: x_train})
qpi = softmax(qpi_logits)
cross_entropy = -np.sum(np.log(qpi) * x_labels)
cross_entropy / M
# Estimate images from latent variables
xsampled = sess.run(hidden_rep, feed_dict={z: zhat})
create_image_array(xsampled).save(os.path.join(IMG_DIR, 'sampled.png'))
np.mean(np.abs(xsampled - x_train)/2)

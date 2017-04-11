#!/usr/bin/env python
"""Semi-supervised learning

Generative semi-supervised model M2 from:

  Semi-supervised Learning with Deep Generative Models, Kingma et al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import linalg as LA
import semi_util, importlib
importlib.reload(semi_util)
from semi_util import *
from tensorflow.examples.tutorials.mnist import input_data
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Dropout
# from keras.layers.convolutional import Conv2DTranspose, Conv2D
from edward.models import Normal, Bernoulli, Categorical


def repeat_unlabelled(a, name):
  """a is of shape (Ml+Mu, None). We expand a to be of shape (Ml+Mu*K, None) by
  repeating the part of a corresponding to the unknown labels."""
  tf.assert_equal(tf.shape(a)[0], Ml + Mu)
  return tf.concat([a[:Ml], tf_repeat(a[-Mu:], K)], axis=0, name=name)


# a = np.arange(Ml+Mu).reshape((Ml+Mu,1))
# a_tf = tf.constant(a)
# a_exp_tf = repeat_unlabelled(a_tf, 'a')
# sess_test = tf.InteractiveSession()
# a_exp = sess_test.run(a_exp_tf)
# a[-20:]
# a_exp[-20:]



#
# Model M2
# Generative semi-supervised model
#

D = 2    # Number of dimensions of z
K = 10   # Number of digits to classify
Ml = 17  # Number of known labels in mini-batch
Mu = 19  # Number of unknown labels in mini-batch
M = Ml + K * Mu  # Mini-batch size (we need K replicates of each unlabelled datum)
DATA_DIR = "data/mnist"
IMG_DIR = "img"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)


# The generative model
#
# Sample latent variables, z, for both labelled and unlabelled data
z = Normal(mu=tf.zeros([(Ml+Mu), D]), sigma=tf.ones([(Ml+Mu), D]), name='z')
z_rep = repeat_unlabelled(z, 'z_exp')
# Define a y separately for the labelled and unlabelled data and concatenate them
yu = tf.one_hot(tf.tile(tf.range(K, dtype=tf.int32), (Mu,)), K, name='yu')  # Use one-hot encoding
yl_ph = tf.placeholder(dtype=yu.dtype, shape=(Ml,K), name='yl')  # The known labels
y = tf.concat([yl_ph, yu], axis=0, name='y')
# Define the generative network and the Bernoulli likelihood
x_logits = generative_network(y, z_rep, K, D, M)
xgen = Bernoulli(logits=x_logits, name='xgen')
x_probs = tf.sigmoid(x_logits, name='xp')

# Recognition model
#
xl_ph = tf.placeholder(dtype=tf.int32, shape=[Ml, 28 * 28], name='xl')
xu_ph = tf.placeholder(dtype=tf.int32, shape=[Mu, 28 * 28], name='xu')
xrec = tf.concat([xl_ph, xu_ph], axis=0, name='xrec')
mu, sigma, y_logits = inference_network(tf.cast(xrec, tf.float32), K, D)
qz = Normal(mu=mu, sigma=sigma, name='qz')


# Inference
#
xrec_rep = repeat_unlabelled(xrec, 'x_repeated')
inference = SemiSuperKLqp(
    K=K, Ml=Ml, Mu=Mu, y=y, y_logits=y_logits, alpha=.1*(Ml+Mu),
    latent_vars={z: qz}, data={xgen: xrec_rep})
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
debug = False
n_samples = 1
inference.initialize(n_samples=n_samples, optimizer=optimizer, debug=debug)
sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


#
# DATA. MNIST batches are fed at training time.
#
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)


#
# Train
#
avg_losses = []
n_epoch = 40
n_iter_per_epoch = 1000
n_epoch = 40
n_iter_per_epoch = 1000
for _ in range(n_epoch):
  epoch = len(avg_losses)
  total_loss = 0.0

  for t in range(n_iter_per_epoch):
    xl_train_soft, yl_train = mnist.validation.next_batch(Ml)
    xu_train_soft, yu_train = mnist.train.next_batch(Mu)
    xl_train = np.random.binomial(1, xl_train_soft)
    xu_train = np.random.binomial(1, xu_train_soft)
    feed_dict = {xl_ph: xl_train, xu_ph: xu_train, yl_ph: yl_train}
    info_dict = inference.update(feed_dict=feed_dict)
    if math.isnan(info_dict['loss']):
      raise ValueError('Loss is NaN')
    # print(np.sum(np.abs(xl_labels - qyl.eval(feed_dict=feed_dict))))
    # pi_logits.eval(feed_dict=feed_dict)[:Ml,:]
    # qyl.eval(feed_dict=feed_dict)
    # qy.log_prob(tf.cast(np.concatenate([xl_labels, xu_labels], axis=0), dtype=tf.float32)).eval(feed_dict)[:Ml]
    # Increment total loss
    total_loss += info_dict['loss']

  # Check misclassification rate on known labels
  known_labels = np.argmax(yl_ph.eval(feed_dict), axis=1)
  unknown_labels = (yu_train * np.arange(K)).sum(axis=1)
  pred_labels = np.argmax(inference.yp.eval(feed_dict), axis=1)
  misclassrate = np.mean(known_labels != pred_labels[:Ml])
  H_qyu = inference.H_qyu.eval(feed_dict)

  # Print a lower bound to the average marginal likelihood for an
  # image (and unknown label for unlabelled data).
  avg_loss = total_loss / n_iter_per_epoch / (Ml + Mu)
  print("Epoch: {:0>3}; avg loss <= {:0.3f}; misclassification rate = {:0.3f}; avg H_qyu = {:0.3f}".format(
    len(avg_losses), avg_loss, misclassrate, H_qyu.mean()))
  avg_losses += [avg_loss]

  # Interpolate between 2 random points in latent z-space and plot one digit for each
  ysampled = np.zeros((M, K))
  zsampled = np.zeros((M, D))
  numstyles = int(M / K)
  numimages = numstyles * K
  # 1 random point in z-space
  z1 = np.random.normal(loc=0, scale=1., size=D)
  z1 = 2 * z1 / LA.norm(z1, 2)  # Make z1 of length 2
  # Interpolate between z1 and -z1
  zstyles = z1 - np.linspace(0, 1, num=numstyles).reshape((numstyles ,1)) * 2 * z1.reshape((1, D))
  zsampled[:numimages] = np.tile(zstyles, (K, 1))
  ysampled[np.arange(numimages), np.repeat(np.arange(K), numstyles)] = 1  # One hot encoding
  imgs = sess.run(x_probs, feed_dict={z_rep: zsampled, y: ysampled})[:numimages]
  create_image_array(imgs, rows=K).save(
      os.path.join(IMG_DIR, 'semi-M2-{:0>3}.png'.format(epoch)))

  # Plot the entropy of the y logits
  plt.figure()
  sns.kdeplot(H_qyu)
  plt.savefig(os.path.join(IMG_DIR, 'H_qyu-{:0>3}.png'.format(epoch)))
  plt.close()

  # raise ValueError()

# Plots of losses over epochs
#
plt.figure()
plt.plot(avg_losses)
plt.savefig('img/avg_losses.png')
plt.close()
plt.figure()
plt.plot(avg_losses[10:])  # Ignore first few losses to zoom in on convergence
plt.savefig('img/avg_losses_last.png')
plt.close()

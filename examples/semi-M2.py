#!/usr/bin/env python
"""Semi-supervised learning

Generative semi-supervised model M2 from:

  Semi-supervised Learning with Deep Generative Models, Kingma, Rezende, Mohamed, Welling
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
# DATA. MNIST batches are fed at training time.
#
DATA_DIR = "data/mnist"
IMG_DIR = "img"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
nlabelled = 3000
K = 10   # Number of digits to classify
idxs_l, idxs_u = choose_labelled(mnist.train, nlabelled, K)
nbatches = 1000
if len(idxs_l) % nbatches:
  raise ValueError('Number of labelled examples not a multiple of nbatches')
if len(idxs_u) % nbatches:
  raise ValueError('Number of unlabelled examples not a multiple of nbatches')
Ml = len(idxs_l) // nbatches
Mu = len(idxs_u) // nbatches
print('Ml = {}; Mu = {}'.format(Ml, Mu))
assert nbatches == len(list(chunked(idxs_l, Ml)))
assert nbatches == len(list(chunked(idxs_u, Mu)))
batches = [
  (
    mnist.train.images[idx_l],
    mnist.train.labels[idx_l],
    mnist.train.images[idx_u],
    mnist.train.labels[idx_u],
  )
  for idx_l, idx_u in zip(chunked(idxs_l, Ml), chunked(idxs_u, Mu))
]
assert nbatches == len(batches)


#
# Model M2
# Generative semi-supervised model
#

D = 2    # Number of dimensions of z
M = Ml + K * Mu  # Mini-batch size for generative network,
                 # we need K replicates of each unlabelled datum in order to marginalise


#
# The generative model
#
# Sample latent variables, z, for both labelled and unlabelled data
z = Normal(mu=tf.zeros([(Ml+Mu), D]), sigma=tf.ones([(Ml+Mu), D]), name='z')
z_rep = repeat_unlabelled(z, 'z_rep')
# Define a y separately for the labelled and unlabelled data and concatenate them
yu = tf.one_hot(tf.tile(tf.range(K, dtype=tf.int32), (Mu,)), K, name='yu')  # Use one-hot encoding
yl_ph = tf.placeholder(dtype=yu.dtype, shape=(Ml,K), name='yl')  # The known labels
y = tf.concat([yl_ph, yu], axis=0, name='y')
# Define the generative network and the Bernoulli likelihood
x_logits = generative_network(y, z_rep, K, D, M)
x = Bernoulli(logits=x_logits, name='x')
xp = tf.sigmoid(x_logits, name='xp')


#
# Recognition model
#
x_ph = tf.placeholder(dtype=tf.int32, shape=[Ml+Mu, 28 * 28], name='x_ph')
mu, sigma, y_logits = inference_network(tf.cast(x_ph, tf.float32), K, D)
qz = Normal(mu=mu, sigma=sigma, name='qz')

#
# For validation
#
with tf.name_scope('valid'):
  yp = tf.nn.softmax(y_logits, name='yp')
  y_labels = tf.placeholder(dtype=tf.float32, shape=(Ml+Mu,K), name='y_labels')
  y_pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_labels, name='y_pred_loss')


#
# Inference
#
x_rep = repeat_unlabelled(x_ph, 'x_rep')
inf_data = {x: x_rep}
inf_latent = {z: qz}
inference = SemiSuperKLqp(
    K=K, Ml=Ml, Mu=Mu, y=y, y_logits=y_logits, alpha=.1*(Ml+Mu),
    latent_vars=inf_latent, data=inf_data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
debug = False
n_samples = 1
inference.initialize(n_samples=n_samples, optimizer=optimizer, debug=debug)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Create session
sess = ed.get_session()
# Restore variables if we know the epoch otherwise initialise them
if 'epoch' in locals():
  saver.restore(sess, ckpt_path('semi-M2', epoch=epoch))
else:
  init = tf.global_variables_initializer()
  init.run()



#
# Train
#
avg_train_losses = []
avg_valid_losses = []
n_epochs = 1000
# nbatches = 100  # Don't use all batches
for _ in range(n_epochs):
  epoch = len(avg_train_losses)
  total_loss = 0.0

  for b in range(nbatches):
    xl_train_soft, yl_train, xu_train_soft, yu_train = batches[b]
    x_train = np.random.binomial(1, np.concatenate([xl_train_soft, xu_train_soft]))
    feed_dict = {x_ph: x_train, yl_ph: yl_train}
    info_dict = inference.update(feed_dict=feed_dict)
    if math.isnan(info_dict['loss']):
      raise ValueError('Loss is NaN')
    # Increment total loss
    total_loss += info_dict['loss']

  # Check misclassification rate on known labels
  known_labels = np.argmax(yl_ph.eval(feed_dict), axis=1)
  unknown_labels = (yu_train * np.arange(K)).sum(axis=1)
  pred_labels = np.argmax(inference.yp.eval(feed_dict), axis=1)
  misclassrate_known = np.mean(unknown_labels != pred_labels[Ml:])

  # Evaluate entropy of y predictions on unknown data
  H_qyu = inference.H_qyu.eval(feed_dict)

  # Check KL divergence of q(z|x) from prior
  klmean = inference.kl.eval(feed_dict).mean()

  # Calculate the average loss on the training data
  avg_loss = total_loss / nbatches / (Ml + Mu)
  avg_train_losses += [avg_loss]

  # Calculate the loss and misclassification rate on the validation set
  valid_loss = 0.0
  misclassified = 0
  for x_valid_soft, y_valid in \
      zip(chunked(mnist.validation.images, Ml + Mu),
          chunked(mnist.validation.labels, Ml + Mu)):
    x_valid = np.random.binomial(1, x_valid_soft)
    y_valid = np.asarray(y_valid)  # Convert from list
    nvalid = len(x_valid)  # Number of validation examples in this chunk
    # If chunk is smaller than recognition network input, we must pad it
    x_valid.resize((Ml+Mu, x_valid.shape[1])),
    y_valid.resize((Ml+Mu, K))
    feed_dict = {x_ph: x_valid, y_labels: y_valid}
    probs, loss = sess.run([yp, y_pred_loss], feed_dict=feed_dict)
    # Resize if needed
    probs = probs[:nvalid]
    loss = loss[:nvalid]
    # Cross entropy loss
    valid_loss += loss.sum()
    # Mis-classification rate
    pred_labels = np.argmax(probs, axis=1)
    valid_labels = (y_valid[:nvalid].astype(np.int32) * np.arange(K)).sum(axis=1)
    misclassified += (valid_labels != pred_labels).sum()
  avg_valid_loss = valid_loss / mnist.validation.num_examples
  avg_valid_losses += [avg_valid_loss]
  misclassrate = misclassified / mnist.validation.num_examples
  misclassrate

  # Print some statistics
  print(
      "Epoch: {:0>3}; "
      "avg train loss <= {:0.3f}; "
      "avg valid loss = {:0.3f}; "
      "misclass = {:0.3f}; "
      "avg H_qyu = {:0.3f}; "
      "avg KL[q(z|x) || p(z)] = {:0.3f}".format(
        epoch, avg_loss, avg_valid_loss, misclassrate, H_qyu.mean(), klmean))

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
  imgs = sess.run(xp, feed_dict={z_rep: zsampled, y: ysampled})[:numimages]
  create_image_array(imgs, rows=K).save(
      os.path.join(IMG_DIR, 'semi-M2-{:0>3}.png'.format(epoch)))

  # Plot the entropy of the y logits
  plt.figure()
  sns.kdeplot(H_qyu)
  plt.savefig(os.path.join(IMG_DIR, 'H_qyu-{:0>3}.png'.format(epoch)))
  plt.close()

  # Save the variables to disk.
  save_path = saver.save(sess, ckpt_path('semi-M2', epoch))

  # Plots of losses over epochs
  #
  plt.figure()
  plt.plot(avg_train_losses)
  plt.savefig('img/avg_train_losses.png')
  plt.close()
  # Ignore first few losses to zoom in on convergence
  ignore_first = 10
  plt.figure()
  plt.plot(range(ignore_first, len(avg_train_losses)), avg_train_losses[ignore_first:])
  plt.savefig('img/avg_losses_last.png')
  plt.close()

# G = tf.get_default_graph()
# G.get_operations()

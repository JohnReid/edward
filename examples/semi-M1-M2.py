#!/usr/bin/env python
"""Semi-supervised learning

Generative semi-supervised model M2 from:

  Semi-supervised Learning with Deep Generative Models, Kingma, Rezende, Mohamed, Welling
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import semi_util, importlib
importlib.reload(semi_util)
from semi_util import *
from edward.models import Normal, Bernoulli, Categorical


def generative_network(y, z, P):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  mu, sigma = neural_network(z)
  """
  with tf.name_scope('generative'):
    net = tf.concat([y, z], axis=1, name='yz')
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
      net = slim.fully_connected(net, 500)
      net = slim.dropout(net, 0.9)
      net = slim.fully_connected(net, P * 2, activation_fn=None)  # mu & sigma for each of P dimensions

    # mu = 20 * tf.nn.sigmoid(net[:, :P]) - 10
    mu = net[:, :P]
    sigma = tf.exp(net[:, P:] / 2)

  return mu, sigma


def inference_network(x, K, D):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma, y_logits = neural_network(x)
  """
  with tf.name_scope('inference'):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'scale': True}):
      net = slim.fully_connected(x, 500)
      net = slim.flatten(net)
      net = slim.fully_connected(net, D * 2 + K, activation_fn=None)

    mu = net[:, :D]
    sigma = tf.nn.softplus(net[:, D:-K])
    y_logits = net[:, -K:]
  return mu, sigma, y_logits


#
# DATA
#
model_tag = 'semi-M1-M2'
DATA_DIR = "data/mnist/dummy"
# DATA_DIR = "data/mnist"
IMG_DIR = os.path.join("img", model_tag)
CKPT_DIR = os.path.join("models", model_tag)
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)
if not os.path.exists(CKPT_DIR):
  os.makedirs(CKPT_DIR)
train = load_mapped_dataset(DATA_DIR, 'train')
trainloc = train['zloc']
trainscale = train['zscale']
trainlabels = train['y']
valid = load_mapped_dataset(DATA_DIR, 'validation')
validloc = valid['zloc']
validscale = valid['zscale']
validlabels = valid['y']
P = trainloc.shape[-1]  # Dimensionality of data
print('Input dimensionality = {}'.format(P))
K = 10   # Number of digits to classify
nlabelled = 300
nbatches = 100
idx_l, idx_u, Ml, Mu = choose_labelled(trainlabels, nlabelled, K, nbatches)


# 1/0
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')
unhotlabels = np.dot(trainlabels, np.arange(K)).astype('int')
# d = 0
# df = pd.DataFrame({
#   'x': trainloc[:,d],
#   'y': unhotlabels})
# plt.figure()
# df['x'].hist(by=df['y'], alpha=0.5)
# plt.savefig('trainloc-d{}.png'.format(d))
# plt.close('all')
# plt.figure()
# sns.kdeplot(trainloc.flatten(), bw=0.5)
# plt.savefig('trainloc.png')
# plt.figure()
# sns.kdeplot(np.log(trainscale).flatten(), bw=0.5)
# plt.savefig('trainscale.png')
def summarise(data, fn):
  df = pd.DataFrame(data)
  df['y'] = unhotlabels
  stats = df.groupby('y').apply(lambda g: fn(g.as_matrix(), axis=0))
  stats = np.asarray([s for s in stats])
  return stats
means = summarise(trainloc, np.mean)
means
stds = summarise(trainloc, np.std)
stds
1/0

#
# Model M2
# Generative semi-supervised model
#

D = 50    # Number of dimensions of z
print('Using {} latent dimensions for z'.format(D))
M = Ml + K * Mu  # Mini-batch size for generative network,
                 # we need K replicates of each unlabelled datum in order to marginalise
                 # over digits


#
# The generative model
#
# Sample latent variables, z, for both labelled and unlabelled data
z = Normal(loc=tf.zeros([(Ml+Mu), D]), scale=tf.ones([(Ml+Mu), D]), name='z')
z_rep = repeat_unlabelled(z, Ml, Mu, K, 'z_rep')
# Define a y separately for the labelled and unlabelled data and concatenate them
yu = tf.one_hot(tf.tile(tf.range(K, dtype=tf.int32), (Mu,)), K, name='yu')  # Use one-hot encoding
yl_ph = tf.placeholder(dtype=yu.dtype, shape=(Ml,K), name='yl')  # The known labels
y = tf.concat([yl_ph, yu], axis=0, name='y')
# Define the generative network and the Gaussian likelihood
x_mu, x_sigma = generative_network(y, z_rep, P)
x = Normal(loc=x_mu, scale=x_sigma, name='x')


#
# Recognition model
#
x_ph = tf.placeholder(dtype=x.dtype, shape=[Ml+Mu, P], name='x_ph')
qz_mu, qz_sigma, y_logits = inference_network(tf.cast(x_ph, tf.float32), K, D)
qz = Normal(loc=qz_mu, scale=qz_sigma, name='qz')


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
x_rep = repeat_unlabelled(x_ph, Ml, Mu, K, 'x_rep')
inf_data = {x: x_rep}
inf_latent = {z: qz}
inference = SemiSuperKLqp(
    K=K, Ml=Ml, Mu=Mu, y=y, y_logits=y_logits, alpha=.1*(Ml+Mu),
    latent_vars=inf_latent, data=inf_data)
with tf.name_scope('optimizer'):
  # The total number of examples we have processed
  total_examples = tf.Variable(0, trainable=False, name="total_examples")
  increment_total_examples_op = tf.assign(total_examples, total_examples+Ml+Mu)
  use_adam = True
  if use_adam:
    learning_rate = tf.Variable(3e-4, trainable=False, name="learning_rate")
    # Define an operation we can execute to reduce the learning rate
    learning_rate_scale = .1
    reduce_learning_rate = tf.assign(learning_rate, learning_rate*learning_rate_scale)
    epsilon = tf.Variable(1., trainable=False, name="epsilon")
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('epsilon', epsilon)
    # Use Adam with fixed learning rate
    # optimizer = tf.train.AdamOptimizer(0.1, epsilon=1.0)
    # optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.AdamOptimizer(.1)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=epsilon)
  else:
    # Use a decaying scale factor with a standard gradient descent optimizer.
    starter_learning_rate = .0000001
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=total_examples,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False,
        name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
debug = False
n_samples = 1
logdir = get_log_dir(model_tag)
inference.initialize(n_samples=n_samples, optimizer=optimizer, debug=debug, logdir=logdir)
inference.summarize = tf.summary.merge_all()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Create session
sess = ed.get_session()
# Restore variables if we know the epoch otherwise initialise them
# epoch = 38
if 'epoch' in locals():
  model_path = os.path.join(CKPT_DIR, ckpt_file(model_tag, epoch))
  print('Loading model from {}'.format(model_path))
  saver.restore(sess, model_path)
  avg_train_losses = [130.] * (epoch + 1)
  avg_valid_losses = [2.]   * (epoch + 1)
else:
  init = tf.global_variables_initializer()
  init.run()
  avg_train_losses = []
  avg_valid_losses = []



#
# Check for NaNs
#
tf.add_check_numerics_ops()


#
# Train
#
n_epochs = 250
# nbatches = 100  # Don't use all batches
for _ in range(n_epochs):
  epoch = len(avg_train_losses)
  total_loss = 0.0

  #
  # Generate some samples from the mean and standard deviation for z
  #
  batches = make_batches_from_M1(trainloc, trainscale, trainlabels, idx_l, idx_u, Ml, Mu)
  for b in range(nbatches):
    xl_train, yl_train, xu_train, yu_train = batches[b]
    feed_dict = {x_ph: np.concatenate([xl_train, xu_train]), yl_ph: yl_train}
    info_dict = inference.update(feed_dict=feed_dict)
    # print('Min x_sigma={}'.format(x_sigma.eval(feed_dict).min()))
    if math.isnan(info_dict['loss']):
      raise ValueError('Loss is NaN')
      model_path = os.path.join(CKPT_DIR, ckpt_file(model_tag, max(0, epoch - 2)))
      print('Loss is NaN, restoring from checkpoint: {}'.format(model_path))
      saver.restore(sess, model_path)
      sess.run(reduce_learning_rate)
    # Increment total loss
    total_loss += info_dict['loss']
    # Increment total examples for decaying learning rates
    sess.run(increment_total_examples_op)

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
  for x_valid, y_valid in \
      zip(map(np.asarray, chunked(validloc, Ml + Mu)),
          map(np.asarray, chunked(validlabels, Ml + Mu))):
    nvalid = len(x_valid)  # Number of validation examples in this chunk
    # If chunk is smaller than recognition network input, we must pad it
    x_valid = np.resize(x_valid, (Ml+Mu, x_valid.shape[1]))
    y_valid = np.resize(y_valid, (Ml+Mu, K))
    feed_dict_2 = {x_ph: x_valid, y_labels: y_valid}
    probs, loss = sess.run([yp, y_pred_loss], feed_dict=feed_dict_2)
    # Resize if needed
    probs = probs[:nvalid]
    loss = loss[:nvalid]
    # Cross entropy loss
    valid_loss += loss.sum()
    # Mis-classification rate
    pred_labels = np.argmax(probs, axis=1)
    valid_labels = (y_valid[:nvalid].astype(np.int32) * np.arange(K)).sum(axis=1)
    misclassified += (valid_labels != pred_labels).sum()
  valid_examples = validlabels.shape[0]
  avg_valid_loss = valid_loss / valid_examples
  avg_valid_losses += [avg_valid_loss]
  misclassrate = misclassified / valid_examples
  misclassrate
  summary = tf.Summary(value=[
    tf.Summary.Value(tag="validation/misclassrate", simple_value=misclassrate),
    tf.Summary.Value(tag="validation/avg_valid_loss", simple_value=avg_valid_loss),
  ])
  inference.train_writer.add_summary(summary, epoch)

  # Print some statistics
  print(
      "Epoch: {:0>3}; "
      "avg train loss <= {:0.3f}; "
      "avg valid loss = {:0.3f}; "
      "misclass = {:0.3f}; "
      "avg H_qyu = {:0.3f}; "
      "avg KL[q(z|x) || p(z)] = {:0.3f}".format(
        epoch, avg_loss, avg_valid_loss, misclassrate, H_qyu.mean(), klmean))

  # Plot the entropy of the y logits
  plt.figure()
  sns.kdeplot(H_qyu)
  plt.savefig(os.path.join(IMG_DIR, 'H_qyu-{:0>3}.png'.format(epoch)))
  plt.close()

  # Save the variables to disk.
  model_path = os.path.join(CKPT_DIR, ckpt_file(model_tag, epoch))
  save_path = saver.save(sess, model_path)

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

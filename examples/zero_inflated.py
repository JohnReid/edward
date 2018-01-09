#!/usr/bin/env python
"""Correlated normal posterior. Inference with Hamiltonian Monte Carlo.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import RandomVariable
from edward.models import Bernoulli
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions import Poisson

from matplotlib import pyplot as plt
plt.style.use("ggplot")


# Check TF select/where functionality
x = tf.constant([-2, 2])
y = tf.constant(1)
def f1(): return -tf.multiply(x, y)
def f2(): return  tf.multiply(x, y)
r = tf.where(tf.less(x, y), f1(), f2())
tf.Session().run(r)


class ZeroInflatedRV(RandomVariable, Distribution):
  """
  A zero-inflated random variable. The prob_zero parameter defines the
  probability of inflation.
  """
  def __init__(self, prob_zero, underlying, *args, **kwargs):
    self.prob_zero = prob_zero
    self.underlying = underlying
    self.bernoulli = Bernoulli(probs = self.prob_zero)  # for sampling
    super(ZeroInflatedRV, self).__init__(
        *args,
        **kwargs,
        dtype=underlying.dtype,
        validate_args=underlying.validate_args,
        allow_nan_stats=underlying.allow_nan_stats,
        reparameterization_type=underlying.reparameterization_type)

  def _log_prob(self, value):
    not_zero_lp = self.underlying.log_prob(value)
    return tf.where(
        tf.equal(value, tf.zeros_like(value)),
        tf.log(self.prob_zero + (1. - self.prob_zero * tf.exp(not_zero_lp))),
        tf.log(1. - self.prob_zero) + not_zero_lp)

  def _sample_n(self, n, seed=None):
    zero = self.bernoulli.sample(n, seed=seed)
    return tf.where(
        tf.equal(tf.constant(1), zero),
        tf.zeros_like(zero, dtype=self.dtype),
        self.underlying.sample(n))

zipois = ZeroInflatedRV(prob_zero = tf.constant((.8, .1)), underlying = Poisson(rate=tf.constant((12.5, 3.))))
s = zipois.sample((5, 3))
s.shape
lp = zipois.log_prob(s)
tf.Session().run((s, lp))

pois = Poisson(rate=tf.constant((12.5, 3.)))
sp = pois.sample((5, 3))
sp.shape
lppois = pois.log_prob(sp)
tf.Session().run((sp, lppois))

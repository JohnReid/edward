#!/usr/bin/env python
"""Use analytic KL for Poisson distributions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.models import Poisson
import tensorflow as tf
import tensorflow.contrib.distributions as dt

@dt.RegisterKL(dt.Poisson, dt.Poisson)
def _kl_poisson(poisson1, poisson2, name=None):
  """KL divergence between two Poisson distributions."""
  with tf.name_scope(name, "KL_Poisson", [poisson1, poisson2]):
    return poisson1.rate * (tf.log(poisson1.rate) - tf.log(poisson2.rate)) + poisson2.rate - poisson1.rate

p1 = Poisson(rate=1.)
p2 = Poisson(rate=2.)
kl = dt.kl_divergence(p1, p2)
tf.Session().run(kl)


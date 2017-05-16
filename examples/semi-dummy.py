#!/usr/bin/env python

#
# Create dummy data for semi-supervised learning
#
import numpy as np

P = 7         # Number of dimensions
K = 10        # Number of digits
varmean = .2  # Mean of the variance for each datum

#
# Sample multivariate normal parameters for each digit
#
mu = np.random.normal(scale=3, size=(K,P))
sigma = np.exp(np.random.normal(loc=-1, size=(K,P)))


def sample_data(N):
  # Choose digits
  k = np.random.choice(K, size=N)
  # Get parameters for data
  xmu = mu[k]
  xsigma = sigma[k]
  # Sample x
  xloc = np.random.normal(loc=xmu, scale=xsigma)
  xscale = np.sqrt(np.random.lognormal(mean=np.log(varmean), size=xmu.shape))
  # One-hot encode the ys
  y = np.zeros((N, K))
  y[np.arange(N),k] = 1
  return xloc, xscale, y


#
# Sample training data
#
xloctrain, xscaletrain, ytrain = sample_data(55000)
xlocvalid, xscalevalid, yvalid = sample_data(5000)


#
# Save training data
#
np.savez('data/mnist/dummy/semi-M1-train.npz', zloc=xloctrain, zscale=xscaletrain, y=ytrain)
np.savez('data/mnist/dummy/semi-M1-validation.npz', zloc=xlocvalid, zscale=xscalevalid, y=yvalid)

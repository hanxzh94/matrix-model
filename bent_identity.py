# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bent Identity bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import distribution_util
from tensorflow.python.ops import control_flow_ops


__all__ = [
    "BentIdentity",
]


class BentIdentity(bijector.Bijector):
  """Bijector which computes `Y = g(X) = Log[exp(2X) + exp(X)]`.

  The bent-identity `Bijector` has the following two useful properties:

  * The domain is all real numbers
  * `Y approx 2X`, for very positive `X`, and `Y approx X`, for very negative `X`,
  so the gradient is always bounded in (1, 2); no overflow
  """

  def __init__(self,
               name="bent_identity"):
    super(BentIdentity, self).__init__(
        forward_min_event_ndims=0,
        name=name)

  def _forward(self, x):
    return tf.where(x > 0, 2. * x + tf.log(1. + tf.exp(-x)), 
                           x + tf.log(1. + tf.exp(x)))

  def _inverse(self, y):
    # x = Log[(Sqrt(1 + 4 exp(y)) - 1) / 2]
    # when y is very negative, the numerator becomes very small, and x approx y
    threshold = np.log(np.finfo(y.dtype.as_numpy_dtype).eps) + 2.
    return tf.where(y > threshold, y / 2. + tf.log( (tf.sqrt(4.+tf.exp(-y)) - tf.exp(-y/2.))/2. ),
                                   y)

  def _inverse_log_det_jacobian(self, y):
    # dx / dy = 2 exp(y) / (1 + 4 exp(y) - sqrt(1 + 4 exp(y)))
    # when y is very negative, the demoninator becomes very small, and dx / dy approx 1
    threshold = np.log(np.finfo(y.dtype.as_numpy_dtype).eps) + 2.
    return tf.where(y > threshold, tf.log(2.) - tf.log( 4. + tf.exp(-y) - tf.sqrt(4.*tf.exp(-y)+tf.exp(-2.*y)) ),
                                   tf.zeros_like(y))

  def _forward_log_det_jacobian(self, x):
    return tf.where(x > 0, tf.log(2. + tf.exp(-x)) - tf.log(1. + tf.exp(-x)),
                           tf.log(1. + 2. * tf.exp(x)) - tf.log(1. + tf.exp(x)))


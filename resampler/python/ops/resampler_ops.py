# pylint: disable=g-bad-file-header
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tensorflow op performing differentiable resampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from resampler.python.ops.native import *

_resampler_so = tf.load_op_library(
  resource_loader.get_path_to_datafile('_resampler_ops.so'))


def resampler(data, warp, name="resampler", validate_warp=False, native=False):
  """Resamples input data at user defined coordinates.

  The resampler currently only supports bilinear interpolation of 2D data.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor of minimum rank 2 containing the coordinates at which
      resampling will be performed. Since only bilinear interpolation is
      currently supported, the last dimension of the `warp` tensor must be 2,
      representing the (x, y) coordinate where x is the index for width and y is
      the index for height.
    validate_warp: validate warp in [0, H or W - 1]
    native: use native tf implementation
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    determined by the shape of the warp tensor. For example, if `data` is of
    shape `[batch_size, data_height, data_width, data_num_channels]` and warp of
    shape `[batch_size, dim_0, ... , dim_n, 2]` the output will be of shape
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ImportError: if the wrapper generated during compilation is not present when
    the function is called.
  """
  with ops.name_scope(name, "resampler", [data, warp]):
    data_tensor = ops.convert_to_tensor(data, name="data")
    warp_tensor = ops.convert_to_tensor(warp, name="warp")

    if native:
      return native_resampler(data, warp, validate_warp)

    if validate_warp:
      return _resampler_so.resampler(data_tensor, warp_tensor)
    else:
      return _resampler_so.fast_resampler(data_tensor, warp_tensor)


@ops.RegisterGradient("Resampler")
def _resampler_grad(op, grad_output):
  data, warp = op.inputs
  grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
  return _resampler_so.resampler_grad(data, warp, grad_output_tensor)


@ops.RegisterGradient("FastResampler")
def _resampler_grad(op, grad_output):
  data, warp = op.inputs
  grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
  return _resampler_so.fast_resampler_grad(data, warp, grad_output_tensor)


ops.NotDifferentiable("ResamplerGrad")
ops.NotDifferentiable("FastResamplerGrad")

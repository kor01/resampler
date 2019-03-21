import tensorflow as tf
import resampler
import numpy as np
import time
from resampler.python.ops.native import *
from fake_consumer import *


def main():

  sz, channels = 1024, 64

  def run_once():
    image = consume(tf.random.normal((2, sz, sz, channels)))
    coord = consume(tf.random.uniform((2, sz, sz, 2)) * (sz - 1))

    start = time.time()
    ret = consume(resampler.resampler(image, coord))

    usage = time.time() - start
    print('time usage', usage)

    start = time.time()
    ret_native = consume(native_resampler(image, coord))
    native_usage = time.time() - start
    print('time usage native: ', native_usage)

    diff = ret - ret_native

    diff = np.abs(diff.numpy())

    print('mean diff', diff.mean(), 'max diff', diff.max())
    print('speed up:', native_usage / usage)

    return usage

  cum = 0.0
  iterations = 16
  for i in range(iterations):
    cum += run_once()
  print('average latency:', cum / iterations)


if __name__ == '__main__':
  with tf.device('/gpu:0'):
    main()

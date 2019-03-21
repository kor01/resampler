import tensorflow as tf

@tf.function
def native_resampler(data, warp, validate_warp=False):

  batch_size = tf.shape(data)[0]

  ret = tf.TensorArray(dtype=tf.float32, size=2)

  for i in tf.range(batch_size):

    d, w = data[i], warp[i]

    floor = tf.cast(tf.floor(w), tf.int32)
    ceil = floor + 1
    delta = w - tf.cast(floor, tf.float32)

    fx, fy = tf.split(floor, [1, 1], axis=-1)
    cx, cy = tf.split(ceil, [1, 1], axis=-1)
    dx, dy = tf.split(delta, [1, 1], axis=-1)

    def gather(x, y):
      return tf.gather_nd(d, tf.concat((y, x), axis=-1))

    fxfy = gather(fx, fy) * (1 - dx) * (1 - dy)
    cxcy = gather(cx, cy) * dx * dy
    fxcy = gather(fx, cy) * (1 - dx) * dy
    cxfy = gather(cx, fy) * dx * (1 - dy)

    ret = ret.write(i, fxfy + cxcy + fxcy + cxfy)

  return ret.stack()


__all__ = ['native_resampler']

## Resampler

factored out bilinear resampler implementation in tensorflow contrib

add faster implementation by removing all boundary checks

#

### code example:

```python

import time
import tensorflow as tf
from resampler import *

# make sure random test data generated before profiling
image = tf.random.normal((2, 1024, 1024, 64))
coord = tf.random.uniform((2, 1024, 1024, 2)) * (1024 - 1)

# fast implementation by default
ret = resampler(image, coord)

# safe implementation
ret = resampler(image, coord, validate_warp=True)

# native tf eager implementation
ret = resampler(image, coord, native=True)

```


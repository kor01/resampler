#! /bin/bash

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' ))

nvcc -O3 -std=c++11 -c -o resampler_ops_gpu.cu.o \
     resampler/kernels/resampler_ops_gpu.cu.cc \
     ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
     -I/usr/local/ -I./ --expt-relaxed-constexpr -DNDEBUG

nvcc -O3 -std=c++11 -c -o fast_resampler_ops_gpu.cu.o \
     resampler/kernels/fast_resampler_ops_gpu.cu.cc \
     ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
     -I/usr/local/ -I./ --expt-relaxed-constexpr -DNDEBUG

g++-4.8 -std=c++11 -Ofast -shared -DGOOGLE_CUDA -o _resampler_ops.so \
    resampler/ops/resampler_ops.cc \
    resampler/kernels/resampler_ops.cc resampler_ops_gpu.cu.o \
    resampler/ops/fast_resampler_ops.cc \
    resampler/kernels/fast_resampler_ops.cc fast_resampler_ops_gpu.cu.o \
     ${TF_CFLAGS[@]} -I./ \
    -fPIC -L/usr/local/cuda/lib64/ -lcudart ${TF_LFLAGS[@]}

mv _resampler_ops.so resampler/python/ops/
rm resampler_ops_gpu.cu.o fast_resampler_ops_gpu.cu.o

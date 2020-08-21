# This code extends codebase from followings:
# https://github.com/dstamoulis/single-path-nas
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

import numpy as np
import tensorflow as tf

from util.math_utils import linear
from util.utils import FLAGS

def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=None,
                        decay_epochs=None,
                        total_epochs=None,
                        warmup_epochs=5):
    """Build learning rate."""
    if lr_decay_type == 'exponential':
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True)
    elif lr_decay_type == 'cosine':
        assert total_epochs is not None
        assert steps_per_epoch is not None
        total_steps = total_epochs * steps_per_epoch
        lr = 0.5 * initial_lr * (
                1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
    elif lr_decay_type == 'constant':
        lr = initial_lr
    else:
        assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

    if warmup_epochs:
        tf.logging.info('Learning rate warmup_epochs: %d' % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_start_steps = 0
        warmup_start_lr = 0.0

        warmup_lr = linear(global_step, start_x=warmup_start_steps, start_y=warmup_start_lr,
                           end_x=warmup_steps, end_y=lr)
        lr = tf.cond(global_step <= warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
    """Build optimizer."""
    if optimizer_name == 'sgd':
        tf.logging.info('Using SGD optimizer')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        tf.logging.info('Using Momentum optimizer')
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        tf.logging.info('Using RMSProp optimizer')
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                              epsilon)
    else:
        tf.logging.fatal('Unknown optimizer:', optimizer_name)
        raise NotImplementedError

    return optimizer


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    tf.logging.info('drop_connect_rate: %s' % drop_connect_rate)
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def swish(features, use_native=True, use_hard=False):
    """Computes the Swish activation function.

    We provide three alternnatives:
      - Native tf.nn.swish, use less memory during training than composable swish.
      - Quantization friendly hard swish.
      - A composable swish, equivalant to tf.nn.swish, but more general for
        finetuning and TF-Hub.

    Args:
      features: A `Tensor` representing preactivation values.
      use_native: Whether to use the native swish from tf.nn that uses a custom
        gradient to reduce memory usage, or to use customized swish that uses
        default TensorFlow gradient computation.
      use_hard: Whether to use quantization-friendly hard swish.

    Returns:
      The activation value.
    """
    if use_native and use_hard:
        raise ValueError('Cannot specify both use_native and use_hard.')

    if use_native:
        return tf.nn.swish(features)

    if use_hard:
        return features * tf.nn.relu6(features + np.float32(3)) * (1. / 6.)

    features = tf.convert_to_tensor(features, name='features')
    return features * tf.nn.sigmoid(features)


def hsigmoid(features):
    """h-sigmoid from slim/nets/mobilenet/mobilenet_v3.py"""
    return tf.nn.relu6(features + 3) * 0.16667


def get_ema_vars():
    """Get all exponential moving average (ema) variables."""
    ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
            ema_vars.append(v)

    ema_vars = list(set(ema_vars))
    return ema_vars


def get_kerasconv_kern_shape_info(keras_conv):
    assert isinstance(keras_conv, tf.keras.layers.Conv2D)
    # TODO : I didn't check which is h and w
    k_h, k_w, input_filters, output_filters = keras_conv.kernel.shape.as_list()
    return k_h, k_w, input_filters, output_filters


def get_denseconv_kern_shape_info(keras_dense):
    assert isinstance(keras_dense, tf.keras.layers.Dense)
    input_units, output_units = keras_dense.kernel.shape.as_list()
    return input_units, output_units
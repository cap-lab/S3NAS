# This code extends codebase from followings:
# https://github.com/dstamoulis/single-path-nas
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def build_is_supergraph_training_tensor(global_step, supergraph_train_steps, is_training):
    tf.logging.info('Dropout rate supergraph train steps: %d' % supergraph_train_steps)
    yes = tf.cast(1, tf.float32)
    no = tf.cast(0, tf.float32)

    if is_training:
        return tf.cond(global_step <= supergraph_train_steps, lambda: yes, lambda: no)
    else:
        return no


def build_latency_lambda(global_step, supergraph_train_steps, final_lambda):
    tf.logging.info('Latency lambda starts after steps: %d' % supergraph_train_steps)
    initial_lambda_ = tf.cast(0.0, tf.float32)
    final_lambda_ = tf.cast(final_lambda, tf.float32)
    latency_lambda = tf.cond(global_step > supergraph_train_steps, lambda: final_lambda_, lambda: initial_lambda_)
    return latency_lambda


def build_latency_loss(latency_val, target_latency, latency_lambda, mul_lamda):
    latency_loss = latency_lambda * tf.log(1 + mul_lamda * tf.nn.relu(latency_val - target_latency))
    return latency_loss

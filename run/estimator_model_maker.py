# This code extends codebase from followings:
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
# https://github.com/dstamoulis/single-path-nas
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf
from absl import flags

import graph.graph_utils
import run.nas_utils as nas_utils
from graph import net_builder

FLAGS = flags.FLAGS


class EstimatorModelMaker(object):
    """
    self.global_step : global_step. initialized in get_model_fn
    self.tensordict_to_write_on_tensorboard : tensors to write on tensorboard. initialized in get_model_fn
    """

    def get_model_fn(self):
        def set_gs_and_td_and_get_model_fn(features, labels, mode, params):
            """params are automatically built by tensorflow, and additionally added in train_eval.py build_estimator"""
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            tf.keras.backend.set_learning_phase(is_training)
            self.global_step = tf.train.get_global_step()

            preprocessed_features = self.preprocess_features(features)
            logits, latency_val, self.tensordict_to_write_on_tensorboard = \
                self.logits_latency_tensordict(preprocessed_features, mode, params, FLAGS.ignore_latency,
                                               FLAGS.log_searchableblock_tensor)

            loss = self.build_losses(logits, labels, latency_val, params, FLAGS.ignore_latency)

            # build lr, estimator, train_op
            train_op = None
            if is_training:
                lr = self.build_learning_rate(params)
                optim = self.build_optimizer(lr)
                train_op = self.build_train_op(optim, loss, clip_gradients=FLAGS.clip_gradients)

            # EMA
            train_op, ema_scaffold_fn = self.build_EMAed_op_scaffold_fn(FLAGS.moving_average_decay, train_op)

            eval_metrics = None
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metrics = (self.metric_fn, [labels, logits])

            # tensorboard
            host_call = None
            if is_training and not FLAGS.skip_host_call:
                gs_t = tf.reshape(self.global_step, [1])
                current_epoch = (tf.cast(self.global_step, tf.float32) / params['steps_per_epoch'])
                summary_dict = OrderedDict(dict(gs=gs_t, current_epoch=current_epoch, lr=lr,
                                                total_loss=loss, latency=latency_val))
                summary_dict.update(self.tensordict_to_write_on_tensorboard)

                host_call = self.build_host_call_for_tensorboard(summary_dict)

            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                host_call=host_call,
                eval_metrics=eval_metrics,
                scaffold_fn=ema_scaffold_fn)

        return set_gs_and_td_and_get_model_fn

    @classmethod
    def preprocess_features(self, features):
        # In most cases, the default data format NCHW instead of NHWC should be
        # used for a significant performance boost on GPU. NHWC should be used
        # only if the network needs to be run on CPU since the pooling operations
        # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
        if FLAGS.data_format == 'channels_first':
            assert not FLAGS.transpose_input  # channels_first only for GPU
            features = tf.transpose(features, [0, 3, 1, 2])
            stats_shape = [3, 1, 1]
        else:
            stats_shape = [1, 1, 3]

        if FLAGS.transpose_input:
            features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

        normalized_features = self.normalize_features(features, net_builder.MEAN_RGB,
                                                      net_builder.STDDEV_RGB, stats_shape)

        return normalized_features

    @classmethod
    def normalize_features(self, features, mean_rgb, stddev_rgb, stats_shape):
        """Normalize the image given the means and stddevs."""
        features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
        features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
        return features

    def logits_latency_tensordict(self, features, mode, params, ignore_latency=False, log_searchableblock_tensor='min'):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        override_params = self._get_override_params_from_FLAGS()
        is_supergraph_training_tensor = nas_utils.build_is_supergraph_training_tensor(
            self.global_step, params['supergraph_train_steps'], is_training)
        override_params['is_supergraph_training_tensor'] = is_supergraph_training_tensor

        logits, latency_val, tensordict_to_write_on_tensorboard = \
            net_builder.build_logits_latency_tensordict(features,
                                                        model_json_path=FLAGS.model_json_path,
                                                        training=is_training,
                                                        override_params=override_params,
                                                        model_dir=FLAGS.model_dir,
                                                        ignore_latency=ignore_latency,
                                                        log_searchableblock_tensor=log_searchableblock_tensor)

        if params['use_bfloat16']:
            with tf.contrib.tpu.bfloat16_scope():
                logits = tf.cast(logits, tf.float32)

        num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('number of trainable parameters: {}'.format(num_params))

        return logits, latency_val, tensordict_to_write_on_tensorboard

    def build_losses(self, logits, labels, latency_val, params, ignore_latency=False):
        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=one_hot_labels,
            label_smoothing=FLAGS.label_smoothing)

        # Add weight decay to the loss for non-batch-normalization variables.
        loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'batch_normalization' not in v.name])

        if not ignore_latency:
            latency_lambda = nas_utils.build_latency_lambda(self.global_step, params['supergraph_train_steps'],
                                                            FLAGS.latency_lambda_val)
            self.tensordict_to_write_on_tensorboard['latency_lambda'] = latency_lambda

            target_latency = FLAGS.target_latency
            latency_loss = nas_utils.build_latency_loss(latency_val, target_latency, latency_lambda, FLAGS.mul_lamda)
            latency_loss = tf.reshape(latency_loss, shape=cross_entropy.shape)

            loss = loss + latency_loss

        return loss

    # lr, estimator, train_op related
    def build_learning_rate(self, params):
        scaled_lr = FLAGS.base_learning_rate * ((FLAGS.train_batch_size / 256.0) * params['train_num_replicas'])

        learning_rate = graph.graph_utils.build_learning_rate(scaled_lr, self.global_step,
                                                              params['steps_per_epoch'],
                                                              lr_decay_type=FLAGS.sched,
                                                              decay_factor=FLAGS.exp_decay_factor,
                                                              decay_epochs=FLAGS.exp_decay_epochs,
                                                              total_epochs=FLAGS.train_epochs,
                                                              warmup_epochs=FLAGS.warmup_epochs)

        return learning_rate

    def build_optimizer(self, learning_rate):
        optimizer = graph.graph_utils.build_optimizer(learning_rate)
        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        return optimizer

    def build_train_op(self, optimizer, loss, clip_gradients=0.0):
        def build_minimize_op(optimizer, loss, clip_gradients, global_step, var_list):
            # clip_gradients: got idea from https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net
            grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
            if clip_gradients > 0.0:
                g, v = zip(*grads_and_vars)
                g, _ = tf.clip_by_global_norm(
                    g, clip_gradients)
                grads_and_vars = zip(g, v)

            return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        var_list = tf.trainable_variables()
        minimize_op = build_minimize_op(optimizer, loss, clip_gradients, self.global_step, var_list)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        return train_op

    def build_EMAed_op_scaffold_fn(self, moving_average_decay, train_op):
        """
        :return: ema_op=None or scaffold_fn=None if not used, otherwise returns some valid instances.
        """
        EMAed_train_op, scaffold_fn = train_op, None

        is_training = train_op is not None
        has_moving_average_decay = (moving_average_decay > 0)
        if has_moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(
                decay=moving_average_decay, num_updates=self.global_step)
            ema_vars = graph.graph_utils.get_ema_vars()

            # EMA for train
            if is_training:
                with tf.control_dependencies([train_op]):
                    EMAed_train_op = ema.apply(ema_vars)

            # EMA for eval
            else:
                # Load moving average variables for eval.
                restore_vars_dict = ema.variables_to_restore(ema_vars)

                def _scaffold_fn():
                    saver = tf.train.Saver(restore_vars_dict)
                    return tf.train.Scaffold(saver=saver)

                scaffold_fn = _scaffold_fn

        return EMAed_train_op, scaffold_fn

    # eval related
    def metric_fn(self, labels, logits):
        """Evaluation metric function. Evaluates accuracy.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the model
        to the `metric_fn`, provide as part of the `eval_metrics`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `eval_metrics`.

        Args:
          labels: `Tensor` with shape `[batch]`.
          logits: `Tensor` with shape `[batch, num_classes]`.

        Returns:
          A dict of the metrics to return from evaluation.
        """
        predictions = tf.argmax(logits, axis=1)
        top_1_accuracy = tf.metrics.accuracy(labels, predictions)
        in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
        top_5_accuracy = tf.metrics.mean(in_top_5)

        return {
            'top_1_accuracy': top_1_accuracy,
            'top_5_accuracy': top_5_accuracy,
        }

    # tensorboard
    def build_host_call_for_tensorboard(self, summary_dict):
        def host_call_fn(**kwargs):
            """
            writes the {"tag_name": tensor} dict to tensorboard. got idea from
            https://github.com/tensorflow/tensor2tensor/blob/bf33311314005528482ea50b098d1aca8da85d84/tensor2tensor/utils/t2t_model.py#L2157
            """
            # Host call fns are executed FLAGS.iterations_per_loop times after one
            # TPU loop is finished, setting max_queue value to the same as number of
            # iterations will make the summary writer only flush the data to storage
            # once per loop.
            gs = kwargs.pop("gs")[0]
            with tf.contrib.summary.create_file_writer(
                    FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
                with tf.contrib.summary.always_record_summaries():
                    for name, tensor in sorted(six.iteritems(kwargs)):
                        half_tensor = tf.cast(tensor, tf.float16)
                        tf.contrib.summary.scalar(name, half_tensor[0], step=gs)

                    return tf.contrib.summary.all_summary_ops()

        half_dtype = tf.bfloat16 if FLAGS.use_tpu or FLAGS.tpu else tf.float16
        for key, value in summary_dict.items():
            if 'use' in key:
                value = tf.cast(value, tf.bool)
            elif key not in ['gs', 'lr', 'total_loss']:
                value = tf.cast(value, half_dtype)
            summary_dict[key] = tf.reshape(value, [1])

        host_call = (host_call_fn, summary_dict)

        return host_call

    def _get_override_params_from_FLAGS(self):
        override_params = {}
        if FLAGS.batch_norm_momentum is not None:
            override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
        if FLAGS.batch_norm_epsilon is not None:
            override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
        if FLAGS.dropout_rate is not None:
            override_params['dropout_rate'] = FLAGS.dropout_rate
        if FLAGS.drop_connect_rate is not None:
            override_params['drop_connect_rate'] = FLAGS.drop_connect_rate
        if FLAGS.data_format:
            override_params['data_format'] = FLAGS.data_format
        if FLAGS.num_label_classes:
            override_params['num_classes'] = FLAGS.num_label_classes

        return override_params

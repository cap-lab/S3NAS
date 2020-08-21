# This code extends codebase from followings:
# https://github.com/dstamoulis/single-path-nas
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import tensorflow as tf
from absl import flags
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

import imagenet_input
from util import io_utils

FLAGS = flags.FLAGS


class TrainNEval(object):
    def __init__(self, model_fn):
        self.params = {}  # We will save class attr in here
        config = self.set_and_get_device_config()
        self.set_train_params()
        self.build_estimator(model_fn, config)

        # Input pipelines are slightly different (with regards to shuffling and
        # preprocessing) between training and evaluation.
        self.imagenet_train = self.build_imagenet_input(is_training=True)
        self.imagenet_eval = self.build_imagenet_input(is_training=False)

    def train_and_eval(self):
        assert FLAGS.mode == 'train_and_eval'

        current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)

        train_epochs = self.params['train_steps'] / self.params['steps_per_epoch']
        tf.logging.info('Training for %d steps (%.2f epochs in total). Current step %d.',
                        self.params['train_steps'], train_epochs, current_step)

        start_timestamp = time.time()  # This time will include compilation time

        eval_results = None
        while current_step < self.params['train_steps']:
            # Train for up to steps_per_eval number of steps.
            # At the end of training, a checkpoint will be written to --model_dir.
            steps_per_eval = int(FLAGS.epochs_per_eval * self.params['steps_per_epoch'])
            next_eval = (current_step // steps_per_eval) * steps_per_eval + steps_per_eval
            print("next eval point : ", next_eval)
            next_checkpoint = min(next_eval, self.params['train_steps'])
            self.est.train(input_fn=self.imagenet_train.input_fn, max_steps=int(next_checkpoint))
            current_step = next_checkpoint

            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            next_checkpoint, int(time.time() - start_timestamp))

            eval_results = self.eval()

        if eval_results is None:
            eval_results = self.eval()

        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        self.params['train_steps'], elapsed_time)

        tf.keras.backend.clear_session()
        tf.reset_default_graph()

        return eval_results['top_1_accuracy'].item()

    def eval(self, eval_name=None):
        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')

        eval_results = self.est.evaluate(
            input_fn=self.imagenet_eval.input_fn,
            steps=FLAGS.num_eval_images // FLAGS.eval_batch_size,
            name=eval_name)
        tf.logging.info('Eval results for final: %s', eval_results)
        io_utils.archive_ckpt(eval_results, eval_results['top_1_accuracy'], FLAGS.model_dir, FLAGS.keep_archive_max)

        return eval_results

    def set_and_get_device_config(self):
        if FLAGS.tpu or FLAGS.use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu,
                zone=FLAGS.tpu_zone,
                project=FLAGS.gcp_project)
        else:
            tpu_cluster_resolver = None

        if FLAGS.use_async_checkpointing:
            save_checkpoints_steps = None
        else:
            save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)

        train_distribution = None
        gpu_options = None
        train_num_replicas = 1
        if not FLAGS.use_tpu:
            train_distribution = tf.contrib.distribute.MirroredStrategy()
            gpu_options = tf.GPUOptions(allow_growth=True)
            train_num_replicas = train_distribution.num_replicas_in_sync
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            log_step_count_steps=FLAGS.log_step_count_steps,
            keep_checkpoint_max=FLAGS.keep_checkpoint_max,
            session_config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                graph_options=tf.GraphOptions(
                    rewrite_options=rewriter_config_pb2.RewriterConfig(
                        disable_meta_optimizer=True))),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
                    .PER_HOST_V2),
            train_distribute=train_distribution,
            eval_distribute=None
        )  # pylint: disable=line-too-long

        self.params.update(dict(train_num_replicas=train_num_replicas))
        return config

    def set_train_params(self):
        # Initializes model parameters.
        steps_per_epoch = int(FLAGS.num_train_images / (FLAGS.train_batch_size * self.params['train_num_replicas']))
        train_steps = int(FLAGS.train_epochs * steps_per_epoch)
        supergraph_train_steps = int(FLAGS.supergraph_train_epochs * steps_per_epoch)
        print("train steps : ", train_steps)

        self.params.update(dict(steps_per_epoch=steps_per_epoch,
                                train_steps=train_steps,
                                supergraph_train_steps=supergraph_train_steps,
                                use_bfloat16=FLAGS.use_bfloat16))

    def build_estimator(self, model_fn, config):
        self.est = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            params=self.params)

    @classmethod
    def build_imagenet_input(self, is_training):
        """Generate ImageNetInput for training and eval."""
        # For imagenet dataset, include background label if number of output classes
        # is 1001
        include_background_label = (FLAGS.num_label_classes == 1001)

        tf.logging.info('Using dataset: %s', FLAGS.data_dir)

        return imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=FLAGS.transpose_input,
            cache=FLAGS.use_cache and is_training,
            image_size=FLAGS.input_image_size,
            num_parallel_calls=FLAGS.num_parallel_calls,
            use_bfloat16=FLAGS.use_bfloat16,
            include_background_label=include_background_label)

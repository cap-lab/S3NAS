# This code extends codebase from followings:
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from run import preprocessing
from etc_utils.base_tocsv import CSVMaker
from graph import net_builder
from graph.jsonnet import Net
from graph.graph_utils import get_ema_vars
from util.string_utils import get_filename

FLAGS = flags.FLAGS

flags.DEFINE_string('model_json_path', default=None, help=('The path for model json file'))
flags.DEFINE_string('data_path', None, 'Image data path')
flags.DEFINE_string('ckpt_dir', None, 'Checkpoint folder')
flags.DEFINE_string('csv_save_dir', None, 'CSV files dir')
flags.DEFINE_boolean('enable_ema', True, 'Enable exponential moving average.')
flags.DEFINE_integer('input_image_size', default=224, help='Input image size.')
## in jsonnet.py
# flags.DEFINE_list(
#     'log_excitation_names_containing', default=None,
#     help=('log values of excitations. If value is all, all SE blocks will be logged.'
#           'If value is stages_0,stages_1/blocks_2 then SE blocks with names containing stage_0 and stage_1/block_2 will be logged')
# )


class ModelBuilder(object):
    @classmethod
    def get_preprocess_fn(cls):
        """Build input dataset."""
        return preprocessing.preprocess_image

    @classmethod
    def build_prob_and_tensordict(cls, sess, images, ckpt_dir, model_json_path, enable_ema):
        probs, tensordict = cls._build_model(model_json_path, images, is_training=False)
        num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print('number of parameters : ', num_params)
        if isinstance(probs, tuple):
            probs = probs[0]

        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        cls._restore_model(sess, checkpoint, enable_ema)

        return probs, tensordict

    @classmethod
    def _build_model(cls, model_json_path, features, is_training):
        """Build model with input features."""
        model_builder = net_builder
        features -= tf.constant(
            model_builder.MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(
            model_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
        logits, _, tensordict = model_builder.build_logits_latency_tensordict(
            features, model_json_path, is_training,
            ignore_latency=True, log_searchableblock_tensor='all')
        probs = tf.nn.softmax(logits)
        probs = tf.squeeze(probs)
        return probs, tensordict

    @classmethod
    def _restore_model(cls, sess, checkpoint, enable_ema=True):
        """Restore variables from checkpoint dir."""
        sess.run(tf.global_variables_initializer())
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = get_ema_vars()
            var_dict = ema.variables_to_restore(ema_vars)
        else:
            var_dict = get_ema_vars()

        tf.train.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)


def build_dataset(image_filenames, image_size, preprocess_fn, batch_size=1, is_training=False):
    """Build input dataset."""
    image_filenames = tf.constant(image_filenames)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames))

    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = preprocess_fn(
            image_string, is_training, image_size=image_size)
        image = tf.cast(image_decoded, tf.float32)
        return image

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    return images


def run_inference_and_make_csv(sess, probs, tensordict, num_images):
    """Build and run inference on the target images and labels."""
    tensordict = OrderedDict(tensordict)  # To preserve orders

    ## initialize csv_makers
    tensor_names = tensordict.keys()
    tensors = list(tensordict.values())
    csv_makers = _init_csv_makers(tensor_names, tensors)

    run_tensors = [probs]
    run_tensors.extend(tensors)

    ## collect datas
    for i in range(num_images):
        run_results = sess.run(run_tensors)
        out_probs = run_results[0]
        tensor_values = run_results[1:]

        for tensor_name, tensor_val in zip(tensor_names, tensor_values):
            csv_makers[tensor_name].fill_row(tensor_val.squeeze())

        if i % 100 == 0:
            print("made image %d" % i)

    _save_csv(csv_makers)


def _save_csv(csv_makers):
    for tensor_name, csv_maker in csv_makers.items():
        base_save_path = FLAGS.csv_save_dir
        if not base_save_path:
            base_save_path = get_filename(FLAGS.model_json_path)
        stage_block_prefix = Net.extract_stage_block_prefix(tensor_name)
        save_name = os.path.join(base_save_path, stage_block_prefix) + '.csv'
        print("saving to %s" % save_name)
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        csv_maker.save(save_name)


def _init_csv_makers(names, tensors):
    csv_makers = {}
    for name, tensor in zip(names, tensors):
        columns = list(range(int(tensor.shape[-1])))
        csv_maker = CSVMaker(columns)
        csv_makers[name] = csv_maker
    return csv_makers


def extract_se_excitations(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS.log_excitation_names_containing = 'all'

    mb = ModelBuilder
    with tf.Graph().as_default(), tf.Session() as sess:
        image_files = sorted(glob.glob(os.path.join(FLAGS.data_path, "*/*")))
        images = build_dataset(image_files, image_size=FLAGS.input_image_size, preprocess_fn=mb.get_preprocess_fn(),
                               is_training=False)
        prob, tensordict = mb.build_prob_and_tensordict(sess, images, FLAGS.ckpt_dir, FLAGS.model_json_path,
                                                        FLAGS.enable_ema)
        run_inference_and_make_csv(sess, prob, tensordict, num_images=len(image_files))


if __name__ == '__main__':
    app.run(extract_se_excitations)

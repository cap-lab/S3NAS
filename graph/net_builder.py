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

import json

import tensorflow as tf
from absl import flags

from graph.blockargs import BlockArgsDecoder
from graph.gparams import GlobalParams
from graph.jsonnet import Net
from util.io_utils import tf_open_file_in_path
from util.utils import AttrDict

FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def build_logits_latency_tensordict(images, model_json_path, training, override_params=None, model_dir=None,
                                    ignore_latency=False, log_searchableblock_tensor='min'):
    """A helper function to creates the NAS Supernet and returns predicted logits.

    Args:
      images: input images tensor.
      model_json_path: string, the model args json path
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        GlobalParams.
      model_dir: If not None, block_args are written to model_params.txt in model_dir.
      ignore_latency: If true, terms related to latency will be ignored
      log_searchableblock_tensor: 'never' : don't log tensordict from model
                    'min' : log only use_conditions in searchable blocks
                    'all' : log all tensordict

    Returns:
      logits: the logits tensor of classes.
      latency: the total latency based on the threshold decisions
      tensordict_to_write_on_tensorboard: tensors you want to watch on tensorboard.
    """
    assert isinstance(images, tf.Tensor)
    model_args, global_params = get_model_args_and_gparams(model_json_path, override_params)
    if model_dir:
        save_model_args(model_args, model_dir)

    in_w = FLAGS.input_image_size
    with tf.variable_scope('model'):
        model = Net(model_args, global_params)
        logits = model(images, training=training)
        logits = tf.identity(logits, 'logits')

        input_shape = (in_w, in_w, 3)
        tf.logging.info("built model with trainable params %d and flops %d" %
                        (model.get_params_considering_bnbias(input_shape),
                         model.get_flops(input_shape)))

        tensordict = {}
        log_searchableblock_tensor = log_searchableblock_tensor.lower()
        if log_searchableblock_tensor == 'all':
            tensordict = model.tensordict_to_write_on_tensorboard()
        elif log_searchableblock_tensor == 'min':
            tensordict = {}
            for name, val in model.tensordict_to_write_on_tensorboard().items():
                is_useconds = 'use' in name
                if is_useconds:
                    tensordict[name] = val
        else:
            assert log_searchableblock_tensor == 'never'

        if ignore_latency:
            total_latency = tf.zeros((1,))
        else:
            from graph.latency_estimator import get_constraint_estimator
            latency_estimator = get_constraint_estimator(FLAGS.constraint.lower(), FLAGS.constraint_parse_key,
                                                         FLAGS.constraint_div_unit)
            total_latency, tensordict_latency = latency_estimator.estim_constraint(model, in_w)

            tensordict.update(tensordict_latency)

        return logits, total_latency, tensordict


def get_model_args_and_gparams(model_json_path, override_params):
    """
    Gets model_args from json file.
    Supports both tensorflow-style stages_args and more human-readable style.
    """
    model_json = json.load(tf_open_file_in_path("", model_json_path, "r"), object_pairs_hook=AttrDict)
    model_args = AttrDict(model_json)

    decoder = BlockArgsDecoder()
    model_args.stages_args = decoder.decode_to_stages_args(model_args.stages_args)

    gparams_dict = parse_gparams_from_model_args(model_args)
    global_params = GlobalParams(**gparams_dict)

    if override_params:
        global_params = global_params._replace(**override_params)

    tf.logging.info('global_params= %s', global_params)
    tf.logging.info('stages_args= %s', model_args.stages_args)
    return model_args, global_params


def parse_gparams_from_model_args(model_args):
    def update_gparams_if_exist_in_modelargs(gparams_dict, model_args, key):
        val = model_args.get(key)
        if val:
            gparams_dict[key] = val
        return gparams_dict

    gparams_dict = {}

    for key in ['act_fn', 'se_inner_act_fn', 'se_gating_fn']:
        gparams_dict = update_gparams_if_exist_in_modelargs(gparams_dict, model_args, key)

    return gparams_dict


def save_model_args(model_args, model_dir, filename='scaled_model_args.json'):
    f = tf_open_file_in_path(model_dir, filename, 'w')
    json.dump(model_args, f, indent=4, ensure_ascii=False)

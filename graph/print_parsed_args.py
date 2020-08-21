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
from absl import flags, app

from graph import parse_netarch

FLAGS = flags.FLAGS
from graph.net_builder import get_model_args_and_gparams
from util.io_utils import tf_open_file_in_path
import json

flags.DEFINE_string(
    'parse_search_dir',
    default=None,
    help=('The directory where the tensorboard result of search process resides.'))

flags.DEFINE_string(
    'search_model_json_path',
    default=None,
    help=('The model json you used for search'))

flags.DEFINE_string(
    'parse_json_name',
    default='parsed_model',
    help=('name of parsed json_model')
)

flags.DEFINE_integer(
    'input_image_size', default=224, help='Input image size.')


def print_parse(unused_args):
    model_path = FLAGS.search_model_json_path
    parse_search_dir = FLAGS.parse_search_dir

    model_args, _ = get_model_args_and_gparams(model_path, None)
    tf.logging.info(model_args)

    stages_args = parse_netarch.parse_stages_args(parse_search_dir, base_model_args=model_args)

    model_args.stages_args = stages_args
    parse_dir = FLAGS.parse_search_dir
    f = tf_open_file_in_path(parse_dir, FLAGS.parse_json_name + '.json', 'w')
    json.dump(model_args, f, indent=4, ensure_ascii=False)
    tf.logging.info(model_args)


if __name__ == "__main__":
    app.run(print_parse)

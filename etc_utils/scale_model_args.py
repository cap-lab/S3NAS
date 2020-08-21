import json

import tensorflow as tf
from absl import flags

from graph.cpscale import compound_scale
from graph.gparams import GlobalParams
from util import io_utils, math_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'input_image_size', default=224, help='Input image size.')

flags.DEFINE_float(
    'depth_coefficient', default=1.0,
    help=('Depth coefficient for scaling number of layers.'))

flags.DEFINE_list(
    'depth_list', default=None,
    help=('Target depth for scaling number of layers.'))

flags.DEFINE_float(
    'width_coefficient', default=1.0,
    help=('WIdth coefficient for scaling channel size.'))

flags.DEFINE_float(
    'resol_coefficient', default=1.0,
    help=('resolution coefficient for scaling input image size.'))

flags.DEFINE_integer(
    'filters_divisor', default=8, help=('Depth divisor (default to 8).'))

flags.DEFINE_integer(
    'img_divisor', default=4, help=('Depth divisor (default to 8).'))

flags.DEFINE_string(
    'model_json_path',
    default=None,
    help=('The path for model json file'))

flags.DEFINE_string(
    'save_dir', default='.',
    help=('directory you want to save in'))

flags.DEFINE_string(
    'save_json_name', default=None,
    help=('name you want to use for saving'))


def main(unused_argv):
    save_json_name = FLAGS.save_json_name
    orig_img_size = FLAGS.input_image_size

    model_args = io_utils.load_json_as_attrdict(FLAGS.model_json_path)
    img_size = math_utils.round_to_multiple_of(orig_img_size * FLAGS.resol_coefficient, FLAGS.img_divisor)

    params_dict = {key: getattr(FLAGS, key) for key in ['filters_divisor']}
    global_params = GlobalParams(**params_dict)

    model_args = compound_scale(model_args, global_params, FLAGS.width_coefficient, FLAGS.depth_coefficient,
                                orig_img_size, FLAGS.depth_list)
    model_args.img_size = img_size

    f = io_utils.tf_open_file_in_path(FLAGS.save_dir, save_json_name, 'w')
    json.dump(model_args, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    tf.app.run(main)

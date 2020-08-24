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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tensorboard_dir',
    default=None,
    help=('The directory where the tensorboard result of search process resides.'))


def get_latency(tb_path):
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }
    event_acc = EventAccumulator(tb_path, tf_size_guidance)
    event_acc.Reload()
    return event_acc.Scalars('latency')[-1].value


def print_latency(unused_args):
    lat = get_latency(FLAGS.tensorboard_dir)
    print("Parsed latency : ", lat)
    tf.logging.info("Parsed latency : %f", lat)


if __name__ == "__main__":
    app.run(print_latency)

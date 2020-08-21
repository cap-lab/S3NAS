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

import tensorflow as tf
from absl import app
from absl import flags

from run.estimator_model_maker import EstimatorModelMaker
from run.train_eval import TrainNEval
from util.log_utils import init_tflog

FLAGS = flags.FLAGS

# MODEL RELATED
flags.DEFINE_string(
    'model_json_path',
    default=None,
    help=('The path for model json file'))

flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))

flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override.'))

flags.DEFINE_bool(
    'use_nas_modelmaker',
    default=False,
    help=('Whether to use nas_modelmaker'))

# TRAIN & EVAL RELATED
flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "eval"}')

## dataset
flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'
flags.DEFINE_string(
    'data_dir', default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored.'))

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'input_image_size', default=224, help='Input image size.')

## epochs
flags.DEFINE_float(
    'train_epochs', default=350,
    help=('The number of epochs to use for search. Default is 8'
          ' with batch size 1024 and with warmup epochs 5'))

flags.DEFINE_float(
    'warmup_epochs', default=5, help='warm_start for learning rate.')

flags.DEFINE_float(
    'supergraph_train_epochs', default=8,
    help='supergraph training epochs. During this period, we do not apply latency loss, and we apply dropout in useconds to select various blocks.')

flags.DEFINE_float(
    'epochs_per_eval', default=5,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

## etc
flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

# TRAINING HYPERPARAMS
flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1000, help='Batch size for evaluation.')

flags.DEFINE_string(
    'sched',
    default='exponential',
    help=('lr schedule'))

flags.DEFINE_float(
    'exp_decay_epochs',
    default=2.4,
    help=('decay epochs for exponential learning rate decay')
)

flags.DEFINE_float(
    'exp_decay_factor',
    default=0.97,
    help=('decay factor for exponential learning rate decay')
)

flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'clip_gradients', default=0.0,
    help=('use gradient clipping for further stabilization'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=0.2,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_float(
    'drop_connect_rate', default=0.2,
    help=('Drop connect rate for the network. Assumes all blocks has residual connection'))

flags.DEFINE_bool(
    'shuffle_tfrecords', default=True,
    help='set this False to shard on multipe devices')

# DATA
flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

flags.DEFINE_bool(
    'use_bfloat16',
    default=False,
    help=('Whether to use bfloat16 as activation for training.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))

# TPU SPECIFIC
flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

# LOG
flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer('keep_checkpoint_max', 6, 'The number of checkpoints you want to keep')

flags.DEFINE_integer('keep_archive_max', 3, 'The number of checkpoints you want to keep')

flags.DEFINE_bool(
    'use_async_checkpointing', default=False, help=('Enable async checkpoint'))

flags.DEFINE_integer('log_step_count_steps', 256, 'The number of steps at '
                                                  'which the global step information is logged.')
## tensorboard
flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_string('log_searchableblock_tensor', default='min',
                    help=('adjust verbosity of logging info in searchableblocks. '
                          'never / min / all'))

# HARDWARE CONSTRAINT
flags.DEFINE_bool('ignore_latency', default=False, help=('ignore latency value when you log on tb'))

flags.DEFINE_float(
    'latency_lambda_val', default=15,
    help=('Lambda val for trading off loss and latency'))

flags.DEFINE_float(
    'mul_lamda', default=100,
    help=('Lambda val for trading off loss and latency'))

flags.DEFINE_float(
    'target_latency', default=5.0,
    help=('target latency we want'))

flags.DEFINE_string(
    'constraint', default='latency',
    help=('target constraint we want'))

flags.DEFINE_string(
    'constraint_parse_key', default='latency',
    help=('in the lookup files, we will access constraint values via this given key'))

flags.DEFINE_float(
    'constraint_div_unit', default=1e6,
    help=('we may have to divide constraint values due to units'))

flags.DEFINE_string(
    'constraint_lut_folder', default=None,
    help=('folder which contains constraint lookup files')
)


def main(unused_argv):
    use_tpu = FLAGS.tpu or FLAGS.use_tpu
    init_tflog(FLAGS.model_dir, use_tpu)

    if FLAGS.use_nas_modelmaker:
        FLAGS.warmup_epochs = -1
        print("We don't use warmup lr when using NASModelMaker")
        FLAGS.drop_connect_rate = 0
        print("Setting drop_connect_rate as 0. We don't use drop_connect when search")
        model_fn = EstimatorModelMaker().get_model_fn()
    else:
        FLAGS.ignore_latency = True
        print("We don't care about latency when we train")
        FLAGS.log_searchableblock_tensor = 'never'
        model_fn = EstimatorModelMaker().get_model_fn()

    trainer = TrainNEval(model_fn)
    if FLAGS.mode == 'train_and_eval':
        trainer.train_and_eval()
    elif FLAGS.mode == 'eval':
        trainer.eval()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)

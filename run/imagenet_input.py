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

import abc
import functools
import os

import tensorflow as tf
from absl import flags

import preprocessing

FLAGS = flags.FLAGS


def build_image_serving_input_fn(image_size):
    """Builds a serving input fn for raw images."""

    def _image_serving_input_fn():
        """Serving input fn for raw images."""

        def _preprocess_image(image_bytes):
            """Preprocess a single raw image."""
            image = preprocessing.preprocess_image(
                image_bytes=image_bytes, is_training=False, image_size=image_size)
            return image

        image_bytes_list = tf.placeholder(
            shape=[None],
            dtype=tf.string,
        )
        images = tf.map_fn(
            _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver(
            images, {'image_bytes': image_bytes_list})

    return _image_serving_input_fn


class ImageNetTFExampleInput(object):
    """Base class for ImageNet input_fn generator.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      num_cores: `int` for the number of TPU cores
      image_size: `int` for image size (both width and height).
      transpose_input: 'bool' for whether to use the double transpose trick
      include_background_label: If true, label #0 is reserved for background.
      autoaugment_name: `string` that is the name of the autoaugment policy
          to apply to the image. If the value is `None` autoaugment will not be
          applied.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 is_training,
                 use_bfloat16,
                 num_cores=8,
                 image_size=224,
                 transpose_input=False,
                 include_background_label=False,
                 autoaugment_name=None):
        self.image_preprocessing_fn = preprocessing.preprocess_image
        self.is_training = is_training
        self.use_bfloat16 = use_bfloat16
        self.num_cores = num_cores
        self.transpose_input = transpose_input
        self.image_size = image_size
        self.include_background_label = include_background_label
        self.autoaugment_name = autoaugment_name

    def set_shapes(self, batch_size, images, labels):
        """Statically set the batch_size dimension."""
        if self.transpose_input:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([None, None, None, batch_size])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))
        else:
            images.set_shape(images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])))
            labels.set_shape(labels.get_shape().merge_with(
                tf.TensorShape([batch_size])))

        return images, labels

    def dataset_parser(self, value):
        """Parses an image and its label from a serialized ResNet-50 TFExample.

        Args:
          value: serialized string containing an ImageNet TFExample.

        Returns:
          Returns a tuple of (image, label) from the TFExample.
        """
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

        image = self.image_preprocessing_fn(
            image_bytes=image_bytes,
            is_training=self.is_training,
            image_size=self.image_size,
            use_bfloat16=self.use_bfloat16,
            autoaugment_name=self.autoaugment_name)

        # The labels will be in range [1,1000], 0 is reserved for background
        label = tf.cast(
            tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

        if not self.include_background_label:
            # Subtract 1 if the background label is discarded.
            label -= 1

        return image, label

    @abc.abstractmethod
    def make_source_dataset(self, index, num_hosts):
        """Makes dataset of serialized TFExamples.

        The returned dataset will contain `tf.string` tensors, but these strings are
        serialized `TFExample` records that will be parsed by `dataset_parser`.

        If self.is_training, the dataset should be infinite.

        Args:
          index: current host index.
          num_hosts: total number of hosts.

        Returns:
          A `tf.data.Dataset` object.
        """
        return

    def input_fn(self, params):
        """Input function which provides a single batch for train or eval.

        Args:
          params: `dict` of parameters passed from the `TPUEstimator`.
              `params['batch_size']` is always provided and should be used as the
              effective batch size.

        Returns:
          A `tf.data.Dataset` object.
        """
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contrib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1

        dataset = self.make_source_dataset(current_host, num_hosts)

        # Use the fused map-and-batch operation.
        #
        # For XLA, we must used fixed shapes. Because we repeat the source training
        # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
        # batches without dropping any training examples.
        #
        # When evaluating, `drop_remainder=True` prevents accidentally evaluating
        # the same image twice by dropping the final batch if it is less than a full
        # batch size. As long as this validation is done with consistent batch size,
        # exactly the same images will be used.
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=self.num_cores, drop_remainder=True))

        # Transpose for performance on TPU
        if self.transpose_input:
            dataset = dataset.map(
                lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
                num_parallel_calls=self.num_cores)

        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset


class ImageNetInput(ImageNetTFExampleInput):
    """Generates ImageNet input_fn from a series of TFRecord files.

    The training data is assumed to be in TFRecord format with keys as specified
    in the dataset_parser below, sharded across 1024 files, named sequentially:

        train-00000-of-01024
        train-00001-of-01024
        ...
        train-01023-of-01024

    The validation data is in the same format but sharded in 128 files.

    The format of the data required is created by the script at:
        https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
    """

    def __init__(self,
                 is_training,
                 use_bfloat16,
                 transpose_input,
                 data_dir,
                 image_size=224,
                 num_parallel_calls=64,
                 cache=False,
                 include_background_label=False,
                 autoaugment_name=None):
        """Create an input from TFRecord files.

        Args:
          is_training: `bool` for whether the input is for training
          use_bfloat16: If True, use bfloat16 precision; else use float32.
          transpose_input: 'bool' for whether to use the double transpose trick
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
          image_size: `int` for image size (both width and height).
          num_parallel_calls: concurrency level to use when reading data from disk.
          cache: if true, fill the dataset by repeating from its cache.
          include_background_label: if true, label #0 is reserved for background.
          autoaugment_name: `string` that is the name of the autoaugment policy
              to apply to the image. If the value is `None` autoaugment will not be
              applied.
        """
        super(ImageNetInput, self).__init__(
            is_training=is_training,
            image_size=image_size,
            use_bfloat16=use_bfloat16,
            transpose_input=transpose_input,
            include_background_label=include_background_label,
            autoaugment_name=autoaugment_name)
        self.data_dir = data_dir
        if self.data_dir == 'null' or not self.data_dir:
            self.data_dir = None
        self.num_parallel_calls = num_parallel_calls
        self.cache = cache

    def _get_null_input(self, data):
        """Returns a null image (all black pixels).

        Args:
          data: element of a dataset, ignored in this method, since it produces
              the same null image regardless of the element.

        Returns:
          a tensor representing a null image.
        """
        del data  # Unused since output is constant regardless of input
        return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
        if self.use_bfloat16 else tf.float32)

    def dataset_parser(self, value):
        """See base class."""
        if not self.data_dir:
            return value, tf.constant(0, tf.int32)
        return super(ImageNetInput, self).dataset_parser(value)

    def make_source_dataset(self, index, num_hosts):
        """See base class."""
        if not self.data_dir:
            tf.logging.info('Undefined data_dir implies null input')
            return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')

        if FLAGS.shuffle_tfrecords and not self.is_training:
            assert num_hosts == 1, "I thought I won't make num_hosts larger than 1"
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        else:
            # For multi-host training, we want each hosts to always process the same
            # subset of files.  Each host only sees a subset of the entire dataset,
            # allowing us to cache larger datasets in memory.
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
            dataset = dataset.shard(num_hosts, index)

        if self.is_training and not self.cache:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

        if self.cache and self.is_training:
            dataset = dataset.cache().apply(
                tf.contrib.data.shuffle_and_repeat(1024 * 16))
        elif self.is_training:
            dataset = dataset.shuffle(1024)
        return dataset

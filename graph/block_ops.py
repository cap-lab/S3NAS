"""
Classes for basic Blocks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy
from functools import reduce

import tensorflow as tf
from absl import flags
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function

from graph.graph_utils import conv_kernel_initializer, dense_kernel_initializer, get_kerasconv_kern_shape_info, \
    get_denseconv_kern_shape_info
from util.math_utils import ceil_div

FLAGS = flags.FLAGS


class MyNetComponents(object):
    def get_output_shape(self, input_shape=(224, 224, 3)):
        raise NotImplementedError

    def get_params(self, input_shape=(224, 224, 3)):
        return self._get_('params', input_shape)

    def get_params_considering_bnbias(self, input_shape=(224, 224, 3)):
        """
        You can get trainable parameters by using tensorflow.
        num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        Use this only for debugging purpose.
        """
        return self._get_('params_considering_bnbias', input_shape)

    def get_flops(self, input_shape=(224, 224, 3)):
        return self._get_('flops', input_shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        raise NotImplementedError

    def _set_data_format_related_stuffs(self, global_params):
        self._data_format = global_params.data_format

        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    # def tensordict_to_write_on_tensorboard(self):
    #     raise NotImplementedError


class BasicOp(MyNetComponents):
    def get_str(self):
        """Need to call this after you built it, because I need to know about input_shape"""
        assert self.built
        return self.construct_str()

    @classmethod
    def construct_str(cls, *args, **kwargs):
        return ''


class Conv2D(tf.keras.layers.Conv2D, BasicOp):
    def __init__(self, global_params, output_filters, kernel_size, strides=1, padding='same', use_bias=False,
                 input_filters=None, **kwargs):
        """
        :param kernel_size: can be a int or list or tuple.
        :param strides: can be a int or list or tuple.
        # I found that they use "normalize_tuple" function to support int.
        """
        if input_filters:
            self._input_filters = input_filters
        super(Conv2D, self).__init__(
            filters=output_filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=conv_kernel_initializer,
            padding=padding,
            data_format=global_params.data_format,
            use_bias=use_bias,
            **kwargs
        )

    def get_output_shape(self, input_shape=(224, 224, 3)):
        assert self.data_format == 'channels_last'
        assert self.padding == 'same'
        shape = (ceil_div(input_shape[0], self.strides[0]), ceil_div(input_shape[1], self.strides[1]), self.filters)
        return tuple(int(d) for d in shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        assert self.data_format == 'channels_last'
        if what == 'params':
            weight_params = input_shape[-1] * self.filters * self.kernel_size[0] * self.kernel_size[1]
            return weight_params
        elif what == 'params_considering_bnbias':
            weight_params = self._get_('params', input_shape)
            bias_params = 0
            if self.use_bias:
                bias_params = int(self.filters)
            return weight_params + bias_params
        elif what == 'flops':
            shape = self.get_output_shape(input_shape)
            flops = reduce(lambda x, y: x * y, shape + self.kernel_size + (input_shape[-1],))
            return flops
        else:
            raise NotImplementedError

    def get_str(self):
        assert self.built
        k, _, input_filters, output_filters = get_kerasconv_kern_shape_info(self)
        return self.construct_str(k, self.strides, input_filters, output_filters)

    @classmethod
    def construct_str(cls, kernel_size, strides, input_filters, output_filters):
        args_list = [
            'k%d' % kernel_size,
            's%d%d' % (strides[0], strides[0]),
            'i%d' % input_filters,
            'o%d' % output_filters,
        ]
        return '_'.join(args_list)

    def call(self, inputs, training=None):
        return super(Conv2D, self).call(inputs)


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, BasicOp):
    def __init__(self, global_params, kernel_size, strides=1, padding='same', use_bias=False, **kwargs):
        """
        :param kernel_size: can be a int or list or tuple.
        :param strides: can be a int or list or tuple.
        """
        super(DepthwiseConv2D, self).__init__(
            kernel_size=kernel_size,
            strides=strides,
            depthwise_initializer=conv_kernel_initializer,
            padding=padding,
            data_format=global_params.data_format,
            use_bias=use_bias,
            **kwargs
        )

    def get_output_shape(self, input_shape=(224, 224, 3)):
        assert self.data_format == 'channels_last'
        assert self.padding == 'same'
        shape = (ceil_div(input_shape[0], self.strides[0]), ceil_div(input_shape[1], self.strides[1]), input_shape[2])
        return tuple(int(d) for d in shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        assert self.data_format == 'channels_last'
        if what == 'params':
            weight_params = int(input_shape[-1] * self.kernel_size[0] * self.kernel_size[1])
            return weight_params
        elif what == 'params_considering_bnbias':
            assert not self.use_bias
            weight_params = self._get_('params', input_shape)
            return weight_params
        elif what == 'flops':
            shape = self.get_output_shape(input_shape)
            flops = reduce(lambda x, y: x * y, shape + self.kernel_size)
            return flops
        else:
            raise NotImplementedError

    def call(self, inputs, training=None):
        return super(DepthwiseConv2D, self).call(inputs)


class GlobalAvgPool(tf.keras.layers.GlobalAveragePooling2D, BasicOp):
    def __init__(self, global_params, keep_dims=False):
        self._set_data_format_related_stuffs(global_params)
        self._keep_dims = keep_dims
        super(GlobalAvgPool, self).__init__()

    def build(self, input_shape):
        """
        Records input_filters to support get_str.
        """
        self._input_filters = int(input_shape[self._channel_axis])
        super(GlobalAvgPool, self).build(input_shape)

    def call(self, x, training=None):
        assert self._input_filters == int(x.shape[self._channel_axis])
        return tf.reduce_mean(x, self._spatial_dims, keep_dims=self._keep_dims)

    def get_output_shape(self, input_shape=(224, 224, 3)):
        assert self._data_format == 'channels_last'
        shape = (1, 1, input_shape[2])
        return tuple(int(d) for d in shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        assert self._data_format == 'channels_last'
        if what in ['params', 'params_considering_bnbias', 'flops']:
            return 0
        else:
            raise NotImplementedError

    def get_str(self):
        assert self.built
        return self.construct_str(self._input_filters)

    @classmethod
    def construct_str(cls, input_filters):
        args_list = [
            'i%d' % input_filters,
        ]
        return '_'.join(args_list)


class Dense(tf.keras.layers.Dense, BasicOp):
    def __init__(self, num_classes, **kwargs):
        """
        :param kernel_size: can be a int or list or tuple.
        :param strides: can be a int or list or tuple.
        """
        super(Dense, self).__init__(
            num_classes,
            kernel_initializer=dense_kernel_initializer,
            **kwargs
        )

    def get_output_shape(self, input_shape=(224, 224, 3)):
        return (self.units)

    def _get_(self, what, input_shape=(224, 224, 3)):
        if what in ['params', 'params_considering_bnbias']:
            return int(input_shape[-1] * self.units +
                       (self.units if self.use_bias else 0))
        elif what == 'flops':
            flops = self.units * input_shape[-1]
            return flops
        else:
            raise NotImplementedError

    def get_str(self):
        """Need to call this after you built it, because I need to know about input_shape"""
        assert self.built
        input_filters, output_filters = get_denseconv_kern_shape_info(self)
        return self.construct_str(input_filters, output_filters)

    @classmethod
    def construct_str(cls, input_filters, output_filters):
        args_list = [
            'i%d' % input_filters,
            'o%d' % output_filters,
        ]
        return '_'.join(args_list)

    def call(self, inputs, training=None):
        return super(Dense, self).call(inputs)


class Activation(tf.keras.layers.Layer, BasicOp):
    def __init__(self, global_params, act_fn):
        self.act_fn = act_fn
        super(Activation, self).__init__()

    def get_output_shape(self, input_shape=(224, 224, 3)):
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        if what in ['params_considering_bnbias', 'params', 'flops']:
            return 0
        else:
            raise NotImplementedError

    def call(self, inputs, training=None):
        return self.act_fn(inputs)


### Code below originated from EfficientNet and MnasNet repo in https://github.com/tensorflow/tpu
### Modified by thnkinbtfly
class TpuBatchNormalization(tf.layers.BatchNormalization, BasicOp):
    # class TpuBatchNormalization(tf.layers.BatchNormalization):
    """Cross replica batch normalization. Mostly got idea from EfficientNet utils code"""

    def __init__(self, global_params, fused=False, **kwargs):
        self._set_data_format_related_stuffs(global_params)
        kwargs['axis'] = self._channel_axis
        kwargs['momentum'] = global_params.batch_norm_momentum
        kwargs['epsilon'] = global_params.batch_norm_epsilon
        if fused in (True, None):
            raise ValueError('TpuBatchNormalization does not support fused=True.')
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                                 % (num_shards, num_shards_per_group))
            num_groups = num_shards // num_shards_per_group
            group_assignment = [[
                x for x in range(num_shards) if x // num_shards_per_group == y
            ] for y in range(num_groups)]
        return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype)

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims)

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
            num_shards_per_group = 1
        else:
            num_shards_per_group = max(8, num_shards // 8)
        tf.logging.info('TpuBatchNormalization with num_shards_per_group %s',
                        num_shards_per_group)
        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(
                shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group)
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return (group_mean, group_variance)
        else:
            return (shard_mean, shard_variance)

    def get_output_shape(self, input_shape=(224, 224, 3)):
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        if what == 'params_considering_bnbias':
            assert self._data_format == 'channels_last'
            return int(input_shape[-1] * 2)
        elif what in ['params', 'flops']:
            return 0
        else:
            raise NotImplementedError


class MixConv(tf.keras.layers.Layer, BasicOp):
    def __init__(self, global_params, kernel_sizes, filter_splits, strides, padding='same', use_bias=False):
        self._convs, self._filter_splits = [], []
        self._padding = padding
        self._strides = strides
        self._set_data_format_related_stuffs(global_params)
        assert len(filter_splits) == len(kernel_sizes)
        for filters, k_size in zip(filter_splits, kernel_sizes):
            self._filter_splits.append(filters)
            self._convs.append(DepthwiseConv2D(global_params, k_size, strides, padding=padding, use_bias=use_bias))
        super(MixConv, self).__init__()

    def build(self, input_shape):
        for filters, conv in zip(self._filter_splits, self._convs):
            shape = list(deepcopy(input_shape))
            shape[self._channel_axis] = filters
            conv.build(tf.TensorShape(shape))
        super(MixConv, self).build(input_shape)

    def call(self, inputs, training=None):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        x_splits = tf.split(inputs, self._filter_splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x

    def get_output_shape(self, input_shape=(224, 224, 3)):
        """ Same as DepthwiseConv """
        assert self._data_format == 'channels_last'
        assert self._padding == 'same'
        shape = (ceil_div(input_shape[0], self._strides[0]), ceil_div(input_shape[1], self._strides[1]), input_shape[2])
        return tuple(int(d) for d in shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        assert self._data_format == 'channels_last'
        result = 0
        chan_splits = self._filter_splits
        for conv, chan_split in zip(self._convs, chan_splits):
            split_input = (input_shape[0], input_shape[1], chan_split)
            result += conv._get_(what, split_input)

        return result

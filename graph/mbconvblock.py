from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
from absl import flags

import graph.graph_utils
from graph.block_ops import DepthwiseConv2D, Conv2D, MixConv, MyNetComponents
from graph.blockargs import BlockArgsDecoder
from graph.op_collections import SE, build_conv_bn_act
from graph.searchableops import SearchableDwiseConv, SearchableBlock, SearchableMixConv
from util.math_utils import argmax
from util.utils import is_iterable

FLAGS = flags.FLAGS


class MBConvBlock(tf.keras.layers.Layer, MyNetComponents):
    def __init__(self, block_args, global_params):
        self._block_args = block_args
        self._global_params = global_params
        self.act_fn = BlockArgsDecoder.get_act_fn_from_string(
            block_args.get("act_fn") if block_args.get("act_fn") else global_params.act_fn)

        self.tensordict = {}

        self._set_data_format_related_stuffs(self._global_params)
        self._has_se = self._block_args.se_ratio is not None and self._block_args.se_ratio > 0
        self._build()
        super(MBConvBlock, self).__init__()

    def _build(self):
        expand_filters = int(self._block_args.input_filters * self._block_args.expand_ratio)
        self._build_expand(output_filters=expand_filters)
        self._build_dwise(output_filters=expand_filters)

        if self._has_se:
            self._build_se(output_filters=expand_filters)

        output_filters = self._block_args.output_filters
        self._build_proj(output_filters=output_filters)

    def _build_expand(self, output_filters):
        if self._block_args.expand_ratio != 1:
            kwargs = dict(
                cls=Conv2D.__name__,
                output_filters=output_filters,
                kernel_size=[1, 1],
                strides=[1, 1])
        else:
            kwargs = None
        self._expand_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=self.act_fn)

    def _build_dwise(self, output_filters):
        kernel_size = self._block_args.kernel_size
        kwargs = dict(
            cls=DepthwiseConv2D.__name__,
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides)

        self._dwise_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=self.act_fn)

    def _build_proj(self, output_filters):
        kwargs = dict(
            cls=Conv2D.__name__,
            output_filters=output_filters,
            kernel_size=[1, 1],
            strides=[1, 1])

        self._proj_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=None)

    def _build_se(self, output_filters):
        se_ratio = self._block_args.se_ratio
        num_reduced_filters = max(
            1, int(self._block_args.input_filters * se_ratio))
        se_inner_act_fn = BlockArgsDecoder.get_se_inner_act_fn_from_string(self._global_params.se_inner_act_fn)
        se_gating_fn = BlockArgsDecoder.get_se_gating_fn_from_string(self._global_params.se_gating_fn)

        self._se = SE(self._global_params, num_reduced_filters, output_filters, se_inner_act_fn, se_gating_fn)

    def call(self, inputs, training=True, drop_connect_rate=None):
        x = self._expand_layers(inputs, training=training)
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        x = self._dwise_layers(x, training=training)
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

        if self._has_se:
            with tf.variable_scope('se'):
                x = self._call_se(x)
                tf.logging.info('SE: %s shape: %s' % (x.name, x.shape))

        x = self._proj_layers(x, training=training)

        if self.can_add_residcon:
            if drop_connect_rate:
                x = graph.graph_utils.drop_connect(x, training, drop_connect_rate)
            x = tf.add(x, inputs)
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x

    def _call_se(self, x):
        outputs = self._se(x)

        log_excitation_names = FLAGS.log_excitation_names_containing
        if log_excitation_names:
            activations = self._se.activations
            this_excitation_name = activations.name.replace(":", "_")
            if 'all' in log_excitation_names or any(
                    [log_name in this_excitation_name for log_name in log_excitation_names]):
                self.tensordict[this_excitation_name] = activations

        return outputs

    def get_output_shape(self, input_shape=(224, 224, 3)):
        for layers in [self._expand_layers, self._dwise_layers, self._proj_layers]:
            input_shape = layers.get_output_shape(input_shape)
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        block_layers = [self._expand_layers, self._dwise_layers, self._proj_layers]
        if self._has_se:
            block_layers.insert(2, self._se)

        result = 0
        for layers in block_layers:
            result += layers._get_(what, input_shape)
            input_shape = layers.get_output_shape(input_shape)
        return result

    @property
    def can_add_residcon(self):
        if self._block_args.id_skip and all(s == 1 for s in self._block_args.strides) and \
                self._block_args.input_filters == self._block_args.output_filters:
            return True
        else:
            return False

    def tensordict_to_write_on_tensorboard(self):
        """
        TF Tensors to write on tensorboard.
        If you return {name, tensor},
        On tensorboard, the tensor will be written under net/name
        """
        return self.tensordict


class MixConvBlock(MBConvBlock):
    def _build(self):
        input_filters = self._block_args.input_filters
        expand_ratios = self._block_args.expand_ratio
        total_expand_ratio = functools.reduce(lambda x, y: x + y, expand_ratios)
        self.expand_filters = int(input_filters * total_expand_ratio)
        self._build_expand(output_filters=self.expand_filters)
        self._build_dwise(output_filters=self.expand_filters)

        if self._has_se:
            self._build_se(output_filters=self.expand_filters)

        output_filters = self._block_args.output_filters
        self._build_proj(output_filters=output_filters)

    def _build_dwise(self, output_filters):
        kernel_sizes = self._block_args.kernel_size
        input_filters = self._block_args.input_filters
        expand_ratios = self._block_args.expand_ratio
        filter_splits = [int(input_filters * er) for er in expand_ratios]
        assert self.expand_filters == sum(filter_splits)
        kwargs = dict(
            cls=MixConv.__name__,
            kernel_sizes=kernel_sizes,
            filter_splits=filter_splits,
            strides=self._block_args.strides)

        self._dwise_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=self.act_fn)


class SearchableMBConvBlock(MBConvBlock, SearchableBlock):
    def __init__(self, block_args, global_params):
        super(SearchableMBConvBlock, self).__init__(block_args, global_params)
        self.is_supergraph_training_tensor = global_params.is_supergraph_training_tensor
        self._change_to_searchableblock(global_params)

    searchable_op_cls = SearchableDwiseConv

    @property
    def core_op(self):
        return self._dwise_layers['conv']

    @core_op.setter
    def core_op(self, value):
        self._dwise_layers['conv'] = value

    def conv_type_for(self, kernel_size, expand_ratio):
        return MBConvBlock.__name__

    @property
    def _k_sel_list(self):
        kernel_size = self._block_args.kernel_size
        k_split_unit = 2
        k_min_size = 3
        return list(range(k_min_size, kernel_size + 1, k_split_unit))

    @property
    def _er_sel_list_with_0_maybe(self):
        er_sel_list = []
        can_zeroout_to_make_skipop = self.can_add_residcon
        if can_zeroout_to_make_skipop:
            er_sel_list.append(0)

        expand_ratio = self._block_args.expand_ratio
        assert int(expand_ratio) == expand_ratio, "We only support int expand_ratios"
        er_split_unit = 2
        er_sel_list.extend(range(er_split_unit, int(expand_ratio) + 1, er_split_unit))
        return er_sel_list

    @property
    def core_C_sel_list(self):
        input_filters = self._block_args.input_filters
        return [input_filters * er for er in self._er_sel_list_with_0_maybe]

    def useconds_for_ksize_expandratio(self, kernel_size, expand_ratio):
        channels = self._block_args.input_filters * expand_ratio
        return self.core_op.useconds_for_ksize_channels(kernel_size, channels)

    def ksize_expandratio_of_useconds(self, useconds_kern, useconds_chan):
        input_filters = self._block_args.input_filters
        k, C = self.core_op.ksize_channels_of_useconds(useconds_kern, useconds_chan)
        er = C // input_filters
        return k, er

    def tensordict_to_write_on_tensorboard(self):
        """Because of multiple inheritance..."""
        self.tensordict.update(self.core_op.tensordict_to_write_on_tensorboard())
        return self.tensordict


class SearchableMixConvBlock(MixConvBlock, SearchableBlock):
    def __init__(self, block_args, global_params):
        super(SearchableMixConvBlock, self).__init__(block_args, global_params)
        self.is_supergraph_training_tensor = global_params.is_supergraph_training_tensor
        self._change_to_searchableblock(global_params)

    searchable_op_cls = SearchableMixConv

    @property
    def core_op(self):
        return self._dwise_layers['conv']

    @core_op.setter
    def core_op(self, value):
        self._dwise_layers['conv'] = value

    def conv_type_for(self, kernel_size, expand_ratio):
        simplified_ksize, simplified_er = self.simplify_kern_expandratio(kernel_size, expand_ratio)
        if isinstance(simplified_ksize, int):
            return MBConvBlock.__name__
        else:
            return MixConvBlock.__name__

    @property
    def _k_sel_list(self):
        k_list_per_branches = []
        k_split_unit = 2
        k_min_size = 3
        for ksize in self._block_args.kernel_size:
            k_list_per_branches.append(list(range(k_min_size, ksize + 1, k_split_unit)))
        return k_list_per_branches

    @property
    def _er_sel_list_with_0_maybe(self):
        er_split_unit = 2

        def build_er_sel_list(expand_ratio, can_zeroout):
            er_sel_list = []
            if can_zeroout:
                er_sel_list.append(0)
            assert int(expand_ratio) == expand_ratio, "We only support int expand_ratios"
            er_sel_list.extend(range(er_split_unit, int(expand_ratio) + 1, er_split_unit))
            return er_sel_list

        largest_kernel_index = argmax(self._block_args.kernel_size)
        er_list_per_branches = []
        for i, er in enumerate(self._block_args.expand_ratio):
            unable_to_zeroout = (i == largest_kernel_index) and not self.can_add_residcon
            er_list_per_branches.append(build_er_sel_list(er, can_zeroout=(not unable_to_zeroout)))

        return er_list_per_branches

    @property
    def core_C_sel_list(self):
        input_filters = self._block_args.input_filters
        C_list_per_branches = []
        for er_list in self._er_sel_list_with_0_maybe:
            C_list_per_branches.append([input_filters * er for er in er_list])
        return C_list_per_branches

    def get_listof_possible_ksize_expandratio_with_0_maybe(self):
        result = []
        for (k, C) in self.core_op.get_listof_possible_ksize_channels():
            result.append((k, self._convert_to_er(C)))
        return result

    def useconds_for_ksize_expandratio(self, kernel_size, expand_ratio):
        k, er = self.simplify_kern_expandratio(kernel_size, expand_ratio)
        return self.core_op.useconds_for_ksize_channels(k, self._convert_to_channels(er))

    def simplify_kern_expandratio(self, kernel_size, expand_ratio):
        k, c = self.core_op.simplify_k_C(kernel_size, self._convert_to_channels(expand_ratio))
        return k, self._convert_to_er(c)

    def ksize_expandratio_of_useconds(self, useconds_kern, useconds_chan):
        """returns simplified k_sizes and expand_ratios"""
        k_list, C_list = self.core_op.ksize_channels_of_useconds(useconds_kern, useconds_chan)
        er_list = self._convert_to_er(C_list)
        return k_list, er_list

    def tensordict_to_write_on_tensorboard(self):
        """Because of multiple inheritance..."""
        self.tensordict.update(self.core_op.tensordict_to_write_on_tensorboard())
        return self.tensordict

    def _convert_to_er(self, channels):
        input_filters = self._block_args.input_filters
        if is_iterable(channels):
            channels = tuple([c // input_filters for c in channels])
        else:
            channels //= input_filters
        return channels

    def _convert_to_channels(self, expand_ratio):
        input_filters = self._block_args.input_filters
        if is_iterable(expand_ratio):
            channels = tuple([er * input_filters for er in expand_ratio])
        else:
            channels = expand_ratio * input_filters
        return channels

    @classmethod
    def get_summary_prefix(cls, branch_idx):
        return cls.searchable_op_cls.get_summary_prefix(branch_idx)

    @classmethod
    def get_label(cls, branch_idx, summary_name):
        label = cls.searchable_op_cls.get_summary_prefix(branch_idx) + summary_name
        return label

    @property
    def num_branches(self):
        return self.core_op.num_branches

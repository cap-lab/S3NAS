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

import tensorflow as tf
from absl import flags

import graph
from graph.block_ops import GlobalAvgPool, Activation, MyNetComponents, Dense
from graph.blockargs import BlockArgsDecoder
from graph.op_collections import build_conv_bn_act
from util.string_utils import add_pre_suf_to_keys_of

FLAGS = flags.FLAGS

# log related
flags.DEFINE_list(
    'log_excitation_names_containing', default=None,
    help=('log values of excitations. If value is all, all SE blocks will be logged.'
          'If value is stages_0,stages_1 then SE blocks with names containing stage_0 and stage_1 will be logged')
)


class JsonNet(tf.keras.Model, MyNetComponents):
    def __init__(self, model_args=None, global_params=None):
        """Initializes an `JsonNet` instance.
        Args:
          model_args: A json args expressing model.
          global_params: GlobalParams, a set of global parameters.
        """

        super(JsonNet, self).__init__()
        self.tensordict = {}
        self._global_params = global_params
        self.act_fn = BlockArgsDecoder.get_act_fn_from_string(global_params.act_fn)
        self._set_data_format_related_stuffs(global_params)
        self.endpoints = None

        self._build(model_args)

    def _build(self, model_args):
        self._build_stem_layers(model_args)

        self._stages = []
        for stage_args in model_args.stages_args:
            self._stages.append(graph.construct_with_args(self._global_params, **stage_args))

        self._build_head_layers(model_args)
        self._classifier = Dense(self._global_params.num_classes)

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def _build_stem_layers(self, model_args):
        self._stem_layers = build_conv_bn_act(self._global_params, model_args.first_conv, add_bn=True,
                                              act_fn=self.act_fn)

    def _build_head_layers(self, model_args):
        self._head_layers = build_conv_bn_act(self._global_params, model_args.feature_mix_layer, add_bn=True,
                                              act_fn=self.act_fn)
        self._head_layers.append(GlobalAvgPool(self._global_params, keep_dims=True))

    def iter_stem_layers(self):
        for layer in self._stem_layers:
            yield layer

    def iter_head_layers(self):
        for layer in self._head_layers:
            yield layer

    def call(self, inputs, training=True):
        with tf.variable_scope('stem'):
            outputs = self._stem_layers(inputs, training=training)
            tf.logging.info('Stem: %s shape: %s' % (outputs.name, outputs.shape))

        total_blocks = 0
        for stage in self._stages:
            total_blocks += stage.num_blocks

        blocks_upto_last_stage = 0
        for idx, stage in enumerate(self._stages):
            blocks_in_stage = stage.num_blocks
            with tf.variable_scope('stages_%s' % idx):
                drop_rate = self._global_params.drop_connect_rate  # This can be None.
                if drop_rate:
                    drop_rate_min = drop_rate * float(blocks_upto_last_stage) / total_blocks
                    drop_rate_max = drop_rate * float(blocks_upto_last_stage + blocks_in_stage) / total_blocks
                    blocks_upto_last_stage += blocks_in_stage
                    drop_range = [drop_rate_min, drop_rate_max]
                    tf.logging.info('stage_%s drop_connect_rate: %s' % (idx, drop_range))
                    outputs = stage(
                        outputs, training=training, drop_connect_rate_range=drop_range)
                else:
                    outputs = stage(
                        outputs, training=training, drop_connect_rate_range=None)

        with tf.variable_scope('head'):
            outputs = self._head_layers(outputs, training=training)
            tf.logging.info('Head: %s shape: %s' % (outputs.name, outputs.shape))

        outputs = tf.squeeze(outputs, self._spatial_dims)
        if self._dropout:
            outputs = self._dropout(outputs, training=training)
        outputs = self._classifier(outputs)
        tf.logging.info('Built classifier %s with : %s' % (outputs.name, outputs.shape))

        return outputs

    def _get_(self, what, input_shape=(224, 224, 3)):
        subparts = []
        subparts.extend(self.iter_stem_layers())
        subparts.extend(self._stages)
        subparts.extend(self.iter_head_layers())
        subparts.append(self._classifier)

        return self._get_from_subparts(subparts, what, input_shape)

    def _get_from_subparts(self, subparts, what, input_shape):
        result = 0
        for subpart in subparts:
            result += subpart._get_(what, input_shape)
            input_shape = subpart.get_output_shape(input_shape)
        return result

    # TODO: have to combine this ftn with get_label... they are closely related each other. Need to follow the hierarchy
    def tensordict_to_write_on_tensorboard(self):
        for idx, stage in enumerate(self._stages):
            global_prefix = 'net/'
            prefix, suffix = global_prefix + 'stage' + str(idx) + '_', ''
            self.tensordict.update(
                add_pre_suf_to_keys_of(stage.tensordict_to_write_on_tensorboard(), prefix, suffix))
        return self.tensordict

    @classmethod
    def get_label(cls, stage_idx, block_idx, summary_name):
        prefix = 'net/'
        label = prefix + 'stage%d_block%d_%s' % (stage_idx, block_idx, summary_name)
        return label

    @classmethod
    def remove_label(cls, label):
        assert isinstance(label, str)
        prefix = 'net/'
        assert label.startswith(prefix)
        label = label[len(prefix):]

        prefix = cls.extract_stage_block_prefix(label)
        label = label.replace(prefix, "")

        return label

    @classmethod
    def extract_stage_block_prefix(cls, label):
        import re
        prefix_pattern = re.compile("stage(\d*)_block(\d*)_")
        m = prefix_pattern.search(label)
        prefix = m.group()
        return prefix


class MBV3CompatJsonNet(JsonNet):
    """
    JsonNet which both supports old-style han model and OFA-style han model.
    """

    def _build_head_layers(self, model_args):
        final_expand_layer = model_args.get('final_expand_layer')
        if final_expand_layer:
            self._head_layers = build_conv_bn_act(self._global_params, model_args.final_expand_layer, add_bn=True,
                                                  act_fn=self.act_fn)
            gap = GlobalAvgPool(self._global_params, keep_dims=True)
            feature_mix_layer = graph.construct_with_args(self._global_params, **model_args.feature_mix_layer)
            final_act_fn = Activation(self._global_params, self.act_fn)
            self._head_layers.extend([gap, feature_mix_layer, final_act_fn])
        else:
            super(MBV3CompatJsonNet, self)._build_head_layers(model_args)


Net = MBV3CompatJsonNet

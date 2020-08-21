# This code extends codebase from followings:
# https://github.com/dstamoulis/single-path-nas
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from graph.block_ops import MyNetComponents
from graph.convblock import *
from graph.mbconvblock import *
from util.string_utils import add_pre_suf_to_keys_of


def _get_conv_cls(conv_type, get_cls_num=False):
    conv_block_map = {0: MBConvBlock, 12: SearchableMBConvBlock,
                      15: ConvBlock_kxk1x1, 16: SearchableConvBlock_kxk1x1,
                      100: MixConvBlock, 102: SearchableMixConvBlock,
                      }
    if isinstance(conv_type, int):
        if get_cls_num:
            return conv_type
        else:
            return conv_block_map[conv_type]
    else:
        assert isinstance(conv_type, str)
        for num, cls in conv_block_map.items():
            if conv_type == cls.__name__:
                if get_cls_num:
                    return num
                else:
                    return cls


class BasicStage(tf.keras.Model, MyNetComponents):
    """
    BasicStage, which contains multiple blocks/ops
    """

    def __init__(self, global_params, blocks_args):
        super(BasicStage, self).__init__()
        self.tensordict = {}
        self._global_params = global_params
        self._blocks = []
        self.changed_spatial_dimension = False
        self._build(blocks_args)

    def _build(self, blocks_args):
        for block_args in blocks_args:
            assert block_args.num_repeat > 0
            conv_cls = _get_conv_cls(block_args.conv_type)

            # The first block needs to take care of stride and filter size increase.
            repeats = block_args.num_repeat
            block_args = block_args._replace(num_repeat=1)
            self._add_block(conv_cls(block_args, self._global_params))

            if repeats > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in range(repeats - 1):
                self._add_block(conv_cls(block_args, self._global_params))

    def _add_block(self, conv_block):
        self._blocks.append(conv_block)

    @property
    def num_blocks(self):
        return len(self._blocks)

    def call(self, inputs, training=True, drop_connect_rate_range=None):
        outputs = inputs

        total_blocks = self.num_blocks
        for idx, block in enumerate(self._blocks):
            with tf.variable_scope('blocks_%s' % idx):
                if drop_connect_rate_range:
                    drop_min, drop_max = drop_connect_rate_range
                    drop_rate = drop_min + float(drop_max - drop_min) * idx / total_blocks
                    outputs = block(
                        outputs, training=training, drop_connect_rate=drop_rate)
                else:
                    outputs = block(
                        outputs, training=training, drop_connect_rate=None)

        return outputs

    def tensordict_to_write_on_tensorboard(self):
        for idx, block in enumerate(self._blocks):
            prefix, suffix = 'block' + str(idx) + '_', ''
            self.tensordict.update(
                add_pre_suf_to_keys_of(block.tensordict_to_write_on_tensorboard(), prefix, suffix))
        return self.tensordict

    def get_output_shape(self, input_shape=(224, 224, 3)):
        for block in self._blocks:
            input_shape = block.get_output_shape(input_shape)
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        result = 0
        for block in self._blocks:
            result += block._get_(what, input_shape)
            input_shape = block.get_output_shape(input_shape)
        return result

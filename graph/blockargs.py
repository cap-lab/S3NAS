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

import collections
import functools
import re

import tensorflow as tf

from graph.graph_utils import swish, hsigmoid
from util.utils import AttrDict, is_iterable


class BlockArgsDecoder(object):
    """A class of decoder to get model configuration."""

    @classmethod
    def _decode_blocks_string(self, block_string):
        """Gets a block through a string notation of arguments.

        E.g. r2_k3_s22_e1_i32_o16_se0.25_noskip_relu: r - number of repeat blocks,
        k - kernel size, s - strides , e - expansion ratio, i - input filters,
        o - output filters, se - squeeze/excitation ratio
        relu - relu/sw/hsw. act_fn
        """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        act_fn = self.decode_act_fn_in_string(block_string)

        def _parse_multiple_(ss, to=int):
            """
            parses mixnet-like list (I use , rather than .) input with backward compatability
            mostly got idea from tensorflow mnasnet code
            :param ss: list of numbers, which are splitted by ',' ex) "3,5" means [3, 5]
            """
            l = tuple([to(k) for k in ss.split(',')])
            if len(l) == 1:
                return l[0]
            else:
                return l

        return AttrDict(
            BlockArgs(
                kernel_size=_parse_multiple_(options['k'], int),
                num_repeat=int(options['r']),
                input_filters=int(options['i']),
                output_filters=int(options['o']),
                expand_ratio=_parse_multiple_(options['e'], float),
                id_skip=('noskip' not in block_string),
                se_ratio=float(options['se']) if 'se' in options else None,
                strides=[int(options['s'][0]), int(options['s'][1])],
                act_fn=act_fn,
                conv_type=int(options['c']) if 'c' in options else 0
            )._asdict()
        )

    _supported_act_fn_dict = {
        'hsw': functools.partial(swish, use_hard=True, use_native=False),
        'sw': functools.partial(swish, use_native=False),  # To avoid memory error in python2 TPU
        'relu6': tf.nn.relu6,
        'relu': tf.nn.relu,
        'hsigmoid': hsigmoid,
        'sigmoid': tf.nn.sigmoid,
    }

    @classmethod
    def decode_act_fn_in_string(cls, block_string):
        result = None
        supported_act_fns = cls._supported_act_fn_dict.keys()
        supported_act_fns = sorted(supported_act_fns, key=lambda x: len(x), reverse=True) # to avoid parsing hsw as sw
        if block_string:
            for act_fn_string in supported_act_fns:
                if act_fn_string in block_string:
                    result = act_fn_string
                    break
        return result

    @classmethod
    def get_act_fn_from_string(cls, string, default=tf.nn.relu):
        string2act_fn = cls._supported_act_fn_dict
        if string is None:
            act_fn = default
        elif string in string2act_fn:
            act_fn = string2act_fn[string]
        else:
            raise NotImplementedError
        return act_fn

    @classmethod
    def get_se_inner_act_fn_from_string(cls, string):
        return cls.get_act_fn_from_string(string, default=tf.nn.relu)

    @classmethod
    def get_se_gating_fn_from_string(cls, string):
        return cls.get_act_fn_from_string(string, default=tf.nn.sigmoid)

    def span_blocks_args(self, blocks_args):
        """
        Remove the 'repeat' term of blocks_args. All the block_args will become repeat=1
        :param blocks_args: list of block_args.
        :return:
        """
        res_args = []
        for block_args in blocks_args:
            assert block_args.num_repeat > 0
            res_args.append(block_args._replace(num_repeat=1))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
            for _ in range(block_args.num_repeat - 1):
                res_args.append(block_args._replace(num_repeat=1))
        return res_args

    def decode_to_stages_args(self, stages_args):
        """Decodes a list of string notations to specify blocks inside the network."""
        is_effnet_style_str = isinstance(stages_args[0], str)
        if is_effnet_style_str:
            for i, blocks_string in enumerate(stages_args):
                blocks_args = self._decode_blocks_string(blocks_string)
                blocks_args = self.span_blocks_args([blocks_args])
                stage_args = AttrDict({
                    'cls': 'BasicStage',
                    'blocks_args': blocks_args
                })
                stages_args[i] = stage_args
        else:
            for stage_args in stages_args:
                is_effnet_style_str = isinstance(stage_args.blocks_args[0], str)
                if is_effnet_style_str:
                    for i, block_string in enumerate(stage_args.blocks_args):
                        block_args = self._decode_blocks_string(block_string)
                        stage_args.blocks_args[i] = block_args

        return stages_args

    @classmethod
    def _encode_block_string(self, block):
        """Encodes a block to a string."""

        def _encode_multiple_ints(arr):
            if not is_iterable(arr):
                arr = [arr]
            for k in arr:
                assert int(k) == k, "I haven't used parsed expand ratio with float values"
            return ','.join([str(int(k)) for k in arr])

        from graph.stage import _get_conv_cls
        conv_type = block.conv_type
        if not isinstance(conv_type, int):
            conv_type = _get_conv_cls(conv_type, get_cls_num=True)

        args = [
            'r%d' % block.num_repeat,
            'k%s' % _encode_multiple_ints(block.kernel_size),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % _encode_multiple_ints(block.expand_ratio),
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % conv_type
        ]
        se_ratio = block.se_ratio
        if se_ratio is not None and (se_ratio > 0):
            args.append('se%s' % se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        if hasattr(block, "act_fn"):
            if block.act_fn is not None:
                args.append(block.act_fn)
        return '_'.join(args)


BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'conv_type',
    'kernel_size', 'input_filters', 'output_filters',
    'expand_ratio', 'strides', 'id_skip', 'se_ratio', 'act_fn',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = \
    BlockArgs(
        num_repeat=1,
        conv_type=0,
        kernel_size=None,
        input_filters=None,
        output_filters=None,
        expand_ratio=None,
        strides=None,
        id_skip=None,
        se_ratio=None,
        act_fn=None,
    )

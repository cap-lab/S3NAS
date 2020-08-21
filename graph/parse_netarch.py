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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from graph.gparams import GlobalParams
from graph.jsonnet import Net
from graph.stage import _get_conv_cls, SearchableMBConvBlock, SearchableMixConvBlock, SearchableConvBlock_kxk1x1
from graph.blockargs import BlockArgsDecoder


def parse_stages_args(tb_path, base_model_args):
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }
    event_acc = EventAccumulator(tb_path, tf_size_guidance)
    event_acc.Reload()

    decoder = BlockArgsDecoder()
    for stage_args in base_model_args.stages_args:
        stage_args['blocks_args'] = decoder.span_blocks_args(stage_args['blocks_args'])

    res_stage_args = []
    for stage_idx, stage_args in enumerate(base_model_args.stages_args):
        blocks_args = stage_args['blocks_args']
        res_blocks_args = []

        for block_idx, block_args in enumerate(blocks_args):
            conv_class = _get_conv_cls(block_args.conv_type)
            search_block_name_prefix = ['Searchable']
            if not any((s in conv_class.__name__) for s in search_block_name_prefix):
                args = block_args

            # If skip_op, the block_args will be changed to 'None'
            elif conv_class in [SearchableMixConvBlock]:
                args = __get_searchable_mixconvblock_args(block_args, stage_idx, block_idx, event_acc)
            elif conv_class in [SearchableMBConvBlock, SearchableConvBlock_kxk1x1]:
                args = __get_searchable_block_args(block_args, stage_idx, block_idx, event_acc)
            else:
                raise NotImplementedError

            is_skipop = args is None
            if not is_skipop:
                res_blocks_args.append(args)

        if len(res_blocks_args) > 0:
            stage_args['blocks_args'] = res_blocks_args
            res_stage_args.append(stage_args)

    return res_stage_args


def __get_searchable_mixconvblock_args(block_args, stage_idx, block_idx, event_acc):
    block = __build_block(block_args)
    assert isinstance(block, SearchableMixConvBlock)

    use_k_list, use_C_list = [], []
    for branch_idx in range(block.num_branches):
        use_k, use_C = __get_useconds(stage_idx, block_idx, event_acc, block_prefix=block.get_summary_prefix(branch_idx))
        use_k_list.append(use_k)
        use_C_list.append(use_C)

    kernel_size, expand_ratio = block.ksize_expandratio_of_useconds(use_k_list, use_C_list)
    conv_type = block.conv_type_for(kernel_size, expand_ratio)

    if expand_ratio == 0:
        return None
    else:
        new_block_args = block_args._replace(conv_type=conv_type, kernel_size=kernel_size, expand_ratio=expand_ratio)
        return new_block_args


def __get_useconds(stage_idx, block_idx, event_acc, block_prefix=''):
    useconds_chan = __get_useconds_of(stage_idx, block_idx, event_acc, 'chan', block_prefix)
    useconds_kern = __get_useconds_of(stage_idx, block_idx, event_acc, 'kernel', block_prefix)

    return useconds_kern, useconds_chan


def __build_block(block_args):
    """
    Makes block with only using block_args. Ignores GlobalParams.
    Shouldn't use this in general case. We only use this for parse purpose
    """
    conv_cls = _get_conv_cls(block_args.conv_type)
    is_supergraph_training_tensor = tf.cast(0, tf.float32)
    block = conv_cls(block_args, GlobalParams(is_supergraph_training_tensor=is_supergraph_training_tensor))
    return block


def __get_searchable_block_args(block_args, stage_idx, block_idx, event_acc, block_prefix='',
                                return_integer_expandratio=True):
    block = __build_block(block_args)

    use_k, use_C = __get_useconds(stage_idx, block_idx, event_acc, block_prefix)
    k, er = block.ksize_expandratio_of_useconds(use_k, use_C)

    conv_type = block.conv_type_for(k, er)
    del block

    if return_integer_expandratio:
        er = int(er)
    if er == 0:
        return None
    else:
        new_block_args = block_args._replace(conv_type=conv_type, kernel_size=k, expand_ratio=er)
        return new_block_args


def __get_useconds_of(stage_idx, block_idx, event_acc, wanted, block_prefix=''):
    """
    gets useconds of chan/kernel of specific stage_idx & block_idx, from tensorboard event_acc
    :param wanted: 'chan' or 'kernel'
    """
    useconds = []

    summary_prefix = block_prefix + 'use_%s_' % wanted
    idx = 0
    while True:
        usecond = get_usecond(summary_prefix + str(idx), stage_idx, block_idx, event_acc)
        if usecond is None: break
        useconds.append(usecond)
        idx += 1

    return useconds


def get_usecond(summary_name, stage_idx, block_idx, event_acc):
    """
    Get usecondition from event_acc.

    :return use_condition (1 or 0) if found, None if not found.
    """
    try:
        label = Net.get_label(stage_idx, block_idx, summary_name)
        useconds = event_acc.Scalars(label)
        usecond = int(useconds[-1].value)
    except KeyError:
        usecond = None

    assert usecond in [None, 0, 1]

    return usecond

# This code extends codebase from followings:
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

from collections import namedtuple
from copy import deepcopy

import tensorflow as tf

from graph import BasicStage
from graph.blockargs import BlockArgsDecoder
from graph.jsonnet import Net
from graph.stage import _get_conv_cls
from util.math_utils import round_to_multiple_of


def compound_scale(model_args, global_params, width_coeff, depth_coeff, image_size, depth_list=None):
    model_args = depth_scale(model_args, global_params, depth_coeff, image_size, depth_list)
    model_args = width_scale(model_args, global_params, width_coeff)
    return model_args


def depth_scale(model_args, global_params, depth_coeff, image_size, depth_list=None):
    """
    scales blocks in BasicStages by depth_coefficient.
    The only difference is that this uses 'round' function, rather than 'ceil', which is used in EfficientNet Code.
    """
    round_ftn = round_to_multiple_of

    if depth_coeff != 1.0 or depth_list is not None:
        model = Net(model_args, global_params)
        input_shape = model._stem_layers.get_output_shape((image_size, image_size, 3))

        result_stages_args = []
        total_result_blocks = 0
        for stage_i, (stage, stage_args) in enumerate(zip(model._stages, model_args.stages_args)):
            assert isinstance(stage, BasicStage)
            num_blocks = stage.num_blocks
            if depth_list is not None:
                required_more_blocks = int(depth_list[stage_i]) - num_blocks
            else:
                required_more_blocks = round_ftn(float(num_blocks * depth_coeff)) - num_blocks
            assert required_more_blocks >= 0, "I only considered making model larger."

            stage_args = _depth_scale_one_stage(stage, stage_args, required_more_blocks, input_shape, global_params)
            result_stages_args.append(stage_args)
            for block_args in stage_args['blocks_args']:
                total_result_blocks += block_args.num_repeat

            input_shape = stage.get_output_shape(input_shape)

        model_args.stages_args = result_stages_args
        del model
        tf.logging.info('total blocks after depth scale ={}'.format(total_result_blocks))

    return model_args


def _depth_scale_one_stage(stage, stage_args, required_more_blocks, input_shape, global_params):
    decoder = BlockArgsDecoder()
    blocks_args = decoder.span_blocks_args(stage_args['blocks_args'])
    num_blocks = stage.num_blocks

    def add_repeat_to_(block_idx):
        block_args = blocks_args[block_idx]
        blocks_args[block_idx] = block_args._replace(num_repeat=(block_args.num_repeat + 1))

    while required_more_blocks >= num_blocks:
        for block_idx in range(num_blocks):
            add_repeat_to_(block_idx)
            required_more_blocks -= 1

    params_idx = namedtuple("params_idx", ["params", "block_idx"])
    params_idx_list = []
    for block_idx, block in enumerate(stage._blocks):
        if block_idx == 0: # we need to consider params of 'will be added' block
            assert global_params.data_format == 'channels_last'
            changed_input_shape = input_shape[:-1] + (blocks_args[block_idx].output_filters,)
            changed_block_args = deepcopy(blocks_args[block_idx])
            changed_block_args = changed_block_args._replace(
                input_filters=changed_block_args.output_filters, strides=[1, 1])

            changed_block = __build_block(changed_block_args, global_params, changed_input_shape)
            changed_params = changed_block.get_params(changed_input_shape)
            params_idx_list.append(params_idx(changed_params, block_idx))

        if block_idx > 0:
            params_idx_list.append(params_idx(block.get_params(input_shape), block_idx))

        input_shape = block.get_output_shape(input_shape)

    params_idx_list.sort()
    while required_more_blocks > 0:
        block_idx = params_idx_list.pop().block_idx
        add_repeat_to_(block_idx)
        required_more_blocks -= 1

    stage_args['blocks_args'] = blocks_args
    return stage_args


def __build_block(block_args, global_params, input_shape):
    conv_cls = _get_conv_cls(block_args.conv_type)
    block = conv_cls(block_args, global_params)
    features = tf.random.normal(shape=(1,) + input_shape)
    block(features)  # build block
    return block


def width_scale(model_args, global_params, width_coeff):
    def update_filters(obj, attr):
        prev = getattr(obj, attr)
        new = round_filters(prev, width_coeff, global_params.filters_divisor, global_params.min_depth)
        setattr(obj, attr, new)

    update_filters(model_args.first_conv, 'output_filters')
    # TODO: what if we change the structure of SuperNet?
    for stage_args in model_args.stages_args:
        blocks_args = stage_args['blocks_args']
        for idx, _ in enumerate(blocks_args):
            update_filters(blocks_args[idx], 'input_filters')
            update_filters(blocks_args[idx], 'output_filters')

    update_filters(model_args.feature_mix_layer, 'output_filters')
    if hasattr(model_args, 'final_expand_layer'):
        update_filters(model_args.final_expand_layer, 'output_filters')

    return model_args


def round_filters(filters, width_coeff, divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    if not width_coeff:
        return filters

    filters *= width_coeff
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    tf.logging.info('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)

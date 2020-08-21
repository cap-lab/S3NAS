# This code extends codebase from followings:
# https://github.com/dstamoulis/single-path-nas
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
#
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================

import collections

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'batch_norm', 'data_format',
    'num_classes', 'drop_connect_rate', 'dropout_rate', 'is_supergraph_training_tensor',
    'filters_divisor', 'min_depth',
    'act_fn', 'se_inner_act_fn', 'se_gating_fn'
])
from graph.block_ops import TpuBatchNormalization

GlobalParams.__new__.__defaults__ = \
    GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        batch_norm=TpuBatchNormalization,

        data_format='channels_last',

        num_classes=1000,
        drop_connect_rate=0.2,
        dropout_rate=0.2,

        is_supergraph_training_tensor=None,  # Must needed to make SearchableBlocks

        filters_divisor=8,
        min_depth=None,

        act_fn='relu',
        se_inner_act_fn='relu',
        se_gating_fn='sigmoid'
    )

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from copy import deepcopy

from absl import flags

from graph.block_ops import TpuBatchNormalization, Activation
from graph.blockargs import BlockArgsDecoder
from graph.convblock import *
from graph.mbconvblock import *
from graph.op_collections import SequentialDict
from graph.stage import _get_conv_cls

FLAGS = flags.FLAGS


class ConstraintEstim(object):
    def estim_constraint(self, net, input_width):
        raise NotImplementedError


class ConstraintGetterFromFolder(object):
    """
    Search for 'simulation_save_folder/cls_name/subpart_str' (json-style)
    and gets value of 'key' in from file.

    subpart can be anything like MBConvBlock, Conv2D, GlobalAvgPool, Dense
    """

    def __init__(self, simulation_save_folder, key='total_true_latencies'):
        self._simulation_save_folder = simulation_save_folder
        self._key = key

    def get_constraint_of_subpart(self, cls_name, subpart_str, input_img_size):
        simul_file = os.path.join(self._simulation_save_folder, cls_name, subpart_str + "_imgsize%d" % input_img_size)

        if not os.path.exists(simul_file):
            raise IOError("Can't find simulation file %s " % simul_file)
        f = open(simul_file, "r")
        simul_dict = json.load(f)
        f.close()
        constraint = simul_dict[self._key]

        return constraint

    def get_constraint_of_block_args(self, block_args, input_img_size):
        block_name, block_args_str = self._get_block_name_and_str(block_args)
        return self.get_constraint_of_subpart(block_name, block_args_str, input_img_size)

    def _get_block_name_and_str(self, block_args):
        decoder = BlockArgsDecoder
        if isinstance(block_args, str):
            block_args = decoder._decode_blocks_string(block_args)
        block_name = _get_conv_cls(block_args.conv_type).__name__
        block_args_str = decoder._encode_block_string(block_args)

        return block_name, block_args_str


class ConstraintEstimWithoutCalc(ConstraintEstim):
    def __init__(self, div_unit=1e6):
        """
        :param div_unit: divide unit you want to use.
        """
        self.div_unit = div_unit

    def estim_constraint(self, net, input_width):
        self.tensordict_to_write_on_tensorboard = {}
        input_shape = (input_width, input_width, 3)
        total_constraint = self._estim_constraint(net, input_shape) / self.div_unit

        return total_constraint, self.tensordict_to_write_on_tensorboard

    def _estim_constraint(self, net, input_shape):
        """
        :param net: Supernet.
        We need to use supernet because we need the dynamically changing threshold values to estimate SearchableBlocks.
        """
        total_constraint = tf.zeros((1,))

        for layer in net.iter_stem_layers():
            total_constraint += self._get_constraint_of_layer(layer, input_shape)
            input_shape = layer.get_output_shape(input_shape)

        for stage in net._stages:
            for block in stage._blocks:
                if self._is_searchable_block(block):
                    total_constraint += self._estim_constraint_of_searchableblock(block, input_shape)
                else:
                    total_constraint += self._get_constraint_of_block(block, input_shape)
                input_shape = block.get_output_shape(input_shape)

        for layer in net.iter_head_layers():
            total_constraint += self._get_constraint_of_layer(layer, input_shape)
            input_shape = layer.get_output_shape(input_shape)

        total_constraint += self._get_constraint_of_layer(net._classifier, input_shape)

        return total_constraint

    def _get_constraint_of_layer(self, layer, input_shape):
        return NotImplementedError

    def _get_constraint_of_block(self, block, input_shape):
        return NotImplementedError

    def _get_constraint_of_block_args(self, block_args, input_shape):
        return NotImplementedError

    def _is_searchable_block(self, block):
        return 'Searchable' in block.__class__.__name__

    def _estim_constraint_of_searchableblock(self, block, input_shape):
        block_args = deepcopy(block._block_args)
        # kernel_size = block_args.kernel_size
        # expand_ratio = block_args.expand_ratio

        constraint = tf.zeros((1,))
        for k, er in block.get_listof_possible_ksize_expandratio_with_0_maybe():
            useconds = block.useconds_for_ksize_expandratio(k, er)
            block_args.kernel_size, block_args.expand_ratio = k, er
            block_args.conv_type = block.conv_type_for(k, er)
            constraint += (useconds * self._get_constraint_of_block_args(block_args, input_shape))

        del block_args

        return constraint


class LatencyEstimLUT(ConstraintEstimWithoutCalc):
    def __init__(self, latency_lut_folder, div_unit=1e6, parse_key='total_true_latencies'):
        """
        :param lut_folder: Folder containing lookup values.
        Must contain subfolders named with class names, like

        lut_folder/MBConvBlock
        lut_folder/MultDwiseBlock ...

        and there have to exist files named with each blockargs.
        """
        super(LatencyEstimLUT, self).__init__(div_unit)
        self.latency_getter = ConstraintGetterFromFolder(latency_lut_folder, parse_key)

    def _get_constraint_of_block(self, block, input_shape):
        return self._get_constraint_of_block_args(block._block_args, input_shape)

    def _get_constraint_of_block_args(self, block_args, input_shape):
        assert input_shape[0] == input_shape[1]
        img_size = input_shape[0]

        if block_args.expand_ratio == 0:
            return 0

        return self.latency_getter.get_constraint_of_block_args(block_args, img_size)

    def _get_constraint_of_layer(self, layer, input_shape):
        assert input_shape[0] == input_shape[1]
        img_size = input_shape[0]

        if isinstance(layer, SequentialDict):
            raise NotImplementedError
        else:
            if isinstance(layer, TpuBatchNormalization) or isinstance(layer, Activation):
                tf.logging.info("I assume that BN or Act latency are already considered in Conv")
                return 0

            cls_name = layer.__class__.__name__
            subpart_str = layer.get_str()
            return self.latency_getter.get_constraint_of_subpart(cls_name, subpart_str, img_size)


def get_constraint_estimator(constraint='latency', parse_key='total_true_latencies', div_unit=1e6):
    if constraint == 'latency':
        return LatencyEstimLUT(FLAGS.constraint_lut_folder, parse_key=parse_key, div_unit=div_unit)
    else:
        raise NotImplementedError

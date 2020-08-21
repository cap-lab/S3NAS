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

import itertools
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import flags

from graph.block_ops import Conv2D, DepthwiseConv2D, MyNetComponents, BasicOp, MixConv
from graph.searchable_utils import SearchableKernelMaker, calc_useconds_of_value_in_list, interpret_useconds
from util import string_utils
from util.utils import get_kwargs, prev_curr


class SearchableBlock(MyNetComponents):
    def _change_to_searchableblock(self, global_params):
        # Learnable Depth-wise convolution Superkernel
        searchable_op = self.searchable_op_cls(
            global_params=global_params,
            op=self.core_op,
            k_sel_list=self._k_sel_list,
            C_sel_list=self.core_C_sel_list
        )
        self.core_op = searchable_op

    @property
    def core_op(self):
        raise NotImplementedError

    @core_op.setter
    def core_op(self, value):
        raise NotImplementedError

    @property
    def _k_sel_list(self):
        raise NotImplementedError

    @property
    def _er_sel_list_with_0_maybe(self):
        """expand_ratio selection list, which can contain 0, which means skipop."""
        raise NotImplementedError

    @property
    def core_C_sel_list(self):
        raise NotImplementedError

    def get_listof_possible_ksize_expandratio_with_0_maybe(self):
        """
        If you get expand_ratio 0, it means skipop.
        """
        result = []
        for k in self._k_sel_list:
            for er in self._er_sel_list_with_0_maybe:
                result.append((k, er))
        return result

    def useconds_for_ksize_expandratio(self, kernel_size, expand_ratio):
        raise NotImplementedError

    def ksize_expandratio_of_useconds(self, useconds_kern, useconds_chan):
        raise NotImplementedError

    @property
    def searchable_op_cls(self):
        raise NotImplementedError

    def conv_type_for(self, kernel_size, expand_ratio):
        raise NotImplementedError


class SearchableOp(tf.keras.layers.Layer, BasicOp):
    def __init__(self, global_params, op, k_sel_list, C_sel_list):
        self._k_sel_list = k_sel_list
        self._C_sel_list = C_sel_list
        self._core_op = op
        self._global_params = global_params
        self.tensordict = {}
        assert isinstance(op, self.op_cls)
        assert global_params.data_format == 'channels_last'
        self.is_supergraph_training_tensor = tf.stop_gradient(global_params.is_supergraph_training_tensor)
        super(SearchableOp, self).__init__()

    def build(self, input_shape):
        """avoided multiple inheritance because of this building procedure.."""
        self._core_op.build(input_shape)
        self._build_searchable_kernel()
        super(SearchableOp, self).build(input_shape)

    def _build_searchable_kernel(self):
        raise NotImplementedError

    def call(self, inputs, training=None, *args, **kwargs):
        return self._core_op(inputs, training=training, *args, **kwargs)

    def get_output_shape(self, input_shape=(224, 224, 3)):
        return self._core_op.get_output_shape(input_shape)

    def _get_(self, what, input_shape=(224, 224, 3)):
        return self._core_op._get_(what, input_shape)

    @property
    def op_cls(self):
        raise NotImplementedError

    @property
    def C_split_target_axis(self):
        raise NotImplementedError

    def get_listof_possible_ksize_channels(self):
        raise NotImplementedError

    def useconds_for_ksize_channels(self, kernel_size, channels):
        raise NotImplementedError

    def add_tensor_to_tensordict(self, name, tensor):
        self.tensordict[name] = tensor

    def tensordict_to_write_on_tensorboard(self):
        return self.tensordict


class SearchableBasicOp(SearchableOp):
    @property
    def kernel_attr(self):
        raise NotImplementedError

    @property
    def core_kernel(self):
        return getattr(self._core_op, self.kernel_attr)

    @core_kernel.setter
    def core_kernel(self, value):
        setattr(self._core_op, self.kernel_attr, value)

    def _build_searchable_kernel(self):
        searchable_kernel_considering_k = self._set_kernels_kernel_and_get_weight(self.core_kernel, self._k_sel_list)
        searchable_kernel = self._set_channels_kernel_and_get_weight(searchable_kernel_considering_k,
                                                                     self.C_split_target_axis)

        self.core_kernel = searchable_kernel

    def _set_kernels_kernel_and_get_weight(self, kernel_w, k_sel_list):
        """Make searchable considering ksizes, and also adds thresholds"""
        kernel_shape = kernel_w.shape
        masks = []
        k_sel_list.sort()
        assert k_sel_list[-1] == kernel_shape[0]

        k_mid = (kernel_shape[0] + 1) // 2
        prev_k_size_filled = np.zeros(kernel_shape)
        for k_size in k_sel_list:
            assert k_size % 2 == 1, "We only support k_size with odd number"
            mask = np.zeros(kernel_shape)

            k_l = k_mid - (k_size) // 2
            k_r = k_mid + (k_size) // 2
            mask[k_l:(k_r + 1), k_l:(k_r + 1), :, :] = 1.0

            # For example, we need to remove [1:(3+1), 1:(3+1), :, :] of 5x5 kernel, when you use [3, 5] for k_sel_list
            prev_k_size_filled, mask = mask, mask - prev_k_size_filled

            masks.append(tf.convert_to_tensor(mask,
                                              dtype=kernel_w.dtype))
        assert len(masks) == len(k_sel_list)

        return self._set_splitted_kernel_and_get_weight(kernel_w, masks, prefix='kernel',
                                                        can_zeroout=False)

    @property
    def C_sel_list_corresponding_to_masks(self):
        """Channel list used for making masks. Placed here to access this list without build"""
        # Even if the C_sel_list is [0, 16, 32], only two masks for channels [:,:,0:16], [:,:,16:32] exists inherently.
        # And each mask represents usage of 16, 32 respectively.
        if self._C_sel_list[0] == 0:
            return self._C_sel_list[1:]
        else:
            return self._C_sel_list

    def _set_channels_kernel_and_get_weight(self, kernel_w, split_target_axis=2):
        """
        Splits the kernel_w by target_axis every self.C_split_units automatically.
        It means... given [3, 3, 32, 64] kernel, if you set C_split_units=16 and axis=2,
        [:, :, 0:16, :] and [:, :, 16:32, :] will be the splitted kernels.
        Refer to set_splitted_kernel_and_get_weight for more details

        :param kernel_w: The kernel weight tensor you want to split.
        :param split_target_axis: what axis you want to split the channels. Typically, only 2 or 3 is eligible.
        axis=2 means in_channels typically. That is, the channels of input feature map.
        axis=3 means out_channels typically. But for depthwise, it seems to mean number filters per group.
        """

        kernel_shape = kernel_w.shape
        num_channels = int(kernel_shape[split_target_axis])

        assert self._C_sel_list[-1] == num_channels

        self.can_zeroout = self._C_sel_list[0] == 0

        masks = []

        for prev_channel, channel in prev_curr([0] + self.C_sel_list_corresponding_to_masks):
            mask = np.zeros(kernel_shape)
            if split_target_axis == 2:
                mask[:, :, prev_channel:channel, :] = 1.0  # from 0% to 50% channels
            elif split_target_axis == 3:
                mask[:, :, :, prev_channel:channel] = 1.0  # from 0% to 50% channels
            masks.append(tf.convert_to_tensor(mask,
                                              dtype=kernel_w.dtype))

        return self._set_splitted_kernel_and_get_weight(kernel_w, masks, prefix='chan',
                                                        can_zeroout=self.can_zeroout)

    def _set_splitted_kernel_and_get_weight(self, kernel_w, masks, prefix, can_zeroout):
        """
        Make thresholds and corresponding useconds for given split setting.
        adds thresholds as trainable variable, with add_weight ftn,
        sets self.t_'prefix' = thresholds, self.use_'prefix' = useconds
        and adds these to self.tensordict.

        each mask corresponds to each threshold and each useconds.
        these corresponds to self.k_sel_list or self.C_sel_list_corresponding_to_masks
        """
        thresholds = []
        for i in range(len(masks)):
            if not can_zeroout and i == 0:
                t = tf.zeros(1)
            else:
                t = self._core_op.add_weight(shape=(1,), initializer='zeros', name="t_%s_%d" % (prefix, i))
            thresholds.append(t)

        sk_maker = SearchableKernelMaker()
        masked_kernel, useconds = \
            sk_maker.make_searchable_kernel_w_and_conds(kernel_w, masks, thresholds, self.is_supergraph_training_tensor,
                                                        can_zeroout, equal_selection=True)

        for i in range(len(masks)):
            self.add_tensor_to_tensordict('t_%s_%d' % (prefix, i), thresholds[i])
            self.add_tensor_to_tensordict('use_%s_%d' % (prefix, i), useconds[i])
        setattr(self, 't_%s' % prefix, thresholds)
        setattr(self, 'use_%s' % prefix, useconds)

        return masked_kernel

    def get_listof_possible_ksize_channels(self):
        """
        If you get expand_ratio 0, it means skipop.
        """
        result = []
        for k in self._k_sel_list:
            for C in self._C_sel_list:
                result.append((k, C))
        return result

    def useconds_for_ksize_channels(self, kernel_size, channels):
        useconds_for_k = calc_useconds_of_value_in_list(kernel_size, self._k_sel_list,
                                                        self.use_kernel, can_zeroout=False)
        useconds_for_C = calc_useconds_of_value_in_list(channels, self.C_sel_list_corresponding_to_masks,
                                                        self.use_chan, self.can_zeroout)

        return useconds_for_k * useconds_for_C

    def ksize_channels_of_useconds(self, useconds_kern, useconds_chan):
        k = interpret_useconds(list(self._k_sel_list), useconds_kern)
        C = interpret_useconds(list(self.C_sel_list_corresponding_to_masks), useconds_chan)

        return k, C


class SearchableDwiseConv(SearchableBasicOp):
    op_cls = DepthwiseConv2D
    kernel_attr = 'depthwise_kernel'
    C_split_target_axis = 2


class SearchableConvFront(SearchableBasicOp):
    """Searchable Conv2D which varies the output channels, expected to be placed in front of a operation"""
    op_cls = Conv2D
    kernel_attr = 'kernel'
    C_split_target_axis = 3


class SearchableConvBack(SearchableBasicOp):
    """Searchable Conv2D which varies the input channels, expected to be placed behind of a operation"""
    op_cls = Conv2D
    kernel_attr = 'kernel'
    C_split_target_axis = 2


class SearchableMixConv(SearchableOp):
    """Searchable Conv2D which varies the input channels, expected to be placed behind of a operation"""
    op_cls = MixConv

    def __init__(self, global_params, op, k_sel_list, C_sel_list):
        kwargs = get_kwargs()
        super(SearchableMixConv, self).__init__(**kwargs)
        for idx, (dconv, k_list, C_list) in enumerate(zip(self._core_op._convs, self._k_sel_list, self._C_sel_list)):
            searchable_dconv = SearchableDwiseConv(self._global_params, dconv, k_list, C_list)
            self._core_op._convs[idx] = searchable_dconv

    def _build_searchable_kernel(self):
        self._generate_useconds_per_block()

    def _generate_useconds_per_block(self):
        branches = self._core_op._convs

        def default_factory():
            return tf.zeros((1,))

        self._useconds_per_k_C = defaultdict(default_factory)  # makes tf.zeros(1) as default

        ksize_channels_of_branches_list = []
        for branch in branches:
            ksize_channels_list = branch.get_listof_possible_ksize_channels()
            ksize_channels_of_branches_list.append(ksize_channels_list)

        all_possible_ksize_channels_for_branches = itertools.product(*ksize_channels_of_branches_list)

        # This will make every possible branch selections.
        # [((k1_brch1, C1_brch1), (k1_brch2, C1_brch2), ..), (k2_brch1, C2_brcn1), (k1_brch2, C1_brch2), ...), ...]
        for ksize_channels_for_branches in all_possible_ksize_channels_for_branches:
            usecond = tf.ones((1,))
            for branch, ksize_channels in zip(branches, ksize_channels_for_branches):
                k, C = ksize_channels
                usecond = usecond * branch.useconds_for_ksize_channels(k, C)

            kernel_sizes, channels = zip(*ksize_channels_for_branches)
            kernel_sizes, channels = self.simplify_k_C(kernel_sizes, channels)

            self._useconds_per_k_C[(kernel_sizes, channels)] += usecond

    def simplify_k_C(self, kernel_size, channels):
        if isinstance(kernel_size, int):
            assert isinstance(channels, int)
            return kernel_size, channels

        C_of_k = defaultdict(int)  # makes 0 as default
        for k, C in zip(kernel_size, channels):
            C_of_k[k] += C

        # Remove expand_ratio = 0, but
        for k, C in list(C_of_k.items()):
            if C == 0:
                del C_of_k[k]
        is_skipop = len(C_of_k) == 0
        if is_skipop:  # Add dummy (ksize0,chan0) to indicate it is skipop
            C_of_k[0] = 0

        simplified_kernel_size, simplified_channels = zip(*sorted(C_of_k.items()))

        def to_tuple_or_int(generator):
            result = tuple(generator)
            if len(result) == 1:
                result = result[0]
            return result

        # Need to support MultDwiseBLock when len > 1, otherwise MBConvBLock, to avoid unnecessary Concat in MIDAP
        return to_tuple_or_int(simplified_kernel_size), to_tuple_or_int(simplified_channels)

    def useconds_for_ksize_channels(self, kernel_size, channels):
        if (kernel_size, channels) in self._useconds_per_k_C:
            return self._useconds_per_k_C[(kernel_size, channels)]
        else:
            raise ValueError(str((kernel_size, channels)))

    def ksize_channels_of_useconds(self, useconds_kern, useconds_chan):
        """ useconds_kern: form of [[1, 1, 0], [1, 0, 0], ..] """
        assert len(useconds_kern) == len(useconds_chan) == len(self._core_op._convs)
        k_list, C_list = [], []
        for searchable_dwise, use_kern, use_chan in zip(self._core_op._convs, useconds_kern, useconds_chan):
            assert isinstance(searchable_dwise, SearchableDwiseConv)
            k = interpret_useconds(searchable_dwise._k_sel_list, use_kern)
            C = interpret_useconds(searchable_dwise.C_sel_list_corresponding_to_masks, use_chan)
            k_list.append(k)
            C_list.append(C)

        return self.simplify_k_C(k_list, C_list)

    def get_listof_possible_ksize_channels(self):
        """
        If you get expand_ratio 0, it means skipop.
        """
        result = []
        for (k, C) in self._useconds_per_k_C.keys():
            result.append((k, C))
        return result

    def tensordict_to_write_on_tensorboard(self):
        self.tensordict = super(SearchableMixConv, self).tensordict_to_write_on_tensorboard()
        branches = self._core_op._convs
        for idx, branch in enumerate(branches):
            prefix, suffix = self.get_summary_prefix(idx), ''
            self.tensordict.update(
                string_utils.add_pre_suf_to_keys_of(branch.tensordict_to_write_on_tensorboard(), prefix, suffix))
        return self.tensordict

    @classmethod
    def get_summary_prefix(cls, branch_idx):
        return 'branch' + str(branch_idx) + '_'

    @property
    def num_branches(self):
        return len(self._core_op._convs)

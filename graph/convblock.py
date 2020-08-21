from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from graph.block_ops import Conv2D
from graph.mbconvblock import MBConvBlock, SearchableMBConvBlock
from graph.op_collections import build_conv_bn_act
from graph.searchableops import *


class ConvBlock(MBConvBlock):
    def __init__(self, block_args, global_params, kernel_front=False, stride_front=True, default_ksize=1):
        self.default_ksize = default_ksize
        self.kernel_front = kernel_front
        self.stride_front = stride_front
        super(ConvBlock, self).__init__(block_args, global_params)

    def _build_expand(self, output_filters):
        kernel_size = self._block_args.kernel_size
        if self.kernel_front:
            expand_ksize = [kernel_size, kernel_size]
        else:
            expand_ksize = [self.default_ksize, self.default_ksize]

        if self.stride_front:
            expand_strides = self._block_args.strides
        else:
            expand_strides = [1, 1]

        kwargs = dict(cls=Conv2D.__name__,
                      output_filters=output_filters,
                      kernel_size=expand_ksize,
                      strides=expand_strides)
        self._expand_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=self.act_fn)

    def _build_dwise(self, output_filters):
        kwargs = None
        self._dwise_layers = build_conv_bn_act(self._global_params, kwargs)

    def _build_proj(self, output_filters):
        kernel_size = self._block_args.kernel_size
        if self.kernel_front:
            proj_ksize = [self.default_ksize, self.default_ksize]
        else:
            proj_ksize = [kernel_size, kernel_size]

        strides = self._block_args.strides
        if self.stride_front:
            proj_strides = [1, 1]
        else:
            proj_strides = strides

        kwargs = dict(cls=Conv2D.__name__,
                      output_filters=output_filters,
                      kernel_size=proj_ksize,
                      strides=proj_strides)

        self._proj_layers = build_conv_bn_act(self._global_params, kwargs, add_bn=True, act_fn=None)


class ConvBlock_kxk1x1(ConvBlock):
    def __init__(self, block_args, global_params):
        super(ConvBlock_kxk1x1, self).__init__(block_args, global_params, kernel_front=True, stride_front=True,
                                               default_ksize=1)


class SearchableConvBlock_kxk1x1(ConvBlock_kxk1x1, SearchableMBConvBlock):
    def __init__(self, *args, **kwargs):
        super(SearchableConvBlock_kxk1x1, self).__init__(*args, **kwargs)

    searchable_op_cls = SearchableConvFront

    @property
    def core_op(self):
        return self._expand_layers['conv']

    @core_op.setter
    def core_op(self, value):
        self._expand_layers['conv'] = value

    def conv_type_for(self, kernel_size, expand_ratio):
        return ConvBlock_kxk1x1.__name__

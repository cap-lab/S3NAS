"""Makes name_to_cls automatically if you only import the class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from inspect import getmembers, isclass

import graph
from graph.block_ops import Conv2D, DepthwiseConv2D, MixConv
from graph.stage import BasicStage

__name_to_cls = {}
for name, cls in getmembers(graph):
    if isclass(cls):
        __name_to_cls[cls.__name__] = cls


def name_to_class(name):
    return __name_to_cls[name]


def construct_with_args(global_params=None, **kwargs):
    cls_name = kwargs.pop('cls')
    cls = name_to_class(cls_name)
    if global_params is not None:
        kwargs['global_params'] = global_params
    return cls(**kwargs)

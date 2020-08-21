from collections import OrderedDict

import tensorflow as tf

import graph
from graph.block_ops import MyNetComponents, GlobalAvgPool, Activation, Conv2D


class SE(tf.keras.layers.Layer, MyNetComponents):
    def __init__(self, global_params, num_reduced_filters, output_filters, se_inner_act_fn, se_gating_fn):
        se_reduce = Conv2D(
            global_params,
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            use_bias=True
        )
        se_expand = Conv2D(
            global_params,
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            use_bias=True
        )
        self._se_layers = SequentialDict([GlobalAvgPool(global_params, keep_dims=True),
                                          se_reduce, Activation(global_params, se_inner_act_fn),
                                          se_expand, Activation(global_params, se_gating_fn)])
        super(SE, self).__init__()

    def get_output_shape(self, input_shape=(224, 224, 3)):
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        return self._se_layers._get_(what, input_shape)

    def call(self, inputs, training=None):
        self.activations = self._se_layers(inputs)

        inputs = inputs * self.activations
        return inputs


class SequentialDict(tf.keras.layers.Layer, MyNetComponents):
    """Similar to ModuleDict in torch.nn. Got idea from torch.nn codes"""
    def __init__(self, args=None):
        super(SequentialDict, self).__init__()
        self._layer_dict = OrderedDict()

        if args:
            if isinstance(args, OrderedDict):
                self.update(args)
            else:
                self.extend(args)

    def update(self, layer_dict):
        assert isinstance(layer_dict, dict)
        for key, layer in layer_dict.items():
            self._layer_dict[key] = layer

    def append(self, layer):
        assert isinstance(layer, MyNetComponents)
        self._layer_dict[str(len(self._layer_dict))] = layer

    def extend(self, args):
        for layer in args:
            self.append(layer)

    def build(self, input_shape):
        """Assume NHWC data type"""
        orig_input_shape = input_shape
        for layer in self:
            layer.build(input_shape)
            input_shape = input_shape.as_list()
            input_shape = [input_shape[0]] + list(layer.get_output_shape(input_shape[1:]))
            input_shape = tf.TensorShape(input_shape)
        super(SequentialDict, self).build(orig_input_shape)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self:
            x = layer(x, *args, **kwargs)

        return x

    def get_output_shape(self, input_shape=(224, 224, 3)):
        for layer in self:
            input_shape = layer.get_output_shape(input_shape)
        return input_shape

    def _get_(self, what, input_shape=(224, 224, 3)):
        result = 0
        for layer in self:
            result += layer._get_(what, input_shape)
            input_shape = layer.get_output_shape(input_shape)
        return result

    def __getitem__(self, key):
        if isinstance(key, int):
            key = str(int)
        return self._layer_dict[key]

    def __setitem__(self, key, layer):
        if isinstance(key, int):
            key = str(int)
        assert isinstance(layer, MyNetComponents)
        self._layer_dict[key] = layer

    def __delitem__(self, key):
        if isinstance(key, int):
            key = str(int)
        del self._layer_dict[key]

    def __len__(self):
        return len(self._layer_dict)

    def __dir__(self):
        keys = super(SequentialDict, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._layer_dict.values())


def build_conv_bn_act(global_params, conv_kwargs, add_bn=True, act_fn=None):
    """if conv_cls is given, it builds a conv_bn_act. else, gives null Sequential"""
    layers = SequentialDict()
    if conv_kwargs:
        layers.update({'conv': graph.construct_with_args(global_params, **conv_kwargs)})
        assert isinstance(add_bn, bool)
        if add_bn:
            layers.update({'bn': global_params.batch_norm(global_params)})
        if act_fn:
            layers.update({'act': Activation(global_params, act_fn)})
    return layers


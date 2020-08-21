from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class SearchableKernelMaker:
    def make_searchable_kernel_w_and_conds(self, kernel_w, tf_masks, thresholds, is_supergraph_training_tensor,
                                           can_zeroout, equal_selection=True):
        """
        Makes searchable_kernel.
        Kernel will be chosen among the followings
        k_0 : kernel_w * tf_masks[0]
        k_1 : kernel_w * (tf_masks[0] + tf_masks[1]),
        k_2 : kernel_w * (tf_masks[0] + tf_masks[1] + tf_masks[2]), ...

        if norm(kernel_w * tf_masks[i]) > thresholds[i] for all i < n, use k_n.
        thresholds[0] will choose whether or not to use k_0, k_1, k_2, ...
        thresholds[1] will choose whether or not to use k_1, k_2, ...

        :param kernel_w: kernel weight to make into searchable form
        :param tf_masks: tensorflow tensor masks which indicate the form of searchable kernel.
        All the elements should be 0 or 1,
        And when summed up, they have to be identical with np.ones(kernel_w.shape)
        :param thresholds: List of tensors with shape [1]. Can be trainable if one wants to change this by training.
        Must be the same length with tf_masks.
        :param is_supergraph_training_tensor: tensor with shape [1], which indicate the supergraph is training.
        :param can_zeroout: if True, this kernel_w will be able to be zeroed out totally.
        It thinks that this will make this layer to be skip-op
        """

        # FIXME: this is awkward... these two variables have to be combined, but how?
        splitted_kernels = [kernel_w * mask for mask in tf_masks]
        self.norms_per_splitted_kernels = [tf.norm(splitted_kernel) for splitted_kernel in splitted_kernels]
        useconds_of_splitted_kernel = self.__diffble_islargerthanT(self.norms_per_splitted_kernels, thresholds)

        if is_supergraph_training_tensor is not None:
            assert equal_selection
            useconds_of_splitted_kernel = self.__add_equal_drop_per_split(useconds_of_splitted_kernel,
                                                                          is_supergraph_training_tensor)

        if not can_zeroout:
            self.__avoid_zerooutall_cond(useconds_of_splitted_kernel)

        return combine_to_nested_form(splitted_kernels, useconds_of_splitted_kernel), useconds_of_splitted_kernel

    def _get_norms_per_kernels(self):
        """
        For debugging purpose. Make sure to call this after you made searchable_kernel
        """
        return self.norms_per_splitted_kernels

    @classmethod
    def __diffble_islargerthanT(cls, values, thresholds):
        assert len(values) == len(thresholds)
        useconds_of_splitted_kernel = [diffible_indicator(v - t) for v, t in zip(values, thresholds)]
        return useconds_of_splitted_kernel

    @classmethod
    def __add_equal_drop_per_split(cls, useconds_of_splitted_kernel, is_supergraph_training):
        """
        This equation was calculated when is_supergraph_training is 1 or 0.
        We will have total split_nums + 1 selections. Let's call this n + 1
        So, to make equal drops, each selection point, i.e. each useconds have to be dropped by
        1/(n+1), 1/n, 1/(n-1), ... 1/2.
        Then, all selections can be selected by prob
        1/(n+1), 1/n * n/(n+1), 1/(n-1) * (n-1)/n * n/(n+1), ...
        This equation works even when the first condition is fixed by 1, by __avoid_zerooutall_cond.
        Because, remaining selections will be selected by prob 1/n.
        """
        dropped_useconds = []
        split_nums = len(useconds_of_splitted_kernel)
        for i in range(split_nums):
            drop_prob = is_supergraph_training * 1 / (split_nums + 1 - i)
            dropped_useconds.append(add_dropout(useconds_of_splitted_kernel[i], drop_prob))

        return dropped_useconds

    @classmethod
    def __avoid_zerooutall_cond(cls, useconds_of_splitted_kernel):
        # Note that zeroout must be done only in the first threshold
        useconds_of_splitted_kernel[0] = tf.ones([1])


def diffible_indicator(x):
    return tf.stop_gradient(tf.to_float(x >= 0) - tf.sigmoid(x)) + tf.sigmoid(x)


def add_dropout(tensor, drop_prob):
    return tf.nn.dropout(tensor, rate=drop_prob)


def combine_to_nested_form(splitted_kernels, useconds_of_splitted_kernel):
    """
    For example, if splitted_kernels = [c50%, c100%], useconds = [use_ehalf, use_efull],
    return use_ehalf * (c50% + use_efull * c100%)

    :param splitted_kernels:
    :param useconds_of_splitted_kernel:
    :return:
    """
    assert len(splitted_kernels) == len(useconds_of_splitted_kernel)

    if len(splitted_kernels) == 1:
        return useconds_of_splitted_kernel[0] * splitted_kernels[0]
    else:
        return useconds_of_splitted_kernel[0] * \
               (splitted_kernels[0] + combine_to_nested_form(splitted_kernels[1:], useconds_of_splitted_kernel[1:]))


def calc_useconds_of_value_in_list(wanted, sel_list, useconds_of_selections, can_zeroout=False):
    """
    If you used combine_to_searchable_form to construct the splitted_kernels,
    You can calculate usecond value for wanted value in sel_list with this ftn
    """
    if wanted == 0:
        assert can_zeroout
        return 1 - useconds_of_selections[0]

    assert wanted in sel_list
    assert len(sel_list) == len(useconds_of_selections)

    result = tf.ones((1,))
    for i, (selection, usecond) in enumerate(zip(sel_list, useconds_of_selections)):
        result = result * usecond
        if selection == wanted:
            break

    not_last_selection = (i + 1 < len(sel_list))
    if not_last_selection:
        result = result * (1 - useconds_of_selections[i + 1])

    return result


def interpret_useconds(sel_list, useconds_of_selections):
    """
    interprets value of useconds.
    returns 0 if all the useconds are zero. i.e. that means skipop (expand_ratio=0)
    """

    assert len(sel_list) == len(useconds_of_selections)

    result = 0
    for selection, usecond in zip(sel_list, useconds_of_selections):
        if usecond == 0:
            break
        result = selection

    return result


def test_interpret_useconds():
    C_sel_list = [32, 64, 128]
    assert 0 == interpret_useconds(C_sel_list, useconds_of_selections=[0, 0, 0])
    assert 32 == interpret_useconds(C_sel_list, useconds_of_selections=[1, 0, 0])
    assert 64 == interpret_useconds(C_sel_list, useconds_of_selections=[1, 1, 0])
    assert 128 == interpret_useconds(C_sel_list, useconds_of_selections=[1, 1, 1])
    assert 32 == interpret_useconds(C_sel_list, useconds_of_selections=[1, 0, 1])
    assert 0 == interpret_useconds(C_sel_list, useconds_of_selections=[0, 1, 1])
    print("passed test_interpret_useconds")


def test_useconds_of_value_in_list():
    C_sel_list = [32, 64]
    with tf.Session() as sess:
        usecond_32, usecond_64 = tf.Variable((0.0)), tf.Variable((0.0))
        sess.run(tf.global_variables_initializer())

        usecond_for_ = {wanted: calc_useconds_of_value_in_list(wanted, C_sel_list, [usecond_32, usecond_64], can_zeroout=True)
                        for wanted in [0, 32, 64]}

        def check_val(tensor, value):
            tensor_val = sess.run(tensor).item()
            assert tensor_val == value, "%f" % tensor_val

        check_val(usecond_for_[0], 1)
        check_val(usecond_for_[32], 0)
        check_val(usecond_for_[64], 0)

        sess.run(usecond_32.assign(1.0))
        check_val(usecond_for_[0], 0)
        check_val(usecond_for_[32], 1)
        check_val(usecond_for_[64], 0)

        sess.run(usecond_64.assign(1.0))
        check_val(usecond_for_[0], 0)
        check_val(usecond_for_[32], 0)
        check_val(usecond_for_[64], 1)

        sess.run(usecond_32.assign(0.0))
        check_val(usecond_for_[0], 1)
        check_val(usecond_for_[32], 0)
        check_val(usecond_for_[64], 0)
    print("passed useconds_of_value_in_list")


if __name__ == '__main__':
    test_useconds_of_value_in_list()
    test_interpret_useconds()

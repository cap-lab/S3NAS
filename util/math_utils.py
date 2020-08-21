import tensorflow as tf


def partitionfunc(n, partition_length, unit_of_num_in_partition=1, min_num_in_partition=None):
    '''
    got idea from https://stackoverflow.com/a/18503391
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    '''
    min_length = 1

    assert n % unit_of_num_in_partition == 0
    if min_num_in_partition is None:
        min_num_in_partition = unit_of_num_in_partition
    assert min_num_in_partition % unit_of_num_in_partition == 0

    if partition_length < min_length:
        raise StopIteration
    if partition_length == min_length:
        if n >= min_num_in_partition:
            yield (n,)
        raise StopIteration

    # i means minimum number of a specific result partition.
    for i in range(min_num_in_partition, n + 1, unit_of_num_in_partition):
        for result in partitionfunc(n - i, partition_length - 1, unit_of_num_in_partition, min_num_in_partition=i):
            yield (i,) + result


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """
    round function which have same behavior in python2 and python3. from
    https://github.com/pytorch/vision/blob/78ed10cc51067f1a6bac9352831ef37a3f842784/torchvision/models/mnasnet.py#L68
    """
    assert 0.0 <= round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2.0) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def round_to_multiple_of(val, divisor=1):
    return _round_to_multiple_of(val, divisor, round_up_bias=0.0)


def argmax(l):
    # from https://towardsdatascience.com/there-is-no-argmax-function-for-python-list-cd0659b05e49
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


def ceil_div(num, divisor):
    return (num + divisor - 1) // divisor


def linear(x, start_x, start_y, end_x, end_y):
    x = tf.cast(x, tf.float32)
    return ((x - start_x) / (end_x - start_x)) * (end_y - start_y) + start_y


def smooth_square(x, start_x, start_y, end_x, end_y):
    x = tf.cast(x, tf.float32)
    x_normalized = (x - start_x) / (end_x - start_x)
    left_square = 2 * x_normalized ** 2
    right_square = 1 - 2 * (1 - x_normalized) ** 2
    square = tf.cond(x_normalized <= 0.5, lambda: left_square, lambda: right_square)
    return square * (end_y - start_y) + start_y
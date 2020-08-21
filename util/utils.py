from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
from copy import deepcopy
from itertools import tee

from absl import flags
from box import Box

FLAGS = flags.FLAGS


class AttrDict(Box):
    """
    Class which supports both Tuple-style and Dict-style usage
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get('ordered_box') is None:
            kwargs['ordered_box'] = True
        super(AttrDict, self).__init__(*args, **kwargs)

    def _replace(self, **kwargs):
        res = deepcopy(self)
        res.update(**kwargs)
        return res


def get_kwargs():
    """
    Gets kwargs of given called function.
    You can use this instead "local()"
    got idea from https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
    """
    frame = inspect.stack()[1][0]
    varnames, _, _, values = inspect.getargvalues(frame)

    called_from_class_method = (varnames[0] == 'self')
    if called_from_class_method:
        varnames = varnames[1:]

    kwargs = {i: values[i] for i in varnames}
    return kwargs


def inclusive_range(min, max, step=1):
    return range(min, max + step, step)


def is_iterable(object):
    # from https://stackoverflow.com/a/4668647
    from collections import Iterable
    return isinstance(object, Iterable)


def prev_curr(iterable):
    # from https://docs.python.org/3/library/itertools.html#itertools-recipes
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

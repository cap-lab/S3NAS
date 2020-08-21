import datetime
import os
import subprocess
import sys

from util.io_utils import tf_open_file_in_path


def log_verbose(folder_path, name_prefix=""):
    verbose_filename = name_prefix + 'verbose'
    gitdiff_filename = name_prefix + 'gitdiff'
    f = tf_open_file_in_path(folder_path, verbose_filename, "w")
    f.write(" ".join(sys.argv) + '\n')
    f.write(get_flags() + '\n')

    write_git_revision(f)
    f.close()

    f = tf_open_file_in_path(folder_path, gitdiff_filename, "wb")
    write_gitdiff(f)
    f.close()


def write_git_revision(opened_file):
    git_tag = _try_getting_output_or_blank(["git", "describe", "--tags", "--exact-match"])
    git_branch = _try_getting_output_or_blank(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    git_revision = _try_getting_output_or_blank(["git", "rev-parse", "HEAD"])
    opened_file.write(git_tag + git_branch + git_revision)


def write_gitdiff(opened_binary_file):
    git_diff = str.encode(_try_getting_output_or_blank(["git", "diff", "--ignore-space-at-eol"]))
    opened_binary_file.write(git_diff)


def _try_getting_output_or_blank(command):
    try:
        output = subprocess.check_output(command, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = ""
    return output


def get_starttime():
    if not hasattr(get_starttime, "starttime"):
        now = datetime.datetime.now()
        get_starttime.starttime = now.strftime('%y-%m-%d.%H.%M.%S')
    return get_starttime.starttime


def init_tflog(save_path, use_tpu=False):
    """
    Helper function to log detailed settings in a experiment.
    - it creates a path named current time in the save_path
    - it creates a symbolic link in it. It's called 'latest'. This makes it easier to cd in latest experiment.
    - (if you use gpus I assume you work on local.) it makes a local log file and let tensorflow to log in that file.
    """
    base_path = os.path.join(save_path, get_starttime())

    log_verbose(folder_path=base_path)

    if not use_tpu:
        # couldn't find method to log file on cloud...
        import logging
        link_path_to_savepath = os.path.join(save_path, "latest")
        if os.path.islink(link_path_to_savepath):
            os.remove(link_path_to_savepath)
        os.symlink(get_starttime(), link_path_to_savepath)

        log_filename = os.path.join(base_path, 'log')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        from tensorflow.python.platform.tf_logging import get_logger
        get_logger().addHandler(file_handler)


def get_flags(format='str', keys=None):
    from absl import flags
    FLAGS = flags.FLAGS
    flagdict = FLAGS.__dict__['__flags']
    resdict = {}
    if keys is None:
        for key in flagdict.keys():
            resdict[key] = getattr(FLAGS, key)
    else:
        for key in keys:
            resdict[key] = getattr(FLAGS, key)

    if format == 'str':
        result = str(sorted(resdict.items()))
    elif format == 'dict':
        result = resdict
    else:
        raise NotImplementedError

    return result

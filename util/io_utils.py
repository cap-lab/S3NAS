# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import os

import tensorflow as tf

from util.string_utils import natural_keys
from util.utils import AttrDict


def tf_open_file_in_path(path, filename_in_path, mode='w'):
    """
    Automatically creates path if the path doesn't exist,
    then open filename_in_path with mode,
    and write content
    """
    filepath = os.path.join(path, filename_in_path)
    if not tf.gfile.Exists(filepath):
        if not tf.gfile.Exists(path):
            tf.gfile.MakeDirs(path)
    return tf.gfile.GFile(filepath, mode)


def load_json_as_attrdict(json_file):
    return json.load(tf_open_file_in_path("", json_file, "r"), object_pairs_hook=AttrDict)


### Code below originated from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py
### Modified by thnkinbtfly
def archive_ckpt(ckpt_eval_result_dict, ckpt_objective, ckpt_dir, keep_archives=2):
    """Archive a checkpoint and ckpt before, if the metric is better."""
    archive_dir = 'archive'
    archive_oldest_available_dir = 'archive_oldest_available'

    saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')

    if not check_is_improved(ckpt_objective, saved_objective_path):
        return False

    all_ckpts_available = get_ckpt_old_to_new(ckpt_dir)
    latest_ckpt = all_ckpts_available[-1]
    if not update_one_ckpt_and_remove_old_ones(latest_ckpt, os.path.join(ckpt_dir, archive_dir),
                                               keep_archives, ckpt_eval_result_dict):
        return False

    oldest_ckpt_available = all_ckpts_available[0]
    if not update_one_ckpt_and_remove_old_ones(oldest_ckpt_available,
                                               os.path.join(ckpt_dir, archive_oldest_available_dir),
                                               keep_archives):
        return False

    # Update the best objective.
    with tf.gfile.GFile(saved_objective_path, 'w') as f:
        f.write('%f' % ckpt_objective)

    return True


def check_is_improved(ckpt_objective, saved_objective_path):
    saved_objective = float('-inf')
    if tf.gfile.Exists(saved_objective_path):
        with tf.gfile.GFile(saved_objective_path, 'r') as f:
            saved_objective = float(f.read())
    if saved_objective > ckpt_objective:
        tf.logging.info('Ckpt %s is worse than %s', ckpt_objective, saved_objective)
        return False
    else:
        return True


def get_ckpt_old_to_new(target_dir):
    """Returns ckpt names from newest to oldest. Returns [] if nothing exists"""
    prev_ckpt_state = tf.train.get_checkpoint_state(target_dir)
    all_ckpts = []
    if prev_ckpt_state:
        all_ckpts = sorted(prev_ckpt_state.all_model_checkpoint_paths, key=natural_keys, reverse=False)
        tf.logging.info('got all_model_ckpt_paths %s' % str(all_ckpts))
    return all_ckpts


def update_one_ckpt_and_remove_old_ones(ckpt_name_path, dst_dir, num_want_to_keep_ckpts, ckpt_eval_result_dict=""):
    """
    :param ckpt_eval_result_dict: provide a evaluation informations if you want to write there.
    """
    filenames = tf.gfile.Glob(ckpt_name_path + '.*')
    if filenames is None:
        tf.logging.info('No files to copy for checkpoint %s', ckpt_name_path)
        return False

    tf.gfile.MakeDirs(dst_dir)

    num_want_to_keep_prev_ckpts = num_want_to_keep_ckpts - 1
    remaining_ckpts = remove_old_ckpts_and_get_remaining_names(
        dst_dir, num_want_to_keep_ckpts=num_want_to_keep_prev_ckpts)

    write_ckpts(ckpt_name_path, dst_dir, remaining_ckpts)

    if ckpt_eval_result_dict:
        with tf.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
            f.write('%s' % ckpt_eval_result_dict)

    return True


def remove_old_ckpts_and_get_remaining_names(dst_dir, num_want_to_keep_ckpts):
    # Remove old ckpt files. get_checkpoint_state returns absolute path. refer to
    # https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/2843a7867d51c2cf065b85899ea0b9564e4d9db9
    all_ckpts = get_ckpt_old_to_new(dst_dir)
    if all_ckpts:
        want_to_rm_ckpts = all_ckpts[:-num_want_to_keep_ckpts]
        for want_to_rm_ckpt in want_to_rm_ckpts:
            want_to_rm = tf.gfile.Glob(want_to_rm_ckpt + "*")
            for f in want_to_rm:
                tf.logging.info('Removing checkpoint %s', f)
                tf.gfile.Remove(f)
        remaining_ckpts = all_ckpts[-num_want_to_keep_ckpts:]
    else:
        remaining_ckpts = []

    return remaining_ckpts


def write_ckpts(ckpt_path, dst_dir, remaining_ckpts):
    filenames = tf.gfile.Glob(ckpt_path + '.*')
    tf.logging.info('Copying checkpoint %s to %s', ckpt_path, dst_dir)
    for f in filenames:
        dest = os.path.join(dst_dir, os.path.basename(f))
        tf.gfile.Copy(f, dest, overwrite=True)

    ckpt_state = tf.train.generate_checkpoint_state_proto(
        dst_dir,
        model_checkpoint_path=ckpt_path,
        all_model_checkpoint_paths=remaining_ckpts)
    with tf.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
        str_ckpt_state = str(ckpt_state)
        str_ckpt_state = str_ckpt_state.replace('../', '')
        tf.logging.info('str_ckpt_state %s' % str_ckpt_state)
        f.write(str_ckpt_state)

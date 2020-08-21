import json
import os
import re


def natural_keys(text):
    # from https://stackoverflow.com/a/5967539
    return [atoi(c) for c in re.split('(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text


def get_filename(file, remove_ext=True):
    name = os.path.split(file)[1]
    if remove_ext:
        return rm_ext(name)
    return name


def rm_ext(filename):
    filename = os.path.splitext(filename)[0]
    return filename


def add_pre_suf_to_keys_of(dict, prefix="", suffix=""):
    result = {}
    for name, value in dict.items():
        result[str(prefix + name + suffix)] = value
    return result


def grab_str_between_pre_suf(string, prefix, suffix):
    assert isinstance(string, str)
    left = string.find(prefix)
    if left < 0:
        return ''
    left += len(prefix)
    right = string.find(suffix, left)
    if right < 0:
        return ''
    return string[left:right]


def is_json_same(json1, json2):
    # got idea from https://stackoverflow.com/a/38722500
    return load_json_with_sorted_keys(json1) == load_json_with_sorted_keys(json2)


def load_json_with_sorted_keys(jsonfile):
    loaded_json = json.load(open(jsonfile, "r"))
    return json.dumps(loaded_json, sort_keys=True, indent=4)

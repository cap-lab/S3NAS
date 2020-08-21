from collections import OrderedDict
from itertools import zip_longest

import pandas as pd


class CSVMaker(object):
    def __init__(self, columns):
        self.columns = columns
        self.data = OrderedDict()
        for column in columns:
            self.data[column] = []

    def fill_row(self, val_list):
        for column, val in zip_longest(self.columns, val_list):
            if val is None:
                val = ""
            if column is None:
                break
            self.data[column].append(val)


    def add_row(self, dict_with_columns_key):
        """Add row, which automatically fills blank columns by empty string"""
        for column in self.columns:
            val = dict_with_columns_key.get(column)
            if val is None:
                val = ""
            self.data[column].append(val)

    def add_rows(self, dict_with_columns_key):
        """Add rows, which automatically fills blank columns by empty string"""
        max_length = 0
        for vals in dict_with_columns_key.values():
            if len(vals) > max_length:
                max_length = len(vals)

        for column in self.columns:
            vals = dict_with_columns_key.get(column)
            vals.extend([""] * (max_length - len(vals)))  # Enforce all columns to have same length
            self.data[column].extend(vals)

    def save(self, csv_file_name):
        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(csv_file_name)
